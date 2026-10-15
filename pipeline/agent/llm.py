"""LLM 客户端:统一接口 + OpenAI 兼容实现 + Mock + 调用追踪。

换模型只改环境变量(DeepSeek / 通义 / OpenAI 都兼容 chat.completions):
    LLM_PROVIDER=openai  OPENAI_BASE_URL=https://api.deepseek.com  OPENAI_API_KEY=sk-...  LLM_MODEL=deepseek-chat
没有 Key 时默认走 MockLLM(脚本化的确定性回复,只用于验证系统管线与护栏,**不代表真实模型的智能**)。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Protocol

from ..db import tx


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    estimated: bool          # True = 不是供应商返回的真实 token 用量
    model: str
    latency_s: float = 0.0


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(self, messages: list[dict], *, json_mode: bool = False,
                 temperature: float = 0.0) -> LLMResult: ...


class OpenAICompatClient:
    provider = "openai-compat"

    def __init__(self, base_url: str | None, api_key: str, model: str, *, http_client=None,
                 timeout: float = 60.0, max_retries: int = 2):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client,
                              timeout=timeout, max_retries=max_retries)

    def complete(self, messages: list[dict], *, json_mode: bool = False,
                 temperature: float = 0.0) -> LLMResult:
        kw: dict = {}
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        t0 = time.time()
        resp = self._client.chat.completions.create(model=self.model, messages=messages,
                                                    temperature=temperature, **kw)
        text = resp.choices[0].message.content or ""
        u = getattr(resp, "usage", None)
        if u is not None:
            return LLMResult(text, u.prompt_tokens, u.completion_tokens, False, self.model, time.time() - t0)
        approx = lambda s: max(1, len(s) // 3)            # noqa: E731  供应商没返回用量时的粗估
        return LLMResult(text, approx(json.dumps(messages, ensure_ascii=False)), approx(text), True,
                         self.model, time.time() - t0)


def make_llm() -> LLMClient:
    """按环境变量构造 LLM。没有 OPENAI_API_KEY 时回落到 MockLLM。"""
    provider = os.environ.get("LLM_PROVIDER") or ("openai" if os.environ.get("OPENAI_API_KEY") else "mock")
    if provider == "openai":
        return OpenAICompatClient(os.environ.get("OPENAI_BASE_URL"), os.environ["OPENAI_API_KEY"],
                                  os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    from .mock_llm import MockLLM
    return MockLLM()


class TracedLLM:
    """包装任意 LLMClient:每次调用把 prompt / 输出 / token / 耗时 / 成本写入 llm_calls 表。

    LLM 调用本身是"贵、慢、不确定"的外部副作用,必须可追溯:出问题时能回答
    "当时给了模型什么、它回了什么、花了多少"。
    """

    def __init__(self, inner: LLMClient, engine, price_in_per_1k: float | None = None,
                 price_out_per_1k: float | None = None):
        self.inner = inner
        self.engine = engine
        self.price_in = float(os.environ.get("LLM_PRICE_IN_PER_1K", 0)) if price_in_per_1k is None else price_in_per_1k
        self.price_out = float(os.environ.get("LLM_PRICE_OUT_PER_1K", 0)) if price_out_per_1k is None else price_out_per_1k

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(self, messages: list[dict], *, purpose: str, task_id: str | None = None,
                 plan_id: int | None = None, json_mode: bool = False, temperature: float = 0.0) -> LLMResult:
        eng = self.engine
        t0 = time.time()
        try:
            res = self.inner.complete(messages, json_mode=json_mode, temperature=temperature)
        except Exception as e:
            self._record(purpose, task_id, plan_id, messages, None, None, None, True, time.time() - t0,
                         "error", f"{type(e).__name__}: {e}")
            raise
        cost = res.prompt_tokens / 1000 * self.price_in + res.completion_tokens / 1000 * self.price_out
        self._record(purpose, task_id, plan_id, messages, res.text, res.prompt_tokens, res.completion_tokens,
                     res.estimated, res.latency_s or (time.time() - t0), "ok", None, cost)
        return res

    def _record(self, purpose, task_id, plan_id, messages, output, pt, ct, est, latency, status, error,
                cost=None) -> None:
        eng = self.engine
        with tx(eng.conn):
            eng.conn.execute(
                "INSERT INTO llm_calls(ts,task_id,plan_id,purpose,provider,model,messages,output,prompt_tokens,"
                "completion_tokens,tokens_estimated,latency_s,cost_usd,status,error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (eng.clock(), task_id, plan_id, purpose, self.inner.provider, self.inner.model,
                 json.dumps(messages, ensure_ascii=False), output, pt, ct, int(est), latency, cost, status, error))


# ---------------------------------------------------------------------------- 追踪查询
def list_llm_calls(engine, task_id: str | None = None, plan_id: int | None = None,
                   include_messages: bool = False, limit: int = 200) -> list[dict]:
    q, args = "SELECT * FROM llm_calls WHERE 1=1", []
    if task_id:
        q += " AND task_id=?"
        args.append(task_id)
    if plan_id is not None:
        q += " AND plan_id=?"
        args.append(plan_id)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    out = []
    for r in engine.conn.execute(q, args).fetchall():
        d = {k: r[k] for k in ("id", "ts", "task_id", "plan_id", "purpose", "provider", "model", "prompt_tokens",
                               "completion_tokens", "latency_s", "cost_usd", "status", "error")}
        d["tokens_estimated"] = bool(r["tokens_estimated"])
        d["output"] = r["output"]
        if include_messages:
            d["messages"] = json.loads(r["messages"])
        out.append(d)
    return out


def llm_usage_summary(engine, task_id: str | None = None) -> dict:
    q = ("SELECT COUNT(*) n, COALESCE(SUM(prompt_tokens),0) pt, COALESCE(SUM(completion_tokens),0) ct,"
         " COALESCE(SUM(cost_usd),0) cost, COALESCE(SUM(tokens_estimated),0) est,"
         " COALESCE(SUM(status='error'),0) errs, COALESCE(AVG(latency_s),0) lat FROM llm_calls")
    args: list = []
    if task_id:
        q += " WHERE task_id=?"
        args.append(task_id)
    r = engine.conn.execute(q, args).fetchone()
    by_purpose = {x["purpose"]: x["n"] for x in engine.conn.execute(
        "SELECT purpose, COUNT(*) n FROM llm_calls" + (" WHERE task_id=?" if task_id else "") + " GROUP BY purpose", args)}
    return {"calls": r["n"], "prompt_tokens": r["pt"], "completion_tokens": r["ct"],
            "cost_usd": round(r["cost"], 6), "estimated_calls": r["est"], "errors": r["errs"],
            "avg_latency_s": round(r["lat"], 3), "by_purpose": by_purpose}
