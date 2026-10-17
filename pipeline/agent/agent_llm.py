"""Pipeline Agent 用的"会调用工具的模型"。

- `ArkChat`:方舟 OpenAI 兼容的 chat/completions + function calling(默认 doubao-seed-2.1-pro)。
  密钥只读环境变量 ARK_API_KEY;用量原样取接口返回的 `usage`(含 reasoning_tokens / cached_tokens)。
- `ScriptedAgentLLM`:确定性的**脚本化策略**,没有任何理解能力,只用于在离线/CI 下验证图、工具、checkpoint 与记账的流程。
  它的 token 是按字符数估算的(`estimated=True`),不能当真实模型的用量或决策质量。

两者输入输出都是 LangChain 的 message 对象(LangGraph 的 `messages` state 里存的就是它们)。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Protocol

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

DEFAULT_AGENT_MODEL = "doubao-seed-2.1-pro"
BASE = "https://ark.cn-beijing.volces.com"
PREFIX = {"ark": "/api/v3", "ark_plan": "/api/plan/v3"}


class AgentLLMError(Exception):
    def __init__(self, msg: str, *, fatal: bool = False):
        super().__init__(msg)
        self.fatal = fatal


class AgentLLM(Protocol):
    provider: str
    model: str

    def chat(self, messages: list[BaseMessage], tools: list[dict]) -> AIMessage: ...


# ----------------------------------------------------------------------------- 消息转换
def to_openai(messages: list[BaseMessage]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            d: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                d["tool_calls"] = [{"id": tc["id"], "type": "function",
                                    "function": {"name": tc["name"], "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                                   for tc in m.tool_calls]
            out.append(d)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        else:
            raise TypeError(f"不支持的消息类型 {type(m)}")
    return out


def from_openai(resp: dict, latency_s: float, model: str) -> AIMessage:
    ch = resp["choices"][0]
    msg = ch["message"]
    calls = []
    for tc in msg.get("tool_calls") or []:
        raw = tc["function"].get("arguments") or "{}"
        try:
            args = json.loads(raw)
            if not isinstance(args, dict):
                raise ValueError("arguments 不是对象")
        except ValueError:
            args = {"__invalid_json__": raw[:300]}     # 交给工具节点回报"参数不是合法 JSON",让模型自己修正
        calls.append({"id": tc["id"], "name": tc["function"]["name"], "args": args, "type": "tool_call"})
    u = resp.get("usage") or {}
    det = u.get("completion_tokens_details") or {}
    pdet = u.get("prompt_tokens_details") or {}
    usage = {"prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
             "total_tokens": u.get("total_tokens"), "reasoning_tokens": det.get("reasoning_tokens"),
             "cached_tokens": pdet.get("cached_tokens"), "estimated": False}
    return AIMessage(
        content=msg.get("content") or "", tool_calls=calls,
        additional_kwargs={"usage": usage, "latency_s": round(latency_s, 3), "model": model,
                           "finish_reason": ch.get("finish_reason"),
                           "reasoning": (msg.get("reasoning_content") or "")[:600]},
        usage_metadata={"input_tokens": u.get("prompt_tokens") or 0, "output_tokens": u.get("completion_tokens") or 0,
                        "total_tokens": u.get("total_tokens") or 0})


# ----------------------------------------------------------------------------- 真实模型
class ArkChat:
    provider = "ark"

    def __init__(self, model: str = DEFAULT_AGENT_MODEL, *, api: str = "ark_plan", api_key: str | None = None,
                 base_url: str = BASE, max_tokens: int = 4096, timeout: float = 120.0, max_retries: int = 3,
                 transport: httpx.BaseTransport | None = None, sleep=time.sleep):
        self.model = model
        self.provider = api
        self.url = f"{base_url.rstrip('/')}{PREFIX[api]}/chat/completions"
        self._key = api_key
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._sleep = sleep

    def chat(self, messages: list[BaseMessage], tools: list[dict]) -> AIMessage:
        key = self._key or os.environ.get("ARK_API_KEY")
        if not key:
            raise AgentLLMError("没有设置环境变量 ARK_API_KEY", fatal=True)
        body = {"model": self.model, "messages": to_openai(messages), "tools": tools,
                "max_tokens": self.max_tokens}
        last = ""
        for n in range(1, self.max_retries + 1):
            t0 = time.time()
            try:
                r = self._client.post(self.url, json=body, headers={"Authorization": f"Bearer {key}"})
            except httpx.HTTPError as e:
                last = f"{type(e).__name__}: {e}"
                self._sleep(min(2.0 * n, 10.0))
                continue
            if r.status_code == 200:
                return from_openai(r.json(), time.time() - t0, self.model)
            last = f"HTTP {r.status_code} {r.text[:300]}"
            if r.status_code in (401, 403):
                raise AgentLLMError(f"决策模型鉴权失败: {last}", fatal=True)
            if r.status_code in (400, 404):
                raise AgentLLMError(f"决策模型请求被拒绝(模型 ID/参数有误?): {last}", fatal=True)
            self._sleep(min(2.0 * n, 10.0))          # 429 / 5xx:退避重试
        raise AgentLLMError(f"决策模型调用失败(已重试 {self.max_retries} 次): {last}")


# ----------------------------------------------------------------------------- 脚本化策略(离线/CI)
def _est(s: str) -> int:
    return max(1, len(s) // 3)


class ScriptedAgentLLM:
    """按固定顺序调用工具的"假模型":用于验证编排流程,不代表任何决策能力。

    - 每块并行风格化、每块并行后处理;
    - 阶段失败时:诊断里 retry_allowed → retry_stage,否则 escalate_to_human;
    - `crash_after`:调用到第 N 轮之后抛出 KeyboardInterrupt,用来模拟 Agent 进程被杀(测试里用)。
    """
    provider = "scripted"
    model = "scripted-policy"

    def __init__(self, *, chunks: int = 2, stride: int = 3, max_gap: int = 8, crash_on_turn: int | None = None):
        self.chunks, self.stride, self.max_gap = chunks, stride, max_gap
        self.crash_on_turn = crash_on_turn
        self.calls = 0

    @staticmethod
    def _results(messages: list[BaseMessage]) -> list[tuple[str, dict]]:
        out = []
        for m in messages:
            if isinstance(m, ToolMessage):
                try:
                    out.append((m.name or "", json.loads(m.content)))
                except ValueError:
                    out.append((m.name or "", {}))
        return out

    def chat(self, messages: list[BaseMessage], tools: list[dict]) -> AIMessage:
        self.calls += 1
        turn = sum(isinstance(m, AIMessage) for m in messages) + 1
        if self.crash_on_turn and turn == self.crash_on_turn:
            self.crash_on_turn = None                    # 只崩一次:恢复后同一轮应当正常进行
            raise KeyboardInterrupt("simulated agent process kill")
        res = self._results(messages)
        done = {(n, r.get("stage_id")) for n, r in res if r.get("status") == "SUCCEEDED"}
        names = {n for n, _ in res}
        failed = [r for n, r in res if r.get("status") == "FAILED"]
        escalated = "escalate_to_human" in names

        def call(tool: str, /, **args) -> dict:
            return {"id": f"call_{uuid.uuid4().hex[:10]}", "name": tool, "args": args, "type": "tool_call"}

        def ok(stage: str) -> bool:
            return any(r.get("stage_id") == stage and r.get("status") == "SUCCEEDED" for _, r in res)

        calls: list[dict] = []
        text = ""
        last_failed = None
        for n, r in reversed(res):
            if r.get("status") == "FAILED" and not ok(r.get("stage_id", "")):
                last_failed = r
                break
        if escalated:
            text = "已标记需要人工处理,停止。"
        elif last_failed is not None:
            d = last_failed.get("diagnosis") or {}
            if d.get("retry_allowed"):
                calls = [call("retry_stage", stage_id=last_failed["stage_id"])]
            else:
                calls = [call("escalate_to_human", reason=f"{last_failed['stage_id']} 失败:{d.get('cause')}")]
        elif "load_skill" not in names:
            calls = [call("load_skill", name="anime-style-pipeline")]
        elif not ok("ingest"):
            calls = [call("ingest_video")]
        elif not ok("keyframes"):
            calls = [call("extract_keyframes", chunks=self.chunks, stride=self.stride, max_gap=self.max_gap)]
        else:
            n_chunks = next(r["chunks"] for _, r in res if r.get("stage_id") == "keyframes" and r.get("status") == "SUCCEEDED")
            todo = [i for i in range(n_chunks) if not ok(f"stylize:{i}")]
            if todo:
                calls = [call("stylize_chunk", chunk=i) for i in todo]          # 并行
            elif not ok("poststats"):
                calls = [call("poststats")]
            elif [i for i in range(n_chunks) if not ok(f"post:{i}")]:
                calls = [call("postprocess_chunk", chunk=i) for i in range(n_chunks) if not ok(f"post:{i}")]
            elif not ok("qc"):
                calls = [call("quality_check")]
            elif not ok("mux"):
                calls = [call("mux_audio")]
            elif not ok("package"):
                calls = [call("package_and_publish")]
            else:
                text = "流水线完成:成品已打包并发布。"
        content = text or "按流水线继续。"
        prompt = _est(json.dumps([getattr(m, "content", "") for m in messages], ensure_ascii=False))
        comp = _est(content + json.dumps(calls, ensure_ascii=False))
        usage = {"prompt_tokens": prompt, "completion_tokens": comp, "total_tokens": prompt + comp,
                 "reasoning_tokens": 0, "cached_tokens": 0, "estimated": True}
        return AIMessage(content=content, tool_calls=calls,
                         additional_kwargs={"usage": usage, "latency_s": 0.0, "model": self.model,
                                            "finish_reason": "tool_calls" if calls else "stop", "reasoning": ""},
                         usage_metadata={"input_tokens": prompt, "output_tokens": comp, "total_tokens": prompt + comp})
