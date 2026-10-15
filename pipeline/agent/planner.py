"""Planner Agent:自然语言需求 → 结构化计划 → 护栏校验 → (失败则把错误回灌给 LLM 修复) → 展开为 DAG → 提交引擎。

设计要点:
1. LLM 只产出"阶段级计划"(用哪些步骤、谁依赖谁、几个业务参数),不碰重试/超时/chaos 等安全相关配置,
   也不决定分块如何展开 —— 这些由系统确定性地完成。
2. LLM 输出一律当作**不可信输入**:Pydantic 结构校验 + 白名单 + DAG 合法性 + 业务策略 + 预算,
   全部违规一次性反馈给模型,最多修复 N 轮,仍不合格则拒绝(REJECTED),不会"勉强执行"。
3. 计划(含 LLM 原始输出和每轮校验错误)落库;任务恢复只依赖库里的规格,**不会重新询问 LLM**
   (LLM 的输出本身也是需要固化的产物:再问一次结果可能不同)。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..db import tx
from ..engine import Engine
from ..models import RetryPolicy, StepSpec, TaskSpec, validate_spec
from .llm import TracedLLM

# ---------------------------------------------------------------------------- 阶段目录(白名单 + 依赖约束)
STAGE_CATALOG: dict[str, dict[str, Any]] = {
    "probe":          {"resource": "cpu",  "requires": [],                                  "desc": "探测并校验输入素材"},
    "extract_frames": {"resource": "cpu",  "requires": ["probe"],                           "desc": "拆帧"},
    "extract_audio":  {"resource": "cpu",  "requires": ["probe"],                           "desc": "提取音轨"},
    "gen_reference":  {"resource": "gpu",  "requires": ["probe"],                           "desc": "按描述生成新角色参考图(生成式,有随机性)"},
    "segment":        {"resource": "gpu",  "requires": ["extract_frames"],  "fanout": True,  "desc": "抠出原角色(按分块并行)"},
    "swap":           {"resource": "gpu",  "requires": ["segment", "gen_reference"], "fanout": True, "desc": "替换为新角色(按分块并行)"},
    "assemble":       {"resource": "cpu",  "requires": ["swap"],                            "desc": "合成视频"},
    "mux":            {"resource": "cpu",  "requires": ["assemble", "extract_audio"],       "desc": "音视频合流"},
    "quality_check":  {"resource": "cpu",  "requires": ["mux"],                             "desc": "质检"},
    "human_approval": {"resource": "none", "requires": ["quality_check"],                   "desc": "人工审批(不占用 worker)"},
    "publish":        {"resource": "cpu",  "requires": ["mux", "quality_check"],            "desc": "发布到素材库(有外部副作用)"},
}
ALLOWED_TARGETS = {"asset-library", "staging"}
PRODUCTION_TARGET = "asset-library"          # 策略:发布到生产库必须经过人工审批
MAX_CHUNKS = 16
MAX_FRAMES = 2000
MAX_GPU_STEPS = 40
SOURCE_RE = re.compile(r"^[\w\-./]{1,200}\.(mp4|mov|webm|mkv)$")


# ---------------------------------------------------------------------------- 计划结构(LLM 的输出格式)
class Stage(BaseModel):
    model_config = ConfigDict(extra="forbid")      # 禁止 LLM 夹带 chaos/timeout/retry 等字段
    id: str
    type: str
    deps: list[str] = Field(default_factory=list)


class PlanParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str
    frames: int = 24
    chunks: int = 4
    prompt: str = "new character"
    target: str = PRODUCTION_TARGET


class PlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = ""
    params: PlanParams
    stages: list[Stage]


def catalog_prompt() -> str:
    return "\n".join(
        f"- {t}: {c['desc']};资源 {c['resource']};必须依赖 {c['requires'] or '无'}"
        for t, c in STAGE_CATALOG.items())


SYSTEM_PROMPT = """[[PLANNER]]
你是动画素材处理流水线的规划器。把用户需求转成**阶段级计划**,只输出一个 JSON 对象,不要任何解释文字。

JSON 格式:
{"summary": "一句话说明", "params": {"source": "xxx.mp4", "frames": 整数, "chunks": 整数, "prompt": "新角色描述", "target": "asset-library 或 staging"},
 "stages": [{"id": "<与 type 相同>", "type": "<阶段类型>", "deps": ["<依赖的阶段 id>"]}]}

可用阶段类型(每种至多出现一次,id 必须等于 type):
%s

规则(系统会严格校验,违反则会被退回要求修正):
1. 只能使用上面列出的阶段类型;deps 必须包含该阶段的"必须依赖",且不能成环。
2. segment / swap 会由系统按 chunks 自动展开为并行子步骤,你不需要自己展开。
3. chunks 范围 1~%d,frames 不超过 %d 且不小于 chunks。
4. target 只能是 %s;发布到 asset-library(生产库)必须包含 human_approval 阶段,且 publish 依赖它。
5. 不要输出 chaos / timeout / retry 等字段,这些由系统决定。
""" % (catalog_prompt(), MAX_CHUNKS, MAX_FRAMES, sorted(ALLOWED_TARGETS))


# ---------------------------------------------------------------------------- 校验
def parse_json_loose(text: str) -> Any:
    """容忍 ```json 围栏和前后多余文字,取第一个完整的 JSON 对象。"""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except ValueError:
        pass
    start = t.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(t)):
            depth += (t[i] == "{") - (t[i] == "}")
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise ValueError("找不到 JSON 对象")


def validate_plan(raw: Any) -> tuple[PlanModel | None, list[str]]:
    """返回 (计划, 错误列表)。错误一次性收集完,方便整体反馈给 LLM。"""
    try:
        plan = PlanModel.model_validate(raw)
    except ValidationError as e:
        return None, [f"结构错误 {'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()]
    errs: list[str] = []
    types = [s.type for s in plan.stages]
    ids = [s.id for s in plan.stages]
    for s in plan.stages:
        if s.type not in STAGE_CATALOG:
            errs.append(f"未知阶段类型 {s.type!r},允许: {sorted(STAGE_CATALOG)}")
        if s.id != s.type:
            errs.append(f"阶段 id 必须等于 type: {s.id!r} != {s.type!r}")
    for t in set(types):
        if types.count(t) > 1:
            errs.append(f"阶段类型 {t} 出现了多次,每种至多一次")
    if "probe" not in types:
        errs.append("必须包含 probe 阶段(先校验输入再消耗 GPU)")
    idset = set(ids)
    by_id = {s.id: s for s in plan.stages}
    for s in plan.stages:
        for d in s.deps:
            if d not in idset:
                errs.append(f"阶段 {s.id} 依赖了不存在的阶段 {d!r}")
        cat = STAGE_CATALOG.get(s.type)
        if cat:
            missing = [r for r in cat["requires"] if r not in s.deps]
            if missing:
                errs.append(f"阶段 {s.id} 缺少必须依赖 {missing}")
    # 环检测
    if idset:
        indeg = {i: len(set(d for d in by_id[i].deps if d in idset)) for i in idset}
        ch: dict[str, list[str]] = {i: [] for i in idset}
        for i in idset:
            for d in set(by_id[i].deps):
                if d in idset:
                    ch[d].append(i)
        q = [i for i, n in indeg.items() if n == 0]
        seen = 0
        while q:
            x = q.pop()
            seen += 1
            for c in ch[x]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    q.append(c)
        if seen != len(idset):
            errs.append(f"依赖关系存在环,涉及: {sorted(i for i, n in indeg.items() if n > 0)}")
    p = plan.params
    if not SOURCE_RE.match(p.source) or ".." in p.source or p.source.startswith("/"):
        errs.append(f"source 不合法(只允许相对路径的 mp4/mov/webm/mkv 文件名): {p.source!r}")
    if not 1 <= p.chunks <= MAX_CHUNKS:
        errs.append(f"chunks 必须在 1~{MAX_CHUNKS},当前 {p.chunks}")
    if not 1 <= p.frames <= MAX_FRAMES:
        errs.append(f"frames 必须在 1~{MAX_FRAMES},当前 {p.frames}")
    if p.frames < p.chunks:
        errs.append(f"frames({p.frames}) 不能小于 chunks({p.chunks})")
    if len(p.prompt) > 200:
        errs.append("prompt 过长(>200)")
    if p.target not in ALLOWED_TARGETS:
        errs.append(f"target 只能是 {sorted(ALLOWED_TARGETS)},当前 {p.target!r}")
    gpu_steps = (types.count("segment") + types.count("swap")) * max(p.chunks, 0) + types.count("gen_reference")
    if gpu_steps > MAX_GPU_STEPS:
        errs.append(f"预计 GPU 步骤数 {gpu_steps} 超过预算 {MAX_GPU_STEPS}")
    if "publish" in types and p.target == PRODUCTION_TARGET:
        pub = by_id.get("publish")
        if "human_approval" not in types or (pub and "human_approval" not in pub.deps):
            errs.append("策略:发布到 asset-library(生产库)必须包含 human_approval 阶段,且 publish 依赖它")
    return (plan if not errs else None), errs


# ---------------------------------------------------------------------------- 展开为步骤级 DAG(确定性)
def expand_plan(plan: PlanModel, *, retry: RetryPolicy | None = None, unit_s: float = 0.5,
                stall_timeout_s: float = 30.0, timeout_s: float = 600.0, name: str = "planned-swap") -> TaskSpec:
    """阶段 → 步骤。fanout 阶段按 chunks 展开:
       fanout→fanout 依赖按块对应(segment_i → swap_i);非 fanout→fanout 则每块都依赖它;
       fanout→非 fanout(依赖方)则依赖全部块。"""
    retry = retry or RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=30.0, jitter=0.2)
    p = plan.params
    n = p.chunks
    per = max(1, p.frames // n)
    fan = {s.id for s in plan.stages if STAGE_CATALOG[s.type].get("fanout")}

    def keys_of(stage_id: str) -> list[str]:
        return [f"{stage_id}_{i}" for i in range(n)] if stage_id in fan else [stage_id]

    steps: list[StepSpec] = []
    for s in plan.stages:
        cat = STAGE_CATALOG[s.type]
        base = dict(type=s.type, resource=cat["resource"], retry=retry, timeout_s=timeout_s,
                    stall_timeout_s=stall_timeout_s)
        params: dict[str, Any]
        if s.type == "probe":
            params = {"source": p.source, "frames": p.frames}
        elif s.type == "gen_reference":
            params = {"prompt": p.prompt, "seed": 0, "nondeterministic": False}
        elif s.type == "quality_check":
            params = {"force_fail": False}
        elif s.type == "publish":
            params = {"target": p.target}
        else:
            params = {}
        for i, key in enumerate(keys_of(s.id)):
            deps: list[str] = []
            for d in s.deps:
                if s.id in fan and d in fan:
                    deps.append(f"{d}_{i}")            # 按块对应
                else:
                    deps.extend(keys_of(d))            # 全部块 / 单个步骤
            sp = dict(params)
            if s.id in fan:
                sp.update({"chunk": i, "frames": per, "unit_s": unit_s})
            steps.append(StepSpec(key=key, deps=deps, params=sp,
                                  cacheable=(s.type != "publish"), **base))
    spec = TaskSpec(name=name, steps=steps, meta={"template": "planned", "summary": plan.summary,
                                                  "chunks": n, "frames": p.frames})
    validate_spec(spec)
    return spec


# ---------------------------------------------------------------------------- Planner
@dataclass
class PlanResult:
    ok: bool
    plan_id: int
    rounds: int
    errors: list[str] = field(default_factory=list)
    task_id: str | None = None
    created: bool | None = None
    spec: TaskSpec | None = None
    attempts: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "plan_id": self.plan_id, "rounds": self.rounds, "errors": self.errors,
                "task_id": self.task_id, "created": self.created, "attempts": self.attempts,
                "steps": len(self.spec.steps) if self.spec else 0}


class Planner:
    def __init__(self, llm: TracedLLM, engine: Engine, max_rounds: int = 3, **expand_kw):
        self.llm = llm
        self.engine = engine
        self.max_rounds = max_rounds
        self.expand_kw = expand_kw

    def plan(self, request: str, *, submit: bool = True, idempotency_key: str | None = None) -> PlanResult:
        eng = self.engine
        with tx(eng.conn):
            plan_id = eng.conn.execute(
                "INSERT INTO plans(ts,request,status) VALUES (?,?,?)", (eng.clock(), request, "PLANNING")).lastrowid
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": request}]
        attempts: list[dict] = []
        plan: PlanModel | None = None
        errors: list[str] = []
        for rnd in range(self.max_rounds):
            res = self.llm.complete(messages, purpose="plan" if rnd == 0 else "plan_repair",
                                    plan_id=plan_id, json_mode=True)
            try:
                raw = parse_json_loose(res.text)
            except ValueError as e:
                plan, errors = None, [f"输出不是合法 JSON: {e}"]
            else:
                plan, errors = validate_plan(raw)
            attempts.append({"output": res.text, "errors": errors})
            if not errors:
                break
            messages += [
                {"role": "assistant", "content": res.text},
                {"role": "user", "content": "校验未通过,请修正后重新输出完整 JSON。错误:\n- " + "\n- ".join(errors)},
            ]
        result = PlanResult(ok=not errors, plan_id=plan_id, rounds=len(attempts), errors=errors, attempts=attempts)
        if errors or plan is None:
            self._finish(plan_id, "REJECTED", attempts, None, None)
            return result
        spec = expand_plan(plan, **self.expand_kw)
        result.spec = spec
        if submit:
            tid, created = eng.submit(spec, idempotency_key=idempotency_key, request=request, plan_id=plan_id)
            result.task_id, result.created = tid, created
            with tx(eng.conn):
                eng.conn.execute("UPDATE llm_calls SET task_id=? WHERE plan_id=?", (tid, plan_id))
        self._finish(plan_id, "OK", attempts, spec, result.task_id)
        return result

    def _finish(self, plan_id: int, status: str, attempts: list[dict], spec: TaskSpec | None,
                task_id: str | None) -> None:
        with tx(self.engine.conn):
            self.engine.conn.execute(
                "UPDATE plans SET status=?, attempts=?, spec=?, task_id=? WHERE id=?",
                (status, json.dumps(attempts, ensure_ascii=False),
                 json.dumps(spec.to_dict(), ensure_ascii=False) if spec else None, task_id, plan_id))


def list_plans(engine: Engine, limit: int = 50) -> list[dict]:
    return [{"id": r["id"], "ts": r["ts"], "request": r["request"], "status": r["status"],
             "task_id": r["task_id"], "rounds": len(json.loads(r["attempts"] or "[]"))}
            for r in engine.conn.execute("SELECT * FROM plans ORDER BY id DESC LIMIT ?", (limit,))]


def get_plan(engine: Engine, plan_id: int) -> dict | None:
    r = engine.conn.execute("SELECT * FROM plans WHERE id=?", (plan_id,)).fetchone()
    if r is None:
        return None
    return {"id": r["id"], "ts": r["ts"], "request": r["request"], "status": r["status"], "task_id": r["task_id"],
            "attempts": json.loads(r["attempts"] or "[]"), "spec": json.loads(r["spec"]) if r["spec"] else None}
