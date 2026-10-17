"""阶段执行器:把"流水线的一个阶段"变成一个引擎任务,并等它结束。

为什么这样设计(而不是 Agent 自己在进程里跑步骤):
- 租约/心跳、fencing、重试退避、步骤缓存、逐帧账本、worker 崩溃恢复都在引擎里,而且有测试。
  Agent 层只做"决策 + 编排",不重复实现这些;Agent 进程被杀,正在跑的阶段仍由 worker 跑完。
- 一个阶段 = 一个小任务,里面的步骤代码与整张 DAG 一次提交时**完全相同**(参数、版本、依赖名),
  上游产物用 `an_import` 按 sha256 引入 → 步骤缓存 key 与整张 DAG 互通,看板的逐帧网格/用量面板照常工作。
- 提交带幂等键(线程 + 阶段 + 参数/上游产物的哈希):Agent 崩溃后重新执行同一个工具调用,
  拿到的是同一个任务(还在跑就继续等,已完成就立刻返回),不会重复提交、不会重复付费。

阶段 id:ingest / keyframes / stylize:<i> / poststats / post:<i> / qc / mux / package
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ..db import tx
from ..engine import Engine
from ..models import RetryPolicy, StepSpec, TaskSpec, canonical_json
from ..templates import frame_style_task
from . import runbook

TERMINAL = ("SUCCEEDED", "FAILED", "CANCELLED")

# 阶段 → 工具组(用于按阶段汇总 token);顺序即流水线顺序
GROUPS = ["plan", "ingest", "keyframes", "stylize", "poststats", "post", "qc", "mux", "package", "control"]


class RunInterrupted(BaseException):
    """用户请求中断。继承 BaseException:工具层的 `except Exception` 不会把它当成普通错误吞掉,
    LangGraph 会让它一路抛出,运行记为 INTERRUPTED,之后可从最近的 checkpoint 继续。"""


def interrupt_requested(data_dir: str | Path, thread_id: str) -> bool:
    eng = Engine(data_dir)
    try:
        r = eng.conn.execute("SELECT status FROM agent_runs WHERE thread_id=?", (thread_id,)).fetchone()
        return bool(r) and r["status"] == "INTERRUPTING"
    finally:
        eng.close()


class StageError(Exception):
    """阶段无法提交(上游未完成、参数不合法等)。信息会原样回给 LLM。"""


def stage_group(stage_id: str) -> str:
    return stage_id.split(":")[0]


def stage_keys(stage_id: str, cfg: dict) -> list[str]:
    g, _, n = stage_id.partition(":")
    if g == "ingest":
        return ["ingest"] + (["reference"] if cfg.get("reference") else [])
    if g in ("keyframes", "poststats", "qc", "mux"):
        return [g]
    if g == "stylize":
        return [f"stylize_{int(n)}"]
    if g == "post":
        return [f"post_{int(n)}"]
    if g == "package":
        return ["package", "publish"]
    raise StageError(f"未知阶段 {stage_id!r}")


def frame_kwargs(cfg: dict) -> dict:
    """运行配置 → frame_style_task 的参数(配置要能存进 JSON/checkpoint,所以 retry 以字典形式保存)。"""
    kw = {k: v for k, v in cfg.items() if k != "name"}
    if isinstance(kw.get("retry"), dict):
        kw["retry"] = RetryPolicy(**kw["retry"])
    return kw


def build_stage_spec(thread_id: str, stage_id: str, cfg: dict, plan: dict, produced: dict[str, dict],
                     *, frame_overrides: dict | None = None) -> TaskSpec:
    """从完整流水线 DAG 中切出一个阶段:选中的步骤原样保留,它们依赖的上游步骤换成 an_import 替身。"""
    chunks = int(plan.get("chunks", 1))
    kw = frame_kwargs(cfg)
    kw.update(chunks=chunks, stride=int(plan.get("stride", kw.get("stride", 6))),
              max_gap=int(plan.get("max_gap", kw.get("max_gap", 24))), require_approval=False)
    if frame_overrides:
        kw["frame_overrides"] = {**(kw.get("frame_overrides") or {}), **frame_overrides}
    full = frame_style_task(name=f"agent:{thread_id}/{stage_id}", **kw)
    by = {s.key: s for s in full.steps}
    g, _, n = stage_id.partition(":")
    if g in ("stylize", "post") and int(n) >= chunks:
        raise StageError(f"{stage_id}: chunk 序号超出范围(共 {chunks} 块)")
    keys = stage_keys(stage_id, cfg)
    sel = [by[k] for k in keys]
    need = sorted({d for s in sel for d in s.deps if d not in keys})
    imports: list[StepSpec] = []
    for d in need:
        outs = produced.get(d)
        if not outs:
            raise StageError(f"阶段 {stage_id} 依赖的上游 {d!r} 还没有产出,请先完成产生它的阶段")
        imports.append(StepSpec(
            key=d, type="an_import", deps=[], resource="cpu", retry=sel[0].retry, timeout_s=600.0,
            stall_timeout_s=120.0,
            params={"refs": {nm: {"sha256": o["sha256"], "size": o["size"]} for nm, o in outs.items()}}))
    spec = TaskSpec(name=f"agent:{thread_id}/{stage_id}", steps=imports + sel,
                    meta={**full.meta, "agent_thread": thread_id, "stage": stage_id})
    return spec


def _idem_key(thread_id: str, stage_id: str, spec: TaskSpec) -> str:
    h = hashlib.sha256(canonical_json({"s": [s.to_dict() for s in spec.steps]}).encode()).hexdigest()[:16]
    return f"agent:{thread_id}:{stage_id}:{h}"


def _loads(s: str | None) -> Any:
    return json.loads(s) if s else None


class StageRunner:
    """一个 Agent 线程的阶段执行器。每次调用自己开 Engine 连接(工具会被并行调用,sqlite 连接不跨线程共用)。"""

    def __init__(self, data_dir: str | Path, thread_id: str, cfg: dict, *, poll: float = 0.4,
                 timeout_s: float = 3600.0, clock=time.time):
        self.data_dir = Path(data_dir)
        self.thread_id = thread_id
        self.cfg = cfg
        self.poll = poll
        self.timeout_s = timeout_s
        self.clock = clock

    # ---------------------------------------------------------------- 状态读取
    def _rows(self, eng: Engine) -> dict[str, dict]:
        out = {}
        for r in eng.conn.execute("SELECT * FROM agent_stage_runs WHERE thread_id=?", (self.thread_id,)):
            out[r["stage_id"]] = {k: r[k] for k in r.keys()}
        return out

    def state(self) -> dict:
        eng = Engine(self.data_dir)
        try:
            rows = self._rows(eng)
        finally:
            eng.close()
        res = {}
        for sid, r in rows.items():
            res[sid] = {"status": r["status"], "task_id": r["task_id"], "retries": r["retries"],
                        "summary": _loads(r["summary"]), "usage": _loads(r["usage"]), "error": _loads(r["error"]),
                        "started_at": r["started_at"], "finished_at": r["finished_at"]}
        return res

    def _produced(self, rows: dict[str, dict]) -> dict[str, dict]:
        produced: dict[str, dict] = {}
        for r in rows.values():
            if r["status"] == "SUCCEEDED" and r["outputs"]:
                produced.update(json.loads(r["outputs"]))
        return produced

    def plan(self, eng: Engine) -> dict:
        r = self._rows(eng).get("keyframes")
        s = _loads(r["summary"]) if r else None
        return {"chunks": s["chunks"], "stride": s["stride"], "max_gap": s["max_gap"]} if s else {}

    # ---------------------------------------------------------------- 执行
    def run(self, stage_id: str, *, params: dict | None = None, frame_overrides: dict | None = None) -> dict:
        params = params or {}
        if interrupt_requested(self.data_dir, self.thread_id):
            raise RunInterrupted(f"用户请求中断,未提交阶段 {stage_id}")
        eng = Engine(self.data_dir)
        try:
            rows = self._rows(eng)
            if stage_id == "keyframes":
                plan = {"chunks": int(params.get("chunks", 4)), "stride": int(params.get("stride", 6)),
                        "max_gap": int(params.get("max_gap", 24))}
            else:
                plan = self.plan(eng)
                if not plan and stage_id != "ingest":
                    raise StageError("还没有关键帧阶段的结果,请先调用 extract_keyframes")
            spec = build_stage_spec(self.thread_id, stage_id, self.cfg, plan, self._produced(rows),
                                    frame_overrides=frame_overrides)
            idem = _idem_key(self.thread_id, stage_id, spec)
            task_id, created = eng.submit(spec, idempotency_key=idem)
            now = self.clock()
            with tx(eng.conn):
                old = eng.conn.execute("SELECT task_id, status FROM agent_stage_runs WHERE thread_id=? AND stage_id=?",
                                       (self.thread_id, stage_id)).fetchone()
                if old is None or old["task_id"] != task_id:       # 参数变了会得到新任务:旧记录被取代
                    eng.conn.execute(
                        "INSERT OR REPLACE INTO agent_stage_runs(thread_id,stage_id,task_id,status,retries,started_at) "
                        "VALUES (?,?,?,?,0,?)", (self.thread_id, stage_id, task_id, "RUNNING", now))
            return self._wait(eng, stage_id, task_id, plan, created)
        finally:
            eng.close()

    def retry(self, stage_id: str) -> dict:
        """重试一个失败的阶段。受处置手册约束:只有手册允许自动重试、且没超过次数时才会执行。"""
        eng = Engine(self.data_dir)
        try:
            r = self._rows(eng).get(stage_id)
            if r is None or r["status"] != "FAILED":
                raise StageError(f"{stage_id} 当前不是失败状态,不需要重试")
            err = _loads(r["error"]) or {}
            pb = runbook.match(err.get("error_kind"), err.get("error_msg"))
            if not pb.auto_retry:
                raise StageError(f"处置手册判定「{pb.title}」重试无益,不允许重试。请调用 escalate_to_human 交给人工;"
                                 f"人工步骤:{'; '.join(pb.human_steps)}")
            if r["retries"] >= pb.max_auto:
                raise StageError(f"{stage_id} 已重试 {r['retries']} 次仍失败(上限 {pb.max_auto}),请调用 escalate_to_human")
            eng.resume_task(r["task_id"])
            with tx(eng.conn):
                eng.conn.execute("UPDATE agent_stage_runs SET status='RUNNING', retries=retries+1, error=NULL "
                                 "WHERE thread_id=? AND stage_id=?", (self.thread_id, stage_id))
            return self._wait(eng, stage_id, r["task_id"], self.plan(eng), False)
        finally:
            eng.close()

    # ---------------------------------------------------------------- 等待与汇总
    def _wait(self, eng: Engine, stage_id: str, task_id: str, plan: dict, created: bool) -> dict:
        deadline = self.clock() + self.timeout_s
        while True:
            t = eng.get_task(task_id)
            if t["status"] in TERMINAL:
                break
            if interrupt_requested(self.data_dir, self.thread_id):
                raise RunInterrupted(f"用户在等待阶段 {stage_id} 时请求中断")
            if self.clock() > deadline:
                raise StageError(f"阶段 {stage_id} 等待超时({self.timeout_s:.0f}s);任务 {task_id} 仍在引擎里运行,可稍后重新调用同一工具继续等待")
            time.sleep(self.poll)
        now = self.clock()
        if t["status"] == "SUCCEEDED":
            keys = stage_keys(stage_id, self.cfg)
            outputs = {s["key"]: s["outputs"] for s in t["steps"] if s["key"] in keys}
            summary, usage = self._summarize(eng, stage_id, t, outputs, plan)
            with tx(eng.conn):
                eng.conn.execute(
                    "UPDATE agent_stage_runs SET status='SUCCEEDED', outputs=?, summary=?, usage=?, error=NULL, "
                    "finished_at=? WHERE thread_id=? AND stage_id=?",
                    (json.dumps(outputs), json.dumps(summary, ensure_ascii=False),
                     json.dumps(usage, ensure_ascii=False) if usage else None, now, self.thread_id, stage_id))
            return {"status": "SUCCEEDED", "stage_id": stage_id, "task_id": task_id, "reused_existing": not created,
                    "cached_steps": t["cached_steps"], **summary}
        failed = next((s for s in t["steps"] if s["status"] == "FAILED"), None)
        err = {"step": failed["key"] if failed else None, "error_kind": failed["error_kind"] if failed else None,
               "error_msg": (failed["error_msg"] or "")[:600] if failed else t["status"]}
        pb = runbook.match(err["error_kind"], err["error_msg"])
        used = (eng.conn.execute("SELECT retries FROM agent_stage_runs WHERE thread_id=? AND stage_id=?",
                                 (self.thread_id, stage_id)).fetchone() or {"retries": 0})["retries"]
        with tx(eng.conn):
            eng.conn.execute("UPDATE agent_stage_runs SET status='FAILED', error=?, finished_at=? "
                             "WHERE thread_id=? AND stage_id=?",
                             (json.dumps(err, ensure_ascii=False), now, self.thread_id, stage_id))
        return {"status": "FAILED", "stage_id": stage_id, "task_id": task_id, **err,
                "diagnosis": {"playbook": pb.code, "cause": pb.title,
                              "retry_allowed": pb.auto_retry and used < pb.max_auto,
                              "retries_used": used, "max_retries": pb.max_auto,
                              "human_steps": list(pb.human_steps)}}

    def _json_out(self, eng: Engine, outputs: dict, key: str, name: str) -> dict:
        sha = outputs[key][name]["sha256"]
        return json.loads(eng.store.path(sha).read_text(encoding="utf-8"))

    def _summarize(self, eng: Engine, stage_id: str, task: dict, outputs: dict, plan: dict) -> tuple[dict, dict | None]:
        g = stage_group(stage_id)
        cfg = self.cfg
        if g == "ingest":
            ing = self._json_out(eng, outputs, "ingest", "ingest.json")
            src = {k: ing["source"].get(k) for k in ("name", "width", "height", "fps", "duration_s", "has_audio")}
            return {"source": src, "standard": ing["standard"], "decode_error_lines": ing["decode_error_lines"],
                    "warnings": ing.get("warnings", [])}, None
        if g == "keyframes":
            tl = self._json_out(eng, outputs, "keyframes", "timeline.json")
            price = float(cfg.get("price_cny") or 0.12)
            return {"chunks": plan["chunks"], "stride": plan["stride"], "max_gap": plan["max_gap"],
                    "n_frames": tl["n_frames"], "n_keyframes": tl["n_keyframes"], "cuts": tl["cuts"],
                    "duplicates_skipped": tl["duplicates_skipped"],
                    "est_cost_cny": round(tl["n_keyframes"] * price, 2), "budget_cny": cfg.get("budget_cny")}, None
        if g == "stylize":
            u = self._json_out(eng, outputs, stage_id.replace(":", "_"), "usage.json")
            use = {k: u.get(k, 0) for k in ("calls", "billed", "maybe_billed", "output_tokens", "reused_output_tokens",
                                            "reused_images", "cost_cny_est", "cost_maybe_extra_cny_est")}
            use.update(model=u.get("model"), provider=u.get("provider"))
            step = next(s for s in task["steps"] if s["key"] == stage_id.replace(":", "_"))
            if step["cached"]:
                # 整块命中步骤缓存:usage.json 是当初生成时的用量,本次没有调用接口、没有花钱 → 全部算"复用"
                use.update(reused_output_tokens=use["output_tokens"] + use["reused_output_tokens"],
                           reused_images=use["billed"] + use["reused_images"], calls=0, billed=0, maybe_billed=0,
                           output_tokens=0, cost_cny_est=0.0, cost_maybe_extra_cny_est=0.0, step_cached=True)
            fr = u.get("frames", {})
            return {"chunk": int(stage_id.split(":")[1]), "frames": fr.get("total"), "done": fr.get("done"),
                    "cached_frames": fr.get("cached"), "resumed_frames": fr.get("resumed"),
                    "degraded": fr.get("degraded"), "maybe_duplicated": fr.get("maybe_duplicated"),
                    "seedream": use}, use
        if g == "poststats":
            return {"ok": True}, None
        if g == "post":
            return {"chunk": int(stage_id.split(":")[1]), "ok": True}, None
        if g == "qc":
            q = self._json_out(eng, outputs, "qc", "qc.json")
            return {"ok": q["ok"], "hard_failures": q["hard_failures"], "warnings": q["warnings"],
                    "flicker": q["flicker"], "structure": q.get("structure"),
                    "degraded_keyframes": q["degraded_keyframes"]}, None
        if g == "mux":
            m = self._json_out(eng, outputs, "mux", "mux.json")
            return {k: m[k] for k in ("frames", "duration_s", "width", "height", "fps", "av_drift_s", "audio", "size_mb")}, None
        if g == "package":
            meta = self._json_out(eng, outputs, "package", "metadata.json")
            return {"final_video": outputs["package"]["final.mp4"]["sha256"][:12] + "…",
                    "published": task["steps_succeeded"] == task["steps_total"],
                    "stylize_usage_lineage": meta["stylize"]}, None
        return {"ok": True}, None
