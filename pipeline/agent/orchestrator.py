"""Pipeline Agent:用 LangGraph 编排的"决策 + 工具调用"状态图。

    START ─► agent(决策模型:读 messages,返回工具调用)
                 │ 有 tool_calls                         │ 没有 → END(最终答复)
                 ▼
             tools(并行执行本轮全部工具调用;每个工具 = 提交一个阶段到引擎并等结果)
                 │ 升级给人 → END
                 └──────────► 回到 agent

- State:`messages`(add_messages 归并,模型与工具之间传递的所有消息)、`stages`(各阶段状态)、`cfg`、`turns`、`escalated`。
- Checkpoint:LangGraph `SqliteSaver`(数据目录下 agent_checkpoints.db)。每个节点执行完写一个 checkpoint;
  进程在任何时刻被杀,`resume(thread_id)` 从最后一个 checkpoint 继续:已完成的 LLM 轮次**不会重新调用**,
  正在执行的工具调用会重新执行,但它们是幂等的(引擎幂等提交 + 步骤缓存 + 逐帧账本),不会重复提交/付费。
- 两份持久化各管一件事:LangGraph checkpoint 存"Agent 推理进度与消息",引擎的数据库存"任务/步骤/产物/账本"。
  以引擎为准;Agent 恢复后会重新向引擎确认阶段状态,而不是相信 state 里的旧值。
- 可观测:`agent_turns`(每轮 LLM 的接口 usage)、`agent_tool_calls`、`agent_stage_runs`,汇总见 `report()`。
"""
from __future__ import annotations

import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from ..db import tx
from ..engine import Engine
from ..templates import frame_style_task
from .agent_llm import AgentLLM
from .pipeline_tools import PipelineToolRegistry, ToolContext, build_tools, compact
from .stages import GROUPS, RunInterrupted, StageRunner, frame_kwargs, interrupt_requested

SYSTEM_PROMPT = """你是视频风格化流水线的决策 Agent。你通过工具(function calling)驱动一条 6 道工序的流水线,把用户的视频逐帧做风格化/人物替换,再合成成品。

工作方式:
1. 先调用 load_skill 读取 "anime-style-pipeline" 技能,严格按其中的阶段顺序、决策规则和失败处置办事。
2. 每一步只依据工具返回的真实结果决策,不要编造数字或假设某阶段已完成。
3. 彼此独立的块(stylize_chunk / postprocess_chunk)必须在同一轮里并行调用,不要一块一轮。
4. 阶段失败时:看返回里的 diagnosis。retry_allowed 为 true 才可以 retry_stage;否则立刻调用 escalate_to_human,不要反复尝试。
5. 质检未通过(ok=false)不得发布;能通过调整参数解决的可以说明理由后重新规划,否则 escalate_to_human。
6. 全部完成后用中文简要汇报:成品情况、关键帧数、图像模型调用与 token、质检结论、遇到的问题。

运行配置(固定,不可更改):
{cfg}
"""


def _merge(a: dict | None, b: dict | None) -> dict:
    return {**(a or {}), **(b or {})}


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    cfg: dict
    stages: Annotated[dict, _merge]
    turns: int
    escalated: dict | None


def _cfg_for_prompt(cfg: dict) -> str:
    keep = ("prompt", "provider", "model", "budget_cny", "price_cny", "concurrency", "interp", "size", "on_reject")
    d = {k: cfg.get(k) for k in keep if k in cfg}
    d["source"] = Path(cfg["source"]).name
    return json.dumps(d, ensure_ascii=False)


class PipelineAgent:
    def __init__(self, data_dir: str | Path, llm: AgentLLM, *, max_turns: int = 40, poll: float = 0.4,
                 stage_timeout_s: float = 3600.0, max_parallel: int = 8, clock=time.time):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.llm = llm
        self.max_turns = max_turns
        self.poll = poll
        self.stage_timeout_s = stage_timeout_s
        self.max_parallel = max_parallel
        self.clock = clock
        Engine(self.data_dir).close()                                  # 确保表存在
        self._ck = sqlite3.connect(str(self.data_dir / "agent_checkpoints.db"), check_same_thread=False)
        self.saver = SqliteSaver(self._ck)
        self.graph = self._build().compile(checkpointer=self.saver)

    def close(self) -> None:
        self._ck.close()

    # ------------------------------------------------------------------ 数据库小工具
    def _conn(self) -> Engine:
        return Engine(self.data_dir)

    def _registry(self, thread_id: str, cfg: dict) -> PipelineToolRegistry:
        runner = StageRunner(self.data_dir, thread_id, cfg, poll=self.poll, timeout_s=self.stage_timeout_s,
                             clock=self.clock)
        return PipelineToolRegistry(ToolContext(self.data_dir, thread_id, cfg, runner))

    # ------------------------------------------------------------------ 图
    def _build(self) -> StateGraph:
        specs = PipelineToolRegistry(ToolContext(self.data_dir, "_", {"source": "x"}, None)).specs()   # type: ignore[arg-type]

        def agent_node(state: AgentState, config) -> dict:
            tid = config["configurable"]["thread_id"]
            turn = int(state.get("turns", 0)) + 1
            if turn > self.max_turns:
                return {"messages": [AIMessage(content=f"已达到最大决策轮数 {self.max_turns},停止。")], "turns": turn}
            if interrupt_requested(self.data_dir, tid):          # 轮与轮之间检查:不再发起新的模型调用
                raise RunInterrupted("用户请求中断,未发起新一轮决策")
            t0 = self.clock()
            ai = self.llm.chat(state["messages"], specs)
            self._record_turn(tid, turn, ai, self.clock() - t0)
            return {"messages": [ai], "turns": turn}

        def route_agent(state: AgentState) -> str:
            last = state["messages"][-1]
            return "tools" if isinstance(last, AIMessage) and last.tool_calls else END

        def tools_node(state: AgentState, config) -> dict:
            tid = config["configurable"]["thread_id"]
            ai = state["messages"][-1]
            reg = self._registry(tid, state["cfg"])
            turn = int(state.get("turns", 0))
            calls = list(ai.tool_calls)
            for c in calls:
                self._tool_start(tid, turn, c)

            def run(c: dict) -> dict:
                r = reg.call(c["name"], c["args"])
                self._tool_end(tid, c["id"], r)
                return r

            with ThreadPoolExecutor(max_workers=min(len(calls), self.max_parallel) or 1) as ex:
                futs = [ex.submit(run, c) for c in calls]
                results = [f.result() for f in futs]               # 进程级中断(KeyboardInterrupt)在这里原样抛出

            msgs, stages, esc = [], {}, None
            for c, r in zip(calls, results):
                msgs.append(ToolMessage(content=compact(r), tool_call_id=c["id"], name=c["name"]))
                if r.get("stage_id") and r.get("status") in ("SUCCEEDED", "FAILED"):
                    stages[r["stage_id"]] = {"status": r["status"], "task_id": r.get("task_id")}
                if r.get("status") == "ESCALATED" and esc is None:
                    esc = {k: r.get(k) for k in ("reason", "stage_id", "human_steps")}
            out: dict[str, Any] = {"messages": msgs, "stages": stages}
            if esc:
                out["escalated"] = esc
            return out

        def route_tools(state: AgentState) -> str:
            return END if state.get("escalated") else "agent"

        g = StateGraph(AgentState)
        g.add_node("agent", agent_node)
        g.add_node("tools", tools_node)
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", route_agent, {"tools": "tools", END: END})
        g.add_conditional_edges("tools", route_tools, {"agent": "agent", END: END})
        return g

    # ------------------------------------------------------------------ 记账
    def _record_turn(self, tid: str, turn: int, ai: AIMessage, latency: float) -> None:
        kw = ai.additional_kwargs or {}
        u = kw.get("usage") or {}
        calls = [{"id": c["id"], "name": c["name"], "args": c["args"]} for c in ai.tool_calls]
        e = self._conn()
        try:
            with tx(e.conn):
                e.conn.execute(
                    "INSERT OR REPLACE INTO agent_turns(thread_id,turn,model,prompt_tokens,completion_tokens,"
                    "reasoning_tokens,cached_tokens,estimated,latency_s,content,reasoning,tool_calls,ts) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (tid, turn, kw.get("model"), u.get("prompt_tokens"), u.get("completion_tokens"),
                     u.get("reasoning_tokens"), u.get("cached_tokens"), int(bool(u.get("estimated"))),
                     kw.get("latency_s", round(latency, 3)), (ai.content or "")[:1500], kw.get("reasoning"),
                     json.dumps(calls, ensure_ascii=False), self.clock()))
        finally:
            e.close()

    def _tool_start(self, tid: str, turn: int, c: dict) -> None:
        e = self._conn()
        try:
            with tx(e.conn):
                e.conn.execute(
                    "INSERT OR REPLACE INTO agent_tool_calls(thread_id,call_id,turn,name,args,status,started_at) "
                    "VALUES (?,?,?,?,?,'RUNNING',?)",
                    (tid, c["id"], turn, c["name"], json.dumps(c["args"], ensure_ascii=False), self.clock()))
        finally:
            e.close()

    def _tool_end(self, tid: str, call_id: str, r: dict) -> None:
        st = {"SUCCEEDED": "OK", "OK": "OK", "FAILED": "FAILED", "ESCALATED": "OK"}.get(r.get("status"), "ERROR")
        brief = {k: v for k, v in r.items() if k not in ("content", "stylize_usage_lineage")}
        e = self._conn()
        try:
            with tx(e.conn):
                e.conn.execute("UPDATE agent_tool_calls SET status=?, result=?, finished_at=? "
                               "WHERE thread_id=? AND call_id=?",
                               (st, json.dumps(brief, ensure_ascii=False, default=str)[:3000], self.clock(), tid, call_id))
        finally:
            e.close()

    def _set_run(self, tid: str, status: str, *, final: str | None = None, error: str | None = None) -> None:
        e = self._conn()
        try:
            with tx(e.conn):
                e.conn.execute("UPDATE agent_runs SET status=?, final=COALESCE(?,final), error=?, updated_at=? "
                               "WHERE thread_id=?", (status, final, error, self.clock(), tid))
        finally:
            e.close()

    # ------------------------------------------------------------------ 对外
    def start(self, thread_id: str, cfg: dict, request: str) -> dict:
        frame_style_task(name="validate", chunks=1, **frame_kwargs(cfg))   # 参数不合法在这里就报错
        e = self._conn()
        try:
            with tx(e.conn):
                if e.conn.execute("SELECT 1 FROM agent_runs WHERE thread_id=?", (thread_id,)).fetchone():
                    raise ValueError(f"thread {thread_id} 已存在,请用 resume 继续")
                now = self.clock()
                e.conn.execute("INSERT INTO agent_runs(thread_id,status,agent_model,config,request,created_at,updated_at) "
                               "VALUES (?,?,?,?,?,?,?)",
                               (thread_id, "RUNNING", self.llm.model, json.dumps(cfg, ensure_ascii=False), request, now, now))
        finally:
            e.close()
        init = {"messages": [SystemMessage(content=SYSTEM_PROMPT.format(cfg=_cfg_for_prompt(cfg))),
                             HumanMessage(content=request)],
                "cfg": cfg, "stages": {}, "turns": 0, "escalated": None}
        return self._drive(thread_id, init)

    def resume(self, thread_id: str) -> dict:
        e = self._conn()
        try:
            row = e.conn.execute("SELECT status FROM agent_runs WHERE thread_id=?", (thread_id,)).fetchone()
        finally:
            e.close()
        if row is None:
            raise KeyError(thread_id)
        self._set_run(thread_id, "RUNNING")
        return self._drive(thread_id, None)

    def _drive(self, thread_id: str, init: dict | None) -> dict:
        config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 200}
        try:
            out = self.graph.invoke(init, config)
        except BaseException as ex:                           # 含 KeyboardInterrupt:记为 INTERRUPTED,可 resume
            self._set_run(thread_id, "INTERRUPTED", error=f"{type(ex).__name__}: {ex}"[:500])
            raise
        stages = StageRunner(self.data_dir, thread_id, out["cfg"]).state()
        last = next((m for m in reversed(out["messages"]) if isinstance(m, AIMessage)), None)
        if out.get("escalated"):
            status = "NEEDS_HUMAN"
        elif stages.get("package", {}).get("status") == "SUCCEEDED":
            status = "SUCCEEDED"
        else:
            status = "INCOMPLETE"
        self._set_run(thread_id, status, final=(last.content if last else None))
        return {"thread_id": thread_id, "status": status, "final": last.content if last else None,
                "escalated": out.get("escalated"), "turns": out.get("turns")}

    # ------------------------------------------------------------------ 观测
    def snapshot(self, thread_id: str) -> dict:
        """LangGraph 里的当前状态:下一个节点、checkpoint 数、消息(精简)。"""
        config = {"configurable": {"thread_id": thread_id}}
        st = self.graph.get_state(config)
        vals = st.values or {}
        msgs = []
        for m in vals.get("messages", []):
            d: dict[str, Any] = {"role": {"SystemMessage": "system", "HumanMessage": "user", "AIMessage": "assistant",
                                          "ToolMessage": "tool"}.get(type(m).__name__, "?"),
                                 "content": (m.content if isinstance(m.content, str) else str(m.content))[:1200]}
            if isinstance(m, AIMessage) and m.tool_calls:
                d["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
            if isinstance(m, ToolMessage):
                d["name"] = m.name
                d["content"] = m.content[:600]
            msgs.append(d)
        n_ck = sum(1 for _ in self.saver.list(config))
        return {"next": list(st.next), "checkpoints": n_ck,
                "checkpoint_id": (st.config or {}).get("configurable", {}).get("checkpoint_id"),
                "step": (st.metadata or {}).get("step"), "turns": vals.get("turns", 0),
                "messages": msgs, "escalated": vals.get("escalated")}

    def report(self, thread_id: str) -> dict:
        return run_report(self.data_dir, thread_id)


# ============================================================================ 报告(只读,API 也用)
def _rows(conn, sql: str, *a) -> list[dict]:
    return [{k: r[k] for k in r.keys()} for r in conn.execute(sql, a).fetchall()]


def run_report(data_dir: str | Path, thread_id: str) -> dict:
    """按阶段汇总:LLM 决策 token(接口返回的 usage)+ 图像模型 token + 耗时。

    LLM 的 token 按"这一轮决策发出了哪些阶段的工具调用"归属;一轮里涉及多个阶段时平均分摊(近似,不是精确归因)。
    没有工具调用的轮次(最终汇报)归入 final。
    """
    e = Engine(data_dir)
    try:
        run = _rows(e.conn, "SELECT * FROM agent_runs WHERE thread_id=?", thread_id)
        if not run:
            raise KeyError(thread_id)
        run = run[0]
        turns = _rows(e.conn, "SELECT * FROM agent_turns WHERE thread_id=? ORDER BY turn", thread_id)
        calls = _rows(e.conn, "SELECT * FROM agent_tool_calls WHERE thread_id=? ORDER BY started_at", thread_id)
        stages = _rows(e.conn, "SELECT * FROM agent_stage_runs WHERE thread_id=? ORDER BY started_at", thread_id)
    finally:
        e.close()
    group_of = {t.name: t.group for t in build_tools()}
    order = GROUPS + ["final"]
    per = {g: {"group": g, "turns": 0, "tool_calls": 0, "tool_errors": 0, "llm_prompt": 0.0, "llm_completion": 0.0,
               "llm_reasoning": 0.0, "stages": [], "seedream": {"calls": 0, "billed": 0, "output_tokens": 0,
                                                                 "reused_output_tokens": 0, "cost_cny_est": 0.0},
               "wall_s": None} for g in order}
    estimated = False
    for t in turns:
        tc = json.loads(t["tool_calls"] or "[]")
        gs = sorted({group_of.get(c["name"], "control") for c in tc}, key=order.index) or ["final"]
        estimated = estimated or bool(t["estimated"])
        for g in gs:
            p = per[g]
            p["turns"] += 1 / len(gs)
            p["llm_prompt"] += (t["prompt_tokens"] or 0) / len(gs)
            p["llm_completion"] += (t["completion_tokens"] or 0) / len(gs)
            p["llm_reasoning"] += (t["reasoning_tokens"] or 0) / len(gs)
    for c in calls:
        p = per[group_of.get(c["name"], "control")]
        p["tool_calls"] += 1
        p["tool_errors"] += 1 if c["status"] in ("ERROR", "FAILED") else 0
    for s in stages:
        g = s["stage_id"].split(":")[0]
        p = per[g]
        p["stages"].append({"stage_id": s["stage_id"], "status": s["status"], "task_id": s["task_id"],
                            "retries": s["retries"], "duration_s": round((s["finished_at"] or 0) - (s["started_at"] or 0), 2)
                            if s["finished_at"] else None})
        u = json.loads(s["usage"]) if s["usage"] else None
        if u:
            for k in ("calls", "billed", "output_tokens", "reused_output_tokens"):
                p["seedream"][k] += u.get(k, 0) or 0
            p["seedream"]["cost_cny_est"] = round(p["seedream"]["cost_cny_est"] + (u.get("cost_cny_est") or 0), 4)
    for g in order:
        p = per[g]
        starts = [s["started_at"] for s in stages if s["stage_id"].split(":")[0] == g and s["started_at"]]
        ends = [s["finished_at"] for s in stages if s["stage_id"].split(":")[0] == g and s["finished_at"]]
        p["wall_s"] = round(max(ends) - min(starts), 2) if starts and ends and len(ends) == len(starts) else None
        for k in ("turns", "llm_prompt", "llm_completion", "llm_reasoning"):
            p[k] = round(p[k], 1)
    tot = {"llm_turns": len(turns),
           "llm_prompt": sum(t["prompt_tokens"] or 0 for t in turns),
           "llm_completion": sum(t["completion_tokens"] or 0 for t in turns),
           "llm_reasoning": sum(t["reasoning_tokens"] or 0 for t in turns),
           "llm_cached": sum(t["cached_tokens"] or 0 for t in turns),
           "llm_estimated": estimated,
           "seedream_calls": sum(p["seedream"]["calls"] for p in per.values()),
           "seedream_output_tokens": sum(p["seedream"]["output_tokens"] for p in per.values()),
           "seedream_reused_tokens": sum(p["seedream"]["reused_output_tokens"] for p in per.values()),
           "seedream_cost_cny_est": round(sum(p["seedream"]["cost_cny_est"] for p in per.values()), 4),
           "wall_s": round(run["updated_at"] - run["created_at"], 1)}
    return {"run": {**run, "config": json.loads(run["config"])},
            "stages": [per[g] for g in order if per[g]["turns"] or per[g]["tool_calls"] or per[g]["stages"]],
            "totals": tot,
            "turns": [{"turn": t["turn"], "model": t["model"], "prompt": t["prompt_tokens"],
                       "completion": t["completion_tokens"], "reasoning": t["reasoning_tokens"],
                       "latency_s": t["latency_s"], "estimated": bool(t["estimated"]),
                       "content": (t["content"] or "")[:300],
                       "tool_calls": [c["name"] + (f"({','.join(f'{k}={v}' for k, v in c['args'].items())})"
                                                   if c["args"] else "") for c in json.loads(t["tool_calls"] or "[]")]}
                      for t in turns],
            "tool_calls": [{"call_id": c["call_id"], "turn": c["turn"], "name": c["name"], "status": c["status"],
                            "args": json.loads(c["args"] or "{}"),
                            "duration_s": round((c["finished_at"] or 0) - (c["started_at"] or 0), 2) if c["finished_at"] else None}
                           for c in calls]}


# ============================================================================ 流程图(只读,API 也用)
NODE_LABELS = {"skill": "读取 skill", "ingest": "1 摄取", "keyframes": "2 关键帧", "poststats": "4a 统计",
               "qc": "5 质检", "mux": "5 合成音频", "package": "6 打包发布"}
_TOOL_STAGE = {"load_skill": "skill", "ingest_video": "ingest", "extract_keyframes": "keyframes",
               "poststats": "poststats", "quality_check": "qc", "mux_audio": "mux", "package_and_publish": "package"}


def _node_for_call(name: str, args: dict) -> str | None:
    if name in _TOOL_STAGE:
        return _TOOL_STAGE[name]
    if name == "stylize_chunk":
        return f"stylize:{args.get('chunk')}"
    if name == "postprocess_chunk":
        return f"post:{args.get('chunk')}"
    return None


def pipeline_graph(data_dir: str | Path | None, thread_id: str | None, alive: bool | None = None) -> dict:
    """流水线流程图:节点 = 阶段(风格化/后处理按分块展开),状态来自阶段记录与工具调用记录。

    thread_id 为 None 时返回"模板"(全部待执行),任务一开始就能先看到整条流程。
    状态:PENDING / RUNNING / SUCCEEDED / FAILED / INTERRUPTED(运行被中断时仍在执行中的节点)。
    """
    stages: dict[str, dict] = {}
    calls: list[dict] = []
    run: dict | None = None
    progress: dict[str, dict] = {}
    if thread_id and data_dir:
        e = Engine(data_dir)
        try:
            rr = _rows(e.conn, "SELECT status, error, updated_at FROM agent_runs WHERE thread_id=?", thread_id)
            run = rr[0] if rr else None
            for r in _rows(e.conn, "SELECT * FROM agent_stage_runs WHERE thread_id=?", thread_id):
                stages[r["stage_id"]] = r
            calls = _rows(e.conn, "SELECT name,args,status,started_at,finished_at FROM agent_tool_calls "
                                  "WHERE thread_id=? ORDER BY started_at", thread_id)
            for sid, r in stages.items():                                   # 执行中的阶段:引擎里的实时进度
                if r["status"] == "RUNNING" and r["task_id"]:
                    try:
                        t = e.get_task(r["task_id"])
                    except Exception:  # noqa: BLE001
                        continue
                    run_steps = [s for s in t["steps"] if s["status"] in ("RUNNING", "RETRY_WAIT")]
                    progress[sid] = {"percent": t["percent"],
                                     "msg": next((s["progress_msg"] for s in run_steps if s.get("progress_msg")), None),
                                     "running": [s["key"] for s in run_steps]}
        finally:
            e.close()
    summary = json.loads(stages["keyframes"]["summary"]) if "keyframes" in stages and stages["keyframes"]["summary"] else None
    n = int(summary["chunks"]) if summary else None
    if n is None:                                                          # 分块数还没定:看工具调用里的参数
        for c in calls:
            if c["name"] == "extract_keyframes":
                n = int(json.loads(c["args"] or "{}").get("chunks") or 0) or None
    last: dict[str, dict] = {}
    for c in calls:
        nid = _node_for_call(c["name"], json.loads(c["args"] or "{}"))
        if nid:
            last[nid] = c
    stopped = bool(run) and (run["status"] in ("INTERRUPTED", "INTERRUPTING") or (run["status"] == "RUNNING" and alive is False))

    def status_of(nid: str) -> str:
        s = stages.get(nid)
        st = "PENDING"
        if s:
            st = s["status"]
        elif nid in last:
            c = last[nid]
            st = "RUNNING" if c["status"] == "RUNNING" else ("SUCCEEDED" if c["status"] == "OK" else "FAILED")
        if st == "RUNNING" and stopped:
            st = "INTERRUPTED"
        return st

    nodes: list[dict] = []

    def add(nid: str, label: str, col: int, row: int, deps: list[str]) -> None:
        s = stages.get(nid)
        nodes.append({"id": nid, "label": label, "col": col, "row": row, "deps": deps, "status": status_of(nid),
                      "task_id": s["task_id"] if s else None, "retries": s["retries"] if s else 0,
                      "progress": progress.get(nid)})

    add("skill", NODE_LABELS["skill"], 0, 0, [])
    add("ingest", NODE_LABELS["ingest"], 1, 0, ["skill"])
    add("keyframes", NODE_LABELS["keyframes"], 2, 0, ["ingest"])
    if n:
        for i in range(n):
            add(f"stylize:{i}", f"3 风格化 #{i}", 3, i, ["keyframes"])
    else:
        add("stylize:*", "3 风格化(分块并行)", 3, 0, ["keyframes"])
    add("poststats", NODE_LABELS["poststats"], 4, 0, [f"stylize:{i}" for i in range(n)] if n else ["stylize:*"])
    if n:
        for i in range(n):
            add(f"post:{i}", f"4b 后处理 #{i}", 5, i, ["poststats"])
    else:
        add("post:*", "4b 后处理(分块并行)", 5, 0, ["poststats"])
    post_ids = [f"post:{i}" for i in range(n)] if n else ["post:*"]
    add("qc", NODE_LABELS["qc"], 6, 0, post_ids)
    add("mux", NODE_LABELS["mux"], 7, 0, ["qc"])
    add("package", NODE_LABELS["package"], 8, 0, ["mux"])
    active = [x["id"] for x in nodes if x["status"] in ("RUNNING", "INTERRUPTED")]
    return {"nodes": nodes, "chunks": n, "stopped": stopped, "active": active,
            "interrupted_at": [x["id"] for x in nodes if x["status"] == "INTERRUPTED"]}