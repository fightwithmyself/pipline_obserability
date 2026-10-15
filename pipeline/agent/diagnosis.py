"""Diagnosis Agent(LangGraph):步骤最终失败后,读证据 → LLM 诊断 → 人工确认(interrupt)→ 执行。

collect ──► diagnose ──(建议重试)──► human_gate ──► act ──► END
                 └──(其它建议)──► report ──► END

关键设计:
- **人在回路**:任何写操作(retry_step)都必须经过 `interrupt` 等待人确认。图状态由 SqliteSaver 持久化,
  进程重启后可用同一 thread_id 继续,且**不会重新调用 LLM**(diagnose 节点已有 checkpoint)。
- **以引擎为准**:LangGraph 的 checkpoint 只保存 Agent 自己的推理进度;真正的任务状态只在引擎里。
  动作执行时由引擎重新校验(步骤已不是 FAILED 则工具报错,记录为"未执行")。
- **LLM 输出不可信**:Pydantic 校验,失败降级为"升级给人工";建议重试但步骤并非 FAILED 时同样降级。
- Agent 只持有 retry_step 这一个写权限,没有 approve/cancel。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field, ValidationError

from ..db import tx
from ..engine import Engine, InvalidAction, NotFound
from .llm import TracedLLM
from .planner import parse_json_loose
from .tools import ToolError, ToolRegistry

DIAG_SYSTEM = """[[DIAGNOSE]]
你是任务执行系统的故障诊断员。根据用户提供的 JSON 证据(任务、失败步骤、尝试历史、事件),判断根因并给出建议动作。
只输出一个 JSON 对象:
{"root_cause": "一句话根因", "category": "bad_input|quality|resource|transient|poison|permanent|unknown",
 "recommended_action": "retry_step|give_up|escalate", "confidence": 0~1 的小数, "explanation": "简短理由"}

动作含义:
- retry_step: 重试失败步骤(仅当错误是可重试类别、重试有意义时)
- give_up: 重试不会有帮助(如输入素材损坏),应终止并告知上游
- escalate: 证据不足或需要人工判断
注意:永久性错误(PERMANENT)盲目重试没有意义;反复导致 worker 失联的步骤不要自动重试。
"""


class Diagnosis(BaseModel):
    root_cause: str
    category: Literal["bad_input", "quality", "resource", "transient", "poison", "permanent", "unknown"]
    recommended_action: Literal["retry_step", "give_up", "escalate"]
    confidence: float = Field(ge=0, le=1)
    explanation: str = ""


class DiagState(TypedDict, total=False):
    task_id: str
    step_key: str
    evidence: dict
    proposal: dict
    human: dict
    result: dict


class DiagnosisAgent:
    def __init__(self, data_dir: str | Path, engine: Engine, llm: TracedLLM):
        self.engine = engine
        self.llm = llm
        self.registry = ToolRegistry(engine)
        self._ck_conn = sqlite3.connect(str(Path(data_dir) / "agent_checkpoints.db"), check_same_thread=False)
        self.saver = SqliteSaver(self._ck_conn)
        self.graph = self._build().compile(checkpointer=self.saver)

    def close(self) -> None:
        self._ck_conn.close()

    # ------------------------------------------------------------------ 图
    def _build(self) -> StateGraph:
        reg, eng = self.registry, self.engine

        def collect(state: DiagState) -> DiagState:
            tid, key = state["task_id"], state["step_key"]
            task = reg.call("get_task", {"task_id": tid})
            step = next((s for s in task["steps"] if s["key"] == key), None)
            if step is None:
                raise ToolError(f"任务 {tid} 中没有步骤 {key}")
            attempts = reg.call("get_step_attempts", {"task_id": tid, "key": key})[-6:]
            for a in attempts:
                a["error_msg"] = (a["error_msg"] or "")[:300]
            events = [e for e in reg.call("get_events", {"task_id": tid, "limit": 200}) if e["step_key"] == key][-15:]
            return {"evidence": {"task": {k: task[k] for k in ("id", "name", "status", "steps_by_status")},
                                 "step": step, "attempts": attempts, "events": events}}

        def diagnose(state: DiagState) -> DiagState:
            ev = state["evidence"]
            msgs = [{"role": "system", "content": DIAG_SYSTEM},
                    {"role": "user", "content": json.dumps(ev, ensure_ascii=False)}]
            res = self.llm.complete(msgs, purpose="diagnose", task_id=state["task_id"], json_mode=True)
            try:
                d = Diagnosis.model_validate(parse_json_loose(res.text)).model_dump()
            except (ValueError, ValidationError) as e:
                d = {"root_cause": "模型输出无法解析", "category": "unknown", "recommended_action": "escalate",
                     "confidence": 0.0, "explanation": f"校验失败: {str(e)[:200]}", "fallback": True}
            if d["recommended_action"] == "retry_step" and ev["step"]["status"] != "FAILED":
                d.update(recommended_action="escalate", note=f"步骤当前为 {ev['step']['status']},不是 FAILED,不执行重试")
            return {"proposal": d}

        def route(state: DiagState) -> str:
            return "human_gate" if state["proposal"]["recommended_action"] == "retry_step" else "report"

        def human_gate(state: DiagState) -> DiagState:
            # 首次执行到这里会暂停并把 payload 抛给调用方;恢复时 interrupt() 返回人给的决定
            decision = interrupt({"kind": "confirm_action", "task_id": state["task_id"],
                                  "step_key": state["step_key"], "action": "retry_step",
                                  "proposal": state["proposal"]})
            return {"human": decision}

        def act(state: DiagState) -> DiagState:
            h = state["human"]
            if not h.get("approve"):
                return {"result": {"executed": False, "reason": "rejected by human", "comment": h.get("comment", "")}}
            try:
                out = reg.call("retry_step", {"task_id": state["task_id"], "key": state["step_key"]}, allow_write=True)
                return {"result": {"executed": True, "action": "retry_step", "tool_result": out,
                                   "by": h.get("who", "human")}}
            except ToolError as e:       # 以引擎为准:状态已变化等情况,记录为未执行
                return {"result": {"executed": False, "reason": f"tool error: {e}"}}

        def report(state: DiagState) -> DiagState:
            a = state["proposal"]["recommended_action"]
            return {"result": {"executed": False, "reason": f"建议动作为 {a},不会自动执行,已通知人工"}}

        g = StateGraph(DiagState)
        g.add_node("collect", collect)
        g.add_node("diagnose", diagnose)
        g.add_node("human_gate", human_gate)
        g.add_node("act", act)
        g.add_node("report", report)
        g.add_edge(START, "collect")
        g.add_edge("collect", "diagnose")
        g.add_conditional_edges("diagnose", route, {"human_gate": "human_gate", "report": "report"})
        g.add_edge("human_gate", "act")
        g.add_edge("act", END)
        g.add_edge("report", END)
        return g

    # ------------------------------------------------------------------ 对外 API
    def start(self, task_id: str, step_key: str) -> dict:
        task = self.engine.get_task(task_id)
        st = next((s for s in task["steps"] if s["key"] == step_key), None)
        if st is None:
            raise NotFound(f"{task_id}/{step_key}")
        if st["status"] != "FAILED":
            raise InvalidAction(f"只能诊断 FAILED 的步骤,当前 {st['status']}")
        thread_id = f"diag-{task_id}-{step_key}-{uuid.uuid4().hex[:6]}"
        cfg = {"configurable": {"thread_id": thread_id}}
        out = self.graph.invoke({"task_id": task_id, "step_key": step_key}, cfg)
        now = self.engine.clock()
        if "__interrupt__" in out:
            proposal = out["__interrupt__"][0].value["proposal"]
            self._upsert(thread_id, task_id, step_key, "AWAITING_HUMAN", proposal, None, None, now)
            self.engine.add_event(task_id, "AGENT_DIAGNOSIS", step_key,
                                  {"thread_id": thread_id, "proposal": proposal, "awaiting_human": True})
            return {"thread_id": thread_id, "status": "AWAITING_HUMAN", "proposal": proposal}
        self._upsert(thread_id, task_id, step_key, "DONE", out["proposal"], None, out["result"], now)
        self.engine.add_event(task_id, "AGENT_DIAGNOSIS", step_key,
                              {"thread_id": thread_id, "proposal": out["proposal"], "awaiting_human": False})
        return {"thread_id": thread_id, "status": "DONE", "proposal": out["proposal"], "result": out["result"]}

    def resume(self, thread_id: str, approve: bool, comment: str = "", who: str = "human") -> dict:
        row = self.engine.conn.execute("SELECT * FROM diagnoses WHERE thread_id=?", (thread_id,)).fetchone()
        if row is None:
            raise NotFound(thread_id)
        if row["status"] != "AWAITING_HUMAN":
            raise InvalidAction(f"诊断已处理: {row['status']}")
        decision = {"approve": approve, "comment": comment, "who": who}
        out = self.graph.invoke(Command(resume=decision), {"configurable": {"thread_id": thread_id}})
        result = out["result"]
        self._upsert(thread_id, row["task_id"], row["step_key"], "DONE", json.loads(row["proposal"]),
                     decision, result, self.engine.clock())
        self.engine.add_event(row["task_id"], "AGENT_ACTION", row["step_key"],
                              {"thread_id": thread_id, "decision": decision, "result": result})
        return {"thread_id": thread_id, "status": "DONE", "result": result}

    def list(self, status: str | None = None, task_id: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM diagnoses WHERE 1=1", []
        if status:
            q += " AND status=?"
            args.append(status)
        if task_id:
            q += " AND task_id=?"
            args.append(task_id)
        q += " ORDER BY created_at DESC"
        return [{"thread_id": r["thread_id"], "task_id": r["task_id"], "step_key": r["step_key"],
                 "status": r["status"], "proposal": json.loads(r["proposal"]) if r["proposal"] else None,
                 "decision": json.loads(r["decision"]) if r["decision"] else None,
                 "result": json.loads(r["result"]) if r["result"] else None}
                for r in self.engine.conn.execute(q, args)]

    def _upsert(self, thread_id, task_id, step_key, status, proposal, decision, result, now) -> None:
        j = lambda v: None if v is None else json.dumps(v, ensure_ascii=False)   # noqa: E731
        with tx(self.engine.conn):
            self.engine.conn.execute(
                "INSERT INTO diagnoses(thread_id,task_id,step_key,status,proposal,decision,result,created_at,updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(thread_id) DO UPDATE SET status=excluded.status,"
                " proposal=excluded.proposal, decision=excluded.decision, result=excluded.result,"
                " updated_at=excluded.updated_at",
                (thread_id, task_id, step_key, status, j(proposal), j(decision), j(result), now, now))
