"""Diagnosis Agent(LangGraph interrupt + checkpoint)、工具权限、MCP Server 测试。"""
from __future__ import annotations

import asyncio
import json

import pytest

from conftest import FAST_RETRY, WorkerPool, step, wait_task
from pipeline.agent.diagnosis import DiagnosisAgent
from pipeline.agent.llm import TracedLLM, list_llm_calls
from pipeline.agent.mock_llm import MockLLM
from pipeline.agent.planner import Planner
from pipeline.agent.tools import ToolError, ToolRegistry
from pipeline.engine import Engine, InvalidAction
from pipeline.mcp_server import build_server
from pipeline.models import RetryPolicy
from pipeline.templates import animation_swap_task

R = RetryPolicy(**FAST_RETRY)


def failed_task(engine, data_dir, **kw):
    kw.setdefault("chunks", 2)
    kw.setdefault("frames", 4)
    kw.setdefault("retry", R)
    tid, _ = engine.submit(animation_swap_task(**kw))
    with WorkerPool(data_dir):
        t = wait_task(engine, tid)
    assert t["status"] == "FAILED"
    return tid


def agent_for(data_dir):
    e = Engine(data_dir)
    return e, DiagnosisAgent(data_dir, e, TracedLLM(MockLLM(), e))


# ----------------------------------------------------------------------------- 诊断 → 人工确认 → 重试
def test_diagnose_waits_for_human_survives_restart_and_retries(engine, data_dir):
    tid = failed_task(engine, data_dir, chaos={"mux": {"fail": {"kind": "TRANSIENT", "until_attempt": 3, "at": 0.0}}})
    e1, a1 = agent_for(data_dir)
    out = a1.start(tid, "mux")
    assert out["status"] == "AWAITING_HUMAN" and out["proposal"]["recommended_action"] == "retry_step"
    # 人没确认之前绝不执行写操作
    assert step(engine.get_task(tid), "mux")["status"] == "FAILED"
    a1.close()
    e1.close()                                                     # 模拟 Agent 进程退出

    e2, a2 = agent_for(data_dir)                                   # 重启后用同一个 thread_id 继续
    pending = a2.list(status="AWAITING_HUMAN")
    assert [p["thread_id"] for p in pending] == [out["thread_id"]]
    done = a2.resume(out["thread_id"], approve=True, comment="LGTM", who="alice")
    assert done["status"] == "DONE" and done["result"]["executed"] is True
    # diagnose 节点有 checkpoint:恢复时没有再次调用 LLM
    assert [c["purpose"] for c in list_llm_calls(e2, task_id=tid)] == ["diagnose"]
    kinds = [x["kind"] for x in e2.get_events(tid) if x["kind"].startswith("AGENT_")]
    assert kinds == ["AGENT_DIAGNOSIS", "AGENT_ACTION"]

    with WorkerPool(data_dir):
        t = wait_task(engine, tid, statuses=("SUCCEEDED",))
    assert step(t, "mux")["attempt_no"] == 4 and step(t, "assemble")["attempt_no"] == 1   # 上游未重跑
    with pytest.raises(InvalidAction):
        a2.resume(out["thread_id"], approve=True)                  # 不能重复处理
    a2.close()
    e2.close()


def test_human_rejection_does_not_execute(engine, data_dir):
    tid = failed_task(engine, data_dir, chaos={"mux": {"fail": {"kind": "TRANSIENT", "until_attempt": 3, "at": 0.0}}})
    e, a = agent_for(data_dir)
    out = a.start(tid, "mux")
    res = a.resume(out["thread_id"], approve=False, comment="先别动")
    assert res["result"]["executed"] is False and "rejected" in res["result"]["reason"]
    assert step(engine.get_task(tid), "mux")["status"] == "FAILED"
    a.close()
    e.close()


def test_engine_is_source_of_truth_when_state_changed_before_human_confirms(engine, data_dir):
    tid = failed_task(engine, data_dir, chaos={"mux": {"fail": {"kind": "TRANSIENT", "until_attempt": 3, "at": 0.0}}})
    e, a = agent_for(data_dir)
    out = a.start(tid, "mux")
    engine.retry_step(tid, "mux")                                  # 有人在确认之前已经手动重试了
    res = a.resume(out["thread_id"], approve=True)
    assert res["result"]["executed"] is False and "tool error" in res["result"]["reason"]
    a.close()
    e.close()


# ----------------------------------------------------------------------------- 不该重试的不重试
def test_bad_input_gets_give_up_without_asking_for_retry(engine, data_dir):
    tid = failed_task(engine, data_dir, source="bad:corrupt.mp4")
    e, a = agent_for(data_dir)
    out = a.start(tid, "probe")
    assert out["status"] == "DONE" and out["proposal"]["recommended_action"] == "give_up"
    assert out["result"]["executed"] is False
    assert step(engine.get_task(tid), "probe")["status"] == "FAILED"
    a.close()
    e.close()


def test_diagnose_only_failed_steps(engine, data_dir):
    tid = failed_task(engine, data_dir, qc_fail=True)
    e, a = agent_for(data_dir)
    with pytest.raises(InvalidAction):
        a.start(tid, "mux")                                       # SUCCEEDED 的步骤不能诊断
    out = a.start(tid, "quality_check")
    assert out["proposal"]["category"] == "quality" and out["proposal"]["recommended_action"] == "escalate"
    a.close()
    e.close()


class _GarbageLLM(MockLLM):
    def complete(self, messages, **kw):
        r = super().complete(messages, **kw)
        r.text = "我觉得应该重试一下"
        return r


def test_invalid_llm_output_degrades_to_escalate(engine, data_dir):
    tid = failed_task(engine, data_dir, chaos={"mux": {"fail": {"kind": "TRANSIENT", "until_attempt": 3, "at": 0.0}}})
    e = Engine(data_dir)
    a = DiagnosisAgent(data_dir, e, TracedLLM(_GarbageLLM(), e))
    out = a.start(tid, "mux")
    assert out["status"] == "DONE" and out["proposal"]["recommended_action"] == "escalate"
    assert out["proposal"]["fallback"] is True
    assert step(engine.get_task(tid), "mux")["status"] == "FAILED"     # 解析失败绝不会触发写操作
    a.close()
    e.close()


# ----------------------------------------------------------------------------- 工具权限
def test_tool_registry_permissions_and_validation(engine, data_dir):
    tid, _ = engine.submit(animation_swap_task(chunks=2, frames=4, retry=R))
    reg = ToolRegistry(engine)
    assert "approve" not in reg.tools and not any("approve" in n for n in reg.tools)    # Agent 不能自己批准
    read_names = {s["function"]["name"] for s in reg.specs()}
    all_names = {s["function"]["name"] for s in reg.specs(include_write=True)}
    assert read_names == {"get_task", "list_tasks", "get_step_attempts", "get_events", "get_metrics", "list_workers"}
    assert {"retry_step", "resume_task", "cancel_task", "submit_request"} <= all_names - read_names
    assert reg.call("get_task", {"task_id": tid})["steps_total"] == 12
    with pytest.raises(ToolError, match="写操作"):
        reg.call("cancel_task", {"task_id": tid})
    with pytest.raises(ToolError, match="未知工具"):
        reg.call("drop_database", {})
    with pytest.raises(ToolError, match="参数错误"):
        reg.call("get_events", {"task_id": tid, "limit": 99999})
    with pytest.raises(ToolError, match="NotFound"):
        reg.call("get_task", {"task_id": "t_nope"})
    with pytest.raises(ToolError, match="Planner"):
        reg.call("submit_request", {"request": "x"}, allow_write=True)
    assert reg.call("cancel_task", {"task_id": tid}, allow_write=True) == {"ok": True}
    # 参数 schema 可直接用于 function calling
    schema = next(s for s in reg.specs() if s["function"]["name"] == "get_events")["function"]["parameters"]
    assert schema["properties"]["limit"]["maximum"] == 200 and schema["required"] == ["task_id"]


# ----------------------------------------------------------------------------- MCP
def _mcp(coro):
    return asyncio.run(coro)


def _text(result) -> dict:
    return json.loads(result.content[0].text)


def test_mcp_server_read_only_by_default(engine, data_dir):
    tid, _ = engine.submit(animation_swap_task(chunks=2, frames=4, retry=R))
    server = build_server(data_dir)
    names = {t.name for t in _mcp(server.list_tools())}
    assert names == {"get_task", "list_tasks", "get_step_attempts", "get_events", "get_metrics", "list_workers"}
    t = _text(_mcp(server.call_tool("get_task", {"task_id": tid})))
    assert t["id"] == tid and t["steps_total"] == 12
    miss = _text(_mcp(server.call_tool("get_task", {"task_id": "t_nope"})))
    assert "NotFound" in miss["error"]


def test_mcp_server_write_tools_only_when_enabled_and_never_approve(engine, data_dir):
    tid, _ = engine.submit(animation_swap_task(chunks=2, frames=4, retry=R))
    server = build_server(data_dir, allow_write=True)
    names = {t.name for t in _mcp(server.list_tools())}
    assert {"retry_step", "resume_task", "cancel_task", "submit_request"} <= names and "approve" not in names
    assert _text(_mcp(server.call_tool("cancel_task", {"task_id": tid}))) == {"ok": True}
    assert engine.get_task(tid)["status"] == "CANCELLED"
    r = _text(_mcp(server.call_tool("submit_request", {"request": "把 demo.mp4 的主角换成红发女孩,发布到测试库"})))
    assert r["ok"] and r["task_id"]
