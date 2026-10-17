"""Pipeline Agent(LangGraph 决策 + 工具调用 + checkpoint 恢复 + 分阶段用量)的端到端测试。

决策模型用 ScriptedAgentLLM(确定性脚本,没有理解能力),图像模型用 MockDoubao:验证的是**编排、并行、
恢复、记账**这些系统行为,不是真实 doubao-seed-2.1-pro 的决策质量(那部分用真实调用另行验证,见 README)。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from pipeline.agent.agent_llm import ArkChat, AgentLLMError, ScriptedAgentLLM, from_openai, to_openai
from pipeline.agent.orchestrator import PipelineAgent, run_report
from pipeline.agent.pipeline_tools import PipelineToolRegistry, ToolContext
from pipeline.agent.stages import StageRunner
from tests.conftest import FAST_RETRY, WorkerPool
from tests.test_anime_pipeline import FPS, billing, clip  # noqa: F401  (clip 是 fixture)

CFG = dict(prompt="日系动漫风格,保持人物与动作不变", provider="mock", target_height=108, fps=FPS, dup_thr=0.0,
           cut_thr=18.0, backoff_base_s=0.05, rpm=60000.0, concurrency=3, budget_cny=50.0, retry=dict(FAST_RETRY))


def cfg_for(clip: Path, **kw) -> dict:
    return {**CFG, "source": str(clip), **kw}


def make_agent(data_dir, llm=None, **kw):
    return PipelineAgent(data_dir, llm or ScriptedAgentLLM(chunks=2, stride=3, max_gap=8), poll=0.1,
                         stage_timeout_s=240, **kw)


# --------------------------------------------------------------------------- 完整运行
def test_full_run_parallel_stylize_checkpoints_and_per_stage_usage(engine, data_dir, clip):
    agent = make_agent(data_dir)
    with WorkerPool(data_dir, n=3):
        out = agent.start("run1", cfg_for(clip), "把素材转成日式动漫风格并发布")
    assert out["status"] == "SUCCEEDED", out

    snap = agent.snapshot("run1")
    assert snap["next"] == [] and snap["checkpoints"] >= 10              # 每个节点执行完都有 checkpoint
    roles = [m["role"] for m in snap["messages"]]
    assert roles[:2] == ["system", "user"] and roles.count("assistant") == out["turns"]
    # 同一轮里并行发出了两块的 stylize 调用
    par = [m for m in snap["messages"] if m["role"] == "assistant" and len(m.get("tool_calls", [])) == 2]
    assert {tuple(sorted(c["name"] for c in m["tool_calls"])) for m in par} >= {("stylize_chunk", "stylize_chunk"),
                                                                                  ("postprocess_chunk", "postprocess_chunk")}

    rep = run_report(data_dir, "run1")
    by = {s["group"]: s for s in rep["stages"]}
    assert set(by) >= {"plan", "ingest", "keyframes", "stylize", "poststats", "post", "qc", "mux", "package", "final"}
    nk = by["keyframes"]["stages"][0]
    assert nk["status"] == "SUCCEEDED"
    sty = by["stylize"]
    assert len(sty["stages"]) == 2 and sty["tool_calls"] == 2
    assert sty["seedream"]["calls"] == len(billing(data_dir)) and sty["seedream"]["output_tokens"] > 0
    assert rep["totals"]["seedream_output_tokens"] == sty["seedream"]["output_tokens"]
    assert rep["totals"]["llm_estimated"] is True                         # 脚本化策略的 token 是估算,必须标出
    assert rep["totals"]["llm_prompt"] > 0 and rep["totals"]["llm_turns"] == out["turns"]
    # 每个阶段都是引擎里一个独立任务(看板可直接查看)
    tasks = {s["task_id"] for g in rep["stages"] for s in g["stages"]}
    assert len(tasks) == 1 + 1 + 2 + 1 + 2 + 1 + 1 + 1
    assert all(engine.get_task(t)["status"] == "SUCCEEDED" for t in tasks)


def test_stage_steps_share_cache_with_a_normal_full_dag(engine, data_dir, clip, tmp_path):
    """阶段化执行用的是同样的步骤与缓存 key:同素材再跑一次 Agent(新线程),图像模型 0 次新调用。"""
    agent = make_agent(data_dir)
    with WorkerPool(data_dir, n=3):
        assert agent.start("a", cfg_for(clip), "x")["status"] == "SUCCEEDED"
        n1 = len(billing(data_dir))
        assert agent.start("b", cfg_for(clip), "x")["status"] == "SUCCEEDED"
    assert len(billing(data_dir)) == n1
    rep = run_report(data_dir, "b")
    assert rep["totals"]["seedream_calls"] == 0 and rep["totals"]["seedream_reused_tokens"] > 0


# --------------------------------------------------------------------------- 中断与恢复
def test_agent_process_killed_mid_run_resumes_from_checkpoint_without_repeating_llm_or_spend(engine, data_dir, clip):
    llm = ScriptedAgentLLM(chunks=2, stride=3, max_gap=8, crash_on_turn=5)   # 第 5 轮决策时"进程被杀"
    agent = make_agent(data_dir, llm)
    with WorkerPool(data_dir, n=3):
        with pytest.raises(KeyboardInterrupt):
            agent.start("r", cfg_for(clip), "x")
        rows = run_report(data_dir, "r")
        assert rows["run"]["status"] == "INTERRUPTED"
        assert [t["turn"] for t in rows["turns"]] == [1, 2, 3, 4]            # 前 4 轮已记录
        paid_before = len(billing(data_dir))
        assert paid_before > 0                                               # 风格化已经完成并付费
        agent.close()

        agent2 = make_agent(data_dir, ScriptedAgentLLM(chunks=2, stride=3, max_gap=8))   # 新"进程":从 checkpoint 恢复
        out = agent2.resume("r")
    assert out["status"] == "SUCCEEDED"
    assert len(billing(data_dir)) == paid_before                             # 恢复后没有重复调用图像模型
    rep = run_report(data_dir, "r")
    assert [t["turn"] for t in rep["turns"]] == list(range(1, out["turns"] + 1))     # 轮次连续,没有重复
    assert rep["run"]["status"] == "SUCCEEDED"
    agent2.close()


def test_tool_call_interrupted_while_stage_is_running_continues_the_same_engine_task(engine, data_dir, clip):
    """阶段还在 worker 里跑的时候 Agent 被杀:恢复后重新执行同一个工具调用,拿到的是同一个引擎任务。"""
    agent = make_agent(data_dir)
    cfg = cfg_for(clip)
    runner = StageRunner(data_dir, "x", cfg, poll=0.1, timeout_s=60)
    with WorkerPool(data_dir, n=2):
        r1 = runner.run("ingest")
        t1 = r1["task_id"]
        r2 = runner.run("ingest")                                            # 模拟恢复后重执行同一工具
    assert r2["task_id"] == t1 and r2["reused_existing"] is True
    agent.close()


# --------------------------------------------------------------------------- 失败处置
def test_non_retryable_failure_escalates_to_human_with_steps(engine, data_dir, clip):
    inj = {"stylize_0": {"doubao": {"frames": {"__first__": {"kind": "auth"}}}}}
    agent = make_agent(data_dir)
    with WorkerPool(data_dir, n=3):
        out = agent.start("f1", cfg_for(clip, chaos=inj, concurrency=1), "x")
    assert out["status"] == "NEEDS_HUMAN"
    assert out["escalated"]["stage_id"] == "stylize:0"
    assert any("ARK_API_KEY" in s for s in out["escalated"]["human_steps"])
    rep = run_report(data_dir, "f1")
    assert rep["run"]["status"] == "NEEDS_HUMAN"
    names = [c["name"] for c in rep["tool_calls"]]
    assert "retry_stage" not in names and names[-1] == "escalate_to_human"   # 手册判定不能重试 → 没有尝试重试
    assert all(g["group"] != "package" for g in rep["stages"])               # 没有往下走


def test_retryable_failure_is_retried_by_the_agent_and_recovers(engine, data_dir, clip):
    retry1 = dict(FAST_RETRY, max_attempts=1)                                # 引擎自己不再重试,逼它走到 FAILED
    inj = {"stylize_0": {"doubao": {"frames": {"__first__": {"kind": "rate_limit", "until_attempt": 1}}}}}
    agent = make_agent(data_dir)
    with WorkerPool(data_dir, n=3):
        out = agent.start("f2", cfg_for(clip, chaos=inj, retry=retry1, frame_max_attempts=2), "x")
    assert out["status"] == "SUCCEEDED", out
    rep = run_report(data_dir, "f2")
    assert "retry_stage" in [c["name"] for c in rep["tool_calls"]]
    st = {s["stage_id"]: s for g in rep["stages"] for s in g["stages"]}
    assert st["stylize:0"]["retries"] == 1


def test_retry_stage_is_refused_by_runbook_for_auth_errors(engine, data_dir, clip):
    inj = {"stylize_0": {"doubao": {"frames": {"__first__": {"kind": "auth"}}}}}
    cfg = cfg_for(clip, chaos=inj, concurrency=1)
    runner = StageRunner(data_dir, "g", cfg, poll=0.1, timeout_s=120)
    reg = PipelineToolRegistry(ToolContext(data_dir, "g", cfg, runner))
    with WorkerPool(data_dir, n=3):
        assert reg.call("ingest_video", {})["status"] == "SUCCEEDED"
        assert reg.call("extract_keyframes", {"chunks": 2, "stride": 3, "max_gap": 8})["status"] == "SUCCEEDED"
        r = reg.call("stylize_chunk", {"chunk": 0})
        assert r["status"] == "FAILED" and r["diagnosis"]["retry_allowed"] is False
        refused = reg.call("retry_stage", {"stage_id": "stylize:0"})
    assert refused["status"] == "ERROR" and "不允许重试" in refused["error"]       # 模型没有办法绕过手册


# --------------------------------------------------------------------------- 工具层
def test_tool_argument_validation_and_dependency_errors_are_returned_to_the_model(engine, data_dir, clip):
    cfg = cfg_for(clip)
    reg = PipelineToolRegistry(ToolContext(data_dir, "v", cfg, StageRunner(data_dir, "v", cfg, poll=0.1, timeout_s=30)))
    assert "参数错误" in reg.call("extract_keyframes", {"chunks": 0})["error"]
    assert "max_gap" in reg.call("extract_keyframes", {"chunks": 2, "stride": 10, "max_gap": 4})["error"]
    assert "未知工具" in reg.call("rm_rf", {})["error"]
    assert "合法 JSON" in reg.call("ingest_video", {"__invalid_json__": "{bad"})["error"]
    assert "没有产出" in reg.call("quality_check", {})["error"] or "关键帧" in reg.call("quality_check", {})["error"]
    skill = reg.call("load_skill", {"name": "anime-style-pipeline"})
    assert skill["status"] == "OK" and "stylize_chunk" in skill["content"]
    assert "没有技能" in reg.call("load_skill", {"name": "nope"})["error"]
    assert reg.call("load_skill", {"name": "../etc"})["status"] == "ERROR"           # 不能路径穿越


def test_validate_config_before_spending_anything(engine, data_dir, clip):
    agent = make_agent(data_dir)
    with pytest.raises(ValueError):
        agent.start("bad", cfg_for(clip, provider="ark_plan"), "x")                  # 非 mock 必须指定 model
    with pytest.raises(ValueError):
        agent.start("bad2", cfg_for(clip, budget_cny=0), "x")


# --------------------------------------------------------------------------- 决策模型客户端
def test_ark_chat_parses_tool_calls_usage_and_retries():
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        if len(seen) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={
            "choices": [{"finish_reason": "tool_calls", "message": {
                "content": "", "reasoning_content": "先读技能", "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "load_skill", "arguments": '{"name":"x"}'}},
                    {"id": "c2", "type": "function", "function": {"name": "bad", "arguments": "{oops"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130,
                      "prompt_tokens_details": {"cached_tokens": 40}, "completion_tokens_details": {"reasoning_tokens": 12}}})

    llm = ArkChat(api="ark_plan", api_key="k", transport=httpx.MockTransport(handler), sleep=lambda s: None)
    msgs = [HumanMessage(content="hi"), AIMessage(content="", tool_calls=[{"id": "p", "name": "t", "args": {"a": 1}, "type": "tool_call"}]),
            ToolMessage(content="{}", tool_call_id="p", name="t")]
    ai = llm.chat(msgs, [])
    assert len(seen) == 2 and seen[0]["model"] == "doubao-seed-2.1-pro"          # 503 后重试成功
    assert llm.url.endswith("/api/plan/v3/chat/completions")
    assert [m["role"] for m in seen[1]["messages"]] == ["user", "assistant", "tool"]
    assert seen[1]["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'
    assert ai.tool_calls[0]["args"] == {"name": "x"} and "__invalid_json__" in ai.tool_calls[1]["args"]
    u = ai.additional_kwargs["usage"]
    assert (u["prompt_tokens"], u["completion_tokens"], u["reasoning_tokens"], u["cached_tokens"]) == (100, 30, 12, 40)
    assert u["estimated"] is False


def test_ark_chat_auth_error_is_fatal_and_key_is_not_in_errors():
    llm = ArkChat(api="ark", api_key="sk-secret", transport=httpx.MockTransport(lambda r: httpx.Response(401, text="bad key")),
                  sleep=lambda s: None)
    with pytest.raises(AgentLLMError) as e:
        llm.chat([HumanMessage(content="hi")], [])
    assert e.value.fatal and "sk-secret" not in str(e.value)


# --------------------------------------------------------------------------- API(只读)
def test_api_exposes_agent_run_report_and_snapshot(engine, data_dir, clip):
    from fastapi.testclient import TestClient
    from pipeline.api import create_app
    agent = make_agent(data_dir)
    with WorkerPool(data_dir, n=3):
        assert agent.start("api-run", cfg_for(clip), "x")["status"] == "SUCCEEDED"
    agent.close()
    c = TestClient(create_app(data_dir))
    runs = c.get("/api/agent/runs").json()
    assert [r["thread_id"] for r in runs] == ["api-run"] and runs[0]["status"] == "SUCCEEDED"
    d = c.get("/api/agent/runs/api-run").json()
    assert d["snapshot"]["checkpoints"] > 5 and d["snapshot"]["next"] == []
    assert d["totals"]["llm_turns"] == len(d["turns"]) and {s["group"] for s in d["stages"]} >= {"stylize", "qc", "package"}
    assert all(s["status"] == "SUCCEEDED" for s in d["stage_runs"])
    assert c.get("/api/agent/runs/nope").status_code == 404