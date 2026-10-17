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

def test_api_start_and_upload_validation(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from pipeline.api import create_app
    media = tmp_path / "media"
    monkeypatch.setenv("PIPELINE_MEDIA_DIR", str(media))
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    c = TestClient(create_app(tmp_path / "data"))
    assert c.get("/api/agent/config").json()["has_ark_key"] is False
    # 上传:只接受纯文件名与媒体扩展名,不覆盖
    assert c.post("/api/media?name=a.mp4", content=b"x" * 10).status_code == 201
    assert c.post("/api/media?name=a.mp4", content=b"x").status_code == 409
    assert c.post("/api/media?name=..%2Fevil.mp4", content=b"x").status_code == 422
    assert c.post("/api/media?name=run.exe", content=b"x").status_code == 422
    assert c.post("/api/media?name=e.mp4", content=b"").status_code == 422
    assert [m["name"] for m in c.get("/api/media").json()] == ["a.mp4"] and not list(media.glob("_upload*"))
    # 没有密钥:真实调用在启动子进程之前就被拒绝(不会产生费用)
    body = {"source": "a.mp4", "prompt": "anime", "provider": "ark_plan", "model": "m"}
    r = c.post("/api/agent/runs", json=body)
    assert r.status_code == 422 and "ARK_API_KEY" in r.json()["detail"]
    assert c.post("/api/agent/runs", json={**body, "source": "../x.mp4"}).status_code == 422
    assert c.post("/api/agent/runs", json={**body, "source": "nope.mp4"}).status_code == 404
    assert c.post("/api/agent/runs", json={**body, "provider": "mock", "agent_llm": "scripted", "thread": "bad thread!"}).status_code == 422

# --------------------------------------------------------------------------- 中断按钮 / 流程图
def _wait_for(cond, timeout=60.0, step=0.05):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return False


def _sql(data_dir, q, *a):
    from pipeline.engine import Engine
    e = Engine(data_dir)
    try:
        return [dict(r) for r in e.conn.execute(q, a).fetchall()]
    finally:
        e.close()


def test_soft_interrupt_then_resume_from_checkpoint_without_double_spend(engine, data_dir, clip):
    from pipeline.agent.stages import RunInterrupted
    from pipeline.agent.orchestrator import pipeline_graph
    from pipeline.db import tx
    agent = make_agent(data_dir)
    box = {}

    def go():
        try:
            box["out"] = agent.start("itr", cfg_for(clip), "x")
        except BaseException as e:  # noqa: BLE001
            box["exc"] = e

    with WorkerPool(data_dir, n=3):
        th = threading.Thread(target=go)
        th.start()
        # 等到风格化阶段已经开始,再请求中断(模拟点击"中断"按钮)
        assert _wait_for(lambda: any(r["stage_id"].startswith("stylize") for r in
                                     _sql(data_dir, "SELECT stage_id FROM agent_stage_runs WHERE thread_id='itr'")))
        from pipeline.engine import Engine
        e = Engine(data_dir)
        with tx(e.conn):
            e.conn.execute("UPDATE agent_runs SET status='INTERRUPTING' WHERE thread_id='itr'")
        e.close()
        th.join(timeout=120)
        assert not th.is_alive()
        assert isinstance(box.get("exc"), RunInterrupted), box
        assert _sql(data_dir, "SELECT status FROM agent_runs WHERE thread_id='itr'")[0]["status"] == "INTERRUPTED"
        g = pipeline_graph(data_dir, "itr", alive=False)
        by = {n["id"]: n["status"] for n in g["nodes"]}
        assert by["skill"] == "SUCCEEDED" and by["ingest"] == "SUCCEEDED" and by["package"] == "PENDING"
        assert g["stopped"] is True
        turns_before = len(_sql(data_dir, "SELECT turn FROM agent_turns WHERE thread_id='itr'"))
        # 从 checkpoint 继续
        out = agent.resume("itr")
    assert out["status"] == "SUCCEEDED"
    rep = run_report(data_dir, "itr")
    n_kf = next(s for s in _sql(data_dir, "SELECT summary FROM agent_stage_runs WHERE thread_id='itr' AND stage_id='keyframes'"))
    n_kf = json.loads(n_kf["summary"])["n_keyframes"]
    assert len(billing(data_dir)) == n_kf, "中断/续跑期间每个关键帧只能调用一次图像模型"
    assert rep["totals"]["llm_turns"] >= turns_before
    assert all(n["status"] == "SUCCEEDED" for n in pipeline_graph(data_dir, "itr", alive=False)["nodes"])


def test_pipeline_graph_template_and_finished_run(engine, data_dir, clip):
    from pipeline.agent.orchestrator import pipeline_graph
    t = pipeline_graph(None, None)
    assert [n["id"] for n in t["nodes"]][:3] == ["skill", "ingest", "keyframes"] and all(n["status"] == "PENDING" for n in t["nodes"])
    agent = make_agent(data_dir)
    with WorkerPool(data_dir, n=3):
        assert agent.start("g", cfg_for(clip), "x")["status"] == "SUCCEEDED"
    g = pipeline_graph(data_dir, "g", alive=False)
    ids = [n["id"] for n in g["nodes"]]
    assert g["chunks"] == 2 and {"stylize:0", "stylize:1", "post:0", "post:1"} <= set(ids)
    assert all(n["status"] == "SUCCEEDED" for n in g["nodes"]) and not g["stopped"]
    assert next(n for n in g["nodes"] if n["id"] == "qc")["deps"] == ["post:0", "post:1"]


def test_pid_helpers():
    import os
    import subprocess
    import sys
    from pipeline.proc import kill_tree, pid_alive
    assert pid_alive(os.getpid()) and not pid_alive(0) and not pid_alive(2 ** 22 + 12345)
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    assert pid_alive(p.pid)
    kill_tree(p.pid)
    p.wait(timeout=10)
    assert not pid_alive(p.pid)


def test_api_interrupt_and_resume_guards(tmp_path, monkeypatch):
    import os
    import subprocess
    from fastapi.testclient import TestClient
    from pipeline.api import create_app
    from pipeline.db import tx
    from pipeline.engine import Engine
    data = tmp_path / "data"
    monkeypatch.setenv("ARK_API_KEY", "dummy")
    spawned = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: spawned.append(a[0]) or None)
    c = TestClient(create_app(data))
    e = Engine(data)
    with tx(e.conn):
        for t, st in (("dead", "RUNNING"), ("done", "SUCCEEDED"), ("live", "RUNNING")):
            e.conn.execute("INSERT INTO agent_runs(thread_id,status,agent_model,config,request,created_at,updated_at) "
                           "VALUES (?,?,?,?,?,?,?)", (t, st, "m", json.dumps({"source": "x"}), "r", 1.0, 1.0))
    e.close()
    (data / "agent_logs").mkdir(parents=True)
    (data / "agent_logs" / "live.pid").write_text(str(os.getpid()))        # 假装 live 的进程还活着(用测试进程自己)
    runs = {r["thread_id"]: r for r in c.get("/api/agent/runs").json()}
    assert runs["dead"]["effective_status"] == "INTERRUPTED" and runs["live"]["effective_status"] == "RUNNING"
    # 进程已不在:中断 = 直接标记 INTERRUPTED;之后允许继续
    assert c.post("/api/agent/runs/dead/interrupt").json()["status"] == "INTERRUPTED"
    assert c.post("/api/agent/runs/dead/resume?agent_llm=scripted").status_code == 202 and spawned
    # 已结束的不能中断;进程还活着的不能重复继续;不存在的 404
    assert c.post("/api/agent/runs/done/interrupt").status_code == 409
    assert c.post("/api/agent/runs/live/resume?agent_llm=scripted").status_code == 409
    assert c.post("/api/agent/runs/nope/interrupt").status_code == 404
    assert c.get("/api/agent/graph").json()["nodes"][0]["id"] == "skill"


def test_real_process_interrupt_via_api_then_cli_resume(tmp_path, clip, monkeypatch):
    """真实子进程:CLI 跑 Agent(含 worker 子进程)→ API 中断(软中断 + 立即强杀兜底)→ CLI 继续 → 成功,图像模型不重复调用。"""
    import os
    import subprocess
    import sys
    from fastapi.testclient import TestClient
    from pipeline.api import create_app
    from pipeline.proc import pid_alive
    data = tmp_path / "data"
    root = Path(__file__).resolve().parent.parent
    base = [sys.executable, "-m", "pipeline.cli", "--data", str(data)]
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    c = TestClient(create_app(data))
    common = ["--agent-llm", "scripted", "--chunks", "2", "--stride", "3", "--max-gap", "8", "--workers", "2"]
    p = subprocess.Popen(base + ["agent-run", "--source", str(clip), "--prompt", "anime", "--thread", "rp",
                                 "--provider", "mock", "--height", "108", *common], cwd=str(root),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _wait_for(lambda: any(r["stage_id"].startswith("stylize") for r in
                                     _sql(data, "SELECT stage_id FROM agent_stage_runs WHERE thread_id='rp'")), 120)
        pid = int((data / "agent_logs" / "rp.pid").read_text())
        assert pid_alive(pid)            # Windows venv 里 Popen 拿到的是启动器 pid,CLI 真实 pid 以 pid 文件为准
        r = c.post("/api/agent/runs/rp/interrupt?force_after_s=0")
        assert r.status_code == 202
        p.wait(timeout=60)                                              # 进程(树)被终止
        assert _wait_for(lambda: c.get("/api/agent/runs/rp").json()["effective_status"] == "INTERRUPTED", 30)
        d = c.get("/api/agent/runs/rp").json()
        assert d["alive"] is False and d["graph"]["stopped"] is True
        assert d["snapshot"]["checkpoints"] >= 3                        # checkpoint 在
    finally:
        if p.poll() is None:
            p.kill()
    p2 = subprocess.run(base + ["agent-resume", "rp", *common], cwd=str(root), capture_output=True, timeout=600)
    assert p2.returncode == 0, p2.stderr.decode("utf-8", "replace")[-800:]
    d = c.get("/api/agent/runs/rp").json()
    assert d["run"]["status"] == "SUCCEEDED" and all(n["status"] == "SUCCEEDED" for n in d["graph"]["nodes"])
    n_kf = json.loads(_sql(data, "SELECT summary FROM agent_stage_runs WHERE thread_id='rp' AND stage_id='keyframes'")[0]["summary"])["n_keyframes"]
    assert d["totals"]["seedream_calls"] <= n_kf + 2     # 硬杀时在途的请求属于"可能多付"(账本会记),这里只做上界检查
