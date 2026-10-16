"""Planner / 护栏 / LLM 追踪测试。LLM 用 MockLLM —— 这里验证的是系统管线与护栏,不是模型智能。"""
from __future__ import annotations

import json

import httpx
import pytest

from conftest import FAST_RETRY, WorkerPool, wait_task
from pipeline.agent.llm import OpenAICompatClient, TracedLLM, list_llm_calls, llm_usage_summary
from pipeline.agent.mock_llm import MockLLM
from pipeline.agent.planner import (MAX_CHUNKS, PlanModel, Planner, expand_plan, get_plan, list_plans,
                                    parse_json_loose, validate_plan)
from pipeline.models import RetryPolicy
from pipeline.templates import animation_swap_task

R = RetryPolicy(**FAST_RETRY)


@pytest.fixture
def planner(engine):
    return Planner(TracedLLM(MockLLM(), engine), engine, retry=R, unit_s=0.05)


# ----------------------------------------------------------------------------- 正常路径
def test_plan_happy_path_submits_and_traces(engine, planner):
    r = planner.plan("把 demo.mp4 的主角换成红发女孩,分成 3 段,发布到测试库")
    assert r.ok and r.rounds == 1 and r.task_id
    t = engine.get_task(r.task_id)
    keys = {s["key"] for s in t["steps"]}
    assert {"segment_0", "segment_1", "segment_2", "swap_2", "publish"} <= keys and "human_approval" not in keys
    assert t["request"].startswith("把 demo.mp4")
    gen = next(s for s in t["steps"] if s["key"] == "gen_reference")
    assert gen["params"]["prompt"] == "红发女孩"
    # LLM 调用已追踪并关联到任务;计划已落库
    calls = list_llm_calls(engine, task_id=r.task_id, include_messages=True)
    assert [c["purpose"] for c in calls] == ["plan"] and calls[0]["tokens_estimated"] is True
    assert calls[0]["messages"][0]["role"] == "system"
    p = get_plan(engine, r.plan_id)
    assert p["status"] == "OK" and p["task_id"] == r.task_id and p["spec"]["steps"]
    assert llm_usage_summary(engine, r.task_id)["calls"] == 1


# ----------------------------------------------------------------------------- 护栏 + 修复回路
def test_policy_violation_is_rejected_then_repaired(engine, planner):
    # 用户要求"不用审批",但发布到生产库的策略要求必须审批
    r = planner.plan("把 demo.mp4 的主角换成红发女孩并发布,不用审批")
    assert r.ok and r.rounds == 2
    assert any("策略" in e for e in r.attempts[0]["errors"])
    assert r.attempts[1]["errors"] == []
    keys = {s["key"] for s in engine.get_task(r.task_id)["steps"]}
    assert "human_approval" in keys                                    # 最终计划被修正为合规
    pub = next(s for s in engine.get_task(r.task_id)["steps"] if s["key"] == "publish")
    assert "human_approval" in pub["deps"]
    assert [c["purpose"] for c in reversed(list_llm_calls(engine, task_id=r.task_id))] == ["plan", "plan_repair"]


def test_skipping_quality_check_is_rejected_then_repaired(planner, engine):
    r = planner.plan("处理 clip.mov,换成蓝发男孩,跳过质检直接发布到测试库")
    assert r.ok and r.rounds == 2
    assert any("缺少必须依赖" in e and "quality_check" in e for e in r.attempts[0]["errors"])
    assert "quality_check" in {s["key"] for s in engine.get_task(r.task_id)["steps"]}


def test_budget_guard_clamps_after_feedback(planner, engine):
    r = planner.plan("把 demo.mp4 的主角换成红发女孩,分成 100 段,发布到测试库")
    assert r.ok and r.rounds == 2
    assert any("chunks" in e for e in r.attempts[0]["errors"])
    t = engine.get_task(r.task_id)
    assert sum(1 for s in t["steps"] if s["key"].startswith("segment_")) == MAX_CHUNKS


@pytest.mark.parametrize("marker,expect", [("[mock:garbage]", "不是合法 JSON"), ("[mock:cycle]", "环"),
                                           ("[mock:inject]", "结构错误")])
def test_malformed_or_malicious_first_output_is_repaired(planner, marker, expect):
    r = planner.plan(f"把 demo.mp4 的主角换成红发女孩,发布到测试库 {marker}")
    assert r.ok and r.rounds == 2
    assert any(expect in e for e in r.attempts[0]["errors"])


def test_persistently_invalid_plan_is_rejected_and_nothing_is_submitted(planner, engine):
    r = planner.plan("把 demo.mp4 的主角换成红发女孩 [mock:evil]")
    assert not r.ok and r.rounds == 3 and r.task_id is None
    assert any("未知阶段类型" in e and "shell" in e for e in r.errors)
    assert engine.list_tasks() == []                                   # 不合格的计划不会被"勉强执行"
    assert get_plan(engine, r.plan_id)["status"] == "REJECTED"
    assert [p["status"] for p in list_plans(engine)] == ["REJECTED"]


# ----------------------------------------------------------------------------- 校验器单测
def _plan(**over):
    stages = [{"id": "probe", "type": "probe", "deps": []}]
    d = {"summary": "", "params": {"source": "a.mp4"}, "stages": stages}
    d.update(over)
    return d


def test_validate_plan_rejects_bad_sources_and_extra_fields():
    for bad in ["../../etc/passwd.mp4", "/abs/path.mp4", "a.exe", "a b.mp4", "http://x/a.mp4"]:
        plan, errs = validate_plan(_plan(params={"source": bad}))
        assert plan is None and any("source" in e for e in errs), bad
    plan, errs = validate_plan(_plan(stages=[{"id": "probe", "type": "probe", "deps": [], "retry": 99}]))
    assert plan is None and any("结构错误" in e for e in errs)
    plan, errs = validate_plan(_plan(params={"source": "a.mp4", "target": "prod-db"}))
    assert any("target" in e for e in errs)
    plan, errs = validate_plan(_plan())
    assert errs == [] and plan is not None


def test_source_must_come_from_the_user_request():
    """真实模型评测发现的漏洞:被拒的文件名可以在"修复"循环里被悄悄换成别的文件。"""
    ok_plan = _plan(params={"source": "demo.mp4"})
    assert validate_plan(ok_plan, "把 DEMO.mp4 的主角换成红发女孩")[1] == []          # 大小写不敏感
    plan, errs = validate_plan(ok_plan, "把 other.mp4 的主角换成红发女孩")
    assert plan is None and any("没有出现在用户请求里" in e for e in errs)
    assert validate_plan(ok_plan)[1] == []                                          # 不传 request 时不检查


def test_repair_loop_cannot_swap_the_file_the_user_asked_for(engine):
    """第 1 轮给了穿越路径被拒,第 2 轮"修复"成用户从没提过的文件 —— 必须仍被拒绝、不提交。"""
    from pipeline.agent.llm import LLMResult

    def reply(source):
        body = {"summary": "x", "params": {"source": source},
                "stages": [{"id": "probe", "type": "probe", "deps": []}]}
        return LLMResult(json.dumps(body), 10, 10, False, "scripted")

    class Scripted:
        provider, model = "scripted", "scripted"
        def __init__(self):
            self.outs = iter(["../../etc/passwd.mp4", "input.mp4", "input.mp4"])
        def complete(self, messages, *, json_mode=False, temperature=0.0):
            return reply(next(self.outs))

    pl = Planner(TracedLLM(Scripted(), engine), engine, retry=R, unit_s=0.05)
    r = pl.plan("把 ../../etc/passwd.mp4 的主角换成红发女孩,发布到测试库")
    assert not r.ok and r.task_id is None and engine.list_tasks() == []
    assert any("没有出现在用户请求里" in e for e in r.errors)


def test_parse_json_loose_tolerates_fences_and_chatter():
    assert parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_loose('好的,结果如下:{"a": {"b": 2}} 以上') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        parse_json_loose("没有 json")


def test_expand_plan_matches_hand_written_template():
    """Planner 生成的 DAG 与手写模板完全一致 —— 说明展开规则(分块依赖)是正确且确定性的。"""
    req = "x"
    plan = PlanModel.model_validate({
        "summary": req, "params": {"source": "demo.mp4", "frames": 24, "chunks": 4, "prompt": "red-haired girl",
                                   "target": "asset-library"},
        "stages": [{"id": t, "type": t, "deps": d} for t, d in [
            ("probe", []), ("extract_frames", ["probe"]), ("extract_audio", ["probe"]),
            ("gen_reference", ["probe"]), ("segment", ["extract_frames"]), ("swap", ["segment", "gen_reference"]),
            ("assemble", ["swap"]), ("mux", ["assemble", "extract_audio"]), ("quality_check", ["mux"]),
            ("human_approval", ["quality_check"]), ("publish", ["mux", "quality_check", "human_approval"])]]})
    assert validate_plan(plan.model_dump())[1] == []
    got = {s.key: s for s in expand_plan(plan).steps}
    want = {s.key: s for s in animation_swap_task(unit_s=0.5, require_approval=True).steps}
    assert set(got) == set(want)
    for k in want:
        g, w = got[k], want[k]
        assert (sorted(g.deps), g.params, g.resource, g.cacheable, g.type) == \
               (sorted(w.deps), w.params, w.resource, w.cacheable, w.type), k


# ----------------------------------------------------------------------------- 规划出来的任务能真正跑完
def test_planned_task_executes_end_to_end(engine, data_dir, planner):
    r = planner.plan("把 demo.mp4 的主角换成红发女孩,分成 2 段,8 帧,发布到测试库")
    assert r.ok
    with WorkerPool(data_dir):
        t = wait_task(engine, r.task_id)
    assert t["status"] == "SUCCEEDED"


# ----------------------------------------------------------------------------- OpenAI 兼容客户端(用 httpx MockTransport,不联网)
def _fake_openai(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_compat_client_request_format_usage_and_tracing(engine):
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["body"] = json.loads(req.content)
        seen["auth"] = req.headers.get("authorization")
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "deepseek-chat",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": '{"ok": true}'}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}})

    c = OpenAICompatClient("https://api.example.com/v1", "sk-test", "deepseek-chat", http_client=_fake_openai(handler),
                           max_retries=0)
    llm = TracedLLM(c, engine, price_in_per_1k=0.001, price_out_per_1k=0.002)
    res = llm.complete([{"role": "user", "content": "hi"}], purpose="plan", task_id=None, json_mode=True)
    assert res.text == '{"ok": true}' and (res.prompt_tokens, res.completion_tokens) == (120, 30) and not res.estimated
    assert seen["url"].endswith("/chat/completions") and seen["auth"] == "Bearer sk-test"
    assert seen["body"]["model"] == "deepseek-chat" and seen["body"]["response_format"] == {"type": "json_object"}
    [call] = list_llm_calls(engine)
    assert call["provider"] == "openai-compat" and call["tokens_estimated"] is False
    assert call["cost_usd"] == pytest.approx(120 / 1000 * 0.001 + 30 / 1000 * 0.002)
    assert llm_usage_summary(engine)["cost_usd"] == pytest.approx(120 / 1000 * 0.001 + 30 / 1000 * 0.002)


def test_llm_error_is_traced_and_raised(engine):
    c = OpenAICompatClient("https://api.example.com/v1", "k", "m", max_retries=0,
                           http_client=_fake_openai(lambda r: httpx.Response(500, json={"error": {"message": "boom"}})))
    llm = TracedLLM(c, engine)
    with pytest.raises(Exception):
        llm.complete([{"role": "user", "content": "hi"}], purpose="plan")
    [call] = list_llm_calls(engine)
    assert call["status"] == "error" and "boom" in call["error"]
    assert llm_usage_summary(engine)["errors"] == 1


# ----------------------------------------------------------------------------- 校验器变异测试(与 MockLLM 无关)
FULL_STAGES = [("probe", []), ("extract_frames", ["probe"]), ("extract_audio", ["probe"]),
               ("gen_reference", ["probe"]), ("segment", ["extract_frames"]), ("swap", ["segment", "gen_reference"]),
               ("assemble", ["swap"]), ("mux", ["assemble", "extract_audio"]), ("quality_check", ["mux"]),
               ("human_approval", ["quality_check"]), ("publish", ["mux", "quality_check", "human_approval"])]


def full_plan(**params):
    p = {"source": "demo.mp4", "frames": 24, "chunks": 4, "prompt": "x", "target": "asset-library"}
    p.update(params)
    return {"summary": "", "params": p, "stages": [{"id": t, "type": t, "deps": list(d)} for t, d in FULL_STAGES]}


def test_validator_accepts_the_canonical_plan_and_valid_subsets():
    assert validate_plan(full_plan())[1] == []
    # 去掉可选的尾部阶段仍然合法
    for drop in (["publish"], ["publish", "human_approval"], ["publish", "human_approval", "quality_check"],
                 ["extract_audio", "mux", "quality_check", "human_approval", "publish"]):
        p = full_plan(target="staging")
        p["stages"] = [s for s in p["stages"] if s["type"] not in drop]
        for s in p["stages"]:
            s["deps"] = [d for d in s["deps"] if d not in drop]
        assert validate_plan(p)[1] == [], drop


def test_validator_catches_every_single_mutation():
    mutations = []
    # 1) 逐个删除每个阶段的每个必须依赖
    for i, (t, deps) in enumerate(FULL_STAGES):
        for d in deps:
            p = full_plan()
            p["stages"][i]["deps"] = [x for x in deps if x != d]
            mutations.append((f"drop dep {t}->{d}", p))
    # 2) 逐个把阶段改成非白名单类型 / 让 id 与 type 不一致 / 重复阶段
    for i, (t, _) in enumerate(FULL_STAGES):
        p = full_plan(); p["stages"][i]["type"] = "shell"; mutations.append((f"unknown type for {t}", p))
        p = full_plan(); p["stages"][i]["id"] = t + "_x"; mutations.append((f"id!=type {t}", p))
        p = full_plan(); p["stages"].append(dict(p["stages"][i])); mutations.append((f"duplicate {t}", p))
    # 3) 依赖不存在的阶段;自依赖;以及后置阶段回指前置阶段形成的环
    p = full_plan(); p["stages"][1]["deps"].append("ghost"); mutations.append(("ghost dep", p))
    p = full_plan(); p["stages"][0]["deps"] = ["publish"]; mutations.append(("cycle probe<-publish", p))
    p = full_plan(); p["stages"][6]["deps"].append("assemble"); mutations.append(("self dep", p))
    # 4) 参数越界 / 越权字段 / 非法 source / 非法 target
    for name, params in [("chunks 0", {"chunks": 0}), ("chunks 17", {"chunks": 17}), ("chunks -1", {"chunks": -1}),
                         ("frames 0", {"frames": 0}), ("frames 2001", {"frames": 2001}),
                         ("frames<chunks", {"frames": 3, "chunks": 4}), ("bad target", {"target": "prod-db"}),
                         ("path traversal", {"source": "../x.mp4"}), ("abs path", {"source": "/x.mp4"}),
                         ("bad ext", {"source": "x.sh"}), ("long prompt", {"prompt": "a" * 201})]:
        mutations.append((name, full_plan(**params)))
    p = full_plan(); p["stages"][0]["timeout_s"] = 1; mutations.append(("extra field on stage", p))
    p = full_plan(); p["params"]["chaos"] = {}; mutations.append(("extra field on params", p))
    p = full_plan(); p["rm"] = "-rf"; mutations.append(("extra top-level field", p))
    # 5) 策略:生产库发布必须审批 —— 去掉审批阶段 / 保留阶段但 publish 不依赖它
    p = full_plan(); p["stages"] = [s for s in p["stages"] if s["type"] != "human_approval"]
    p["stages"][-1]["deps"] = ["mux", "quality_check"]; mutations.append(("prod without approval", p))
    p = full_plan(); p["stages"][-1]["deps"] = ["mux", "quality_check"]; mutations.append(("approval not wired", p))
    # 6) 缺 probe
    p = full_plan(); p["stages"] = p["stages"][1:]; p["stages"][0]["deps"] = []; mutations.append(("no probe", p))

    escaped = [name for name, plan in mutations if validate_plan(plan)[1] == []]
    assert not escaped, f"校验器漏过了这些变异: {escaped}"
    assert len(mutations) >= 50


def test_gpu_budget_guard():
    # chunks=16 时 segment+swap+gen_reference=33 个 GPU 步骤,在预算内;人为压低预算后必须被拦住
    import pipeline.agent.planner as pl
    assert validate_plan(full_plan(chunks=16, frames=32))[1] == []
    old = pl.MAX_GPU_STEPS
    pl.MAX_GPU_STEPS = 20
    try:
        errs = validate_plan(full_plan(chunks=16, frames=32))[1]
    finally:
        pl.MAX_GPU_STEPS = old
    assert any("GPU" in e for e in errs)
