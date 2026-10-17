"""失败诊断 Agent 对"逐帧风格化流水线"的处置:能自动恢复的自动恢复,不能的显式标记并给人工恢复步骤。

错误文本取自真实抛出点(pipeline/steps/anime.py、pipeline/doubao.py),文本被改了这里会红。
LLM 一律用 MockLLM(确定性规则),所以这里验证的是**系统护栏与流程**,不是真实模型的诊断质量。
"""
from __future__ import annotations

import pytest

from pipeline.agent import runbook
from pipeline.agent.diagnosis import DiagnosisAgent
from pipeline.agent.llm import TracedLLM
from pipeline.agent.mock_llm import MockLLM
from pipeline.engine import Engine
from pipeline.models import RetryPolicy
from pipeline.templates import frame_style_inject_presets
from tests.conftest import FAST_RETRY, step
from tests.test_anime_pipeline import clip, run_to_end, submit  # noqa: F401  (clip 是 fixture)


# --------------------------------------------------------------------------- 手册匹配(真实错误文本)
@pytest.mark.parametrize("kind,msg,code,auto", [
    ("PERMANENT", "豆包接口返回任务级错误,已停止继续调用: auth: 401 AuthenticationError bad key", "auth", False),
    ("PERMANENT", "豆包接口返回任务级错误,已停止继续调用: auth: 没有设置环境变量 ARK_API_KEY", "auth", False),
    ("PERMANENT", "豆包接口返回任务级错误,已停止继续调用: quota: 402 AccountOverdueError", "quota", False),
    ("PERMANENT", "豆包接口返回任务级错误,已停止继续调用: model_not_open: 404 ModelNotOpen", "model_not_open", False),
    ("PERMANENT", "豆包接口返回任务级错误,已停止继续调用: not_found: 404 InvalidEndpointOrModel.NotFound", "model_not_found", False),
    ("PERMANENT", "豆包接口返回任务级错误,已停止继续调用: bad_request: 400 InvalidParameter", "bad_request", False),
    ("PERMANENT", "本块累计费用估算已触及预算保护线(≈1.00 元),已停止;已完成的帧会保留", "budget", False),
    ("PERMANENT", "1 帧被内容审核拒绝([3]),on_reject=fail:请对这些帧用 frame_overrides 换提示词", "moderation", False),
    ("PERMANENT", "降级(原图直通)的帧 5/7 超过上限 25%,请检查提示词或素材", "too_many_degraded", False),
    ("PERMANENT", "质检未通过: 帧数不一致", "qc_failed", False),
    ("PERMANENT", "素材文件不存在: x.mp4", "bad_input", False),
    ("TRANSIENT", "ffmpeg: x264 malloc failed / Could not open encoder", "host_memory", True),
    ("TRANSIENT", "1 帧失败、0 帧未处理,将重试(已完成的帧不会重做): 3:429", "frames_failed", True),
    ("LEASE_LOST", "worker 失联", "poison", False),
    ("PERMANENT", "某个从没见过的错误", "unknown", False),
])
def test_runbook_matches_real_error_texts(kind, msg, code, auto):
    pb = runbook.match(kind, msg)
    assert pb.code == code and pb.auto_retry is auto
    assert pb.human_steps, "每个场景都必须给出人工恢复步骤"
    assert (pb.max_auto > 0) == auto


# --------------------------------------------------------------------------- 无法自动恢复 → 标记需要人工
def _agent(data_dir, *, auto):
    e = Engine(data_dir)
    return e, DiagnosisAgent(data_dir, e, TracedLLM(MockLLM(), e), auto_recover=auto)


def test_auth_failure_is_marked_needs_human_with_steps_and_never_auto_retried(engine, data_dir, clip):
    inj = {"stylize_0": {"doubao": {"frames": {"__first__": {"kind": "auth"}}}}}
    tid = submit(engine, clip, chaos=inj, concurrency=1)
    assert run_to_end(engine, data_dir, tid)["status"] == "FAILED"

    e, a = _agent(data_dir, auto=True)               # 即使开启自动恢复,鉴权失败也不能自动重试
    out = a.handle_failures(tid)
    assert [o["step_key"] for o in out] == ["stylize_0"]
    o = out[0]
    assert o["proposal"]["playbook"] == "auth"
    assert o["result"]["executed"] is False and o["result"]["needs_human"] is True
    assert any("ARK_API_KEY" in s for s in o["result"]["human_steps"])
    assert step(engine.get_task(tid), "stylize_0")["status"] == "FAILED"          # 没有被动过
    ev = [x for x in e.get_events(tid) if x["kind"] == "AGENT_NEEDS_HUMAN"]
    assert len(ev) == 1 and ev[0]["detail"]["human_steps"]
    assert a.handle_failures(tid) == []                                          # 同一次失败只诊断一次
    a.close()
    e.close()


def test_moderation_failure_gets_override_instructions(engine, data_dir, clip):
    tid = submit(engine, clip, chaos=frame_style_inject_presets(2)["moderation"], on_reject="fail")
    assert run_to_end(engine, data_dir, tid)["status"] == "FAILED"
    e, a = _agent(data_dir, auto=True)
    o = a.handle_failures(tid)[0]
    assert o["proposal"]["playbook"] == "moderation" and o["result"]["needs_human"] is True
    assert any("frame_overrides" in s for s in o["result"]["human_steps"])
    a.close()
    e.close()


# --------------------------------------------------------------------------- 可自动恢复 → 自己重试并成功
def test_transient_frame_failure_is_auto_recovered(engine, data_dir, clip):
    retry = RetryPolicy(**{**FAST_RETRY, "max_attempts": 1})            # 引擎自己不再重试,逼它走到 FAILED
    tid = submit(engine, clip, chaos=frame_style_inject_presets(2)["rate_limit"], frame_max_attempts=3, retry=retry)
    t = run_to_end(engine, data_dir, tid)
    assert t["status"] == "FAILED" and step(t, "stylize_0")["error_kind"] == "TRANSIENT"

    e, a = _agent(data_dir, auto=True)
    o = a.handle_failures(tid)[0]
    assert o["proposal"]["auto_eligible"] is True
    assert o["result"]["executed"] is True and o["result"]["by"] == "agent-auto"
    assert any(x["kind"] == "AGENT_ACTION" for x in e.get_events(tid))
    t2 = run_to_end(engine, data_dir, tid, timeout=120)                  # 重试后 429 已过去,任务完成
    assert t2["status"] == "SUCCEEDED"
    a.close()
    e.close()


def test_without_auto_flag_the_agent_only_proposes_and_waits_for_human(engine, data_dir, clip):
    retry = RetryPolicy(**{**FAST_RETRY, "max_attempts": 1})
    tid = submit(engine, clip, chaos=frame_style_inject_presets(2)["rate_limit"], frame_max_attempts=3, retry=retry)
    assert run_to_end(engine, data_dir, tid)["status"] == "FAILED"
    e, a = _agent(data_dir, auto=False)
    o = a.handle_failures(tid)[0]
    assert o["status"] == "AWAITING_HUMAN" and o["proposal"]["auto_eligible"] is False
    assert step(engine.get_task(tid), "stylize_0")["status"] == "FAILED"     # 没有人确认,什么都没动
    a.close()
    e.close()


def test_auto_recovery_is_capped_then_marked_needs_human(engine, data_dir, clip):
    """一直失败的瞬时错误:最多自动重试 max_auto 次,之后必须转人工,不能无限循环。"""
    inj = {"stylize_0": {"doubao": {"frames": {"__first__": {"kind": "rate_limit"}}}}}     # 每次都 429
    retry = RetryPolicy(**{**FAST_RETRY, "max_attempts": 1})
    tid = submit(engine, clip, chaos=inj, frame_max_attempts=2, retry=retry)
    assert run_to_end(engine, data_dir, tid, timeout=120)["status"] == "FAILED"
    e, a = _agent(data_dir, auto=True)
    pb = runbook.match("TRANSIENT", "1 帧失败、0 帧未处理,将重试(已完成的帧不会重做): 3:429")
    results = []
    for _ in range(pb.max_auto + 2):
        out = a.handle_failures(tid)
        if not out:
            break
        results.append(out[0]["result"])
        if out[0]["result"].get("executed"):
            assert run_to_end(engine, data_dir, tid, timeout=120)["status"] == "FAILED"
    assert [bool(r.get("executed")) for r in results] == [True] * pb.max_auto + [False]
    assert results[-1]["needs_human"] is True
    assert any(x["kind"] == "AGENT_NEEDS_HUMAN" for x in e.get_events(tid))
    a.close()
    e.close()


def test_garbage_llm_never_triggers_auto_action_but_still_gives_human_steps(engine, data_dir, clip):
    class Garbage(MockLLM):
        def complete(self, messages, **kw):
            r = super().complete(messages, **kw)
            r.text = "随便重试一下吧"
            return r

    retry = RetryPolicy(**{**FAST_RETRY, "max_attempts": 1})
    tid = submit(engine, clip, chaos=frame_style_inject_presets(2)["rate_limit"], frame_max_attempts=3, retry=retry)
    assert run_to_end(engine, data_dir, tid)["status"] == "FAILED"
    e = Engine(data_dir)
    a = DiagnosisAgent(data_dir, e, TracedLLM(Garbage(), e), auto_recover=True)
    o = a.handle_failures(tid)[0]
    assert o["proposal"]["fallback"] is True and o["result"]["executed"] is False
    assert o["result"]["needs_human"] is True and o["result"]["human_steps"]
    assert step(engine.get_task(tid), "stylize_0")["status"] == "FAILED"
    a.close()
    e.close()
