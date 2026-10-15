"""引擎单元/语义测试:规格校验、状态机、租约预算、审批、资源路由、非确定性步骤复用。"""
from __future__ import annotations

import time

import pytest

from conftest import FAST_RETRY, WorkerPool, step, wait_for, wait_task
from pipeline.db import tx
from pipeline.engine import Engine, IllegalTransition, InvalidAction
from pipeline.models import MAX_LOST, RetryPolicy, SpecError, StepSpec, TaskSpec, validate_spec
from pipeline.templates import animation_swap_task

R = RetryPolicy(**FAST_RETRY)


def spec(**kw):
    kw.setdefault("chunks", 2)
    kw.setdefault("frames", 4)
    kw.setdefault("retry", R)
    return animation_swap_task(**kw)


# ----------------------------------------------------------------------------- 规格校验
def test_validate_spec_rejects_bad_dags():
    S = StepSpec
    with pytest.raises(SpecError, match="至少"):
        validate_spec(TaskSpec("x", []))
    with pytest.raises(SpecError, match="重复"):
        validate_spec(TaskSpec("x", [S("a", "probe"), S("a", "probe")]))
    with pytest.raises(SpecError, match="不存在"):
        validate_spec(TaskSpec("x", [S("a", "probe", deps=["zzz"])]))
    with pytest.raises(SpecError, match="依赖自身"):
        validate_spec(TaskSpec("x", [S("a", "probe", deps=["a"])]))
    with pytest.raises(SpecError, match="环"):
        validate_spec(TaskSpec("x", [S("a", "probe", deps=["c"]), S("b", "probe", deps=["a"]),
                                     S("c", "probe", deps=["b"])]))
    with pytest.raises(SpecError, match="resource"):
        validate_spec(TaskSpec("x", [S("a", "probe", resource="tpu")]))


def test_template_is_a_valid_dag_with_expected_shape():
    s = spec(chunks=4, frames=8, require_approval=True)
    keys = {x.key: x for x in s.steps}
    # probe/extract_frames/extract_audio/gen_reference + 4 segment + 4 swap + assemble/mux/qc/approval/publish
    assert len(s.steps) == 4 + 4 + 4 + 5
    assert keys["swap_2"].deps == ["segment_2", "gen_reference"]
    assert keys["publish"].deps == ["mux", "quality_check", "human_approval"]
    assert keys["publish"].cacheable is False and keys["segment_0"].resource == "gpu"


# ----------------------------------------------------------------------------- 状态机
def test_illegal_transition_is_rejected(data_dir):
    eng = Engine(data_dir)
    eng.submit(spec())
    c = eng.claim("w")
    with pytest.raises(IllegalTransition):
        with tx(eng.conn):
            eng._transition(c.step_id, "PENDING")       # RUNNING -> PENDING 不在转换表里
    assert step(eng.get_task(c.task_id), "probe")["status"] == "RUNNING"   # 事务回滚,状态未变


def test_task_status_is_derived_from_steps(data_dir):
    eng = Engine(data_dir)
    tid, _ = eng.submit(spec())
    assert eng.get_task(tid)["status"] == "PENDING"
    eng.claim("w")
    assert eng.get_task(tid)["status"] == "RUNNING"


# ----------------------------------------------------------------------------- 租约与预算
def test_lease_expiry_is_reclaimed_without_consuming_failure_budget(data_dir):
    now = [100.0]
    eng = Engine(data_dir, clock=lambda: now[0])
    tid, _ = eng.submit(spec())
    c1 = eng.claim("A", lease_ttl=5)
    now[0] += 4
    assert eng.heartbeat(c1.step_id, c1.attempt_no, 5)          # 续租后租约顺延
    now[0] += 4
    assert eng.sweep() == 0                                     # 还没过期
    now[0] += 2
    assert eng.sweep() == 1                                     # 过期,被回收
    st = step(eng.get_task(tid), "probe")
    assert st["status"] == "RETRY_WAIT" and st["failures"] == 0 and st["lost_count"] == 1
    c2 = eng.claim("B", lease_ttl=5)                            # LOST 不退避,立即可被认领
    assert c2.key == "probe" and c2.attempt_no == 2


def test_poison_step_stops_after_lost_budget(data_dir):
    now = [0.0]
    eng = Engine(data_dir, clock=lambda: now[0])
    tid, _ = eng.submit(spec())
    for _ in range(MAX_LOST + 1):
        assert eng.claim("W", lease_ttl=1) is not None
        now[0] += 2
        eng.sweep()
    st = step(eng.get_task(tid), "probe")
    assert st["status"] == "FAILED" and st["lost_count"] == MAX_LOST + 1
    reasons = [e["detail"].get("reason") for e in eng.get_events(tid)
               if e["kind"] == "STEP_STATUS" and e["to"] == "FAILED"]
    assert reasons and "lost budget" in reasons[0]
    assert eng.get_task(tid)["status"] == "FAILED"


def test_backoff_delays_the_retry(data_dir):
    now = [50.0]
    eng = Engine(data_dir, clock=lambda: now[0])
    s = spec(retry=RetryPolicy(max_attempts=3, base_delay=4.0, max_delay=100.0, jitter=0.0))
    tid, _ = eng.submit(s)
    c = eng.claim("w")
    eng.fail(c.step_id, c.attempt_no, "TRANSIENT", "boom")
    assert eng.claim("w") is None                      # 退避期间不可被认领
    now[0] += 3.9
    assert eng.claim("w") is None
    now[0] += 0.2
    c2 = eng.claim("w")
    assert c2 and c2.attempt_no == 2
    eng.fail(c2.step_id, c2.attempt_no, "TRANSIENT", "boom2")
    now[0] += 7.9                                      # 第二次退避 = 8s(指数)
    assert eng.claim("w") is None
    now[0] += 0.2
    assert eng.claim("w") is not None


# ----------------------------------------------------------------------------- 人工审批
def test_human_approval_does_not_occupy_worker_and_can_resume(engine, data_dir):
    tid, _ = engine.submit(spec(require_approval=True))
    with WorkerPool(data_dir):
        t = wait_for(lambda: (lambda x: x if x["waiting_approval"] else None)(engine.get_task(tid)), 60,
                     what="等待审批")
        assert t["status"] == "RUNNING" and t["waiting_approval"] == ["human_approval"]
        assert step(t, "publish")["status"] == "PENDING"
        time.sleep(0.5)
        assert engine.get_task(tid)["status"] == "RUNNING"      # 一直等待,不超时、不占 worker
        engine.approve(tid, "human_approval", "approve", who="alice", comment="LGTM")
        t = wait_task(engine, tid, statuses=("SUCCEEDED",))
    assert step(t, "publish")["attempt_no"] == 1
    with pytest.raises(InvalidAction):
        engine.approve(tid, "human_approval", "approve")        # 已处理的不能重复审批


def test_rejected_approval_blocks_publish_until_retried(engine, data_dir):
    tid, _ = engine.submit(spec(require_approval=True))
    with WorkerPool(data_dir):
        wait_for(lambda: engine.get_task(tid)["waiting_approval"], 60, what="等待审批")
        engine.approve(tid, "human_approval", "reject", who="bob", comment="角色边缘不对")
        t = wait_task(engine, tid, statuses=("FAILED",))
        assert step(t, "human_approval")["status"] == "FAILED" and step(t, "publish")["status"] == "BLOCKED"
        engine.retry_step(tid, "human_approval")                # 修复后重新请求审批
        wait_for(lambda: engine.get_task(tid)["waiting_approval"], 30, what="重新等待审批")
        engine.approve(tid, "human_approval", "approve", who="bob")
        t = wait_task(engine, tid, statuses=("SUCCEEDED",))
    assert step(t, "mux")["attempt_no"] == 1                    # 审批被拒不会让上游重算
    rows = engine.conn.execute("SELECT decision FROM approvals ORDER BY id").fetchall()
    assert [r[0] for r in rows] == ["reject", "approve"]


# ----------------------------------------------------------------------------- 资源路由
def test_cpu_worker_never_runs_gpu_steps(engine, data_dir):
    tid, _ = engine.submit(spec())
    with WorkerPool(data_dir, resources=("cpu",)):
        wait_for(lambda: step(engine.get_task(tid), "extract_audio")["status"] == "SUCCEEDED", 60, what="cpu 步骤")
        time.sleep(0.5)
        t = engine.get_task(tid)
        assert t["status"] == "RUNNING"
        for s in t["steps"]:
            if s["resource"] == "gpu":
                assert s["attempt_no"] == 0, s["key"]
    with WorkerPool(data_dir, resources=("cpu", "gpu")):
        assert wait_task(engine, tid)["status"] == "SUCCEEDED"
    ws = {w["id"]: w["resources"] for w in engine.list_workers()}
    assert ws["tw0"] == ["cpu", "gpu"]


# ----------------------------------------------------------------------------- 非确定性步骤必须靠“固化产物”复用
def test_nondeterministic_step_is_reused_not_recomputed(engine, data_dir):
    s = spec(nondeterministic_ref=True,
             chaos={"swap_0": {"fail": {"kind": "PERMANENT", "until_attempt": 1, "at": 0.0}}})
    tid, _ = engine.submit(s)
    with WorkerPool(data_dir):
        t = wait_task(engine, tid)
        assert t["status"] == "FAILED"
        ref_sha = step(t, "gen_reference")["outputs"]["ref.bin"]["sha256"]
        engine.resume_task(tid)
        t2 = wait_task(engine, tid, statuses=("SUCCEEDED",))
    ref2 = step(t2, "gen_reference")
    assert ref2["attempt_no"] == 1 and ref2["outputs"]["ref.bin"]["sha256"] == ref_sha   # 新角色长相没变

    # 对照:真的重新执行一次生成步骤,得到的是不同的结果 —— 说明不能靠“重算”来复用
    s2 = spec(nondeterministic_ref=True, extra_params={"gen_reference": {"salt": 1}})
    tid2, _ = engine.submit(s2)
    with WorkerPool(data_dir):
        t3 = wait_task(engine, tid2, statuses=("SUCCEEDED",))
    assert step(t3, "gen_reference")["outputs"]["ref.bin"]["sha256"] != ref_sha
