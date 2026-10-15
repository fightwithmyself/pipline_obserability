"""故障矩阵 F1~F12(F13 混沌测试见 test_chaos.py)。每个测试对应 docs/DESIGN.md 第 8 节的一行。"""
from __future__ import annotations

import json
import time

import pytest

from conftest import FAST_RETRY, WorkerPool, pid_alive, step, wait_for, wait_task
from pipeline.engine import Engine, InvalidAction
from pipeline.models import RetryPolicy
from pipeline.templates import animation_swap_task

R = RetryPolicy(**FAST_RETRY)


def spec(**kw):
    kw.setdefault("chunks", 2)
    kw.setdefault("frames", 4)
    kw.setdefault("retry", R)
    return animation_swap_task(**kw)


def run(engine, data_dir, s, **pool_kw):
    tid, _ = engine.submit(s)
    with WorkerPool(data_dir, **pool_kw):
        t = wait_task(engine, tid)
    return tid, t


def outcomes(engine, tid, key):
    return [a["outcome"] for a in engine.get_attempts(tid, key)]


# ----------------------------------------------------------------------------- F1
def test_f1_transient_failure_retries_only_failed_step(engine, data_dir):
    s = spec(chaos={"segment_0": {"fail": {"kind": "TRANSIENT", "until_attempt": 2, "at": 0.0}}})
    tid, t = run(engine, data_dir, s)
    assert t["status"] == "SUCCEEDED"
    seg = step(t, "segment_0")
    assert seg["attempt_no"] == 3 and seg["failures"] == 2
    assert outcomes(engine, tid, "segment_0") == ["FAILED", "FAILED", "SUCCEEDED"]
    # 上游与无关步骤只执行一次
    for k in ("probe", "extract_frames", "extract_audio", "gen_reference", "segment_1"):
        assert step(t, k)["attempt_no"] == 1, k
    kinds = [e["detail"].get("kind") for e in engine.get_events(tid)
             if e["to"] == "RETRY_WAIT"]
    assert kinds == ["TRANSIENT", "TRANSIENT"]


def test_f1b_resource_error_is_retried(engine, data_dir):
    s = spec(extra_params={"gen_reference": {"oom_until_attempt": 2}})
    tid, t = run(engine, data_dir, s)
    assert t["status"] == "SUCCEEDED"
    atts = engine.get_attempts(tid, "gen_reference")
    assert [a["error_kind"] for a in atts] == ["RESOURCE", "RESOURCE", None]


# ----------------------------------------------------------------------------- F2
def test_f2_permanent_failure_not_retried_and_blocks_downstream(engine, data_dir):
    tid, t = run(engine, data_dir, spec(source="bad:corrupt.mp4"))
    assert t["status"] == "FAILED"
    probe = step(t, "probe")
    assert probe["status"] == "FAILED" and probe["attempt_no"] == 1    # 永久错误:不重试
    assert all(s["status"] == "BLOCKED" for s in t["steps"] if s["key"] != "probe")
    assert t["has_failed_steps"]


def test_f2b_permanent_failure_keeps_unrelated_branches_running_and_resume_works(engine, data_dir):
    # quality_check 第 1 次永久失败,第 2 次正常
    s = spec(chaos={"quality_check": {"fail": {"kind": "PERMANENT", "until_attempt": 1, "at": 0.0}}})
    tid, _ = engine.submit(s)
    with WorkerPool(data_dir):
        t = wait_task(engine, tid)
        assert t["status"] == "FAILED"
        assert step(t, "publish")["status"] == "BLOCKED"
        # 与失败无关的分支都已完成
        assert all(step(t, k)["status"] == "SUCCEEDED" for k in ("mux", "extract_audio", "assemble"))
        before = {x["key"]: x["attempt_no"] for x in t["steps"]}
        engine.resume_task(tid)
        t2 = wait_task(engine, tid, statuses=("SUCCEEDED",))
    assert step(t2, "quality_check")["attempt_no"] == 2
    assert step(t2, "publish")["attempt_no"] == 1
    for k, n in before.items():     # 其它已成功步骤没有被重跑
        if k not in ("quality_check", "publish"):
            assert step(t2, k)["attempt_no"] == n, k


# ----------------------------------------------------------------------------- F3
def test_f3_budget_exhausted_then_manual_retry_reuses_upstream(engine, data_dir):
    s = spec(chaos={"mux": {"fail": {"kind": "TRANSIENT", "until_attempt": 3, "at": 0.0}}})
    tid, _ = engine.submit(s)
    with WorkerPool(data_dir):
        t = wait_task(engine, tid)
        assert t["status"] == "FAILED"
        mux = step(t, "mux")
        assert mux["status"] == "FAILED" and mux["failures"] == 3
        assert outcomes(engine, tid, "mux") == ["FAILED"] * 3
        assemble_sha = step(t, "assemble")["outputs"]["video.bin"]["sha256"]
        engine.retry_step(tid, "mux")
        t2 = wait_task(engine, tid, statuses=("SUCCEEDED",))
    assert step(t2, "mux")["attempt_no"] == 4
    assert step(t2, "assemble")["attempt_no"] == 1
    assert step(t2, "assemble")["outputs"]["video.bin"]["sha256"] == assemble_sha
    # 历史完整保留
    assert outcomes(engine, tid, "mux") == ["FAILED", "FAILED", "FAILED", "SUCCEEDED"]
    with pytest.raises(InvalidAction):
        engine.retry_step(tid, "mux")        # 已成功的步骤不能再"重试"


# ----------------------------------------------------------------------------- F4 / F5 / F7b
def test_f4_f5_kill9_worker_resumes_from_checkpoint_and_reuses_done_steps(engine, data_dir, procs):
    s = animation_swap_task(chunks=1, frames=8, unit_s=0.5, retry=R, stall_timeout_s=20)
    tid, _ = engine.submit(s)
    wa = procs.spawn("wA", time_scale=1.0, lease_ttl=2.0)

    def mid_segment():
        st = step(engine.get_task(tid), "segment_0")
        ck = st["checkpoint"]
        return st if st["status"] == "RUNNING" and ck and ck["done"] >= 3 else None

    wait_for(mid_segment, 60, what="segment_0 过半")
    snap = {x["key"]: (x["attempt_no"], x["outputs"]) for x in engine.get_task(tid)["steps"]
            if x["status"] == "SUCCEEDED"}
    assert snap, "应该已有已完成步骤"
    child_pid = engine.get_attempts(tid, "segment_0")[0]["child_pid"]
    assert child_pid and pid_alive(child_pid)

    procs.kill9(wa)                                              # kill -9 整个 worker
    wait_for(lambda: not pid_alive(child_pid), 10, what="孤儿步骤子进程自杀(F7b)")

    procs.spawn("wB", time_scale=1.0, lease_ttl=2.0)           # “重启”:新进程、新连接
    t = wait_task(engine, tid, timeout=90)
    assert t["status"] == "SUCCEEDED"

    seg = step(t, "segment_0")
    atts = engine.get_attempts(tid, "segment_0")
    assert atts[0]["outcome"] == "LOST" and atts[0]["error_kind"] == "LEASE_LOST"
    assert atts[-1]["outcome"] == "SUCCEEDED"
    resumed = atts[-1]["notes"]["resumed_from"]
    assert resumed >= 3                                        # 从检查点继续,而不是从头
    assert atts[-1]["notes"]["frames_computed"] == 8 - resumed
    assert seg["lost_count"] == 1 and seg["failures"] == 0     # 租约丢失不占失败预算
    for k, (n, outs) in snap.items():                          # F5:已完成步骤没有重跑、产物未变
        cur = step(t, k)
        assert cur["attempt_no"] == n and cur["outputs"] == outs, k


# ----------------------------------------------------------------------------- F6
def test_f6_zombie_commit_is_fenced(data_dir, tmp_path):
    now = [1000.0]
    eng = Engine(data_dir, clock=lambda: now[0])
    tid, _ = eng.submit(spec())

    ca = eng.claim("A", lease_ttl=10)
    assert ca.key == "probe" and ca.attempt_no == 1
    now[0] += 11                                   # A 失联,租约过期(例如被 SIGSTOP/GC 停顿)
    cb = eng.claim("B", lease_ttl=10)              # B 认领前会回收 A 的租约
    assert cb.key == "probe" and cb.attempt_no == 2

    f = tmp_path / "x.bin"
    f.write_bytes(b"zombie result")
    sha, size = eng.store.put_file(f)
    outs = {"meta.json": {"sha256": sha, "size": size}}

    assert eng.heartbeat(ca.step_id, ca.attempt_no, 10) is False        # 僵尸续租被拒
    assert eng.complete(ca.step_id, ca.attempt_no, outs) is False       # 僵尸提交被拒
    assert eng.fail(ca.step_id, ca.attempt_no, "TRANSIENT", "late") is False
    st = step(eng.get_task(tid), "probe")
    assert st["status"] == "RUNNING" and st["lease_owner"] == "B"       # B 仍然持有

    assert eng.complete(cb.step_id, cb.attempt_no, outs) is True        # 合法持有者提交成功
    assert eng.complete(cb.step_id, cb.attempt_no, outs) is False       # 重复提交也被拒
    kinds = [e["kind"] for e in eng.get_events(tid)]
    assert kinds.count("COMMIT_REJECTED") == 2
    assert [a["outcome"] for a in eng.get_attempts(tid, "probe")] == ["LOST", "SUCCEEDED"]


# ----------------------------------------------------------------------------- F7
def test_f7_stall_is_killed_but_slow_progress_is_not(engine, data_dir):
    # 卡死:attempt 1 在 50% 后停止上报进展 -> 被判 STALL、强杀、重试成功
    s = spec(stall_timeout_s=1.0,
             chaos={"segment_0": {"hang": {"at": 0.5, "on_attempts": [1]}}})
    tid, t = run(engine, data_dir, s)
    assert t["status"] == "SUCCEEDED"
    atts = engine.get_attempts(tid, "segment_0")
    assert atts[0]["error_kind"] == "STALL" and atts[1]["outcome"] == "SUCCEEDED"
    assert not pid_alive(atts[0]["child_pid"])                      # 卡死的子进程确实被杀掉

    # 慢但有进展:总耗时 3s > stall_timeout 1s,但每 0.3s 上报一次 -> 不能被误杀
    s2 = animation_swap_task(chunks=1, frames=10, unit_s=0.3, retry=R, stall_timeout_s=1.0,
                             prompt="slow-but-alive")
    tid2, _ = engine.submit(s2)
    with WorkerPool(data_dir, time_scale=1.0, lease_ttl=1.5):
        t2 = wait_task(engine, tid2, timeout=90)
    assert t2["status"] == "SUCCEEDED"
    assert step(t2, "segment_0")["attempt_no"] == 1 and step(t2, "segment_0")["failures"] == 0


def test_f7b_total_timeout(engine, data_dir):
    s = animation_swap_task(chunks=1, frames=10, unit_s=0.3, timeout_s=1.0, stall_timeout_s=5.0,
                            retry=RetryPolicy(max_attempts=2, base_delay=0.1, max_delay=0.2, jitter=0))
    tid, _ = engine.submit(s)
    with WorkerPool(data_dir, time_scale=1.0):
        t = wait_task(engine, tid, timeout=60)
    assert t["status"] == "FAILED"
    assert [a["error_kind"] for a in engine.get_attempts(tid, "segment_0")] == ["TIMEOUT", "TIMEOUT"]


# ----------------------------------------------------------------------------- F8 / F9
def test_f8_cache_reuse_across_tasks(engine, data_dir):
    base = dict(chunks=2, frames=4, retry=R)
    tid1, t1 = run(engine, data_dir, animation_swap_task(**base))
    assert t1["status"] == "SUCCEEDED" and t1["cached_steps"] == 0

    # 完全相同的任务:除 publish(有副作用、不缓存)外全部命中缓存,且没有任何 attempt
    tid2, t2 = run(engine, data_dir, animation_swap_task(**base))
    assert t2["status"] == "SUCCEEDED"
    assert t2["cached_steps"] == t2["steps_total"] - 1
    for st in t2["steps"]:
        if st["key"] == "publish":
            continue
        assert st["cached"] and st["attempt_no"] == 0, st["key"]
    pub = step(t2, "publish")
    assert pub["attempt_no"] == 1
    assert engine.get_attempts(tid2, "publish")[0]["notes"]["publish"] == "deduplicated"   # 幂等键去重
    assert len(list((data_dir / "published").glob("*.json"))) == 1

    # 只改下游相关参数(新角色描述):抠图/拆帧等上游全部复用,生成与替换重算
    tid3, t3 = run(engine, data_dir, animation_swap_task(prompt="blue-haired boy", **base))
    assert t3["status"] == "SUCCEEDED"
    for k in ("probe", "extract_frames", "extract_audio", "segment_0", "segment_1"):
        assert step(t3, k)["cached"], k
    for k in ("gen_reference", "swap_0", "swap_1", "assemble", "mux"):
        assert not step(t3, k)["cached"], k


def test_f9_corrupted_artifact_invalidates_cache_and_is_repaired(engine, data_dir):
    base = dict(chunks=2, frames=4, retry=R)
    _, t1 = run(engine, data_dir, animation_swap_task(**base))
    sha = step(t1, "segment_0")["outputs"]["mask.bin"]["sha256"]
    p = engine.store.path(sha)
    p.write_bytes(b"tampered!")                       # 篡改产物
    assert not engine.store.verify(sha)

    tid2, t2 = run(engine, data_dir, animation_swap_task(**base))
    assert t2["status"] == "SUCCEEDED"
    seg = step(t2, "segment_0")
    assert not seg["cached"] and seg["attempt_no"] == 1                   # 未命中,重新计算
    assert any(e["kind"] == "CACHE_INVALID" and e["step_key"] == "segment_0" for e in engine.get_events(tid2))
    assert seg["outputs"]["mask.bin"]["sha256"] == sha                      # 确定性:重算得到同一 hash
    assert engine.store.verify(sha)                                         # 存储里的坏文件已被修复


# ----------------------------------------------------------------------------- F10
def test_f10_concurrent_claim_runs_each_step_exactly_once(engine, data_dir):
    # source 不同 => 任务之间不共享任何缓存,每个步骤都必须真实执行恰好一次
    ids = [engine.submit(spec(source=f"clip{i}.mp4", prompt=f"p{i}"))[0] for i in range(3)]
    with WorkerPool(data_dir, n=4):
        ts = [wait_task(engine, i, timeout=90) for i in ids]
    for t in ts:
        assert t["status"] == "SUCCEEDED"
        assert all(s["attempt_no"] == 1 for s in t["steps"])
    n_steps = sum(t["steps_total"] for t in ts)
    n_attempts = engine.conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
    assert n_attempts == n_steps                                           # 无重复执行
    assert not [e for e in engine.get_events() if e["kind"] in ("COMMIT_REJECTED", "FAIL_REJECTED")]


# ----------------------------------------------------------------------------- F11
def test_f11_cancel_stops_running_child_and_blocks_new_steps(engine, data_dir):
    s = animation_swap_task(chunks=1, frames=8, unit_s=0.5, retry=R, stall_timeout_s=20)
    tid, _ = engine.submit(s)
    with WorkerPool(data_dir, time_scale=1.0, lease_ttl=1.5):
        st = wait_for(lambda: (lambda x: x if x["status"] == "RUNNING" and x["type"] == "segment"
                               else None)(step(engine.get_task(tid), "segment_0")), 60, what="segment 运行")
        pid = engine.get_attempts(tid, "segment_0")[0]["child_pid"]
        engine.cancel_task(tid, "test")
        wait_for(lambda: not pid_alive(pid), 10, what="取消后子进程被杀")
        time.sleep(0.5)
    t = engine.get_task(tid)
    assert t["status"] == "CANCELLED"
    assert step(t, "segment_0")["status"] == "CANCELLED"
    assert engine.get_attempts(tid, "segment_0")[0]["outcome"] == "CANCELLED"
    assert all(step(t, k)["attempt_no"] == 0 for k in ("swap_0", "assemble", "publish"))   # 没有新步骤被启动
    with pytest.raises(InvalidAction):
        engine.resume_task(tid)


# ----------------------------------------------------------------------------- F12
def test_f12_idempotent_submit(engine):
    a, c1 = engine.submit(spec(), idempotency_key="req-1")
    b, c2 = engine.submit(spec(), idempotency_key="req-1")
    assert a == b and c1 is True and c2 is False
    assert len(engine.list_tasks()) == 1
