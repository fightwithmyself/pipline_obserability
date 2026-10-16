"""主流水线(豆包逐帧风格化)的端到端测试。provider 一律用 MockDoubao:不联网、不花钱、结果可复现。

重点验证的是"恢复语义":单帧失败只重试那一帧、崩溃后复用已完成的帧、响应丢失时最多多付一次并留痕、
同素材再提交不再付费。这些都是对着 Mock 服务端的**计费账本**(mock_doubao_billing.jsonl)断言的。
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline.models import RetryPolicy
from pipeline.steps.anime import select_keyframes
from pipeline.templates import frame_style_task, frame_style_inject_presets
from tests.clipgen import make_clip
from tests.conftest import FAST_RETRY, WorkerPool, step, wait_task

FPS = 12
BASE = dict(chunks=2, stride=3, max_gap=8, dup_thr=0.0, cut_thr=18.0, target_height=108, fps=FPS,
            backoff_base_s=0.05, rpm=60000.0, concurrency=3, prompt="日系动漫风格,保持人物与动作不变",
            budget_cny=50.0)


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    return make_clip(tmp_path_factory.mktemp("clips") / "walk.mp4", w=192, h=108, fps=FPS, secs=4.0,
                     audio=True, cut_at=2.0)


def submit(engine, clip, **kw):
    args = {**BASE, **kw}
    args.setdefault("retry", RetryPolicy(**FAST_RETRY))
    spec = frame_style_task(source=str(clip), **args)
    tid, _ = engine.submit(spec)
    return tid


def billing(data_dir: Path) -> list[dict]:
    d = data_dir / "mock_doubao_billing"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def out_json(engine, task_id: str, key: str, name: str) -> dict:
    t = engine.get_task(task_id)
    sha = step(t, key)["outputs"][name]["sha256"]
    return json.loads(engine.store.path(sha).read_text(encoding="utf-8"))


def run_to_end(engine, data_dir, tid, n_workers=2, timeout=240.0):
    with WorkerPool(data_dir, n=n_workers):
        return wait_task(engine, tid, timeout=timeout)


# --------------------------------------------------------------------------- 纯函数
def test_select_keyframes_rules():
    n = 60
    g = np.zeros((n, 8, 8), dtype=np.uint8)
    g[30:] = 200                                   # 第 30 帧镜头切换
    r = select_keyframes(g, stride=6, max_gap=20, dup_thr=1.0, cut_thr=18.0)
    k = r["keyframes"]
    assert k[0] == 0 and k[-1] == n - 1            # 首尾必选
    assert 29 in k and 30 in k                     # 切换点及其前一帧都是关键帧 → 补间不会跨镜头
    assert r["cuts"] == [30]
    # 画面完全静止的区段:候选帧被当作重复帧跳过,直到达到 max_gap 才强制取
    assert r["duplicates_skipped"] > 0
    gaps = np.diff(k)
    assert gaps.max() <= 20


def test_select_keyframes_dense_motion_uses_stride():
    rng = np.random.default_rng(1)
    g = rng.integers(0, 255, size=(48, 6, 6), dtype=np.uint8)
    k = select_keyframes(g, stride=6, max_gap=24, dup_thr=0.5, cut_thr=300.0)["keyframes"]
    assert np.diff(k)[:-1].min() >= 6              # 没有切换时,间隔不小于 stride(末帧强制加入除外)


# --------------------------------------------------------------------------- 端到端
def test_end_to_end_mock(engine, data_dir, clip):
    tid = submit(engine, clip)
    t = run_to_end(engine, data_dir, tid)
    assert t["status"] == "SUCCEEDED", [(s["key"], s["status"], s["error_msg"]) for s in t["steps"] if s["status"] != "SUCCEEDED"]
    tl = out_json(engine, tid, "keyframes", "timeline.json")
    nk = tl["n_keyframes"]
    assert nk >= 4
    # 镜头切换(2.0s = 第 24 帧)两侧都是关键帧
    assert 23 in tl["keyframes"] and 24 in tl["keyframes"]

    meta = out_json(engine, tid, "package", "metadata.json")
    assert meta["stylize"]["billed_images"] == nk == len(billing(data_dir))
    assert meta["output"]["frames"] == tl["n_frames"]
    assert meta["output"]["audio"] == "source"
    assert meta["qc"]["ok"] is True
    assert meta["stylize"]["cost_cny_est"] == pytest.approx(nk * 0.12, abs=1e-3)

    # 逐帧账本:每个关键帧一行,全部 DONE,且各自调用了 1 次
    items = engine.list_items(tid, prefix="stylize_")
    assert len(items) == nk and {i["status"] for i in items} == {"DONE"}
    assert all(i["calls"] == 1 and i["billed"] == 1 for i in items)

    # 风格化确实改了像素(Mock 的色阶分层),且尺寸与源一致
    sha_src = step(engine.get_task(tid), "keyframes")["outputs"]["kf_0.zip"]["sha256"]
    sha_out = step(engine.get_task(tid), "stylize_0")["outputs"]["styled.zip"]["sha256"]
    zs, zo = zipfile.ZipFile(engine.store.path(sha_src)), zipfile.ZipFile(engine.store.path(sha_out))
    assert sorted(zs.namelist()) == sorted(zo.namelist())
    name = sorted(zs.namelist())[0]
    a = np.asarray(Image.open(io.BytesIO(zs.read(name))).convert("RGB")).astype(int)
    b = np.asarray(Image.open(io.BytesIO(zo.read(name))).convert("RGB")).astype(int)
    assert a.shape == b.shape and np.abs(a - b).mean() > 3

    # 成品能解码:视频 + 音轨,帧数与标准化素材一致
    final = engine.store.path(step(engine.get_task(tid), "package")["outputs"]["final.mp4"]["sha256"])
    from pipeline.media import count_decoded_frames, probe_video
    assert count_decoded_frames(final) == tl["n_frames"]
    assert probe_video(final)["has_audio"]
    # 发布是幂等的
    assert step(engine.get_task(tid), "publish")["status"] == "SUCCEEDED"


def test_same_source_again_costs_nothing(engine, data_dir, clip):
    """同一素材换一种分块方式再提交:步骤缓存不命中(输入不同),但帧级缓存命中 → 接口 0 次调用。"""
    t1 = run_to_end(engine, data_dir, submit(engine, clip))
    assert t1["status"] == "SUCCEEDED"
    n1 = len(billing(data_dir))
    tid2 = submit(engine, clip, chunks=3)
    t2 = run_to_end(engine, data_dir, tid2)
    assert t2["status"] == "SUCCEEDED"
    assert len(billing(data_dir)) == n1                      # 没有任何新的计费
    meta = out_json(engine, tid2, "package", "metadata.json")
    assert meta["stylize"]["billed_images"] == 0
    assert meta["stylize"]["cached_frames"] == meta["keyframes"]["count"]
    assert meta["stylize"]["cost_cny_est"] == 0


def test_prompt_change_invalidates_frame_cache(engine, data_dir, clip):
    run_to_end(engine, data_dir, submit(engine, clip))
    n1 = len(billing(data_dir))
    t2 = run_to_end(engine, data_dir, submit(engine, clip, prompt="水彩风格"))
    assert t2["status"] == "SUCCEEDED"
    assert len(billing(data_dir)) == 2 * n1                  # 提示词变了,缓存 key 变了,全部重算


# --------------------------------------------------------------------------- 失败与恢复
def test_transient_errors_retry_inside_the_frame(engine, data_dir, clip):
    inj = frame_style_inject_presets(2)["transient"]
    tid = submit(engine, clip, chaos=inj)
    t = run_to_end(engine, data_dir, tid)
    assert t["status"] == "SUCCEEDED"
    nk = out_json(engine, tid, "keyframes", "timeline.json")["n_keyframes"]
    assert len(billing(data_dir)) == nk                      # 瞬时失败没有计费,成功那次才计费
    assert step(t, "stylize_0")["attempt_no"] == 1           # 没有惊动引擎级重试
    items = engine.list_items(tid, step_key="stylize_0")
    assert sorted(i["calls"] for i in items)[-1] == 3        # 有一帧调用了 3 次,其余 1 次


def test_one_frame_exhausts_retries_only_that_frame_is_redone(engine, data_dir, clip):
    """题目里的例子:某一帧失败,只重试它;其他帧不重做、不重复付费。"""
    inj = frame_style_inject_presets(2)["rate_limit"]       # 第一帧前 3 次调用都是 429
    tid = submit(engine, clip, chaos=inj, frame_max_attempts=3)
    t = run_to_end(engine, data_dir, tid)
    assert t["status"] == "SUCCEEDED"
    nk = out_json(engine, tid, "keyframes", "timeline.json")["n_keyframes"]
    assert len(billing(data_dir)) == nk                      # 每个关键帧恰好计费一次
    assert step(t, "stylize_0")["attempt_no"] == 2           # 第一次尝试里这一帧用完了 3 次机会 → 引擎级重试
    assert step(t, "stylize_1")["attempt_no"] == 1
    items = {i["item_key"]: i for i in engine.list_items(tid, step_key="stylize_0")}
    calls = sorted(i["calls"] for i in items.values())
    assert calls[-1] == 4 and calls[:-1] == [1] * (len(calls) - 1)    # 3 次 429 + 1 次成功;其余帧各 1 次
    att = engine.get_attempts(tid, "stylize_0")
    assert att[1]["notes"]["frames_summary"]["resumed"] == len(calls) - 1    # 第二次尝试直接复用了其余帧


def test_crash_resumes_without_redoing_finished_frames(engine, data_dir, clip, tmp_path):
    base_dir = tmp_path / "base"
    from pipeline.engine import Engine
    e0 = Engine(base_dir)
    t0 = run_to_end(e0, base_dir, submit(e0, clip))
    assert t0["status"] == "SUCCEEDED"

    tid = submit(engine, clip, chaos=frame_style_inject_presets(2)["crash"])
    t = run_to_end(engine, data_dir, tid)
    assert t["status"] == "SUCCEEDED"
    s0 = step(t, "stylize_0")
    assert s0["attempt_no"] >= 2
    nk0 = len(engine.list_items(tid, step_key="stylize_0"))
    nk = out_json(engine, tid, "keyframes", "timeline.json")["n_keyframes"]
    extra = len(billing(data_dir)) - nk
    assert 0 <= extra <= BASE["concurrency"]                 # 最多重做崩溃时在途的那几帧
    att = engine.get_attempts(tid, "stylize_0")
    assert att[-1]["notes"]["frames_summary"]["resumed"] >= 1   # 崩溃前已完成的帧被复用
    assert nk0 >= 2
    # 结果与无故障基线逐字节一致
    for i in range(2):
        a = step(e0.get_task(t0["id"]), f"stylize_{i}")["outputs"]["styled.zip"]["sha256"]
        b = step(t, f"stylize_{i}")["outputs"]["styled.zip"]["sha256"]
        assert a == b
    e0.close()


def test_lost_response_may_double_charge_but_is_recorded(engine, data_dir, clip):
    tid = submit(engine, clip, chaos=frame_style_inject_presets(2)["response_lost"])
    t = run_to_end(engine, data_dir, tid)
    assert t["status"] == "SUCCEEDED"
    nk = out_json(engine, tid, "keyframes", "timeline.json")["n_keyframes"]
    assert len(billing(data_dir)) == nk + 1                  # 服务端实际多收了一次
    lost = [i for i in engine.list_items(tid, step_key="stylize_0") if i["maybe_billed"]]
    assert len(lost) == 1 and lost[0]["calls"] == 2           # 账本里如实留痕
    meta = out_json(engine, tid, "package", "metadata.json")
    assert meta["stylize"]["cost_maybe_extra_cny_est"] == pytest.approx(0.12, abs=1e-6)
    assert any("重复计费" in w for w in out_json(engine, tid, "qc", "qc.json")["warnings"])


def test_moderation_rejection_degrades_one_frame(engine, data_dir, clip):
    tid = submit(engine, clip, chaos=frame_style_inject_presets(2)["moderation"], max_degraded_ratio=0.25)
    t = run_to_end(engine, data_dir, tid)
    assert t["status"] == "SUCCEEDED"
    nk = out_json(engine, tid, "keyframes", "timeline.json")["n_keyframes"]
    assert len(billing(data_dir)) == nk - 1                  # 被审核拒绝的图不计费、也不重试
    meta = out_json(engine, tid, "package", "metadata.json")
    assert len(meta["stylize"]["degraded_frames"]) == 1
    deg = [i for i in engine.list_items(tid, step_key="stylize_0") if i["degraded"]]
    assert len(deg) == 1 and deg[0]["error_kind"] == "moderation" and deg[0]["calls"] == 1


def test_moderation_fail_mode_then_resubmit_with_override_pays_for_one_frame(engine, data_dir, clip):
    inj = frame_style_inject_presets(2)["moderation"]
    tid = submit(engine, clip, chaos=inj, on_reject="fail")
    t = run_to_end(engine, data_dir, tid)
    assert t["status"] == "FAILED"
    s0 = step(t, "stylize_0")
    assert s0["status"] == "FAILED" and s0["error_kind"] == "PERMANENT" and s0["attempt_no"] == 1   # 重试没用,不重试
    assert step(t, "stylize_1")["status"] == "SUCCEEDED"     # 另一块不受影响
    assert step(t, "mux")["status"] == "BLOCKED"
    bad = [i for i in engine.list_items(tid, step_key="stylize_0") if i["status"] == "FAILED"]
    assert len(bad) == 1
    idx = bad[0]["idx"]
    paid_before = len(billing(data_dir))

    # 对被拒的那一帧换提示词重新提交;其余帧命中帧级缓存
    tid2 = submit(engine, clip, on_reject="fail", frame_overrides={str(idx): {"prompt": "更温和的日系动漫风格"}})
    t2 = run_to_end(engine, data_dir, tid2)
    assert t2["status"] == "SUCCEEDED"
    assert len(billing(data_dir)) == paid_before + 1         # 只为那一帧付费
    meta = out_json(engine, tid2, "package", "metadata.json")
    assert meta["stylize"]["billed_images"] == 1


def test_fatal_auth_error_stops_spending(engine, data_dir, clip):
    inj = {"stylize_0": {"doubao": {"frames": {"__first__": {"kind": "auth"}}}}}
    tid = submit(engine, clip, chaos=inj, concurrency=1)
    t = run_to_end(engine, data_dir, tid)
    assert t["status"] == "FAILED"
    s0 = step(t, "stylize_0")
    assert s0["error_kind"] == "PERMANENT" and "auth" in s0["error_msg"] and s0["attempt_no"] == 1
    assert len(billing(data_dir)) <= 1 + len(engine.list_items(tid, step_key="stylize_1"))   # 没有继续在这一块上烧钱
    assert step(t, "stylize_0")["status"] == "FAILED"


def test_budget_guard_rejects_before_any_spend(engine, data_dir, clip):
    tid = submit(engine, clip, budget_cny=0.5)
    t = run_to_end(engine, data_dir, tid, timeout=120)
    assert t["status"] == "FAILED"
    k = step(t, "keyframes")
    assert k["error_kind"] == "PERMANENT" and "预算" in k["error_msg"]
    assert billing(data_dir) == []


def test_bad_source_fails_fast(engine, data_dir, tmp_path):
    bad = tmp_path / "broken.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    tid = submit(engine, bad)
    t = run_to_end(engine, data_dir, tid, timeout=60)
    assert t["status"] == "FAILED"
    s = step(t, "ingest")
    assert s["error_kind"] == "PERMANENT" and s["attempt_no"] == 1
    assert billing(data_dir) == []


def test_source_modified_after_submit_is_rejected(engine, data_dir, tmp_path):
    p = make_clip(tmp_path / "a.mp4", w=192, h=108, fps=FPS, secs=2.0)
    tid = submit(engine, p)
    p.write_bytes(p.read_bytes() + b"tamper")
    t = run_to_end(engine, data_dir, tid, timeout=60)
    assert t["status"] == "FAILED" and "修改" in step(t, "ingest")["error_msg"]
