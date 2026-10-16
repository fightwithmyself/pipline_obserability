"""序列帧流水线(真实处理)与素材检测的测试。

覆盖:
  - 素材检测:正例通过;每一类反例各自被拒绝且原因可读(验证检测本身,而不是只验证管线);
  - 滑窗扫描能把"好段"与"坏段"分开;
  - 端到端:真实 ffmpeg/numpy 处理,产物规格(帧数 2n-1、分辨率 ×2、alpha 视频)可独立验证;
  - 缓存复用、QC 失败的下游行为、坏素材在 probe 永久失败且不浪费后面的计算;
  - 崩溃后从步骤内检查点继续,且最终产物与无故障基线逐字节相同;
  - 回归:子进程里 import numpy 不能被"监听父进程"的线程卡死(Windows 上曾真实发生)。
"""
from __future__ import annotations

import io
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pipeline.engine import Engine
from pipeline.media import MediaError, assess_video, count_decoded_frames, probe_video, scan_windows
from pipeline.models import RetryPolicy
from pipeline.steps.frames import blend_frames, pack_zip
from pipeline.templates import frame_pipeline_task

from clipgen import make_clip
from conftest import FAST_RETRY, ROOT, WorkerPool, step, wait_task

R = RetryPolicy(**FAST_RETRY)


def spec(src: Path, **kw):
    kw.setdefault("chunks", 3)
    kw.setdefault("retry", R)
    return frame_pipeline_task(source=str(src), **kw)


@pytest.fixture(scope="module")
def good_clip(tmp_path_factory) -> Path:
    return make_clip(tmp_path_factory.mktemp("clips") / "good.mp4", secs=3.0)       # 36 帧


# ============================================================================ 素材检测
def test_assess_accepts_static_camera_clip_with_moving_subject(good_clip):
    r = assess_video(good_clip)
    assert r["ok"], r["reasons"]
    assert r["metrics"]["static_bg_ratio"] > 0.9
    assert 0.05 < r["metrics"]["fg_ratio_mean"] < 0.3          # 主体大约占画面 13%
    assert r["info"]["width"] == 320 and r["info"]["fps"] == 12


@pytest.mark.parametrize("kw,expect", [
    (dict(pan_after=0.0), "背景不固定"),                              # 机位全程平移
    (dict(subject=False), "没有运动主体"),                            # 空镜头
    (dict(secs=1.0), "时长"),
    (dict(fps=5, secs=3.0), "帧率"),
    (dict(w=96, h=48), "太小"),
    (dict(w=1288, h=724, fps=24, secs=2.5), "分辨率"),
])
def test_assess_rejects_each_kind_of_bad_material(tmp_path, kw, expect):
    clip = make_clip(tmp_path / "bad.mp4", **{"secs": 3.0, **kw})
    r = assess_video(clip)
    assert not r["ok"]
    assert any(expect in x for x in r["reasons"]), r["reasons"]


def test_assess_rejects_non_video_and_missing_files(tmp_path):
    junk = tmp_path / "not_video.mp4"
    junk.write_bytes(b"this is not a video" * 100)
    with pytest.raises(MediaError):
        assess_video(junk)
    with pytest.raises(MediaError):
        assess_video(tmp_path / "missing.mp4")


def test_scan_separates_good_and_bad_segments(tmp_path):
    # 前 3 秒固定机位,之后机位平移:好窗口应排在前面,最后一个窗口应不合格
    clip = make_clip(tmp_path / "half.mp4", secs=6.0, pan_after=3.0)
    res = scan_windows(clip, win_s=3.0, step_s=1.0, sample_fps=6.0)
    assert res[0]["ok"] and res[0]["start_s"] <= 0.5
    worst = max(res, key=lambda r: r["start_s"])
    assert not worst["ok"]
    assert any("背景不固定" in x for x in worst["reasons"])


# ============================================================================ 算法的单元性质
def test_blend_uses_premultiplied_alpha_and_keeps_color():
    red_opaque = np.zeros((2, 2, 4), np.uint8)
    red_opaque[..., 0] = 255
    red_opaque[..., 3] = 255
    transparent = np.zeros((2, 2, 4), np.uint8)            # 透明像素的 RGB 是黑色,但不应把红色"染黑"
    mid = blend_frames(red_opaque, transparent)
    assert abs(int(mid[0, 0, 3]) - 128) <= 1
    assert mid[0, 0, 0] >= 254                              # 颜色保持为红色,只是半透明


def test_pack_zip_is_deterministic():
    items = [("00000.png", b"aaa"), ("00001.png", b"bbb")]
    a, b = io.BytesIO(), io.BytesIO()
    # 两次写入(间隔不同时间)内容应逐字节相同,否则内容寻址缓存永远命不中
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = Path(d) / "1.zip", Path(d) / "2.zip"
        pack_zip(p1, items)
        pack_zip(p2, items)
        assert p1.read_bytes() == p2.read_bytes()
        assert zipfile.ZipFile(p1).namelist() == ["00000.png", "00001.png"]


# ============================================================================ 端到端
def test_pipeline_end_to_end_with_real_processing_and_cache(engine, data_dir, good_clip):
    s = spec(good_clip)
    assert s.meta["template"] == "frame_pipeline"
    tid, _ = engine.submit(s)
    with WorkerPool(data_dir, n=2, resources=("cpu",)):
        t = wait_task(engine, tid, timeout=180)
        assert t["status"] == "SUCCEEDED", [(x["key"], x["status"], x["error_msg"]) for x in t["steps"]]

        # 产物规格:独立于步骤自己的 QC,这里再用 ffmpeg 重新验证一遍
        n_src = 36
        asm = step(t, "assemble")
        mp4 = engine.store.path(asm["outputs"]["final.mp4"]["sha256"])
        webm = engine.store.path(asm["outputs"]["alpha.webm"]["sha256"])
        v = probe_video(mp4)
        assert (v["width"], v["height"], round(v["fps"])) == (640, 360, 24)
        assert count_decoded_frames(mp4) == 2 * n_src - 1
        assert count_decoded_frames(webm, decoder="libvpx-vp9") == 2 * n_src - 1

        # alpha 真的有内容:解出一帧,透明与不透明像素都存在(抠图不是全透明/全不透明)
        raw = subprocess.run(
            [__import__("pipeline.media", fromlist=["ffmpeg_exe"]).ffmpeg_exe(), "-loglevel", "error", "-c:v", "libvpx-vp9",
             "-i", str(webm), "-vf", "select=eq(n\\,40),format=rgba", "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgba", "-"],
            capture_output=True).stdout
        alpha = np.frombuffer(raw, np.uint8).reshape(360, 640, 4)[..., 3]
        assert 0.03 < (alpha > 127).mean() < 0.4 and (alpha < 30).mean() > 0.5

        # 补帧产物的帧编号连续:0..2n-2
        names = []
        for i in range(3):
            z = zipfile.ZipFile(engine.store.path(step(t, f"interp_{i}")["outputs"]["frames.zip"]["sha256"]))
            names += z.namelist()
        assert sorted(names) == [f"{k:05d}.png" for k in range(2 * n_src - 1)]
        im = Image.open(io.BytesIO(zipfile.ZipFile(engine.store.path(
            step(t, "interp_0")["outputs"]["frames.zip"]["sha256"])).read("00000.png")))
        assert im.mode == "RGBA" and im.size == (640, 360)

        # 发布走幂等键;publish 不进缓存
        assert len(list((data_dir / "published").glob("*.json"))) == 1

        # 再提交一个"只有质检会失败"的同源任务:上游全部命中缓存,QC 永久失败,publish 被阻塞
        tid2, _ = engine.submit(spec(good_clip, qc_fail=True))
        t2 = wait_task(engine, tid2, timeout=120)
        assert t2["status"] == "FAILED"
        for k in ("probe", "extract_frames", "bg_plate", "matte_0", "upscale_1", "interp_2", "assemble"):
            assert step(t2, k)["cached"] and step(t2, k)["attempt_no"] == 0, k
        qc = step(t2, "quality_check")
        assert qc["status"] == "FAILED" and qc["error_kind"] == "PERMANENT" and "强制失败" in qc["error_msg"]
        assert step(t2, "publish")["status"] == "BLOCKED"


def test_bad_material_fails_permanently_at_probe_and_wastes_nothing(engine, data_dir, tmp_path):
    bad = make_clip(tmp_path / "pan.mp4", secs=3.0, pan_after=0.0)
    tid, _ = engine.submit(spec(bad))
    with WorkerPool(data_dir, n=1, resources=("cpu",)):
        t = wait_task(engine, tid, timeout=60)
    assert t["status"] == "FAILED"
    p = step(t, "probe")
    assert p["status"] == "FAILED" and p["error_kind"] == "PERMANENT" and p["failures"] == 1   # 不重试
    assert "素材不适合本流水线" in p["error_msg"] and "背景不固定" in p["error_msg"]
    for s in t["steps"]:
        if s["key"] != "probe":
            assert s["status"] == "BLOCKED" and s["attempt_no"] == 0, s["key"]               # 后面一步都没跑


def test_source_modified_after_submit_is_rejected(engine, data_dir, tmp_path):
    clip = make_clip(tmp_path / "c.mp4", secs=3.0)
    s = spec(clip)
    make_clip(clip, secs=3.0, w=336, h=192)                    # 提交后素材被替换
    tid, _ = engine.submit(s)
    with WorkerPool(data_dir, n=1, resources=("cpu",)):
        t = wait_task(engine, tid, timeout=60)
    p = step(t, "probe")
    assert t["status"] == "FAILED" and p["error_kind"] == "PERMANENT" and "被修改" in p["error_msg"]


def test_template_rejects_missing_source_and_bad_chunks(tmp_path):
    with pytest.raises(ValueError):
        frame_pipeline_task(source=str(tmp_path / "nope.mp4"))
    clip = make_clip(tmp_path / "x.mp4", secs=2.0)
    with pytest.raises(ValueError):
        frame_pipeline_task(source=str(clip), chunks=0)


# ============================================================================ 故障恢复(真实计算上)
def _run_clean(data_dir: Path, clip: Path) -> dict[str, str]:
    e = Engine(data_dir)
    try:
        tid, _ = e.submit(spec(clip, chunks=2))
        with WorkerPool(data_dir, n=1, resources=("cpu",)):
            t = wait_task(e, tid, timeout=180)
        assert t["status"] == "SUCCEEDED"
        return {s["key"]: s["outputs"].get("frames.zip", {}).get("sha256") or s["outputs"].get("stats.json", {}).get("sha256")
                for s in t["steps"] if s["key"] in ("matte_0", "matte_1", "upscale_1", "interp_0", "interp_1", "assemble")}
    finally:
        e.close()


def test_crash_mid_matte_resumes_from_checkpoint_and_result_equals_clean_run(tmp_path):
    clip = make_clip(tmp_path / "c.mp4", secs=2.0)             # 24 帧,分 2 块 → 每块 12 帧
    baseline = _run_clean(tmp_path / "clean", clip)

    data = tmp_path / "crashy"
    e = Engine(data)
    try:
        tid, _ = e.submit(spec(clip, chunks=2, chaos={"matte_1": {"crash": {"at": 0.5, "on_attempts": [1]}}}))
        with WorkerPool(data, n=1, resources=("cpu",)):
            t = wait_task(e, tid, timeout=180)
        assert t["status"] == "SUCCEEDED"
        atts = e.get_attempts(tid, "matte_1")
        assert len(atts) == 2 and atts[0]["outcome"] != "SUCCEEDED" and atts[-1]["outcome"] == "SUCCEEDED"
        resumed = atts[-1]["notes"]["resumed_from"]
        assert resumed >= 4                                              # 检查点每 4 帧一次,崩溃发生在第 6 帧之后
        assert atts[-1]["notes"]["frames_computed"] == 12 - resumed       # 只算剩下的,不从头来
        # 其他步骤没有被重跑
        assert step(t, "matte_0")["attempt_no"] == 1 and step(t, "probe")["attempt_no"] == 1
        crashed = {s["key"]: s["outputs"].get("frames.zip", {}).get("sha256") or s["outputs"].get("stats.json", {}).get("sha256")
                   for s in t["steps"] if s["key"] in baseline}
    finally:
        e.close()
    assert crashed == baseline, "崩溃恢复后的产物必须与无故障运行逐字节一致"


# ============================================================================ 回归
def test_step_child_can_import_numpy_while_parent_watcher_runs():
    """Windows 上曾发生:监听父进程的线程阻塞在 os.read(0),占住 CRT 的 fd 锁,子进程 import numpy 永远不返回。
    现在 Windows 用 PeekNamedPipe 轮询;本测试同时验证:导入成功,且父进程关闭管道后子进程自杀。"""
    code = ("import threading, sys\n"
            "from pipeline.step_runner import _watch_parent\n"
            "threading.Thread(target=_watch_parent, daemon=True).start()\n"
            "import numpy, PIL.Image, PIL.ImageFilter\n"
            "print('imported', flush=True)\n"
            "import time; time.sleep(30)\n")
    p = subprocess.Popen([sys.executable, "-u", "-c", code], cwd=str(ROOT), stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        import threading
        got: list[bytes] = []
        threading.Thread(target=lambda: got.append(p.stdout.readline()), daemon=True).start()
        import time
        t0 = time.time()
        while not got and time.time() - t0 < 20:
            time.sleep(0.1)
        assert got and got[0].strip() == b"imported", "子进程 import numpy 卡死(监听线程与 CRT 锁死锁)"
        p.stdin.close()                                    # 模拟父进程死亡
        p.wait(timeout=10)
        assert p.returncode == 1                           # 监听线程触发的自杀
    finally:
        if p.poll() is None:
            p.kill()

# ============================================================================ API
def test_api_media_listing_assess_and_submit_are_path_safe(tmp_path, monkeypatch, good_clip):
    from fastapi.testclient import TestClient
    from pipeline.api import create_app
    mdir = tmp_path / "media"
    mdir.mkdir()
    (mdir / "good.mp4").write_bytes(good_clip.read_bytes())
    (mdir / "_tmp.mp4").write_bytes(b"x")              # 下划线开头:临时文件,不列出、不可提交
    (mdir / "notes.txt").write_text("hi")
    (tmp_path / "secret.mp4").write_bytes(good_clip.read_bytes())    # 素材目录之外的合法视频
    monkeypatch.setenv("PIPELINE_MEDIA_DIR", str(mdir))
    c = TestClient(create_app(tmp_path / "data"))

    assert [m["name"] for m in c.get("/api/media").json()] == ["good.mp4"]
    r = c.get("/api/media/assess", params={"name": "good.mp4"}).json()
    assert r["ok"] and r["info"]["width"] == 320

    for evil in ("../secret.mp4", "..\\secret.mp4", str(tmp_path / "secret.mp4"), "_tmp.mp4", ".hidden.mp4", ""):
        assert c.post("/api/tasks/frames", json={"source": evil}).status_code == 422, evil
        assert c.get("/api/media/assess", params={"name": evil}).status_code == 422, evil
    assert c.post("/api/tasks/frames", json={"source": "missing.mp4"}).status_code == 404

    ok = c.post("/api/tasks/frames", json={"source": "good.mp4", "chunks": 2, "inject": "crash"})
    assert ok.status_code == 201
    t = c.get(f"/api/tasks/{ok.json()['id']}").json()
    assert t["steps_total"] == 3 + 2 * 3 + 3                       # probe/extract/plate + (matte,upscale,interp)×2 + assemble/qc/publish
    assert {s["type"] for s in t["steps"]} >= {"fp_probe", "fp_matte", "fp_interp", "fp_assemble", "publish"}
    e = Engine(tmp_path / "data")
    import json as _j
    chaos = _j.loads(e.conn.execute("SELECT chaos FROM steps WHERE task_id=? AND key='matte_1'", (t["id"],)).fetchone()[0])
    e.close()
    assert chaos["crash"]["at"] == 0.5                             # inject 预设确实落到了 matte_1 上
    assert c.post("/api/tasks/frames", json={"source": "good.mp4", "chunks": 99}).status_code == 422


