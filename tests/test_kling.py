"""外部视频生成步骤(kling_video)测试:用本地假服务器,不联网、不花钱。

验证的是"和一个异步、计费的外部 API 打交道"该有的工程性质,不是模型效果:
  1. 每轮轮询都有观测记录,用量(usage)与费用估算被记录,视频被下载并用 ffmpeg 复核
  2. 步骤进程在轮询中途崩溃 → 重试**沿用检查点里的外部 task_id,不重复提交**(重复提交 = 重复付费)
  3. 外部返回"产品未开通/参数错误" → PERMANENT,不重试、不浪费预算,原因原样保留
  4. 没有密钥 → PERMANENT;密钥不会出现在任务参数、观测记录里
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from clipgen import make_clip
from conftest import FAST_RETRY, WorkerPool, step, wait_task
from pipeline.models import RetryPolicy
from pipeline.templates import kling_video_task

R = RetryPolicy(**FAST_RETRY)

SECRET = "sk-test-secret-should-never-leak"


class FakeKling:
    """极简假服务:POST 提交 → 返回 task_id;GET 查询 → 提交后 ready_after 秒才 SUCCEEDED。"""

    def __init__(self, video: Path, ready_after: float = 0.4, fail: dict | None = None):
        self.video, self.ready_after, self.fail = video, ready_after, fail
        self.submits: list[dict] = []
        self.polls = 0
        self.t_submit: float | None = None
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body: bytes, ctype="application/json"):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n))
                assert self.headers.get("X-DashScope-Async") == "enable"
                assert self.headers.get("Authorization") == "Bearer " + SECRET
                outer.submits.append(body)
                outer.t_submit = time.time()
                self._send(200, json.dumps({"request_id": "req-1", "output": {"task_id": "ext-1",
                                                                              "task_status": "PENDING"}}).encode())

            def do_GET(self):
                if self.path == "/video.mp4":
                    return self._send(200, outer.video.read_bytes(), "video/mp4")
                outer.polls += 1
                age = time.time() - (outer.t_submit or time.time())
                if outer.fail:
                    out = {"task_id": "ext-1", "task_status": "FAILED", **outer.fail}
                    return self._send(200, json.dumps({"output": out}).encode())
                if age < outer.ready_after:
                    st = "PENDING" if age < outer.ready_after / 2 else "RUNNING"
                    return self._send(200, json.dumps({"output": {"task_id": "ext-1", "task_status": st}}).encode())
                url = f"http://127.0.0.1:{outer.port}/video.mp4"
                self._send(200, json.dumps({
                    "output": {"task_id": "ext-1", "task_status": "SUCCEEDED", "video_url": url},
                    "usage": {"duration": 3, "size": "320*180", "fps": 12, "video_count": 1, "audio": False,
                              "SR": "720"}}).encode())

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def video(tmp_path):
    return make_clip(tmp_path / "fake.mp4", w=320, h=180, fps=12, secs=3.0)


def _env(monkeypatch, fake):
    monkeypatch.setenv("KLING_BASE_URL", f"http://127.0.0.1:{fake.port}")
    monkeypatch.setenv("DASHSCOPE_API_KEY", SECRET)


def _spec(**kw):
    return kling_video_task("k", prompt="a cat running", poll_interval_s=0.5, max_wait_s=3.0,
                            retry=R, **kw)


def test_polls_are_recorded_usage_and_cost_estimated_and_video_downloaded(engine, data_dir, video, monkeypatch):
    fake = FakeKling(video, ready_after=0.6)
    try:
        _env(monkeypatch, fake)
        tid, _ = engine.submit(_spec())
        with WorkerPool(data_dir, n=1, resources=("cpu",)):
            t = wait_task(engine, tid, timeout=60)
        assert t["status"] == "SUCCEEDED", t
        att = engine.get_attempts(tid, "generate")
        assert len(att) == 1 and att[0]["outcome"] == "SUCCEEDED"
        n = att[0]["notes"]
        polls = [n[k] for k in sorted(n) if k.startswith("poll_")]
        assert len(polls) >= 3 and polls[-1]["status"] == "SUCCEEDED"
        assert [p["status"] for p in polls][0] in ("PENDING", "RUNNING")      # 状态流转被逐轮记录
        assert all(p["tokens"] is None for p in polls)                        # 该模型不返回 token,不能编造
        assert n["usage"]["duration"] == 3 and n["cost_cny_est"] == 1.8       # std、无声:0.6 元/秒 × 3 秒
        assert n["video_info"]["width"] == 320 and n["video_bytes"] > 1000    # ffmpeg 独立复核了下载的视频
        assert n["submitted_this_attempt"] is True and len(fake.submits) == 1
        # 请求体符合文档:模型名、prompt、参数
        b = fake.submits[0]
        assert b["model"] == "kling/kling-v3-omni-video-generation" and b["input"]["prompt"] == "a cat running"
        assert b["parameters"]["mode"] == "std" and b["parameters"]["duration"] == 3 and b["parameters"]["audio"] is False
        # 密钥不在任务参数里,也不在观测记录里
        assert SECRET not in json.dumps(t, default=str) and SECRET not in json.dumps(att, default=str)
    finally:
        fake.close()


def test_crash_while_polling_resumes_same_external_task_without_resubmitting(engine, data_dir, video, monkeypatch):
    fake = FakeKling(video, ready_after=1.8)
    try:
        _env(monkeypatch, fake)
        spec = kling_video_task("k", prompt="a cat running", poll_interval_s=0.5, max_wait_s=3.0, retry=R,
                                chaos={"generate": {"crash": {"at": 0.5, "on_attempts": [1]}}})
        tid, _ = engine.submit(spec)
        with WorkerPool(data_dir, n=1, resources=("cpu",)):
            t = wait_task(engine, tid, timeout=90)
        assert t["status"] == "SUCCEEDED", t
        att = engine.get_attempts(tid, "generate")
        assert [a["outcome"] for a in att][-1] == "SUCCEEDED" and len(att) >= 2   # 第 1 次崩了
        assert len(fake.submits) == 1                                            # 外部任务只创建了一次 = 只付一次钱
        last = att[-1]["notes"]
        assert last["submitted_this_attempt"] is False and last["resumed_external_task"] == "ext-1"
    finally:
        fake.close()


def test_product_not_activated_is_permanent_and_keeps_the_reason(engine, data_dir, video, monkeypatch):
    msg = "The product is not activated, please confirm that you have activated products"
    fake = FakeKling(video, fail={"code": "InvalidParameter", "message": msg})
    try:
        _env(monkeypatch, fake)
        tid, _ = engine.submit(_spec())
        with WorkerPool(data_dir, n=1, resources=("cpu",)):
            t = wait_task(engine, tid, timeout=60)
        s = step(t, "generate")
        assert t["status"] == "FAILED" and s["error_kind"] == "PERMANENT" and s["failures"] == 1   # 不重试
        assert "not activated" in s["error_msg"] and "InvalidParameter" in s["error_msg"]
        att = engine.get_attempts(tid, "generate")
        assert len(att) == 1 and att[0]["notes"]["external_failure"]["code"] == "InvalidParameter"
        assert len(fake.submits) == 1
    finally:
        fake.close()


def test_missing_api_key_is_permanent(engine, data_dir, monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    tid, _ = engine.submit(_spec())
    with WorkerPool(data_dir, n=1, resources=("cpu",)):
        t = wait_task(engine, tid, timeout=60)
    s = step(t, "generate")
    assert s["error_kind"] == "PERMANENT" and "DASHSCOPE_API_KEY" in s["error_msg"] and s["failures"] == 1


def test_template_validates_cost_driving_parameters():
    for bad in [dict(duration=2), dict(duration=16), dict(mode="8k"), dict(prompt="")]:
        kw = dict(prompt="x", duration=3)
        kw.update(bad)
        with pytest.raises(ValueError):
            kling_video_task("k", **kw)
    spec = kling_video_task("k", prompt="x")
    assert spec.steps[0].cacheable is False                     # 花钱的外部副作用不能被缓存吞掉
    assert "sk-" not in json.dumps(spec.to_dict())
