"""视频风格重绘任务链测试(百炼 video-style-transform + 多模态评审):本地假服务器,不联网、不花钱。

验证链路的工程性质,不是模型效果:
  - 5 步链路跑通:素材检测 → 上传临时存储 → 风格重绘 → 结构复核 → 多模态评审
  - 每步的输入/输出是引擎里的产物,能通过 API 取到(图片/视频/JSON 预览)
  - 用量写在步骤详情(attempt.notes.usage):风格重绘是秒数、评审是 token;每次尝试各记一份
  - 素材不满足接口要求 → 在第一步永久失败,不上传、不提交、不花钱
  - 评审模型输出不合约定 → 重试;失败的那次尝试的 token 也被如实记录(花了就是花了)
  - 密钥不出现在任务参数、产物、观测记录里
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from clipgen import make_clip
from conftest import FAST_RETRY, WorkerPool, step, wait_task
from pipeline.api import create_app
from pipeline.models import RetryPolicy
from pipeline.templates import style_transform_task

SECRET = "sk-test-secret-should-never-leak"
R = RetryPolicy(**FAST_RETRY)
VERDICT = {"is_japanese_anime_style": True, "character_preserved": True, "score": 4,
           "issues": [], "summary": "风格明显,人物保留"}


class FakeDashScope:
    def __init__(self, styled: Path, review_replies: list[str], ready_after: float = 0.3):
        self.styled, self.review_replies, self.ready_after = styled, list(review_replies), ready_after
        self.uploads: list[int] = []
        self.submits: list[dict] = []
        self.submit_headers: list[dict] = []
        self.chat_bodies: list[dict] = []
        self.polls = 0
        self.t_submit = None
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, body: bytes, ctype="application/json", code=200):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> bytes:
                return self.rfile.read(int(self.headers.get("Content-Length", 0)))

            def _auth(self) -> bool:
                return self.headers.get("Authorization") == "Bearer " + SECRET

            def do_GET(self):
                import time
                u = urlparse(self.path)
                if u.path == "/styled.mp4":
                    return self._send(outer.styled.read_bytes(), "video/mp4")
                assert self._auth()
                if u.path == "/api/v1/uploads":
                    assert parse_qs(u.query)["model"] == ["video-style-transform"]
                    return self._send(json.dumps({"data": {
                        "policy": "p", "signature": "s", "upload_dir": "dashscope-instant/x/2026-01-01/y",
                        "upload_host": f"http://127.0.0.1:{outer.port}/oss", "expire_in_seconds": 300,
                        "max_file_size_mb": 100, "oss_access_key_id": "AK", "x_oss_object_acl": "private",
                        "x_oss_forbid_overwrite": "true"}}).encode())
                outer.polls += 1
                age = time.time() - outer.t_submit
                if age < outer.ready_after:
                    return self._send(json.dumps({"output": {"task_id": "ext-1", "task_status": "RUNNING"}}).encode())
                return self._send(json.dumps({"output": {
                    "task_id": "ext-1", "task_status": "SUCCEEDED",
                    "output_video_url": f"http://127.0.0.1:{outer.port}/styled.mp4"},
                    "usage": {"duration": 3, "SR": 540}}).encode())

            def do_POST(self):
                import time
                raw = self._body()
                if self.path == "/oss":                                   # 假 OSS:只记录收到多少字节
                    outer.uploads.append(len(raw))
                    return self._send(b"", code=200)
                assert self._auth()
                if self.path == "/api/v1/services/aigc/video-generation/video-synthesis":
                    assert self.headers.get("X-DashScope-Async") == "enable"
                    outer.submits.append(json.loads(raw))
                    outer.submit_headers.append(dict(self.headers))
                    outer.t_submit = time.time()
                    return self._send(json.dumps({"request_id": "r1", "output": {"task_id": "ext-1",
                                                                                 "task_status": "PENDING"}}).encode())
                if self.path == "/compat/chat/completions":
                    outer.chat_bodies.append(json.loads(raw))
                    text = outer.review_replies.pop(0)
                    return self._send(json.dumps({
                        "id": "c1", "object": "chat.completion", "created": 1, "model": "qwen3.8-omni-flash",
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
                        "usage": {"prompt_tokens": 902, "completion_tokens": 55, "total_tokens": 957,
                                  "prompt_tokens_details": {"image_tokens": 882},
                                  "completion_tokens_details": {"reasoning_tokens": 0}}}).encode())
                self._send(b"{}", code=404)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def clips(tmp_path):
    src = make_clip(tmp_path / "src.mp4", w=640, h=400, fps=12, secs=3.0)
    styled = make_clip(tmp_path / "styled.mp4", w=960, h=540, fps=15, secs=3.0, subject=True)
    return src, styled


def _env(monkeypatch, fake):
    monkeypatch.setenv("DASHSCOPE_BASE_URL", f"http://127.0.0.1:{fake.port}")
    monkeypatch.setenv("DASHSCOPE_COMPAT_BASE_URL", f"http://127.0.0.1:{fake.port}/compat")
    monkeypatch.setenv("DASHSCOPE_API_KEY", SECRET)


def _spec(src):
    return style_transform_task("st", source=str(src), poll_interval_s=0.3, max_wait_s=10.0, retry=R)


def test_whole_chain_runs_and_every_step_exposes_inputs_outputs_and_usage(engine, data_dir, clips, monkeypatch):
    src, styled = clips
    fake = FakeDashScope(styled, [json.dumps(VERDICT, ensure_ascii=False)])
    try:
        _env(monkeypatch, fake)
        tid, _ = engine.submit(_spec(src))
        with WorkerPool(data_dir, n=1, resources=("cpu",)):
            t = wait_task(engine, tid, timeout=120)
        assert t["status"] == "SUCCEEDED", [(s["key"], s["status"], s["error_msg"]) for s in t["steps"]]
        assert [s["key"] for s in t["steps"]] == ["probe", "upload", "style", "quality_check", "review"]

        # 上传:收到了完整文件;提交带了 oss 解析头;请求体符合文档
        assert fake.uploads and fake.uploads[0] > src.stat().st_size
        assert len(fake.submits) == 1
        body = fake.submits[0]
        assert body["model"] == "video-style-transform" and body["input"]["video_url"].startswith("oss://")
        assert body["parameters"]["style"] == 0 and body["parameters"]["min_len"] == 540
        assert any(k.lower() == "x-dashscope-ossresourceresolve" and v == "enable"
                   for k, v in fake.submit_headers[0].items())

        notes = {k: engine.get_attempts(tid, k)[-1]["notes"] for k in ("probe", "upload", "style", "quality_check", "review")}
        # 风格重绘:按秒计费,没有 token;费用 = 0.2 元/秒(540P)× 3 秒
        assert notes["style"]["usage"]["duration"] == 3 and notes["style"]["usage"]["SR"] == 540
        assert notes["style"]["cost_cny_est"] == 0.6 and notes["style"]["polls"] >= 2
        assert all(notes["style"][k]["tokens"] is None for k in notes["style"] if k.startswith("poll_"))
        # 评审:真实 token 用量写在步骤详情里
        u = notes["review"]["usage"]
        assert (u["prompt_tokens"], u["completion_tokens"], u["total_tokens"]) == (902, 55, 957)
        assert u["image_tokens"] == 882 and u["reasoning_tokens"] == 0
        # 评审请求:两张图 + 提示词,且关闭了思考模式(省 token)
        chat = fake.chat_bodies[0]
        parts = chat["messages"][0]["content"]
        assert sum(1 for x in parts if x["type"] == "image_url") == 2
        assert chat["enable_thinking"] is False and chat["model"] == "qwen3.8-omni-flash"

        # 通过 API 点开步骤:输入 = 上游产物,输出 = 本步产物,JSON 内嵌预览,图片/视频可按哈希取回
        c = TestClient(create_app(data_dir))
        io = c.get(f"/api/tasks/{tid}/steps/quality_check/io").json()
        assert [g["from"] for g in io["inputs"]] == ["probe", "style"]
        assert {a["name"] for a in io["outputs"]} == {"compare.png", "out.jpg", "src.jpg", "qc.json"}
        qc = next(a for a in io["outputs"] if a["name"] == "qc.json")
        assert json.loads(qc["preview"])["passed"] is True
        png = next(a for a in io["outputs"] if a["name"] == "compare.png")
        r = c.get(f"/api/artifacts/{png['sha256']}", params={"name": "compare.png"})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
        vid = next(a for g in io["inputs"] if g["from"] == "style" for a in g["artifacts"])
        assert vid["kind"] == "video" and c.get(f"/api/artifacts/{vid['sha256']}").status_code == 200
        rv = json.loads(next(a for a in c.get(f"/api/tasks/{tid}/steps/review/io").json()["outputs"])["preview"])
        assert rv["verdict"]["score"] == 4 and rv["model"] == "qwen3.8-omni-flash"
        # 产物接口只认引擎登记过的哈希
        assert c.get("/api/artifacts/" + "0" * 64).status_code == 404
        assert c.get("/api/artifacts/../../etc/passwd").status_code in (404, 422)
        assert c.get("/api/artifacts/not-a-hash").status_code == 422

        # 密钥哪里都没有
        blob = json.dumps(t, default=str) + json.dumps(notes, default=str) + json.dumps(io, default=str)
        assert SECRET not in blob
    finally:
        fake.close()


def test_unacceptable_source_is_rejected_at_probe_without_uploading_or_paying(engine, data_dir, tmp_path, monkeypatch):
    bad = make_clip(tmp_path / "wide.mp4", w=640, h=256, fps=12, secs=3.0)        # 短边刚好 256 合格,长短边比 2.5 > 1.8
    fake = FakeDashScope(bad, [])
    try:
        _env(monkeypatch, fake)
        tid, _ = engine.submit(_spec(bad))
        with WorkerPool(data_dir, n=1, resources=("cpu",)):
            t = wait_task(engine, tid, timeout=60)
        p = step(t, "probe")
        assert t["status"] == "FAILED" and p["error_kind"] == "PERMANENT" and p["failures"] == 1
        assert "长短边比" in p["error_msg"]
        assert fake.uploads == [] and fake.submits == [] and fake.chat_bodies == []        # 一分钱没花
        for k in ("upload", "style", "quality_check", "review"):
            assert step(t, k)["status"] == "BLOCKED" and step(t, k)["attempt_no"] == 0
        # 失败的步骤不登记产物(只有成功才入库),但检测报告在 notes 里:点开步骤能看到具体哪几条不满足
        rep = engine.get_attempts(tid, "probe")[-1]["notes"]["report"]
        assert rep["accepted"] is False and [c["name"] for c in rep["checks"] if not c["ok"]] == ["长短边比"]
        assert TestClient(create_app(data_dir)).get(f"/api/tasks/{tid}/steps/probe/io").json()["outputs"] == []
    finally:
        fake.close()


def test_review_with_bad_format_is_retried_and_tokens_of_every_attempt_are_recorded(engine, data_dir, clips, monkeypatch):
    src, styled = clips
    fake = FakeDashScope(styled, ["我觉得很好看!", json.dumps(VERDICT, ensure_ascii=False)])
    try:
        _env(monkeypatch, fake)
        tid, _ = engine.submit(_spec(src))
        with WorkerPool(data_dir, n=1, resources=("cpu",)):
            t = wait_task(engine, tid, timeout=120)
        assert t["status"] == "SUCCEEDED"
        att = engine.get_attempts(tid, "review")
        assert [a["outcome"] for a in att] == ["FAILED", "SUCCEEDED"]
        assert "不符合约定格式" in att[0]["error_msg"]
        assert [a["notes"]["usage"]["total_tokens"] for a in att] == [957, 957]      # 失败那次的 token 也记了
        assert len(fake.submits) == 1 and len(fake.chat_bodies) == 2                 # 评审重试不会重提交风格重绘
    finally:
        fake.close()


def test_template_validates_and_hides_nothing_secret():
    with pytest.raises(ValueError):
        style_transform_task("x", source="/no/such/file.mp4")
    import tempfile
    d = Path(tempfile.mkdtemp())
    f = d / "a.mp4"
    f.write_bytes(b"x")
    for bad in (dict(style=9), dict(min_len=480), dict(video_fps=30)):
        with pytest.raises(ValueError):
            style_transform_task("x", source=str(f), **bad)
    spec = style_transform_task("x", source=str(f))
    cacheable = {s.key: s.cacheable for s in spec.steps}
    assert cacheable == {"probe": True, "upload": False, "style": False, "quality_check": True, "review": False}
    assert "sk-" not in json.dumps(spec.to_dict())
