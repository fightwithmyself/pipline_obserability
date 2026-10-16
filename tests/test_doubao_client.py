"""ArkDoubao 客户端:对着本地假服务器验证请求形态与错误映射。

注意:这只能证明"我们按文档假设发请求、按假设解析响应";文档假设本身是否正确,
要等真实密钥的冒烟测试(M3)才能确认。
"""
from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from pipeline.doubao import (ArkDoubao, FatalError, Moderation, RateLimited, Transient, classify_http_error,
                              make_client)


def png(color=(10, 120, 200), size=(32, 24)) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", size, color).save(b, "PNG")
    return b.getvalue()


class Fake:
    def __init__(self):
        self.reply = (200, {}, {})       # status, json, headers
        self.seen: list[dict] = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("content-length", 0))
                outer.seen.append({"path": self.path, "auth": self.headers.get("authorization"),
                                   "body": json.loads(self.rfile.read(n))})
                st, js, hd = outer.reply
                raw = json.dumps(js).encode()
                self.send_response(st)
                self.send_header("content-type", "application/json")
                for k, v in hd.items():
                    self.send_header(k, v)
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.srv = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()


@pytest.fixture
def fake():
    f = Fake()
    yield f
    f.srv.shutdown()


def client(fake, **kw):
    return ArkDoubao("doubao-seedream-x", base_url=fake.url, api_key="k-secret", timeout_s=5, **kw)


def ok_payload(img: bytes) -> dict:
    return {"data": [{"b64_json": base64.b64encode(img).decode(), "size": "32x24"}],
            "usage": {"generated_images": 1, "output_tokens": 77}}


def test_request_shape_and_success(fake):
    fake.reply = (200, ok_payload(png((200, 30, 30))), {"x-request-id": "rid-1"})
    r = client(fake).edit(image=png(), prompt="日系动漫", reference=None, frame_key="f0")
    s = fake.seen[0]
    assert s["path"] == "/api/v3/images/generations" and s["auth"] == "Bearer k-secret"
    b = s["body"]
    assert b["model"] == "doubao-seedream-x" and b["prompt"] == "日系动漫"
    assert b["response_format"] == "b64_json" and b["watermark"] is False
    assert isinstance(b["image"], str) and b["image"].startswith("data:image/jpeg;base64,")
    assert Image.open(io.BytesIO(r.image)).size == (32, 24)
    assert r.usage["output_tokens"] == 77 and r.request_id == "rid-1"


def test_reference_sends_two_images(fake):
    fake.reply = (200, ok_payload(png()), {})
    client(fake).edit(image=png(), prompt="p", reference=png((1, 2, 3)), frame_key="f0")
    assert len(fake.seen[0]["body"]["image"]) == 2


@pytest.mark.parametrize("status,payload,hd,exc", [
    (429, {"error": {"code": "RateLimitExceeded", "message": "slow down"}}, {"retry-after": "7"}, RateLimited),
    (401, {"error": {"code": "AuthenticationError", "message": "bad key"}}, {}, FatalError),
    (500, {"error": {"code": "InternalServiceError", "message": "boom"}}, {}, Transient),
    (400, {"error": {"code": "InputImageSensitiveContentDetected", "message": "x"}}, {}, Moderation),
    (400, {"error": {"code": "InvalidParameter", "message": "x"}}, {}, FatalError),
])
def test_error_mapping(fake, status, payload, hd, exc):
    fake.reply = (status, payload, hd)
    with pytest.raises(exc) as ei:
        client(fake).edit(image=png(), prompt="p", reference=None, frame_key="f0")
    assert type(ei.value) is exc
    if exc is RateLimited:
        assert ei.value.retry_after == 7.0
    assert "k-secret" not in str(ei.value)


def test_bad_bodies_are_transient_and_flagged_maybe_sent(fake):
    for payload in ({"data": [{}]}, {"data": [{"b64_json": base64.b64encode(b"not an image").decode()}]}):
        fake.reply = (200, payload, {})
        with pytest.raises(Transient) as ei:
            client(fake).edit(image=png(), prompt="p", reference=None, frame_key="f0")
        assert ei.value.request_maybe_sent is True        # 请求已被服务端处理,可能已计费


def test_connect_error_means_not_sent(monkeypatch):
    import socket
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()                                       # 端口已关闭 → 连接被拒绝
    c = ArkDoubao("m", base_url=f"http://127.0.0.1:{port}", api_key="k", timeout_s=2)
    with pytest.raises(Transient) as ei:
        c.edit(image=png(), prompt="p", reference=None, frame_key="f")
    assert ei.value.request_maybe_sent is False


def test_missing_key_and_model(monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    with pytest.raises(FatalError) as ei:
        ArkDoubao("m").edit(image=png(), prompt="p", reference=None, frame_key="f")
    assert ei.value.kind == "auth"
    with pytest.raises(FatalError):
        ArkDoubao("")


def test_classify_real_404_codes():
    """这两个错误码是用真实密钥实测到的(见 README 9.2)。"""
    a = classify_http_error(404, "ModelNotOpen", "Your account has not activated the model x")
    b = classify_http_error(404, "InvalidEndpointOrModel.NotFound", "does not exist or you do not have access")
    assert isinstance(a, FatalError) and a.kind == "model_not_open"
    assert isinstance(b, FatalError) and b.kind == "not_found"


def test_classify_is_conservative():
    assert isinstance(classify_http_error(400, "SomethingNew", "?"), FatalError)     # 未知 4xx 不重试烧钱
    assert isinstance(classify_http_error(503, "", ""), Transient)
    assert classify_http_error(503, "", "").request_maybe_sent is False
    assert classify_http_error(402, "", "").kind == "quota"


def test_make_client_dispatch(tmp_path):
    c = make_client("mock", model=None, data_dir=tmp_path, chaos=None, attempt=1, sleep=lambda s: None)
    assert c.name == "mock" if hasattr(c, "name") else True
    assert isinstance(make_client("ark", model="m", data_dir=tmp_path, chaos=None, attempt=1,
                                  sleep=lambda s: None), ArkDoubao)
