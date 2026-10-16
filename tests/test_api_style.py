"""主流水线的 HTTP 接口:提交、帧账本、图片、用量、报告。"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import WorkerPool, wait_for
from pipeline.api import create_app
from tests.clipgen import make_clip


@pytest.fixture
def media(tmp_path, monkeypatch):
    d = tmp_path / "media"
    make_clip(d / "walk.mp4", w=192, h=108, fps=12, secs=3.0, audio=True)
    monkeypatch.setenv("PIPELINE_MEDIA_DIR", str(d))
    return d


@pytest.fixture
def client(data_dir, media):
    return TestClient(create_app(data_dir))


BODY = dict(source="walk.mp4", prompt="日系动漫风格", chunks=2, stride=3, max_gap=8, budget_cny=20,
            target_height=108, base_delay=0.1, max_attempts=3, concurrency=2)


def test_submit_validation(client):
    assert client.get("/api/media").json() == [{"name": "walk.mp4", "size_mb": pytest.approx(0.0, abs=1)}]
    assert client.post("/api/tasks/style", json={**BODY, "source": "../etc/passwd"}).status_code == 422
    assert client.post("/api/tasks/style", json={**BODY, "source": "nope.mp4"}).status_code == 404
    assert client.post("/api/tasks/style", json={**BODY, "provider": "ark"}).status_code == 422      # ark 必须指定模型
    assert client.post("/api/tasks/style", json={**BODY, "budget_cny": 0}).status_code == 422
    assert client.post("/api/tasks/style", json={**BODY, "provider": "ark", "model": "x",
                                                  "inject": "moderation"}).status_code == 422       # 注入只对 mock 有效


def test_end_to_end_via_api(client, data_dir):
    r = client.post("/api/tasks/style", json={**BODY, "inject": "response_lost"}, headers={"Idempotency-Key": "s1"})
    assert r.status_code == 201
    tid = r.json()["id"]
    assert client.post("/api/tasks/style", json=BODY, headers={"Idempotency-Key": "s1"}).json() == {"id": tid, "created": False}
    with WorkerPool(data_dir, n=2):
        t = wait_for(lambda: (lambda x: x if x["status"] in ("SUCCEEDED", "FAILED") else None)(
            client.get(f"/api/tasks/{tid}").json()), 180, what="任务完成")
    assert t["status"] == "SUCCEEDED", [(s["key"], s["error_msg"]) for s in t["steps"] if s["error_msg"]]

    fr = client.get(f"/api/tasks/{tid}/frames").json()
    assert fr["total"] >= 4 and fr["by_status"] == {"DONE": fr["total"]}
    f0 = fr["frames"][0]
    assert f0["step_key"] == "stylize_0" and f0["idx"] == 0

    for kind in ("src", "out"):
        img = client.get(f"/api/tasks/{tid}/frames/{f0['step_key']}/{f0['item_key']}/image", params={"kind": kind})
        assert img.status_code == 200 and img.headers["content-type"] == "image/png"
        assert img.content[:4] == b"\x89PNG"
    th = client.get(f"/api/tasks/{tid}/frames/stylize_0/{f0['item_key']}/image", params={"kind": "out", "w": 64})
    assert th.headers["content-type"] == "image/jpeg"
    # 参数校验:不接受任意路径
    assert client.get(f"/api/tasks/{tid}/frames/stylize_0/..%2F..%2Fx/image").status_code in (404, 422)
    assert client.get(f"/api/tasks/{tid}/frames/keyframes/{f0['item_key']}/image").status_code == 422
    assert client.get(f"/api/tasks/{tid}/frames/stylize_0/999999/image").status_code == 404

    u = client.get(f"/api/tasks/{tid}/usage").json()
    assert u["total"]["billed"] == fr["total"] + 0 and u["total"]["maybe_billed"] == 1      # 注入的"响应丢失"
    assert u["budget_cny"] == 20 and u["provider"] == "mock"
    rep = client.get(f"/api/tasks/{tid}/report").json()
    assert {s["key"] for s in rep["steps"]} >= {"ingest", "keyframes", "stylize_0", "mux", "package", "publish"}
    assert rep["usage"]["total"]["calls"] == u["total"]["calls"]
    assert rep["wall_s"] > 0
    assert "/api/tasks/style" in client.get("/openapi.json").text
