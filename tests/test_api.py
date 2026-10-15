from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from conftest import WorkerPool, wait_for
from pipeline.api import create_app


@pytest.fixture
def client(data_dir):
    return TestClient(create_app(data_dir))


def body(**kw):
    b = dict(chunks=2, frames=4, unit_s=0.05, base_delay=0.1, max_attempts=3)
    b.update(kw)
    return b


def test_submit_query_and_history(client, data_dir):
    r = client.post("/api/tasks/animation", json=body(
        chaos={"segment_0": {"fail": {"kind": "TRANSIENT", "until_attempt": 1, "at": 0.0}}}),
        headers={"Idempotency-Key": "k1"})
    assert r.status_code == 201 and r.json()["created"] is True
    tid = r.json()["id"]
    r2 = client.post("/api/tasks/animation", json=body(), headers={"Idempotency-Key": "k1"})
    assert r2.json() == {"id": tid, "created": False}

    with WorkerPool(data_dir):
        t = wait_for(lambda: (lambda x: x if x["status"] == "SUCCEEDED" else None)(
            client.get(f"/api/tasks/{tid}").json()), 60, what="任务完成")
    assert t["percent"] == 100.0 and t["steps_succeeded"] == t["steps_total"]
    assert any(s["key"] == "segment_0" and s["attempt_no"] == 2 for s in t["steps"])

    atts = client.get(f"/api/tasks/{tid}/steps/segment_0/attempts").json()
    assert [a["outcome"] for a in atts] == ["FAILED", "SUCCEEDED"]
    ev = client.get(f"/api/tasks/{tid}/events").json()
    assert ev[0]["kind"] == "TASK_CREATED" and ev[-1]["to"] == "SUCCEEDED"
    last = ev[len(ev) // 2]["seq"]
    assert all(e["seq"] > last for e in client.get(f"/api/tasks/{tid}/events", params={"after": last}).json())

    assert client.get("/api/tasks").json()[0]["id"] == tid
    m = client.get("/api/metrics").json()
    assert m["tasks_by_status"]["SUCCEEDED"] == 1 and m["errors_by_kind"]["TRANSIENT"] == 1
    assert {w["id"] for w in client.get("/api/workers").json()} == {"tw0"}
    assert client.get("/").status_code == 200 and "任务管线" in client.get("/").text


def test_error_mapping(client):
    assert client.get("/api/tasks/nope").status_code == 404
    assert client.get("/api/tasks/nope/steps/x/attempts").status_code == 404
    tid = client.post("/api/tasks/animation", json=body()).json()["id"]
    assert client.post(f"/api/tasks/{tid}/resume").status_code == 409            # 没有失败步骤
    assert client.post(f"/api/tasks/{tid}/steps/probe/retry").status_code == 409
    assert client.post(f"/api/tasks/{tid}/steps/probe/approve", json={"decision": "approve"}).status_code == 409
    assert client.post("/api/tasks", json={"name": "x", "steps": [
        {"key": "a", "type": "probe", "deps": ["b"]}, {"key": "b", "type": "probe", "deps": ["a"]}]}
    ).status_code == 422                                                         # 成环
    assert client.post("/api/tasks/animation", json=body(chunks=8, frames=4)).status_code == 422
    assert client.post(f"/api/tasks/{tid}/cancel").status_code == 200
    assert client.get(f"/api/tasks/{tid}").json()["status"] == "CANCELLED"
    assert client.post(f"/api/tasks/{tid}/cancel").status_code == 409
