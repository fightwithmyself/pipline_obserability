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


# ----------------------------------------------------------------------------- Agent 接口
def test_plan_endpoint_guardrails_trace_and_diagnosis_flow(client, data_dir, monkeypatch):
    from pipeline.models import RetryPolicy  # noqa: F401
    r = client.post("/api/plan", json={"request": "把 demo.mp4 的主角换成红发女孩并发布,不用审批"})
    assert r.status_code == 201
    j = r.json()
    assert j["ok"] and j["rounds"] == 2 and any("策略" in e for e in j["attempts"][0]["errors"])
    tid = j["task_id"]
    assert "human_approval" in {s["key"] for s in client.get(f"/api/tasks/{tid}").json()["steps"]}

    calls = client.get("/api/llm_calls", params={"task_id": tid, "include_messages": True}).json()
    assert [c["purpose"] for c in reversed(calls)] == ["plan", "plan_repair"] and calls[0]["messages"]
    u = client.get("/api/llm_usage", params={"task_id": tid}).json()
    assert u["calls"] == 2 and u["estimated_calls"] == 2
    assert client.get(f"/api/plans/{j['plan_id']}").json()["status"] == "OK"
    assert client.get("/api/plans/9999").status_code == 404

    bad = client.post("/api/plan", json={"request": "换角色 [mock:evil]"})
    assert bad.status_code == 422 and bad.json()["ok"] is False
    assert any("shell" in e for e in bad.json()["errors"])

    # 诊断流程:先造一个失败任务
    body_ = dict(chunks=2, frames=4, unit_s=0.05, base_delay=0.1, max_attempts=3,
                 chaos={"mux": {"fail": {"kind": "TRANSIENT", "until_attempt": 3, "at": 0.0}}})
    t2 = client.post("/api/tasks/animation", json=body_).json()["id"]
    with WorkerPool(data_dir):
        wait_for(lambda: client.get(f"/api/tasks/{t2}").json()["status"] == "FAILED", 60, what="任务失败")
    d = client.post(f"/api/tasks/{t2}/steps/mux/diagnose")
    assert d.status_code == 201 and d.json()["status"] == "AWAITING_HUMAN"
    th = d.json()["thread_id"]
    assert [x["thread_id"] for x in client.get("/api/diagnoses", params={"status": "AWAITING_HUMAN"}).json()] == [th]
    assert client.get(f"/api/tasks/{t2}").json()["steps"][-1]["status"] in ("BLOCKED",)       # 人没确认前不执行
    ok = client.post(f"/api/diagnoses/{th}/decision", json={"approve": True, "who": "tester"})
    assert ok.status_code == 200 and ok.json()["result"]["executed"] is True
    assert client.post(f"/api/diagnoses/{th}/decision", json={"approve": True}).status_code == 409
    assert client.post(f"/api/tasks/{t2}/steps/probe/diagnose").status_code == 409            # 非 FAILED 不能诊断
