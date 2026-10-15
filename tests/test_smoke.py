from conftest import WorkerPool, wait_task, step
from pipeline.templates import animation_swap_task


def test_happy_path(engine, data_dir):
    tid, created = engine.submit(animation_swap_task(chunks=2, frames=4))
    assert created
    with WorkerPool(data_dir, n=2):
        t = wait_task(engine, tid)
    assert t["status"] == "SUCCEEDED", [(s["key"], s["status"], s["error_msg"]) for s in t["steps"]]
    assert t["steps_succeeded"] == t["steps_total"]
    assert step(t, "publish")["outputs"]["receipt.json"]["sha256"]
