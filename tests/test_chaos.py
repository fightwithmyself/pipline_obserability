"""F13 混沌测试:多个任务并发运行期间反复 kill -9 随机 worker,检查不变量。

不变量:
  I1 所有任务最终 SUCCEEDED
  I2 最终产物 hash 与"无故障基线"逐一相同(确定性步骤 + 正确恢复)
  I3 每个步骤至多一次有效提交(attempts 中 SUCCEEDED 恰好 1 条;缓存命中的为 0 条)
  I4 没有遗留 RUNNING 的 attempt
  I5 外部发布每个任务恰好一次(幂等键去重)
  I6 没有孤儿步骤进程残留
"""
from __future__ import annotations

import random
import time

from conftest import FAST_RETRY, WorkerPool, pid_alive, step, wait_for, wait_task
from pipeline.engine import Engine
from pipeline.models import RetryPolicy
from pipeline.templates import animation_swap_task

N_TASKS = 5


def mk(i: int, chaos: bool):
    return animation_swap_task(
        chunks=3, frames=6, unit_s=0.3, prompt=f"chaos-{i}", source=f"clip{i}.mp4",
        retry=RetryPolicy(max_attempts=5, base_delay=0.1, max_delay=0.5, jitter=0.0), stall_timeout_s=10,
        chaos={"segment_1": {"fail": {"kind": "TRANSIENT", "until_attempt": 1, "at": 0.3}}}
        if chaos and i % 2 == 0 else None)


def finals(t: dict) -> tuple[str, str]:
    return (step(t, "mux")["outputs"]["final.bin"]["sha256"],
            step(t, "publish")["outputs"]["receipt.json"]["sha256"])


def test_f13_random_worker_kills_preserve_invariants(data_dir, tmp_path, procs):
    # ---- 基线:同样的任务,无故障、无 kill
    base_dir = tmp_path / "base"
    be = Engine(base_dir)
    base_ids = [be.submit(mk(i, chaos=False))[0] for i in range(N_TASKS)]
    with WorkerPool(base_dir, n=3):
        baseline = [finals(wait_task(be, tid, timeout=90)) for tid in base_ids]
    be.close()

    # ---- 混沌:独立进程 worker,随机 kill -9 + 补位
    eng = Engine(data_dir)
    ids = [eng.submit(mk(i, chaos=True))[0] for i in range(N_TASKS)]
    rng = random.Random(20260715)
    alive = {f"c{i}": procs.spawn(f"c{i}", time_scale=1.0, lease_ttl=1.5) for i in range(3)}
    seq = 3
    kills = 0
    for _ in range(5):
        time.sleep(rng.uniform(1.0, 2.0))
        if all(eng.get_task(i)["status"] == "SUCCEEDED" for i in ids):
            break
        victim = rng.choice(sorted(alive))
        procs.kill9(alive.pop(victim))
        kills += 1
        wid = f"c{seq}"
        seq += 1
        alive[wid] = procs.spawn(wid, time_scale=1.0, lease_ttl=1.5)

    tasks = [wait_task(eng, i, timeout=180) for i in ids]

    # I1 / I2
    for i, t in enumerate(tasks):
        assert t["status"] == "SUCCEEDED", t["id"]
        assert finals(t) == baseline[i], f"task {i} 的最终产物与基线不一致"

    c = eng.conn
    # I3
    dup = c.execute("SELECT step_id, COUNT(*) n FROM attempts WHERE outcome='SUCCEEDED' "
                    "GROUP BY step_id HAVING n > 1").fetchall()
    assert not dup
    missing = c.execute(
        "SELECT s.id FROM steps s WHERE s.status='SUCCEEDED' AND s.cached=0 AND s.type != 'human_approval' "
        "AND (SELECT COUNT(*) FROM attempts a WHERE a.step_id=s.id AND a.outcome='SUCCEEDED') != 1").fetchall()
    assert not missing
    # I4
    assert c.execute("SELECT COUNT(*) FROM attempts WHERE outcome='RUNNING'").fetchone()[0] == 0
    # I5
    assert len(list((data_dir / "published").glob("*.json"))) == N_TASKS
    # I6
    procs.cleanup()
    pids = [r[0] for r in c.execute("SELECT child_pid FROM attempts WHERE child_pid IS NOT NULL")]
    wait_for(lambda: not any(pid_alive(p) for p in pids), 15, what="无孤儿步骤进程")

    m = eng.metrics()
    print(f"\n[chaos] kills={kills} attempts={m['attempts_by_outcome']} errors={m['errors_by_kind']} "
          f"waste_ratio={m['waste_ratio']}")
    assert kills >= 3
    assert m["attempts_by_outcome"].get("LOST", 0) >= 1, "kill 没有打中任何运行中的步骤,测试无效"
    eng.close()
