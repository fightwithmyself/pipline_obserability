"""恢复基准:kill -9 一个正在执行长步骤的 worker,测量
  - 恢复延迟:kill 时刻 → 该步骤重新开始执行(受 lease_ttl 影响)
  - 重复计算:恢复后重新计算的帧数(有检查点 vs 人为清除检查点)

用法: python scripts/bench_recovery.py [--ttls 1,2,4] [--frames 12] [--unit 0.4]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.engine import Engine  # noqa: E402
from pipeline.models import RetryPolicy  # noqa: E402
from pipeline.templates import animation_swap_task  # noqa: E402


def spawn(data: Path, wid: str, ttl: float) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "pipeline.cli", "--data", str(data), "worker", "--id", wid, "--lease-ttl", str(ttl),
         "--poll", "0.05"], cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def wait(cond, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(0.02)
    raise TimeoutError


def one_run(ttl: float, frames: int, unit: float, clear_ckpt: bool) -> dict:
    data = Path(tempfile.mkdtemp(prefix="bench_"))
    eng = Engine(data)
    spec = animation_swap_task(chunks=1, frames=frames, unit_s=unit, stall_timeout_s=30,
                               retry=RetryPolicy(max_attempts=3, base_delay=0.1, max_delay=0.5, jitter=0))
    tid, _ = eng.submit(spec)
    wa = spawn(data, "A", ttl)
    wait(lambda: (lambda s: s["status"] == "RUNNING" and s["checkpoint"] and s["checkpoint"]["done"] >= frames // 2)(
        next(x for x in eng.get_task(tid)["steps"] if x["key"] == "segment_0")))
    done_at_kill = next(x for x in eng.get_task(tid)["steps"] if x["key"] == "segment_0")["checkpoint"]["done"]
    t_kill = time.time()
    wa.kill()
    wa.wait()
    if clear_ckpt:
        with eng.conn:
            eng.conn.execute("UPDATE steps SET checkpoint=NULL WHERE key='segment_0'")
    wb = spawn(data, "B", ttl)
    try:
        atts = wait(lambda: (lambda a: a if len(a) >= 2 else None)(eng.get_attempts(tid, "segment_0")))
        latency = atts[1]["started_at"] - t_kill
        wait(lambda: eng.get_task(tid)["status"] == "SUCCEEDED")
        atts = eng.get_attempts(tid, "segment_0")
        return {"latency": latency, "done_at_kill": done_at_kill, "recomputed": atts[1]["notes"]["frames_computed"],
                "resumed_from": atts[1]["notes"]["resumed_from"]}
    finally:
        wb.kill()
        wb.wait()
        eng.close()
        shutil.rmtree(data, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ttls", default="1,2,4")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--unit", type=float, default=0.4)
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    print(f"frames={a.frames} unit={a.unit}s (整步骤 {a.frames * a.unit:.1f}s),每组重复 {a.repeat} 次")
    print(f"{'lease_ttl':>9} {'检查点':>6} {'恢复延迟(s)':>12} {'kill时已完成帧':>14} {'恢复后重算帧':>12} {'浪费占比':>8}")
    for ttl in [float(x) for x in a.ttls.split(",")]:
        for clear in (False, True):
            rs = [one_run(ttl, a.frames, a.unit, clear) for _ in range(a.repeat)]
            lat = sum(r["latency"] for r in rs) / len(rs)
            rec = sum(r["recomputed"] for r in rs) / len(rs)
            dk = sum(r["done_at_kill"] for r in rs) / len(rs)
            waste = rec - (a.frames - dk)      # 重算帧数 - 理论最少重算量 = 白白重复计算的帧
            print(f"{ttl:>9} {'无' if clear else '有':>6} {lat:>12.2f} {dk:>14.1f} {rec:>12.1f} {max(waste, 0) / a.frames:>8.0%}")


if __name__ == "__main__":
    main()
