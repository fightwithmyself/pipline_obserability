"""命令行入口:  python -m pipeline.cli {worker,api,submit,status,events}"""
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

from .engine import Engine
from .models import RetryPolicy
from .templates import animation_swap_task

DEFAULT_DATA = os.environ.get("PIPELINE_DATA", "data")


def cmd_worker(a) -> int:
    from .worker import Worker
    w = Worker(a.data, worker_id=a.id, resources=tuple(a.resources.split(",")), lease_ttl=a.lease_ttl,
               poll_interval=a.poll, time_scale=a.time_scale)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    n = w.run_forever(stop, idle_exit=a.idle_exit)
    print(f"worker {w.id} exit, steps done: {n}")
    return 0


def cmd_api(a) -> int:
    import uvicorn
    from .api import create_app
    uvicorn.run(create_app(a.data), host=a.host, port=a.port, log_level="info")
    return 0


INJECT_PRESETS = {
    "transient": {"segment_0": {"fail": {"kind": "TRANSIENT", "until_attempt": 2, "at": 0.3}}},
    "crash": {"segment_1": {"crash": {"at": 0.5, "on_attempts": [1]}}},
    "hang": {"swap_0": {"hang": {"at": 0.5, "on_attempts": [1]}}},
}


def cmd_submit(a) -> int:
    eng = Engine(a.data)
    chaos = json.loads(a.chaos) if a.chaos else None
    if a.inject:
        chaos = {**(chaos or {}), **INJECT_PRESETS[a.inject]}
    spec = animation_swap_task(
        a.name, source=a.source, frames=a.frames, chunks=a.chunks, prompt=a.prompt,
        require_approval=a.approval, qc_fail=a.qc_fail, nondeterministic_ref=a.nondeterministic_ref,
        unit_s=a.unit_s, stall_timeout_s=a.stall_timeout, chaos=chaos,
        retry=RetryPolicy(max_attempts=a.max_attempts, base_delay=a.base_delay, max_delay=30, jitter=0.2))
    tid, created = eng.submit(spec, idempotency_key=a.key)
    print(tid if created else f"{tid} (已存在,幂等返回)")
    return 0


def cmd_submit_frames(a) -> int:
    from .media import MediaError
    from .templates import FRAME_INJECT_PRESETS, frame_pipeline_task
    eng = Engine(a.data)
    chaos = json.loads(a.chaos) if a.chaos else None
    if a.inject:
        chaos = {**(chaos or {}), **FRAME_INJECT_PRESETS[a.inject]}
    try:
        spec = frame_pipeline_task(
            a.name, source=a.source, chunks=a.chunks, require_approval=a.approval, qc_fail=a.qc_fail,
            stall_timeout_s=a.stall_timeout, chaos=chaos,
            retry=RetryPolicy(max_attempts=a.max_attempts, base_delay=a.base_delay, max_delay=30, jitter=0.2))
    except (ValueError, MediaError) as e:
        print(f"提交失败: {e}", file=sys.stderr)
        return 2
    tid, created = eng.submit(spec, idempotency_key=a.key)
    print(tid if created else f"{tid} (已存在,幂等返回)")
    return 0


def _fmt_ts(ts) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts)) if ts else "-"


def cmd_status(a) -> int:
    eng = Engine(a.data)
    if not a.task_id:
        for t in eng.list_tasks():
            print(f"{t['id']}  {t['status']:<10} {t['percent']:>5}%  {t['name']}  "
                  f"steps={t['steps_succeeded']}/{t['steps_total']} cached={t['cached_steps']}")
        return 0
    t = eng.get_task(a.task_id)
    print(f"{t['id']}  {t['name']}  {t['status']}  {t['percent']}%")
    for s in t["steps"]:
        extra = f"  ! {s['error_kind']}: {(s['error_msg'] or '')[:60]}" if s["error_kind"] else ""
        print(f"  {s['key']:<16} {s['status']:<18} attempt={s['attempt_no']} fail={s['failures']} "
              f"lost={s['lost_count']} {'cached' if s['cached'] else ''}{extra}")
    return 0


def cmd_events(a) -> int:
    eng = Engine(a.data)
    for e in eng.get_events(a.task_id, a.after):
        print(f"#{e['seq']:<5} {_fmt_ts(e['ts'])} {e['kind']:<16} {e['step_key'] or '-':<16} "
              f"{e['from'] or ''}->{e['to'] or ''} a={e['attempt_no']} {json.dumps(e['detail'], ensure_ascii=False) if e['detail'] else ''}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="pipeline")
    p.add_argument("--data", default=DEFAULT_DATA, help="数据目录(含 pipeline.db 与产物)")
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("worker")
    w.add_argument("--id")
    w.add_argument("--resources", default="cpu,gpu", help="本 worker 能力,如 cpu 或 gpu 或 cpu,gpu")
    w.add_argument("--lease-ttl", type=float, default=10.0)
    w.add_argument("--poll", type=float, default=0.3)
    w.add_argument("--time-scale", type=float, default=1.0)
    w.add_argument("--idle-exit", type=float, default=None)
    w.set_defaults(fn=cmd_worker)

    s = sub.add_parser("api")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_api)

    m = sub.add_parser("submit")
    m.add_argument("--name", default="character-swap")
    m.add_argument("--source", default="demo.mp4")
    m.add_argument("--frames", type=int, default=24)
    m.add_argument("--chunks", type=int, default=4)
    m.add_argument("--prompt", default="red-haired girl")
    m.add_argument("--unit-s", type=float, default=0.5)
    m.add_argument("--stall-timeout", type=float, default=30.0)
    m.add_argument("--max-attempts", type=int, default=4)
    m.add_argument("--base-delay", type=float, default=1.0)
    m.add_argument("--approval", action="store_true")
    m.add_argument("--qc-fail", action="store_true")
    m.add_argument("--nondeterministic-ref", action="store_true")
    m.add_argument("--chaos", help='JSON,如 {"segment_1":{"crash":{"at":0.5,"on_attempts":[1]}}}')
    m.add_argument("--inject", choices=sorted(INJECT_PRESETS), help="故障注入预设:瞬时失败/进程崩溃/卡死")
    m.add_argument("--key", help="幂等键")
    m.set_defaults(fn=cmd_submit)

    f = sub.add_parser("submit-frames", help="提交序列帧处理任务(拆帧→抠背景→超分→补帧→合成,真实处理)")
    f.add_argument("--source", required=True, help="本地视频文件路径(先用 python -m pipeline.media assess 检测)")
    f.add_argument("--name", default="frame-pipeline")
    f.add_argument("--chunks", type=int, default=4)
    f.add_argument("--stall-timeout", type=float, default=60.0)
    f.add_argument("--max-attempts", type=int, default=3)
    f.add_argument("--base-delay", type=float, default=1.0)
    f.add_argument("--approval", action="store_true")
    f.add_argument("--qc-fail", action="store_true")
    f.add_argument("--chaos", help='JSON,如 {"matte_1":{"crash":{"at":0.5,"on_attempts":[1]}}}')
    from .templates import FRAME_INJECT_PRESETS
    f.add_argument("--inject", choices=sorted(FRAME_INJECT_PRESETS), help="故障注入预设")
    f.add_argument("--key", help="幂等键")
    f.set_defaults(fn=cmd_submit_frames)

    st = sub.add_parser("status")
    st.add_argument("task_id", nargs="?")
    st.set_defaults(fn=cmd_status)

    ev = sub.add_parser("events")
    ev.add_argument("task_id", nargs="?")
    ev.add_argument("--after", type=int, default=0)
    ev.set_defaults(fn=cmd_events)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
