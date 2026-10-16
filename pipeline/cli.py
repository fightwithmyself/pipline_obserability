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


def cmd_submit_kling(a) -> int:
    from .templates import kling_video_task
    eng = Engine(a.data)
    try:
        spec = kling_video_task(a.name, prompt=a.prompt, mode=a.mode, duration=a.duration, audio=a.audio,
                                aspect_ratio=a.aspect_ratio, poll_interval_s=a.poll_interval, max_wait_s=a.max_wait,
                                retry=RetryPolicy(max_attempts=a.max_attempts, base_delay=2.0, max_delay=30, jitter=0.2))
    except ValueError as e:
        print(f"提交失败: {e}", file=sys.stderr)
        return 2
    from .steps.kling import PRICE_CNY_PER_S
    est = PRICE_CNY_PER_S[(a.mode, a.audio)] * a.duration
    print(f"预估费用 ≈ {est:.2f} 元(按文档标价 {PRICE_CNY_PER_S[(a.mode, a.audio)]} 元/秒 × {a.duration} 秒;失败不计费)")
    tid, created = eng.submit(spec, idempotency_key=a.key)
    print(tid if created else f"{tid} (已存在,幂等返回)")
    return 0


def cmd_submit_style(a) -> int:
    from .media import MediaError, probe_video
    from .steps.vst import PRICE_CNY_PER_S, STYLE_NAMES
    from .templates import style_transform_task
    eng = Engine(a.data)
    try:
        spec = style_transform_task(
            a.name, source=a.source, style=a.style, min_len=a.min_len, video_fps=a.video_fps, use_sr=a.use_sr,
            review_model=a.review_model, poll_interval_s=a.poll_interval, max_wait_s=a.max_wait,
            retry=RetryPolicy(max_attempts=a.max_attempts, base_delay=2.0, max_delay=30, jitter=0.2))
        dur = probe_video(a.source)["duration_s"]
    except (ValueError, MediaError) as e:
        print(f"提交失败: {e}", file=sys.stderr)
        return 2
    est = PRICE_CNY_PER_S[a.min_len] * dur
    print(f"风格: {STYLE_NAMES[a.style]};素材时长 {dur}s;预估上限 ≈ {est:.2f} 元"
          f"(按标价 {PRICE_CNY_PER_S[a.min_len]} 元/秒 × 输出时长;该模型有 600 秒免费额度,额度内不扣费;失败不计费)")
    tid, created = eng.submit(spec, idempotency_key=a.key)
    print(tid if created else f"{tid} (已存在,幂等返回)")
    return 0


def cmd_kling_report(a) -> int:
    """按轮打印外部视频生成任务的观测记录:每次尝试、每轮轮询、用量与费用估算。"""
    eng = Engine(a.data)
    t = eng.get_task(a.task_id)
    print(f"任务 {t['id']}  {t['name']}  {t['status']}")
    for att in eng.get_attempts(a.task_id, "generate"):
        notes = att.get("notes") or {}
        print(f"\n== 第 {att['attempt_no']} 次尝试  {att['outcome']}  耗时 {att.get('duration', 0):.1f}s  "
              f"{(att.get('error_kind') or '')} {(att.get('error_msg') or '')[:140]}")
        print(f"   本次是否提交了新的外部任务: {notes.get('submitted_this_attempt')}   "
              f"外部 task_id: {notes.get('external_task_id') or notes.get('resumed_external_task')}")
        for k in sorted(k for k in notes if k.startswith("poll_")):
            r = notes[k]
            print(f"   轮询 {k[5:]}  状态={r['status']:<10} 已等待={r['elapsed_s']:>6}s  "
                  f"token={r['tokens']}  用量(usage)={r['usage']}")
        for k in ("usage", "cost_cny_est", "video_info", "video_bytes", "external_failure"):
            if k in notes:
                print(f"   {k}: {notes[k]}")
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

    k = sub.add_parser("submit-kling", help="提交一个外部视频生成任务(可灵 omni,按秒计费,会花钱)")
    k.add_argument("--prompt", required=True)
    k.add_argument("--name", default="kling-video")
    k.add_argument("--mode", choices=["std", "pro", "4k"], default="std", help="std=720P(最便宜) pro=1080P")
    k.add_argument("--duration", type=int, default=3, help="秒,3..15;直接决定费用")
    k.add_argument("--audio", action="store_true", help="生成有声视频(更贵)")
    k.add_argument("--aspect-ratio", default="16:9")
    k.add_argument("--poll-interval", type=float, default=15.0)
    k.add_argument("--max-wait", type=float, default=900.0)
    k.add_argument("--max-attempts", type=int, default=3)
    k.add_argument("--key", help="幂等键")
    k.set_defaults(fn=cmd_submit_kling)

    sv = sub.add_parser("submit-style", help="提交视频风格重绘任务链(百炼 video-style-transform,按输出秒数计费)")
    sv.add_argument("--source", required=True, help="本地视频;先经 vst_probe 检查是否满足接口要求")
    sv.add_argument("--name", default="style-transform")
    sv.add_argument("--style", type=int, default=0, help="0=日式漫画(默认) 1=美式 2=清新 3=3D卡通 4=国风卡通 5=纸艺 6=简易插画 7=水墨")
    sv.add_argument("--min-len", type=int, choices=[540, 720], default=540, help="输出短边;540P 0.2元/秒,720P 0.5元/秒")
    sv.add_argument("--video-fps", type=int, default=15)
    sv.add_argument("--use-sr", action="store_true", help="超分(540P 开启后可达 1080P 画质,仍按 540P 计费)")
    sv.add_argument("--review-model", default="qwen3.8-omni-flash")
    sv.add_argument("--poll-interval", type=float, default=10.0)
    sv.add_argument("--max-wait", type=float, default=900.0)
    sv.add_argument("--max-attempts", type=int, default=3)
    sv.add_argument("--key", help="幂等键")
    sv.set_defaults(fn=cmd_submit_style)

    kr = sub.add_parser("kling-report", help="按轮打印外部视频生成任务的观测记录")
    kr.add_argument("task_id")
    kr.set_defaults(fn=cmd_kling_report)

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
