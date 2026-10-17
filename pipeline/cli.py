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


def cmd_submit_style(a) -> int:
    """提交主流水线任务:素材 → 逐帧风格化 → 成品视频。provider=mock 不花钱;provider=ark 调用豆包(密钥在环境变量)。"""
    from .templates import DEFAULT_PRICE_CNY, frame_style_inject_presets, frame_style_task
    eng = Engine(a.data)
    chaos = json.loads(a.chaos) if a.chaos else None
    if a.inject:
        chaos = {**(chaos or {}), **frame_style_inject_presets(a.chunks)[a.inject]}
    if a.provider != "mock" and not os.environ.get("ARK_API_KEY"):
        print("提示:还没有设置环境变量 ARK_API_KEY,worker 运行时会在工序 3 立即失败(不会产生费用)", file=sys.stderr)
    try:
        spec = frame_style_task(
            a.name, source=a.source, prompt=a.prompt, provider=a.provider, model=a.model, reference=a.reference,
            chunks=a.chunks, stride=a.stride, interp=a.interp, max_gap=a.max_gap, target_height=a.height, price_cny=a.price,
            budget_cny=a.budget, concurrency=a.concurrency, on_reject=a.on_reject,
            require_approval=a.approval, chaos=chaos,
            retry=RetryPolicy(max_attempts=a.max_attempts, base_delay=a.base_delay, max_delay=60, jitter=0.2))
    except ValueError as e:
        print(f"提交失败: {e}", file=sys.stderr)
        return 2
    price = DEFAULT_PRICE_CNY if a.price is None else a.price
    print(f"provider={a.provider} 单价按 {price} 元/张估算,预算上限 {a.budget} 元;关键帧数在工序 2 之后才知道,"
          f"超预算会在调用接口之前就失败")
    tid, created = eng.submit(spec, idempotency_key=a.key)
    print(tid if created else f"{tid} (已存在,幂等返回)")
    return 0


def cmd_frames(a) -> int:
    """打印逐帧账本与用量(与 Dashboard 的帧网格同一份数据)。"""
    eng = Engine(a.data)
    items = eng.list_items(a.task_id, prefix="stylize_")
    items.sort(key=lambda x: x.get("idx", 0))
    by: dict[str, int] = {}
    for it in items:
        by[it["status"]] = by.get(it["status"], 0) + 1
        if a.all or it["status"] not in ("DONE",) or it.get("maybe_duplicated") or it.get("calls", 0) > 1:
            print(f"帧 {it['idx']:>6} [{it['step_key']}] {it['status']:<9} calls={it.get('calls')} billed={it.get('billed')} "
                  f"maybe_billed={it.get('maybe_billed')} cached={it.get('cached')} resumed={it.get('resumed')} "
                  f"{(it.get('error') or '')[:80]}")
    cost = sum(float(i.get("cost_cny_est") or 0) for i in items)
    extra = sum(float(i.get("cost_maybe_extra_cny_est") or 0) for i in items)
    print(f"共 {len(items)} 帧 {by};费用估算 ≈ {cost:.2f} 元(另有可能多付 ≈ {extra:.2f} 元;以平台账单为准)")
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


def cmd_agent_watch(a) -> int:
    """持续扫描失败步骤并让 Diagnosis Agent 处理:--auto 时低风险场景自动重试,其余标记"需要人工"并打印人工恢复步骤。"""
    from .agent.diagnosis import DiagnosisAgent
    from .agent.llm import TracedLLM, make_llm
    eng = Engine(a.data)
    ag = DiagnosisAgent(a.data, eng, TracedLLM(make_llm(), eng), auto_recover=a.auto)
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    deadline = time.time() + a.duration if a.duration else None
    try:
        while not stop.is_set() and (deadline is None or time.time() < deadline):
            for o in ag.handle_failures(a.task):
                p, r = o["proposal"], o.get("result") or {}
                tag = "自动重试" if r.get("executed") else ("需要人工" if r.get("needs_human") else "等待人工确认")
                print(f"[{tag}] {o['task_id']}/{o['step_key']}: {p['root_cause']} (手册={p['playbook']})", flush=True)
                if r.get("needs_human"):
                    for i, s in enumerate(r.get("human_steps", []), 1):
                        print(f"    {i}. {s}", flush=True)
            stop.wait(a.interval)
    finally:
        ag.close()
        eng.close()
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

    sv = sub.add_parser("submit-style", help="提交主流水线:素材 → 逐帧风格化(豆包) → 成品视频")
    sv.add_argument("--source", required=True, help="本地视频文件")
    sv.add_argument("--prompt", required=True, help="风格/替换指令,如:日系动漫风格,保持人物与动作不变")
    sv.add_argument("--name", default="anime-style")
    sv.add_argument("--provider", choices=["mock", "ark", "ark_plan"], default="mock",
                    help="mock=离线替身(不花钱);ark=豆包按量付费;ark_plan=豆包 Agent Plan(专属路径与专属密钥,扣套餐额度)")
    sv.add_argument("--model", help="provider=ark 时必填:方舟控制台里的模型 ID 或 Endpoint ID")
    sv.add_argument("--reference", help="锚点/人物参考图(可选)")
    sv.add_argument("--chunks", type=int, default=4)
    sv.add_argument("--stride", type=int, default=6, help="关键帧间隔(帧);越大越省钱、补间越多")
    sv.add_argument("--interp", choices=["mc", "blend"], default="mc",
                    help="补帧方式:mc=按源视频运动搬运关键帧(默认,无重影);blend=线性混合(旧,有重影)")
    sv.add_argument("--max-gap", type=int, default=24)
    sv.add_argument("--height", type=int, default=1080, help="标准化目标高度")
    sv.add_argument("--price", type=float, help="单张价格(元),默认 0.12;只用于估算与预算保护")
    sv.add_argument("--budget", type=float, default=20.0, help="预算上限(元),必须设置")
    sv.add_argument("--concurrency", type=int, default=4)
    sv.add_argument("--on-reject", choices=["passthrough", "fail"], default="passthrough")
    sv.add_argument("--approval", action="store_true")
    sv.add_argument("--max-attempts", type=int, default=3)
    sv.add_argument("--base-delay", type=float, default=2.0)
    sv.add_argument("--chaos", help="JSON 故障注入")
    sv.add_argument("--inject", choices=["transient", "crash", "moderation", "response_lost", "rate_limit", "auth_fail"],
                    help="故障注入预设(mock)")
    sv.add_argument("--key", help="幂等键")
    sv.set_defaults(fn=cmd_submit_style)

    fr = sub.add_parser("frames", help="打印逐帧账本与用量")
    fr.add_argument("task_id")
    fr.add_argument("--all", action="store_true", help="包括正常完成的帧")
    fr.set_defaults(fn=cmd_frames)

    st = sub.add_parser("status")
    st.add_argument("task_id", nargs="?")
    st.set_defaults(fn=cmd_status)

    ev = sub.add_parser("events")
    ev.add_argument("task_id", nargs="?")
    ev.add_argument("--after", type=int, default=0)
    ev.set_defaults(fn=cmd_events)

    aw = sub.add_parser("agent-watch", help="失败诊断 Agent:扫描失败步骤,可自动恢复的自动重试,其余标记需要人工并给出恢复步骤")
    aw.add_argument("--task", help="只处理这个任务(默认所有 FAILED 任务)")
    aw.add_argument("--auto", action="store_true", help="允许对手册白名单内的低风险失败自动重试(每个步骤最多 2 次)")
    aw.add_argument("--interval", type=float, default=3.0)
    aw.add_argument("--duration", type=float, help="最多运行多少秒(默认一直运行)")
    aw.set_defaults(fn=cmd_agent_watch)

    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
