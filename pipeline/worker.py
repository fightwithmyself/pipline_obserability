"""Worker:拉取式认领 → 起子进程执行 → 监督(续租/卡死/超时/取消)→ 提交结果。

worker 是唯一访问数据库的一方;子进程只负责计算。fencing(attempt_no)只在这里和引擎之间保证。
"""
from __future__ import annotations

import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .engine import Claim, Engine
from .models import ErrorKind

log = logging.getLogger("pipeline.worker")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EMPTY = object()


def kill_tree(proc: subprocess.Popen) -> None:
    """强杀子进程及其后代(真实步骤可能再起 ffmpeg 等进程)。"""
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


class Worker:
    def __init__(self, data_dir: str | Path, worker_id: str | None = None,
                 resources: tuple[str, ...] = ("cpu", "gpu"), lease_ttl: float = 10.0,
                 poll_interval: float = 0.3, time_scale: float = 1.0):
        self.data_dir = Path(data_dir)
        self.engine = Engine(self.data_dir)
        self.id = worker_id or f"w-{os.getpid()}-{uuid.uuid4().hex[:4]}"
        self.resources = tuple(resources)
        self.lease_ttl = lease_ttl
        self.poll_interval = poll_interval
        self.time_scale = time_scale
        self.hb_interval = max(0.05, lease_ttl / 3)
        self._last_touch = 0.0

    # ------------------------------------------------------------------ 主循环
    def run_forever(self, stop: threading.Event | None = None, idle_exit: float | None = None,
                    max_steps: int | None = None) -> int:
        """返回处理的步骤数。idle_exit: 连续空闲这么多秒后退出(测试/批处理用)。"""
        stop = stop or threading.Event()
        eng = self.engine
        eng.register_worker(self.id, self.resources, os.getpid())
        log.info("worker %s started resources=%s lease_ttl=%s", self.id, self.resources, self.lease_ttl)
        done = 0
        last_busy = time.time()
        while not stop.is_set():
            claim = eng.claim(self.id, self.resources, self.lease_ttl)
            if claim is not None:
                eng.touch_worker(self.id, claim.step_id)
                self._execute(claim)
                eng.touch_worker(self.id, None)
                done += 1
                last_busy = time.time()
                if max_steps and done >= max_steps:
                    break
                continue
            now = time.time()
            if now - self._last_touch > 1.0:
                eng.touch_worker(self.id, None)
                self._last_touch = now
            if idle_exit is not None and now - last_busy > idle_exit:
                break
            stop.wait(self.poll_interval)
        return done

    # ------------------------------------------------------------------ 执行一次 attempt
    def _execute(self, claim: Claim) -> None:
        eng = self.engine
        adir = self.data_dir / "attempts" / f"{claim.step_id}-{claim.attempt_no}"
        shutil.rmtree(adir, ignore_errors=True)
        (adir / "out").mkdir(parents=True)
        work = self.data_dir / "work" / str(claim.step_id)
        work.mkdir(parents=True, exist_ok=True)
        job = {
            "task_id": claim.task_id, "key": claim.key, "type": claim.type, "params": claim.params,
            "chaos": claim.chaos, "attempt": claim.attempt_no, "checkpoint": claim.checkpoint,
            "inputs": claim.inputs, "work_dir": str(work), "out_dir": str(adir / "out"),
            "data_dir": str(self.data_dir), "time_scale": self.time_scale,
        }
        job_path = adir / "job.json"
        job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        stderr_f = open(adir / "stderr.log", "wb")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        kw: dict = {}
        if sys.platform != "win32":
            kw["start_new_session"] = True
        proc = subprocess.Popen(
            [sys.executable, "-u", "-m", "pipeline.step_runner", str(job_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_f, cwd=str(PROJECT_ROOT),
            env=env, **kw)
        try:
            self._supervise(claim, proc, adir, stderr_f)
        except BaseException as e:
            # worker 被中断(Ctrl+C/异常):杀子进程并主动让出租约,让别的 worker 立即接手
            kill_tree(proc)
            try:
                eng.fail(claim.step_id, claim.attempt_no, ErrorKind.LEASE_LOST,
                         f"worker interrupted: {type(e).__name__}")
            except Exception:
                pass
            raise
        finally:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except Exception:
                pass
            kill_tree(proc)
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
            stderr_f.close()
            shutil.rmtree(adir, ignore_errors=True)

    def _stderr_tail(self, adir: Path, n: int = 600) -> str:
        try:
            return (adir / "stderr.log").read_bytes()[-n:].decode("utf-8", "replace").strip()
        except OSError:
            return ""

    def _supervise(self, claim: Claim, proc: subprocess.Popen, adir: Path, stderr_f) -> None:
        eng = self.engine
        sid, att = claim.step_id, claim.attempt_no
        eng.set_child_pid(sid, att, proc.pid)

        q: queue.Queue = queue.Queue()

        def reader() -> None:
            try:
                for raw in proc.stdout:
                    try:
                        msg = json.loads(raw.decode("utf-8", "replace"))
                    except ValueError:
                        continue          # 噪声行,忽略
                    if isinstance(msg, dict) and "t" in msg:
                        q.put(msg)
            finally:
                q.put(None)

        threading.Thread(target=reader, daemon=True).start()

        start = last_progress = last_hb = time.time()
        cur_p: float | None = None
        cur_msg: str | None = None
        notes: dict = {}
        result: dict | None = None
        error: dict | None = None
        verdict: tuple[str, str] | None = None     # (ErrorKind, msg):由监督逻辑判定的失败
        lost = False

        while True:
            try:
                msg = q.get(timeout=0.05)
            except queue.Empty:
                msg = _EMPTY
            now = time.time()
            if msg is None:
                break
            if msg is not _EMPTY:
                t = msg.get("t")
                if t == "progress":
                    last_progress = now
                    cur_p, cur_msg = msg.get("p"), msg.get("msg")
                elif t == "checkpoint":
                    last_progress = now
                    if not eng.save_checkpoint(sid, att, msg["data"], claim.lease_ttl):
                        lost = True
                        break
                    last_hb = now
                elif t == "item":
                    # 工作项账本(观测用):被 fencing 拒绝只说明这次 attempt 已过期,忽略即可
                    last_progress = now
                    eng.upsert_item(sid, att, str(msg["k"]), str(msg["status"]), msg.get("data"))
                elif t == "note":
                    notes[msg["k"]] = msg["v"]
                elif t == "done":
                    result = msg
                elif t == "error":
                    error = msg
            if now - start > claim.timeout_s:
                verdict = (ErrorKind.TIMEOUT, f"超出总时限 {claim.timeout_s}s")
                break
            if now - last_progress > claim.stall_timeout_s:
                verdict = (ErrorKind.STALL, f"{claim.stall_timeout_s}s 无进展(最后进度 {cur_p})")
                break
            if now - last_hb >= self.hb_interval:
                last_hb = now
                if not eng.heartbeat(sid, att, claim.lease_ttl, cur_p, cur_msg, last_progress):
                    lost = True
                    break
                eng.touch_worker(self.id, sid)

        if lost:
            log.warning("step %s/%s attempt %s: lease lost or cancelled, killing child", claim.task_id, claim.key, att)
            kill_tree(proc)
            return
        if verdict:
            kill_tree(proc)
            eng.fail(sid, att, verdict[0], verdict[1], notes or None)
            return
        try:
            rc = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            rc = proc.wait()
        if result is not None and rc == 0:
            try:
                outputs = {}
                for name, rel in result.get("outputs", {}).items():
                    p = (adir / "out" / rel).resolve()
                    if (adir / "out").resolve() not in p.parents:
                        raise ValueError(f"产物路径越界: {rel}")
                    sha, size = eng.store.put_file(p)
                    outputs[name] = {"sha256": sha, "size": size}
            except Exception as e:
                eng.fail(sid, att, ErrorKind.TRANSIENT, f"产物入库失败: {e}", notes or None)
                return
            if not eng.complete(sid, att, outputs, notes or None):
                log.warning("step %s/%s attempt %s: result discarded (fenced)", claim.task_id, claim.key, att)
            return
        if error is not None:
            eng.fail(sid, att, error.get("kind", ErrorKind.TRANSIENT), str(error.get("msg", "")), notes or None)
            return
        tail = self._stderr_tail(adir)
        eng.fail(sid, att, ErrorKind.TRANSIENT,
                 f"步骤进程异常退出 exit={rc}" + (f"; stderr: {tail}" if tail else ""), notes or None)
