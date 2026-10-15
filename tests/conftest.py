from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.engine import Engine  # noqa: E402
from pipeline.worker import Worker  # noqa: E402

TIME_SCALE = 0.1      # 把 MOCK 步骤的耗时压缩 10 倍
LEASE_TTL = 1.5


@pytest.fixture(autouse=True)
def _force_mock_llm(monkeypatch):
    """测试永远用 MockLLM:不能因为本机设置了真实 API Key 就联网、花钱、产生不可复现的结果。"""
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def engine(data_dir: Path) -> Engine:
    e = Engine(data_dir)
    yield e
    e.close()


class WorkerPool:
    """在线程里跑若干 Worker(每个 worker 仍然为每个步骤起真实子进程)。"""

    def __init__(self, data_dir: Path, n: int = 1, resources=("cpu", "gpu"), lease_ttl: float = LEASE_TTL,
                 time_scale: float = TIME_SCALE):
        self.stop = threading.Event()
        self.workers = [Worker(data_dir, worker_id=f"tw{i}", resources=resources, lease_ttl=lease_ttl,
                               poll_interval=0.05, time_scale=time_scale) for i in range(n)]
        self.threads = [threading.Thread(target=w.run_forever, args=(self.stop,), daemon=True)
                        for w in self.workers]

    def start(self) -> "WorkerPool":
        for t in self.threads:
            t.start()
        return self

    def shutdown(self) -> None:
        self.stop.set()
        for t in self.threads:
            t.join(timeout=30)

    def __enter__(self) -> "WorkerPool":
        return self.start()

    def __exit__(self, *a) -> None:
        self.shutdown()


def wait_for(cond, timeout: float = 60.0, interval: float = 0.05, what: str = "condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(interval)
    raise TimeoutError(f"等待超时: {what}")


def wait_task(engine: Engine, task_id: str, statuses=("SUCCEEDED", "FAILED", "CANCELLED"),
              timeout: float = 60.0) -> dict:
    def cond():
        t = engine.get_task(task_id)
        return t if t["status"] in statuses else None
    try:
        return wait_for(cond, timeout, what=f"task {task_id} in {statuses}")
    except TimeoutError:
        t = engine.get_task(task_id)
        brief = [(s["key"], s["status"], s["attempt_no"]) for s in t["steps"] if s["status"] != "SUCCEEDED"]
        raise TimeoutError(f"task {task_id} 状态 {t['status']},未完成步骤: {brief}")


def pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def step(task: dict, key: str) -> dict:
    return next(s for s in task["steps"] if s["key"] == key)


FAST_RETRY = dict(max_attempts=3, base_delay=0.1, max_delay=0.5, jitter=0.0)


class ProcWorkers:
    """以独立 OS 进程启动 worker(python -m pipeline.cli worker),用于真实 kill -9 测试。"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.procs: list[subprocess.Popen] = []

    def spawn(self, wid: str, *, time_scale: float = 1.0, lease_ttl: float = 2.0,
              resources: str = "cpu,gpu") -> subprocess.Popen:
        p = subprocess.Popen(
            [sys.executable, "-m", "pipeline.cli", "--data", str(self.data_dir), "worker", "--id", wid,
             "--resources", resources, "--lease-ttl", str(lease_ttl), "--poll", "0.05",
             "--time-scale", str(time_scale)],
            cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(p)
        return p

    @staticmethod
    def kill9(p: subprocess.Popen) -> None:
        p.kill()          # Windows: TerminateProcess;POSIX: SIGKILL —— 不给任何清理机会
        p.wait(timeout=10)

    def cleanup(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.kill()
        for p in self.procs:
            try:
                p.wait(timeout=10)
            except Exception:
                pass


@pytest.fixture
def procs(data_dir: Path):
    pw = ProcWorkers(data_dir)
    yield pw
    pw.cleanup()
