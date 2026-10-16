"""步骤开发工具包:StepError / StepContext。

独立成模块是为了避免 step_runner 以 __main__ 运行时,步骤模块再 import 它而得到
另一份同名类(导致 except StepError 捕获不到)。
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path

from .models import ErrorKind


class StepError(Exception):
    """步骤主动抛出的带分类的错误。"""

    def __init__(self, kind: str, msg: str):
        super().__init__(msg)
        self.kind = kind
        self.msg = msg


class StepContext:
    def __init__(self, job: dict, out):
        self._out = out
        self._lock = threading.Lock()
        self.task_id: str = job["task_id"]
        self.key: str = job["key"]
        self.type: str = job["type"]
        self.params: dict = job["params"]
        self.chaos: dict = job.get("chaos") or {}
        self.attempt: int = job["attempt"]
        self.checkpoint: dict | None = job.get("checkpoint")
        self.inputs: dict = job["inputs"]
        self.work_dir = Path(job["work_dir"])      # 跨 attempt 保留,用于步骤内检查点
        self.out_dir = Path(job["out_dir"])        # 本次 attempt 私有,成功后由父进程入库
        self.data_dir = Path(job["data_dir"])
        self.time_scale: float = job.get("time_scale", 1.0)
        self.outputs: dict[str, str] = {}

    # ---- 协议
    def _send(self, msg: dict) -> None:
        line = json.dumps(msg, ensure_ascii=False)
        with self._lock:
            self._out.write(line + "\n")
            self._out.flush()

    def progress(self, p: float, msg: str = "") -> None:
        self._apply_chaos(p)
        self._send({"t": "progress", "p": round(min(max(p, 0.0), 1.0), 4), "msg": msg})

    def save_checkpoint(self, data: dict) -> None:
        self._send({"t": "checkpoint", "data": data})

    def note(self, k: str, v) -> None:
        self._send({"t": "note", "k": k, "v": v})

    # ---- 辅助
    def sleep(self, seconds: float) -> None:
        time.sleep(max(0.0, seconds * self.time_scale))

    def simulate(self, total_s: float, ticks: int = 4, msg: str = "") -> None:
        """模拟耗时计算:分 ticks 段 sleep,每段上报一次进度。"""
        self.progress(0.0, msg)
        for i in range(1, ticks + 1):
            self.sleep(total_s / ticks)
            self.progress(i / ticks, msg)

    def input_sha(self, dep: str, name: str) -> str:
        return self.inputs[dep][name]["sha256"]

    def input_path(self, dep: str, name: str) -> Path:
        return Path(self.inputs[dep][name]["path"])

    def input_digest(self) -> str:
        """所有输入产物 hash 的摘要,用于生成与输入相关的确定性输出。"""
        h = hashlib.sha256()
        for dep in sorted(self.inputs):
            for n in sorted(self.inputs[dep]):
                h.update(f"{dep}/{n}:{self.inputs[dep][n]['sha256']};".encode())
        return h.hexdigest()

    def output(self, name: str, data: bytes) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        p = self.out_dir / name
        p.write_bytes(data)
        self.outputs[name] = name
        return p

    def output_path(self, name: str) -> Path:
        """登记一个产物并返回应写入的路径(适合直接写入大文件,避免先读进内存)。"""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.outputs[name] = name
        return self.out_dir / name

    # ---- 故障注入(MOCK 用,不参与缓存 key)
    def _apply_chaos(self, p: float) -> None:
        c = self.chaos
        if not c:
            return
        fail = c.get("fail")
        if fail and self.attempt <= fail.get("until_attempt", 1) and p >= fail.get("at", 0.0):
            raise StepError(fail.get("kind", ErrorKind.TRANSIENT), fail.get("msg", "injected failure"))
        crash = c.get("crash")
        if crash and self.attempt in crash.get("on_attempts", [1]) and p >= crash.get("at", 0.5):
            os._exit(137)    # 模拟被 OOM-kill / kill -9:不留遗言
        hang = c.get("hang")
        if hang and self.attempt in hang.get("on_attempts", [1]) and p >= hang.get("at", 0.5):
            while True:      # 卡死:不再上报进展
                time.sleep(1)
