"""进程小工具:判断某个 pid 是否还活着、杀掉整棵进程树(Agent 运行进程 + 它拉起的 worker 与步骤子进程)。

不用 psutil(不想为此多一个依赖)。Windows 上**不能**用 os.kill(pid, 0) 探测:那会真的终止进程,所以走 ctypes。
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = wintypes.HANDLE
        h = k32.OpenProcess(0x1000, False, pid)           # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
            return bool(ok) and code.value == 259         # STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def kill_tree(pid: int) -> None:
    """强杀进程树。这是"硬中断":正在进行的模型请求会丢失,恢复靠引擎的租约回收与逐帧账本。"""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
