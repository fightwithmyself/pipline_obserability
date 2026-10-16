"""步骤子进程入口:  python -m pipeline.step_runner <job.json>

职责边界:
- 子进程**不访问数据库**,只通过 stdout 向父进程(worker)发 JSON 行协议消息:
    {"t":"progress","p":0.5,"msg":"..."}   进展(父进程据此续租、判断是否卡死)
    {"t":"checkpoint","data":{...}}        步骤内检查点(父进程落库)
    {"t":"note","k":"...","v":...}         诊断信息(记入 attempts.notes)
    {"t":"done","outputs":{name: relpath}} 成功
    {"t":"error","kind":"...","msg":"..."} 失败(带分类)
- 父进程被杀 → 本进程 stdin 收到 EOF → 立即自杀,避免孤儿进程继续占用资源。
- 步骤代码里的 print 会被重定向到 stderr,不污染协议。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from .models import ErrorKind
from .stepkit import StepContext, StepError


def _watch_parent_win() -> None:
    """Windows:轮询管道状态,而不是阻塞读。

    阻塞在 os.read(0) 的线程会长期占着 C 运行时对 fd 0 的锁;步骤里 import numpy 这类库时,
    初始化代码一旦碰到 stdin 就会与它死锁(实测:import numpy 永远不返回,步骤被判 STALL)。
    PeekNamedPipe 直接问操作系统句柄:父进程退出/关闭管道时返回失败,不占 CRT 的锁。
    """
    import ctypes
    import msvcrt
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.PeekNamedPipe.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                                  ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
                                  ctypes.POINTER(wintypes.DWORD)]
    k32.PeekNamedPipe.restype = wintypes.BOOL
    handle = msvcrt.get_osfhandle(0)
    avail = wintypes.DWORD(0)
    try:
        while k32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None):
            time.sleep(0.2)
    except Exception:
        pass
    os._exit(1)      # 管道断开 = 父进程已死/关闭


def _watch_parent() -> None:
    if sys.platform == "win32":
        return _watch_parent_win()
    try:
        while os.read(0, 1024):     # 直接读 fd,不经过带锁的 BufferedReader(否则解释器退出时会死锁)
            pass
    except Exception:
        pass
    os._exit(1)      # 父进程已死/关闭管道


def main(argv: list[str]) -> int:
    job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    real_out = sys.stdout
    sys.stdout = sys.stderr          # 步骤里的 print 不能污染协议通道
    threading.Thread(target=_watch_parent, daemon=True).start()

    from .steps import REGISTRY
    from .steps import animation  # noqa: F401  注册 MOCK 步骤(含通用的 publish)
    if str(job["type"]).startswith("fp_"):
        # 序列帧的真实步骤依赖 numpy/Pillow,导入有成本:只在需要时加载,不拖慢其他步骤的子进程启动
        from .steps import frames  # noqa: F401

    ctx = StepContext(job, real_out)
    try:
        fn = REGISTRY.get(job["type"])
        if fn is None:
            raise StepError(ErrorKind.PERMANENT, f"未知步骤类型: {job['type']}")
        fn(ctx)
        ctx._send({"t": "done", "outputs": ctx.outputs})
        return 0
    except StepError as e:
        ctx._send({"t": "error", "kind": e.kind, "msg": e.msg})
        return 2
    except MemoryError:
        ctx._send({"t": "error", "kind": ErrorKind.RESOURCE, "msg": "MemoryError"})
        return 2
    except Exception as e:  # 未分类异常按 TRANSIENT 处理(有重试预算兜底),并保留堆栈
        ctx._send({"t": "error", "kind": ErrorKind.TRANSIENT,
                   "msg": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"})
        return 2


if __name__ == "__main__":
    code = main(sys.argv)
    try:
        sys.__stdout__.flush()
        sys.stderr.flush()
    finally:
        os._exit(code)     # 不走正常解释器关闭流程,避免守护线程阻塞退出
