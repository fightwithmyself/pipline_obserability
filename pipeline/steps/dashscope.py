"""阿里云百炼(DashScope)的通用工具:HTTP、异步任务"提交一次 + 逐轮轮询"、临时文件上传、下载。

被 kling.py(可灵视频生成)和 vst.py(视频风格重绘链路)共用。

异步任务的工程要点(都是"有钱的外部副作用"带来的):
  1. 提交成功后立刻把外部 task_id 写进检查点;步骤被回收/崩溃后重试,读到 task_id 就**继续轮询而不是重新提交**。
     残余风险:服务端已接受提交、但检查点还没写下来时进程崩溃 → 会重复提交一次(文档里没有幂等键)。
  2. 每一轮轮询都留一条观测记录(poll_001…),存在 attempt.notes,失败也保留。
  3. 外部返回"参数/权限/余额/未开通"这类错误码 → PERMANENT,不重试、不浪费预算。

密钥只从环境变量 DASHSCOPE_API_KEY 读取,不写入任何任务参数、产物、日志或数据库。
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

from ..models import ErrorKind
from ..stepkit import StepContext, StepError

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com"
SUBMIT_PATH = "/api/v1/services/aigc/video-generation/video-synthesis"
TASK_PATH = "/api/v1/tasks/"
UPLOAD_POLICY_PATH = "/api/v1/uploads"

# 这些错误码重试没有意义(参数/权限/余额/未开通),直接判永久失败。带后缀的(如 InvalidParameter.FileDownload)按前缀匹配
PERMANENT_CODES = {"InvalidParameter", "InvalidApiKey", "AccessDenied", "Arrearage", "DataInspectionFailed",
                   "UnsupportedOperation", "ModelNotFound", "InvalidInput"}


def base_url() -> str:
    return os.environ.get("DASHSCOPE_BASE_URL", os.environ.get("KLING_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")


def api_key() -> str:
    k = os.environ.get("DASHSCOPE_API_KEY")
    if not k:
        raise StepError(ErrorKind.PERMANENT, "缺少环境变量 DASHSCOPE_API_KEY(密钥不会写进任务参数)")
    return k


def http_json(method: str, path: str, body: dict | None = None, *, headers: dict | None = None,
              timeout: float = 30.0) -> dict:
    h = {"Authorization": "Bearer " + api_key()}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h.update({"Content-Type": "application/json", "X-DashScope-Async": "enable"})
    h.update(headers or {})
    req = urllib.request.Request(base_url() + path, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        text = e.read().decode("utf-8", "replace")[:300]
        kind = ErrorKind.PERMANENT if e.code in (400, 401, 403, 404) else ErrorKind.TRANSIENT   # 429/5xx 可重试
        raise StepError(kind, f"HTTP {e.code}: {text}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise StepError(ErrorKind.TRANSIENT, f"网络错误: {type(e).__name__}: {e}") from None


def submit_and_poll(ctx: StepContext, body: dict, *, poll_s: float, max_wait_s: float,
                    submit_headers: dict | None = None) -> dict:
    """提交一个百炼异步任务并轮询到终态。返回成功时的完整查询响应(含 output / usage)。"""
    ck = ctx.checkpoint or {}
    ext_id = ck.get("external_task_id")
    if ext_id:
        ctx.note("resumed_external_task", ext_id)
        ctx.note("submitted_this_attempt", False)
        ctx.progress(0.05, f"恢复:沿用已提交的外部任务 {ext_id[:8]}…,不重复提交")
    else:
        ctx.progress(0.02, "提交外部任务")
        resp = http_json("POST", SUBMIT_PATH, body, headers=submit_headers)
        ext_id = (resp.get("output") or {}).get("task_id")
        if not ext_id:
            raise StepError(ErrorKind.PERMANENT, f"提交响应里没有 task_id: {json.dumps(resp, ensure_ascii=False)[:300]}")
        ctx.save_checkpoint({"external_task_id": ext_id, "submitted_at": time.time(),
                             "request_id": resp.get("request_id")})
        ctx.note("submitted_this_attempt", True)
        ctx.note("external_task_id", ext_id)
        ctx.note("submit_request_id", resp.get("request_id"))
        ctx.progress(0.05, f"已提交 {ext_id[:8]}…(task_id 已写入检查点)")

    t0 = time.time()
    n = 0
    while True:
        n += 1
        d = http_json("GET", TASK_PATH + ext_id)
        o = d.get("output") or {}
        st = o.get("task_status", "UNKNOWN")
        elapsed = round(time.time() - t0, 1)
        # tokens 恒为 None:这类视频接口按秒计费,响应里没有 token 数,不能编造
        ctx.note(f"poll_{n:03d}", {"status": st, "elapsed_s": elapsed, "tokens": None, "usage": d.get("usage")})
        ctx.note("polls", n)
        ctx.progress(min(0.9, 0.05 + elapsed / max_wait_s), f"第 {n} 轮轮询: {st} ({elapsed}s)")
        if st == "SUCCEEDED":
            return d
        if st in ("FAILED", "CANCELED", "UNKNOWN"):
            code, msg = o.get("code", st), o.get("message", "")
            permanent = st == "UNKNOWN" or str(code).split(".")[0] in PERMANENT_CODES
            ctx.note("external_failure", {"status": st, "code": code, "message": msg})
            raise StepError(ErrorKind.PERMANENT if permanent else ErrorKind.TRANSIENT,
                            f"外部任务 {st}: {code}: {msg}")
        if elapsed > max_wait_s:
            raise StepError(ErrorKind.TIMEOUT, f"等待外部任务超过 {max_wait_s}s(task_id={ext_id},可继续查询)")
        ctx.sleep(poll_s)


def download(url: str, dest: Path, *, timeout: float = 120.0) -> tuple[str, int]:
    """流式下载到 dest,返回 (sha256, 字节数)。外部结果链接会过期,成功后必须立刻下载。"""
    h = hashlib.sha256()
    n = 0
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
            for blk in iter(lambda: r.read(1 << 20), b""):
                f.write(blk)
                h.update(blk)
                n += len(blk)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise StepError(ErrorKind.TRANSIENT, f"下载失败(外部任务已成功,重试时会直接取结果): {e}") from None
    return h.hexdigest(), n


def upload_temp_file(path: Path, model: str, *, timeout: float = 300.0) -> tuple[str, dict]:
    """把本地文件传到百炼免费临时存储(48 小时有效),返回 (oss:// URL, 细节)。

    使用 oss:// URL 时,提交任务必须带请求头 X-DashScope-OssResourceResolve: enable。
    """
    q = urllib.parse.urlencode({"action": "getPolicy", "model": model})
    pol = http_json("GET", f"{UPLOAD_POLICY_PATH}?{q}", headers={"Content-Type": "application/json"})["data"]
    size = path.stat().st_size
    limit = float(pol.get("max_file_size_mb") or 0)
    if limit and size > limit * 1048576:
        raise StepError(ErrorKind.PERMANENT, f"文件 {size / 1048576:.1f}MB 超过该模型允许的上传上限 {limit}MB")
    key = f"{pol['upload_dir']}/{uuid.uuid4().hex[:8]}_{path.name}"
    boundary = "----pipeline" + uuid.uuid4().hex
    fields = {"OSSAccessKeyId": pol["oss_access_key_id"], "Signature": pol["signature"], "policy": pol["policy"],
              "x-oss-object-acl": pol["x_oss_object_acl"], "x-oss-forbid-overwrite": pol["x_oss_forbid_overwrite"],
              "key": key, "success_action_status": "200"}
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in fields.items()]
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
                  f"Content-Type: application/octet-stream\r\n\r\n").encode())
    parts += [path.read_bytes(), f"\r\n--{boundary}--\r\n".encode()]
    req = urllib.request.Request(pol["upload_host"], data=b"".join(parts), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
    except urllib.error.HTTPError as e:
        raise StepError(ErrorKind.TRANSIENT, f"上传失败 HTTP {e.code}: {e.read()[:200]!r}") from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise StepError(ErrorKind.TRANSIENT, f"上传失败: {type(e).__name__}: {e}") from None
    return f"oss://{key}", {"bytes": size, "upload_s": round(time.time() - t0, 2),
                            "max_file_size_mb": limit, "valid_hours": 48, "host": pol.get("upload_host")}
