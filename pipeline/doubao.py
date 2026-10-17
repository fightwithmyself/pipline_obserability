"""豆包 Seedream(火山方舟)图生图客户端 + 离线 Mock。

为什么自己封装 HTTP,而不是用官方 SDK:
  这个接口是**同步**的,没有任务号。能不能"重发"取决于我们能否准确知道"请求是否已经发出",
  SDK 的自动重试会把这件事藏起来,让账单对不上。所以重试分类和账本都放在我们自己的代码里。

错误分类(决定调用方怎么处理):
  Transient / RateLimited  瞬时:退避后重试这一帧(RateLimited 带建议等待时间)
  Moderation               这一帧被内容审核拒绝:重试没用,按策略降级或失败
  FatalError               整个任务级错误(鉴权失败、欠费、模型不存在、请求参数非法):立刻停,别继续烧钱

**真实接口部分(ArkDoubao)尚未用真实密钥验证**:请求体和错误码依据官方文档编写,
上线前用 3~10 帧冒烟实测,尤其是审核错误码的具体取值。MockDoubao 才是测试的依据。
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter


# --------------------------------------------------------------------------- 错误
class DoubaoError(Exception):
    pass


class Transient(DoubaoError):
    """可重试。request_maybe_sent=True 表示请求可能已到达服务端(超时、断连):此时可能已被计费。"""

    def __init__(self, msg: str, request_maybe_sent: bool = False):
        super().__init__(msg)
        self.request_maybe_sent = request_maybe_sent


class RateLimited(Transient):
    def __init__(self, msg: str, retry_after: float = 2.0):
        super().__init__(msg, request_maybe_sent=False)
        self.retry_after = retry_after


class Moderation(DoubaoError):
    pass


class FatalError(DoubaoError):
    def __init__(self, msg: str, kind: str = "fatal"):
        super().__init__(msg)
        self.kind = kind          # auth / quota / bad_request / not_found


@dataclass
class EditResult:
    image: bytes                  # PNG
    usage: dict = field(default_factory=dict)     # 接口返回的用量原样保留(generated_images / output_tokens ...)
    request_id: str | None = None
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- Mock
def mock_stylize(png: bytes, prompt: str) -> bytes:
    """确定性的"伪风格化":色阶分层 + 提取轮廓线 + 由 prompt 决定的色偏。

    只用于测试和演示流水线;**它不代表真实模型的效果**。同样的 (图, prompt) 永远得到同样的字节。
    """
    im = Image.open(io.BytesIO(png)).convert("RGB")
    a = np.asarray(im).astype(np.float32)
    h = hashlib.sha256(prompt.encode("utf-8")).digest()
    tint = np.array([(h[0] / 255 - 0.5) * 36, (h[1] / 255 - 0.5) * 36, (h[2] / 255 - 0.5) * 36], dtype=np.float32)
    levels = 5.0
    q = (np.floor(a / 256.0 * levels) + 0.5) / levels * 255.0 + tint
    edges = np.asarray(im.convert("L").filter(ImageFilter.FIND_EDGES)).astype(np.float32)
    q[edges > 40] = 24.0
    out = Image.fromarray(np.clip(q, 0, 255).astype(np.uint8), "RGB")
    buf = io.BytesIO()
    out.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


class MockDoubao:
    """离线替身,并且可以注入故障。

    故障规则来自步骤的 chaos["doubao"](不参与缓存 key):
      {"frames": {"<关键帧序号>": {"kind": K, "until_attempt": A, "first_calls": N}},
       "latency_s": 0.05, "price_cny": 0.12}
    kind:
      transient              返回 5xx 类瞬时错误(没有计费)
      rate_limit             返回 429
      moderation             内容审核拒绝(没有计费)
      timeout_after_success  服务端已生成并计费,但响应丢了(客户端只看到超时)——"多付一次"的来源
      garbage                返回无法解码的内容
      auth / quota           任务级致命错误
    until_attempt: 只在引擎的第 1..A 次尝试里触发;first_calls: 只对该帧在本进程内的前 N 次调用触发。
    不写这两个字段则每次都触发。

    服务端计费账本写在 data_dir/mock_doubao_billing/(每次计费一个文件),测试用它断言"实际被计费了几次"。
    """

    name = "mock"

    def __init__(self, data_dir: Path, model: str, chaos: dict | None, attempt: int,
                 sleep: Callable[[float], None] = time.sleep):
        self.data_dir = Path(data_dir)
        self.model = model or "mock-seedream"
        self.chaos = chaos or {}
        self.attempt = attempt
        self.sleep = sleep
        self._calls: dict[str, int] = {}

    def _bill(self, frame_key: str, prompt: str) -> None:
        # 一条计费记录 = 一个独立文件。不用"追加写同一个文件":Windows 上 O_APPEND 不是原子的,
        # 多线程/多进程并发追加会丢行(实测),账本就不可信了。
        d = self.data_dir / "mock_doubao_billing"
        d.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "frame": frame_key, "model": self.model,
               "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest()[:12]}
        (d / f"{time.time_ns()}-{os.getpid()}-{threading.get_ident()}-{frame_key}.json").write_text(
            json.dumps(rec), encoding="utf-8")

    def _rule(self, frame_key: str) -> dict | None:
        r = (self.chaos.get("frames") or {}).get(str(int(frame_key)))
        if not r:
            return None
        n = self._calls.get(frame_key, 0)
        if "until_attempt" in r and self.attempt > r["until_attempt"]:
            return None
        if "first_calls" in r and n > r["first_calls"]:
            return None
        return r

    def edit(self, *, image: bytes, prompt: str, reference: bytes | None, frame_key: str) -> EditResult:
        self._calls[frame_key] = self._calls.get(frame_key, 0) + 1
        self.sleep(float(self.chaos.get("latency_s", 0.0)))
        r = self._rule(frame_key)
        kind = r["kind"] if r else None
        if kind == "transient":
            raise Transient("HTTP 503 (mock)")
        if kind == "rate_limit":
            raise RateLimited("HTTP 429 (mock)", retry_after=float(r.get("retry_after", 0.2)))
        if kind == "moderation":
            raise Moderation("OutputImageSensitiveContentDetected (mock)")
        if kind == "auth":
            raise FatalError("HTTP 401 (mock)", "auth")
        if kind == "quota":
            raise FatalError("AccountOverdueError (mock)", "quota")
        out = mock_stylize(image, prompt)
        self._bill(frame_key, prompt)       # 服务端到这里就已经计费了
        if kind == "timeout_after_success":
            raise Transient("read timeout (mock: 服务端已生成,响应丢失)", request_maybe_sent=True)
        if kind == "garbage":
            return EditResult(image=b"not an image", usage={"generated_images": 1, "output_tokens": 0})
        w, h = Image.open(io.BytesIO(out)).size
        # 模拟 output_tokens:按输出像素估算,只是让用量字段有值,不是真实 token 数
        return EditResult(image=out, usage={"generated_images": 1, "output_tokens": int(w * h / 256)},
                          request_id=f"mock-{frame_key}-{self._calls[frame_key]}")


# --------------------------------------------------------------------------- 真实接口
MODERATION_MARKERS = ("sensitive", "moderation", "contentfilter", "content_filter", "violat", "risk")
QUOTA_MARKERS = ("quota", "balance", "overdue", "arrears", "insufficient", "exceeded_limit")


def classify_http_error(status: int, code: str, message: str, retry_after: float | None = None) -> DoubaoError:
    """把 HTTP 状态码 + 错误码映射成我们的错误类型。纯函数,单独测试。

    审核/欠费的错误码是按文档与命名习惯用子串匹配的,**待实测校准**:匹配不上的 400 会被当作
    FatalError(bad_request)——宁可整个任务停下来让人看,也不对一个未知错误无限重试烧钱。
    """
    c = (code or "").lower()
    text = f"{status} {code} {message}"[:300]
    if status == 429 and any(m in c for m in QUOTA_MARKERS):
        # 实测(Agent Plan + seedream pro):配额用尽也是 HTTP 429,code=AccountQuotaExceeded。它和"瞬时限流"
        # (AccountRateLimitExceeded)不同:等几秒重试没有用,按致命的 quota 处理,让人去看配额/重置时间。
        return FatalError(text, "quota")
    if status == 429:
        return RateLimited(text, retry_after=retry_after or 2.0)
    if status in (401, 403):
        if any(m in c for m in QUOTA_MARKERS):
            return FatalError(text, "quota")
        return FatalError(text, "auth")
    if status == 404:
        # 实测:ModelNotOpen = 账号没在方舟控制台开通该模型;InvalidEndpointOrModel.NotFound = 模型不存在/已下线/无权限
        return FatalError(text, "model_not_open" if "modelnotopen" in c else "not_found")
    if status == 402 or any(m in c for m in QUOTA_MARKERS):
        return FatalError(text, "quota")
    if any(m in c for m in MODERATION_MARKERS):
        return Moderation(text)
    if status == 408 or status >= 500:
        return Transient(text, request_maybe_sent=status != 503)
    if 400 <= status < 500:
        return FatalError(text, "bad_request")
    return Transient(text)


def _to_jpeg_bytes(png: bytes, max_side: int = 4096) -> bytes:
    im = Image.open(io.BytesIO(png)).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _data_uri(b: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(b).decode()


class ArkDoubao:
    """火山方舟 Seedream 图生图:POST {base}/api/v3/images/generations(同步)。

    密钥只从环境变量 ARK_API_KEY 读取,不进任务参数、不进数据库、不进日志。
    返回用 b64_json 而不是 url:省掉"链接 24 小时失效、下载失败"这一整类故障。
    """

    name = "ark"

    def __init__(self, model: str, *, size: str = "2K", watermark: bool = False, timeout_s: float = 180.0,
                 base_url: str | None = None, api_key: str | None = None, api_prefix: str = "/api/v3"):
        self.api_prefix = api_prefix
        if not model:
            raise FatalError("缺少模型 ID(方舟控制台里的 Seedream 型号或 Endpoint ID)", "bad_request")
        self.model = model
        self.size = size
        self.watermark = watermark
        self.timeout_s = timeout_s
        self.base_url = (base_url or os.environ.get("ARK_BASE_URL") or "https://ark.cn-beijing.volces.com").rstrip("/")
        self._key = api_key

    def _api_key(self) -> str:
        k = self._key or os.environ.get("ARK_API_KEY")
        if not k:
            raise FatalError("没有设置环境变量 ARK_API_KEY", "auth")
        return k

    def build_body(self, image: bytes, prompt: str, reference: bytes | None) -> dict:
        imgs = [_data_uri(_to_jpeg_bytes(image))]
        if reference is not None:
            imgs.append(_data_uri(_to_jpeg_bytes(reference)))
        return {"model": self.model, "prompt": prompt, "image": imgs if len(imgs) > 1 else imgs[0],
                "size": self.size, "response_format": "b64_json", "watermark": self.watermark}
        # 不传 sequential_image_generation:默认就是 disabled,而 5.0-flash 实测会因为带了这个参数返回 400

    def edit(self, *, image: bytes, prompt: str, reference: bytes | None, frame_key: str) -> EditResult:
        import httpx
        body = self.build_body(image, prompt, reference)
        headers = {"Authorization": f"Bearer {self._api_key()}", "Content-Type": "application/json"}
        try:
            r = httpx.post(f"{self.base_url}{self.api_prefix}/images/generations", json=body, headers=headers,
                           timeout=self.timeout_s)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:       # 连接阶段失败 = 请求没有到达服务端
            raise Transient(f"connect error: {e}", request_maybe_sent=False)
        except httpx.HTTPError as e:
            raise Transient(f"{type(e).__name__}: {e}", request_maybe_sent=True)
        try:
            payload = r.json()
        except ValueError:
            payload = {}
        if r.status_code != 200:
            err = payload.get("error") or {}
            ra = r.headers.get("retry-after")
            raise classify_http_error(r.status_code, str(err.get("code", "")), str(err.get("message", r.text[:200])),
                                      float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
        data = (payload.get("data") or [{}])[0]
        if data.get("error"):
            e = data["error"]
            raise classify_http_error(400, str(e.get("code", "")), str(e.get("message", "")))
        b64 = data.get("b64_json")
        if not b64:
            raise Transient("响应里没有 b64_json", request_maybe_sent=True)
        try:
            im = Image.open(io.BytesIO(base64.b64decode(b64)))
            im.load()
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="PNG", compress_level=3)
        except Exception as e:
            raise Transient(f"返回的图片无法解码: {e}", request_maybe_sent=True)
        return EditResult(image=buf.getvalue(), usage=payload.get("usage") or {},
                          request_id=r.headers.get("x-request-id"), meta={"size": data.get("size")})


def make_client(provider: str, *, model: str | None, data_dir: Path, chaos: dict | None, attempt: int,
                sleep: Callable[[float], None], size: str = "2K", watermark: bool = False):
    if provider == "mock":
        return MockDoubao(data_dir, model or "mock-seedream", chaos, attempt, sleep)
    if provider == "ark":
        return ArkDoubao(model or "", size=size, watermark=watermark)
    if provider == "ark_plan":      # Agent Plan:专属路径 /api/plan/v3 + 专属密钥(同样放在 ARK_API_KEY)
        return ArkDoubao(model or "", size=size, watermark=watermark, api_prefix="/api/plan/v3")
    raise FatalError(f"未知 provider: {provider}", "bad_request")
