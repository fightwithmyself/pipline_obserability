"""动画素材批量风格化流水线的步骤(真实计算 + 豆包逐帧改图):

    an_ingest ─► an_keyframes ─► an_stylize_i ×N ─► an_poststats ─► an_post_i ×N ─► an_qc ─► an_mux ─► an_package
    1 素材校验与标准化  2 关键帧提取与拆分   3 豆包逐帧风格化(按块并行)   4 后处理与补间              质量校验   5 合成与音轨   6 打包与元数据

设计要点(详见 docs/DESIGN.md):
- 工序 3 是唯一花钱、也是唯一"主恢复目标"的步骤。它把每个关键帧当作一个独立的工作项:
    · 逐帧账本(work_dir/ledger/<帧>.json)才是恢复所依赖的真相:先落"在途"标记再发请求;
    · 逐帧结果同时写入跨任务的帧级内容缓存(data/framecache),key 含 源帧哈希/提示词/模型/参考图;
    · 数据库里的 items 表只是派生的观测视图(经父进程写入),不参与恢复决策。
- 帧在步骤之间以确定性 zip 传递(内容寻址产物),所以任何步骤都能在任何 worker 上重跑。
- 工序 4 按块把"风格化关键帧 + 夹在中间的补间帧"直接编码成视频片段,不落 1440 张 PNG。
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageFilter

from ..doubao import (EditResult, FatalError, Moderation, RateLimited, Transient, make_client)
from ..interp import blend, interpolate
from ..media import MediaError, count_decoded_frames, ffmpeg_exe, probe_video, run_ffmpeg
from ..models import ErrorKind, canonical_json
from ..stepkit import StepContext, StepError
from . import register

ANALYSIS_W = 96            # 关键帧分析用的灰度缩略图宽度
CKPT_EVERY = 4             # 工序 3 每完成几帧写一次检查点(同时是续租心跳)
FRAME_CACHE_VERSION = "1"  # 帧级缓存的逻辑版本;改变风格化逻辑时递增以使旧缓存失效


# --------------------------------------------------------------------------- 工具
def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}.tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def pack_zip(dest: Path, items: list[tuple[str, bytes]]) -> None:
    """确定性 zip:同样的内容 → 同样的字节(固定顺序、固定时间戳、不压缩)。"""
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as z:
        for name, data in items:
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)


def _open_zip(ctx: StepContext, dep: str, name: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(ctx.input_path(dep, name))


def _frame_name(idx: int) -> str:
    return f"{idx:06d}.png"


def _read_raw_frames(path: Path, vf: str, frame_bytes: int, pix_fmt: str) -> Iterator[bytes]:
    """逐帧读 ffmpeg 解码出的原始像素(经管道,不落盘)。解码错误计入 stderr 临时文件,结束时一并检查退出码。"""
    err = tempfile.TemporaryFile()
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path), "-an",
           "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix_fmt, "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL)
    assert p.stdout is not None
    exhausted = False
    try:
        while True:
            buf = p.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                exhausted = True
                break
            yield buf
    finally:
        p.stdout.close()
        if not exhausted:               # 调用方提前停止读取:直接结束 ffmpeg,不把"管道被关"当成错误
            p.kill()
        rc = p.wait()
        err.seek(0)
        tail = err.read()[-300:].decode("utf-8", "replace")
        err.close()
        if exhausted and rc != 0:
            raise MediaError(f"ffmpeg 解码失败(退出码 {rc}): {tail}")


def _even(x: float) -> int:
    return max(2, int(round(x / 2.0)) * 2)


# --------------------------------------------------------------------------- 工序 1:素材校验与标准化
@register("an_ingest")
def an_ingest(ctx: StepContext) -> None:
    """校验素材可读,统一成 恒定帧率 + 目标高度 + MP4(H.264),并分离音轨。坏素材在这里永久失败。"""
    p = ctx.params
    src = Path(str(p.get("source", "")))
    ctx.progress(0.0, "checking source")
    if not src.is_file():
        raise StepError(ErrorKind.PERMANENT, f"素材文件不存在: {src}")
    expect = p.get("source_sha256")
    if expect and _sha_file(src) != expect:
        raise StepError(ErrorKind.PERMANENT, "素材文件在提交后被修改(sha256 与提交时不一致),请重新提交")
    try:
        info = probe_video(src)
    except MediaError as e:
        raise StepError(ErrorKind.PERMANENT, f"素材无法读取: {e}")
    fps = int(p.get("fps", 24))
    target_h = int(p.get("target_height", 1080))
    problems = []
    if not (float(p.get("min_duration_s", 0.5)) <= info["duration_s"] <= float(p.get("max_duration_s", 180))):
        problems.append(f"时长 {info['duration_s']}s 不在 {p.get('min_duration_s', 0.5)}-{p.get('max_duration_s', 180)}s 内")
    if min(info["width"], info["height"]) < 64:
        problems.append(f"分辨率 {info['width']}x{info['height']} 太小(短边需 >= 64)")
    if problems:
        raise StepError(ErrorKind.PERMANENT, "素材不满足要求: " + "; ".join(problems))

    H = _even(target_h)
    W = _even(H * info["width"] / info["height"])
    expected = int(round(info["duration_s"] * fps))
    errs: list[str] = []
    ctx.progress(0.05, "transcoding")
    try:
        run_ffmpeg(["-i", str(src), "-an",
                    "-vf", f"fps={fps},scale={W}:{H}:flags=lanczos,setsar=1",
                    "-c:v", "libx264", "-crf", "14", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(ctx.output_path("standard.mp4"))],
                   lambda f: ctx.progress(0.05 + 0.7 * min(f / max(expected, 1), 1.0), f"transcoded {f}"),
                   collect=errs)
        if info["has_audio"]:
            ctx.progress(0.8, "audio")
            run_ffmpeg(["-i", str(src), "-vn", "-c:a", "aac", "-b:a", "128k", str(ctx.output_path("audio.m4a"))])
    except MediaError as e:
        raise StepError(ErrorKind.TRANSIENT, str(e))
    ctx.progress(0.9, "counting frames")
    try:
        decoded = count_decoded_frames(ctx.out_dir / "standard.mp4")
    except MediaError as e:
        raise StepError(ErrorKind.PERMANENT, f"标准化后的素材无法解码: {e}")
    broken = len(errs)
    dev = abs(decoded - expected) / max(expected, 1)
    warnings = []
    if broken:
        warnings.append(f"解码过程中 ffmpeg 报告了 {broken} 条错误/警告(坏帧已被跳过或遮盖)")
    if not info["has_audio"]:
        warnings.append("素材没有音轨:成品将使用静音轨")
    if dev > float(p.get("max_frame_deviation", 0.02)):
        # 容器时长可能包含比视频更长的音轨,所以帧数与时长推算不符只告警,不当作损坏
        warnings.append(f"解出 {decoded} 帧,按容器时长推算约 {expected} 帧(偏差 {dev:.1%})")
    if decoded < 2 or broken > max(1, decoded * 0.02):
        raise StepError(ErrorKind.PERMANENT,
                        f"素材疑似损坏:解出 {decoded} 帧,ffmpeg 报告 {broken} 条解码错误(超过 2%)。请换素材")
    report = {"source": {**info, "name": src.name, "sha256": _sha_file(src)},
              "standard": {"width": W, "height": H, "fps": fps, "frames": decoded,
                           "duration_s": round(decoded / fps, 3)},
              "expected_frames": expected, "decoded_frames": decoded, "decode_error_lines": broken,
              "errors_sample": errs[:5], "has_audio": bool(info["has_audio"]), "warnings": warnings}
    ctx.note("report", report)
    ctx.output("ingest.json", canonical_json(report).encode())
    ctx.progress(1.0, "ingest ok")


# --------------------------------------------------------------------------- 工序 2:关键帧提取与拆分
def select_keyframes(gray: np.ndarray, *, stride: int, max_gap: int, dup_thr: float,
                     cut_thr: float) -> dict:
    """由灰度缩略图序列选关键帧(纯函数,单独测试)。

    规则(都是可解释、可调的):
      1. 首帧、末帧一定是关键帧(补间需要两端);
      2. 镜头切换(相邻帧差异 >= cut_thr):切换点及其前一帧都是关键帧,这样补间永远不会跨镜头混合;
      3. 否则每隔 stride 帧是候选;候选与上一个关键帧的差异 < dup_thr(几乎没变,如定格/重复的过渡帧)就跳过;
      4. 距上一个关键帧已达 max_gap 则强制取,避免长时间静止后的快速动作被漏掉。
    """
    n = len(gray)
    g = gray.astype(np.int16)
    d = np.zeros(n, dtype=np.float32)
    if n > 1:
        d[1:] = np.abs(g[1:] - g[:-1]).mean(axis=(1, 2))
    cuts = [int(i) for i in range(1, n) if d[i] >= cut_thr]
    forced = {0, n - 1}
    for c in cuts:
        forced.update((c, c - 1))
    sel = [0]
    last = 0
    dup = 0
    for i in range(1, n):
        if i in forced:
            sel.append(i)
            last = i
            continue
        gap = i - last
        if gap >= max_gap:
            sel.append(i)
            last = i
        elif gap >= stride:
            if float(np.abs(g[i] - g[last]).mean()) >= dup_thr:
                sel.append(i)
                last = i
            else:
                dup += 1
    return {"keyframes": sel, "diff": d, "cuts": cuts, "duplicates_skipped": dup,
            "lum": g.mean(axis=(1, 2))}


@register("an_keyframes")
def an_keyframes(ctx: StepContext) -> None:
    p = ctx.params
    rep = _load_json(ctx.input_path("ingest", "ingest.json"))
    std = rep["standard"]
    W, H, fps, n = std["width"], std["height"], std["fps"], std["frames"]
    video = ctx.input_path("ingest", "standard.mp4")
    chunks = int(p["chunks"])
    stride, max_gap = int(p.get("stride", 6)), int(p.get("max_gap", 24))
    dup_thr, cut_thr = float(p.get("dup_thr", 0.8)), float(p.get("cut_thr", 18.0))

    # 第一遍:低分辨率灰度分析(整段视频,只在内存里留小图)
    lw = ANALYSIS_W
    lh = _even(lw * H / W)
    ctx.progress(0.0, "analysing")
    gray = np.empty((n, lh, lw), dtype=np.uint8)
    got = 0
    for buf in _read_raw_frames(video, f"scale={lw}:{lh}:flags=area,format=gray", lw * lh, "gray"):
        if got >= n:
            break
        gray[got] = np.frombuffer(buf, dtype=np.uint8).reshape(lh, lw)
        got += 1
        if got % 50 == 0:
            ctx.progress(0.4 * got / n, f"analysed {got}/{n}")
    if got != n:
        raise StepError(ErrorKind.PERMANENT, f"分析时只读到 {got} 帧,期望 {n}")
    sel = select_keyframes(gray, stride=stride, max_gap=max_gap, dup_thr=dup_thr, cut_thr=cut_thr)
    keys: list[int] = sel["keyframes"]
    nk = len(keys)
    price = float(p.get("price_cny", 0.0))
    budget = float(p.get("budget_cny", 0.0))
    est = round(nk * price, 2)
    ctx.note("estimate", {"keyframes": nk, "frames": n, "price_cny": price, "cost_cny_est": est, "budget_cny": budget})
    if nk < chunks * 2:
        raise StepError(ErrorKind.PERMANENT,
                        f"只选出 {nk} 个关键帧,不足以分成 {chunks} 块(每块至少 2 个)。请减少 chunks 或减小 stride")
    if budget and est > budget:
        raise StepError(ErrorKind.PERMANENT,
                        f"预估费用 {est} 元(关键帧 {nk} × {price} 元/张)超过预算 {budget} 元;"
                        f"请增大 stride(当前 {stride})或提高预算")

    # 第二遍:全分辨率解码,只保存被选中的帧
    ctx.progress(0.4, f"extracting {nk} keyframes")
    keyset = set(keys)
    tmp = ctx.work_dir / "kf"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    saved = 0
    for idx, buf in enumerate(_read_raw_frames(video, "format=rgb24", W * H * 3, "rgb24")):
        if idx in keyset:
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(H, W, 3)
            (tmp / _frame_name(idx)).write_bytes(_png_bytes(Image.fromarray(arr, "RGB")))
            saved += 1
            ctx.progress(0.4 + 0.5 * saved / nk, f"saved {saved}/{nk}")
    if saved != nk:
        raise StepError(ErrorKind.PERMANENT, f"提取到 {saved} 个关键帧,期望 {nk}")
    bounds = [i * nk // chunks for i in range(chunks + 1)]
    chunk_meta = []
    for i in range(chunks):
        s, e = bounds[i], bounds[i + 1]
        pack_zip(ctx.output_path(f"kf_{i}.zip"),
                 [(_frame_name(k), (tmp / _frame_name(k)).read_bytes()) for k in keys[s:e]])
        chunk_meta.append({"i": i, "start": s, "end": e})
    shutil.rmtree(tmp, ignore_errors=True)
    timeline = {
        "n_frames": n, "fps": fps, "width": W, "height": H, "keyframes": keys, "n_keyframes": nk,
        "time_s": [round(k / fps, 3) for k in keys], "chunks": chunk_meta,
        "cuts": sel["cuts"], "duplicates_skipped": sel["duplicates_skipped"],
        "params": {"stride": stride, "max_gap": max_gap, "dup_thr": dup_thr, "cut_thr": cut_thr},
        "diff": [round(float(x), 2) for x in sel["diff"]], "lum": [round(float(x), 2) for x in sel["lum"]],
        "estimate": {"price_cny": price, "cost_cny_est": est},
    }
    ctx.output("timeline.json", canonical_json(timeline).encode())
    ctx.progress(1.0, "keyframes done")


# --------------------------------------------------------------------------- 工序 3:豆包逐帧风格化
@register("an_reference")
def an_reference(ctx: StepContext) -> None:
    """锚点/人物参考图:提交时记录 sha256,这里校验后规范成 PNG。"""
    src = Path(str(ctx.params.get("path", "")))
    if not src.is_file():
        raise StepError(ErrorKind.PERMANENT, f"参考图不存在: {src}")
    expect = ctx.params.get("sha256")
    if expect and _sha_file(src) != expect:
        raise StepError(ErrorKind.PERMANENT, "参考图在提交后被修改,请重新提交")
    try:
        im = Image.open(src).convert("RGB")
    except Exception as e:
        raise StepError(ErrorKind.PERMANENT, f"参考图无法读取: {e}")
    if max(im.size) > 2048:
        s = 2048 / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    ctx.output("ref.png", _png_bytes(im))


def frame_cache_key(*, provider: str, model: str, prompt: str, size: str, watermark: bool,
                    ref_sha: str | None, src_sha: str) -> str:
    return _sha_bytes(canonical_json({
        "v": FRAME_CACHE_VERSION, "provider": provider, "model": model, "prompt": prompt, "size": size,
        "watermark": watermark, "ref": ref_sha, "src": src_sha}).encode())


class FrameCache:
    """跨任务、跨 worker 的帧级内容缓存(文件系统,不经数据库)。读时校验 sha256,坏了当作没命中。"""

    def __init__(self, root: Path):
        self.root = root

    def _p(self, key: str) -> tuple[Path, Path]:
        d = self.root / key[:2]
        return d / f"{key}.png", d / f"{key}.json"

    def get(self, key: str) -> tuple[bytes, dict] | None:
        png, meta = self._p(key)
        try:
            data = png.read_bytes()
            m = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if _sha_bytes(data) != m.get("sha256"):
            return None
        return data, m

    def put(self, key: str, data: bytes, meta: dict) -> None:
        png, mp = self._p(key)
        _atomic_write(png, data)
        _atomic_write(mp, json.dumps({**meta, "sha256": _sha_bytes(data)}, ensure_ascii=False).encode())


def fit_size(png: bytes, size: tuple[int, int]) -> tuple[bytes, list[int] | None]:
    """把返回图规整到标准尺寸。比例相差 <3% 直接缩放;否则等比缩放后居中补黑边。返回 (PNG, 调整前尺寸或 None)。"""
    im = Image.open(io.BytesIO(png)).convert("RGB")
    if im.size == size:
        return _png_bytes(im), None
    before = list(im.size)
    W, H = size
    if abs(im.width / im.height - W / H) / (W / H) < 0.03:
        return _png_bytes(im.resize(size, Image.LANCZOS)), before
    s = min(W / im.width, H / im.height)
    r = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    canvas = Image.new("RGB", size, (0, 0, 0))
    canvas.paste(r, ((W - r.width) // 2, (H - r.height) // 2))
    return _png_bytes(canvas), before


class RateLimiter:
    """进程内限速 + 全局暂停(收到 429 时所有线程一起等)。

    只在单个步骤进程内生效:多个块同时运行时总速率是它们的和。默认配额(500 张/分钟)远大于
    本流水线的并发量,所以这里没有做跨进程的令牌桶——这是已知的简化。
    """

    def __init__(self, rpm: float, scale: float):
        self.interval = 60.0 / max(rpm, 1.0) * scale
        self.scale = scale
        self._lock = threading.Lock()
        self._next = 0.0
        self._pause_until = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            t = max(self._next, self._pause_until, now)
            self._next = t + self.interval
        d = t - time.monotonic()
        if d > 0:
            time.sleep(d)

    def pause(self, seconds: float) -> None:
        with self._lock:
            self._pause_until = max(self._pause_until, time.monotonic() + seconds * self.scale)


def _led_view(led: dict, price: float) -> dict:
    """账本 → 观测字段(经 item 消息写入 items 表)。"""
    billed = int(led.get("billed", 0))
    maybe = int(led.get("maybe_billed", 0))
    return {
        "calls": int(led.get("calls", 0)), "billed": billed, "maybe_billed": maybe,
        "output_tokens": int(led.get("output_tokens", 0)),
        "cost_cny_est": round(billed * price, 4), "cost_maybe_extra_cny_est": round(maybe * price, 4),
        "cached": bool(led.get("cached")), "resumed": bool(led.get("resumed")),
        "degraded": led.get("state") == "DEGRADED", "maybe_duplicated": bool(led.get("maybe_duplicated")),
        "error": led.get("error"), "error_kind": led.get("error_kind"),
        "size_adjusted": led.get("size_adjusted"), "latency_s": led.get("latency_s"),
        "request_id": led.get("request_id"),
    }


@register("an_stylize")
def an_stylize(ctx: StepContext) -> None:
    p = ctx.params
    ci = int(p["chunk"])
    tl = _load_json(ctx.input_path("keyframes", "timeline.json"))
    ch = tl["chunks"][ci]
    idxs: list[int] = tl["keyframes"][ch["start"]:ch["end"]]
    total = len(idxs)
    W, H = tl["width"], tl["height"]
    z = _open_zip(ctx, "keyframes", f"kf_{ci}.zip")
    ref = ctx.input_path("reference", "ref.png").read_bytes() if "reference" in ctx.inputs else None
    ref_sha = _sha_bytes(ref) if ref else None

    provider, model = p.get("provider", "mock"), p.get("model")
    size, watermark = str(p.get("size", "2K")), bool(p.get("watermark", False))
    price = float(p.get("price_cny", 0.0))
    max_calls = int(p.get("frame_max_attempts", 3))
    on_reject = p.get("on_reject", "passthrough")
    max_deg = float(p.get("max_degraded_ratio", 0.1))
    overrides = p.get("frame_overrides") or {}
    conc = max(1, min(int(p.get("concurrency", 4)), 16))
    backoff_base = float(p.get("backoff_base_s", 2.0))
    budget = float(p.get("budget_cny", 0.0))
    chaos = json.loads(json.dumps(ctx.chaos.get("doubao") or {}))
    fr = chaos.get("frames") or {}
    if "__first__" in fr:                 # 预设里用 __first__ 指代"本块第一个关键帧",提交时还不知道序号
        fr[str(idxs[0])] = fr.pop("__first__")
    try:
        client = make_client(provider, model=model, data_dir=ctx.data_dir, chaos=chaos, attempt=ctx.attempt,
                             sleep=ctx.sleep, size=size, watermark=watermark)
    except FatalError as e:
        raise StepError(ErrorKind.PERMANENT, str(e))
    model_name = getattr(client, "model", model or "")
    limiter = RateLimiter(float(p.get("rpm", 300)), ctx.time_scale)
    cache = FrameCache(ctx.data_dir / "framecache")
    work = ctx.work_dir / "styled"
    led_dir = ctx.work_dir / "ledger"
    work.mkdir(parents=True, exist_ok=True)
    led_dir.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    leds: dict[int, dict] = {}
    abort = threading.Event()
    fatal: list[str] = []
    budget_hit = threading.Event()
    chunk_cap = (budget * total / max(tl["n_keyframes"], 1) * 1.25 + 3 * price) if (budget and price) else 0.0

    def read_led(idx: int) -> dict:
        try:
            return json.loads((led_dir / f"{idx:06d}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def write_led(idx: int, led: dict) -> None:
        _atomic_write(led_dir / f"{idx:06d}.json", json.dumps(led, ensure_ascii=False).encode())
        with lock:
            leds[idx] = led

    def emit(idx: int, led: dict, status: str | None = None) -> None:
        ctx.item(f"{idx:06d}", status or led.get("state", "PENDING"), idx=idx, chunk=ci, **_led_view(led, price))

    def spent_est() -> float:
        with lock:
            return sum((v.get("billed", 0) + v.get("maybe_billed", 0)) * price for v in leds.values())

    def new_led(fkey: str, prev: dict) -> dict:
        keep = {k: prev.get(k, 0) for k in ("calls", "billed", "maybe_billed", "output_tokens")}
        return {"fkey": fkey, "state": "PENDING", **keep}

    def run_frame(idx: int) -> dict:
        key = f"{idx:06d}"
        src = z.read(_frame_name(idx))
        prompt = (overrides.get(str(idx)) or {}).get("prompt") or p["prompt"]
        fkey = frame_cache_key(provider=provider, model=model_name, prompt=prompt, size=size,
                               watermark=watermark, ref_sha=ref_sha, src_sha=_sha_bytes(src))
        led = read_led(idx)
        if led.get("fkey") != fkey:        # 没有账本,或账本属于别的提示词/模型:从头开始
            led = new_led(fkey, {})
        out_path = work / _frame_name(idx)
        # 1) 之前的 attempt 已经完成这一帧:校验文件后直接复用
        if led.get("state") in ("DONE", "DEGRADED") and out_path.is_file() and _sha_file(out_path) == led.get("out_sha"):
            led["resumed"] = True
            write_led(idx, led)
            emit(idx, led)
            return led
        # 2) 跨任务帧缓存
        hit = cache.get(fkey)
        if hit is not None:
            data, _ = hit
            _atomic_write(out_path, data)
            led.update(state="DONE", out_sha=_sha_bytes(data), cached=True, error=None)
            write_led(idx, led)
            emit(idx, led)
            return led
        # 3) 调用接口
        if led.get("state") == "INFLIGHT":
            # 上一个进程在请求途中死了:服务端可能已经生成并计费。这里重发,所以可能"多付一次",如实记录
            led["maybe_billed"] = led.get("maybe_billed", 0) + 1
            led["maybe_duplicated"] = True
        for n in range(1, max_calls + 1):
            if abort.is_set():
                return {**led, "state": "SKIPPED"}
            if chunk_cap and spent_est() >= chunk_cap:
                budget_hit.set()
                abort.set()
                return {**led, "state": "SKIPPED"}
            limiter.wait()
            led.update(state="INFLIGHT", calls=led.get("calls", 0) + 1, cached=False, resumed=False)
            write_led(idx, led)                       # 先落"在途"标记,再发请求
            emit(idx, led)
            t0 = time.time()
            try:
                res: EditResult = client.edit(image=src, prompt=prompt, reference=ref, frame_key=key)
                out, adjusted = fit_size(res.image, (W, H))
            except RateLimited as e:
                led.update(state="RETRY", error=str(e), error_kind="rate_limit")
                write_led(idx, led)
                emit(idx, led)
                limiter.pause(e.retry_after)
                continue
            except Transient as e:
                if e.request_maybe_sent:             # 请求可能已到达服务端并计费:接下来的重发可能是重复计费
                    led["maybe_billed"] = led.get("maybe_billed", 0) + 1
                    led["maybe_duplicated"] = True
                led.update(state="RETRY", error=str(e), error_kind="transient")
                write_led(idx, led)
                emit(idx, led)
                ctx.sleep(min(backoff_base * 2 ** (n - 1), 30.0))
                continue
            except Moderation as e:
                led.update(state="DEGRADED" if on_reject == "passthrough" else "FAILED",
                           error=str(e), error_kind="moderation")
                if on_reject == "passthrough":
                    data, _ = fit_size(src, (W, H))
                    _atomic_write(out_path, data)
                    led["out_sha"] = _sha_bytes(data)
                write_led(idx, led)
                emit(idx, led)
                return led
            except FatalError as e:
                fatal.append(f"{e.kind}: {e}")
                abort.set()
                led.update(state="FAILED", error=str(e), error_kind=e.kind)
                write_led(idx, led)
                emit(idx, led)
                return led
            except Exception as e:                    # 返回内容无法解码等:服务端已处理完,按瞬时处理并记为"可能已计费"
                led["maybe_billed"] = led.get("maybe_billed", 0) + 1
                led["maybe_duplicated"] = True
                led.update(state="RETRY", error=f"{type(e).__name__}: {e}", error_kind="bad_response")
                write_led(idx, led)
                emit(idx, led)
                ctx.sleep(min(backoff_base * 2 ** (n - 1), 30.0))
                continue
            _atomic_write(out_path, out)
            led.update(state="DONE", out_sha=_sha_bytes(out), error=None, error_kind=None,
                       billed=led.get("billed", 0) + int(res.usage.get("generated_images", 1)),
                       output_tokens=led.get("output_tokens", 0) + int(res.usage.get("output_tokens", 0)),
                       size_adjusted=adjusted, latency_s=round(time.time() - t0, 3), request_id=res.request_id)
            cache.put(fkey, out, {"provider": provider, "model": model_name, "usage": res.usage})
            write_led(idx, led)
            emit(idx, led)
            return led
        led["state"] = "FAILED"
        write_led(idx, led)
        emit(idx, led)
        return led

    # 先把整块的账本同步给观测视图(含上次已完成的帧),让帧网格一开始就是完整的
    for idx in idxs:
        old = read_led(idx)
        with lock:
            leds[idx] = old
        emit(idx, old if old else {"state": "PENDING"}, old.get("state", "PENDING") if old else "PENDING")
    ctx.progress(0.0, f"stylize chunk {ci}: {total} frames")

    done = 0
    results: dict[int, dict] = {}
    pool = ThreadPoolExecutor(max_workers=conc)
    try:
        futs = {pool.submit(run_frame, i): i for i in idxs}
        pending = set(futs)
        while pending:
            fin, pending = wait(pending, timeout=1.0)
            for f in fin:
                idx = futs[f]
                try:
                    results[idx] = f.result()
                except Exception as e:                 # 不应发生;当作瞬时失败留给重试
                    results[idx] = {"state": "FAILED", "error": f"{type(e).__name__}: {e}"}
                done += 1
                if done % CKPT_EVERY == 0:
                    ctx.save_checkpoint({"done": done, "total": total})
            ctx.progress(done / total, f"chunk {ci}: {done}/{total}")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    summary = {"total": total, "done": 0, "cached": 0, "resumed": 0, "degraded": [], "failed": [],
               "skipped": [], "maybe_duplicated": []}
    tot = {"calls": 0, "billed": 0, "maybe_billed": 0, "output_tokens": 0}
    for idx in idxs:
        led = results.get(idx) or leds.get(idx) or {}
        st = led.get("state")
        for k in tot:
            tot[k] += int(led.get(k, 0))
        if led.get("maybe_duplicated"):
            summary["maybe_duplicated"].append(idx)
        if st == "DONE":
            summary["done"] += 1
            summary["cached"] += 1 if led.get("cached") else 0
            summary["resumed"] += 1 if led.get("resumed") else 0
        elif st == "DEGRADED":
            summary["degraded"].append(idx)
        elif st == "FAILED":
            summary["failed"].append({"idx": idx, "kind": led.get("error_kind"), "error": (led.get("error") or "")[:160]})
        else:
            summary["skipped"].append(idx)
    usage = {**tot, "price_cny": price, "cost_cny_est": round(tot["billed"] * price, 4),
             "cost_maybe_extra_cny_est": round(tot["maybe_billed"] * price, 4),
             "cost_note": "按标价估算;被审核拒绝的图不计费(以账单为准)。maybe_billed 是'响应丢失/进程中途死亡'导致的可能多付",
             "provider": provider, "model": model_name, "chunk": ci, "frames": summary}
    ctx.note("usage", usage)
    ctx.note("frames_summary", summary)

    if fatal:
        raise StepError(ErrorKind.PERMANENT, "豆包接口返回任务级错误,已停止继续调用: " + fatal[0])
    if budget_hit.is_set():
        raise StepError(ErrorKind.PERMANENT, f"本块累计费用估算已触及预算保护线(≈{chunk_cap:.2f} 元),已停止;已完成的帧会保留")
    rejected = [f for f in summary["failed"] if f["kind"] == "moderation"]
    if rejected:
        raise StepError(ErrorKind.PERMANENT,
                        f"{len(rejected)} 帧被内容审核拒绝({[r['idx'] for r in rejected][:8]}),on_reject=fail:"
                        "请对这些帧用 frame_overrides 换提示词后重新提交(其余帧命中帧缓存,不会重新计费)")
    if summary["failed"] or summary["skipped"]:
        raise StepError(ErrorKind.TRANSIENT,
                        f"{len(summary['failed'])} 帧失败、{len(summary['skipped'])} 帧未处理,将重试(已完成的帧不会重做): "
                        + "; ".join(f"{f['idx']}:{f['error'][:40]}" for f in summary["failed"][:4]))
    if len(summary["degraded"]) > max_deg * total:
        raise StepError(ErrorKind.PERMANENT,
                        f"降级(原图直通)的帧 {len(summary['degraded'])}/{total} 超过上限 {max_deg:.0%},请检查提示词或素材")
    pack_zip(ctx.output_path("styled.zip"), [(_frame_name(i), (work / _frame_name(i)).read_bytes()) for i in idxs])
    ctx.output("usage.json", canonical_json(usage).encode())
    ctx.progress(1.0, f"chunk {ci} done")


# --------------------------------------------------------------------------- 工序 4:后处理与补间
def match_color(arr: np.ndarray, t_mean, t_std, strength: float) -> np.ndarray:
    """把每个通道的均值/方差向目标靠拢(strength 0~1)。用于抵消逐帧生成带来的整体色偏闪烁。"""
    f = arr.astype(np.float32)
    out = f.copy()
    for c in range(3):
        m, s = float(f[..., c].mean()), float(f[..., c].std())
        if s < 1.0:
            continue
        new = (f[..., c] - m) * (float(t_std[c]) / s) + float(t_mean[c])
        out[..., c] = f[..., c] + strength * (new - f[..., c])
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def smooth_stats(idxs: list[int], means: list, stds: list, cuts: list[int], win: int = 5) -> dict:
    """沿关键帧序列做滑动平均得到"目标色彩统计";不跨镜头切换,避免把两个场景的色调混在一起。"""
    seg = [sum(1 for c in cuts if c <= i) for i in idxs]
    m, s = np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)
    out = {}
    for j, idx in enumerate(idxs):
        lo, hi = max(0, j - win // 2), min(len(idxs), j + win // 2 + 1)
        ids = [k for k in range(lo, hi) if seg[k] == seg[j]]
        out[str(idx)] = {"mean": [round(float(x), 3) for x in m[ids].mean(axis=0)],
                         "std": [round(float(x), 3) for x in s[ids].mean(axis=0)]}
    return out


@register("an_poststats")
def an_poststats(ctx: StepContext) -> None:
    tl = _load_json(ctx.input_path("keyframes", "timeline.json"))
    N = len(tl["chunks"])
    idxs, means, stds = [], [], []
    for i in range(N):
        with _open_zip(ctx, f"stylize_{i}", "styled.zip") as z:
            for name in sorted(z.namelist()):
                im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
                im.thumbnail((256, 256))
                a = np.asarray(im).astype(np.float32)
                idxs.append(int(name.split(".")[0]))
                means.append(a.mean(axis=(0, 1)))
                stds.append(a.std(axis=(0, 1)))
        ctx.progress((i + 1) / N, f"stats chunk {i + 1}/{N}")
    targets = smooth_stats(idxs, means, stds, tl["cuts"])
    ctx.output("targets.json", canonical_json(targets).encode())
    ctx.output("stats.json", canonical_json({"idx": idxs, "mean": [[round(float(x), 2) for x in m] for m in means],
                                             "std": [[round(float(x), 2) for x in s] for s in stds]}).encode())


class _Encoder:
    """把原始 RGB 帧通过管道喂给 ffmpeg,编码成片段。"""

    def __init__(self, out: Path, w: int, h: int, fps: float, errlog: Path):
        self._err = open(errlog, "wb")
        self.p = subprocess.Popen(
            [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "pipe:0", "-c:v", "libx264", "-crf", "14",
             "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)],
            stdin=subprocess.PIPE, stderr=self._err, stdout=subprocess.DEVNULL)
        self.n = 0

    def write(self, arr: np.ndarray) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(np.ascontiguousarray(arr).tobytes())
        self.n += 1

    def close(self) -> None:
        try:
            if self.p.stdin:
                self.p.stdin.close()
        finally:
            rc = self.p.wait()
            self._err.close()
            if rc != 0:
                raise MediaError(f"ffmpeg 编码失败(退出码 {rc})")


@register("an_post")
def an_post(ctx: StepContext) -> None:
    """一块的后处理:颜色匹配 + 轻度去噪 + 在相邻风格化关键帧之间补出其余帧,直接编码成视频片段。

    补间(interp=mc,默认):用**源视频的运动**把相邻两张风格化关键帧搬到中间时刻再融合(pipeline/interp.py),
    不再是直接混合,所以运动的人物不会出现双影;interp=blend 保留旧的线性混合(基线/对照)。
    镜头切换处不会补间(关键帧选择保证切换点两侧各有一个关键帧,相邻、中间没有补间帧)。
    """
    p = ctx.params
    i = int(p["chunk"])
    tl = _load_json(ctx.input_path("keyframes", "timeline.json"))
    targets = _load_json(ctx.input_path("poststats", "targets.json"))
    N = len(tl["chunks"])
    W, H, fps = tl["width"], tl["height"], tl["fps"]
    keys = tl["keyframes"]
    ch = tl["chunks"][i]
    my = keys[ch["start"]:ch["end"]]
    nxt_first = keys[tl["chunks"][i + 1]["start"]] if i + 1 < N else None
    f0 = my[0]
    f1 = nxt_first if nxt_first is not None else tl["n_frames"]
    strength, den = float(p.get("color_strength", 0.6)), float(p.get("denoise", 0.3))

    zs = {i: _open_zip(ctx, f"stylize_{i}", "styled.zip")}
    if i + 1 < N:
        zs[i + 1] = _open_zip(ctx, f"stylize_{i + 1}", "styled.zip")

    def load(idx: int) -> np.ndarray:
        name = _frame_name(idx)
        z = next(zz for zz in zs.values() if name in zz.namelist())
        im = Image.open(io.BytesIO(z.read(name))).convert("RGB")
        arr = np.asarray(im)
        t = targets[str(idx)]
        arr = match_color(arr, t["mean"], t["std"], strength)
        if den > 0:
            blur = np.asarray(Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(0.7))).astype(np.float32)
            arr = np.clip(np.rint(arr.astype(np.float32) * (1 - den) + blur * den), 0, 255).astype(np.uint8)
        return arr

    # 非最后一块:下一块的第一个关键帧只用来补间,它本身由下一块写出;最后一块要把末帧写出
    anchors = my + ([nxt_first] if nxt_first is not None else [])
    last_src = anchors[-1]
    mode = str(p.get("interp", "mc"))
    src_it = None
    if mode == "mc":
        # 按需只解码本块范围内的源帧(select + setpts 重编号,避免 cfr 复制帧);逐帧流式读取,内存只留一个关键帧间隔
        std = ctx.input_path("ingest", "standard.mp4")
        src_it = _read_raw_frames(
            std, f"select='between(n,{f0},{last_src})',setpts=N/{fps}/TB", W * H * 3, "rgb24")

    def next_src() -> np.ndarray:
        try:
            return np.frombuffer(next(src_it), np.uint8).reshape(H, W, 3)
        except StopIteration:
            raise StepError(ErrorKind.PERMANENT, "标准化素材的帧数少于时间线记录的帧数,无法按源帧补间")

    seg = ctx.output_path("seg.mp4")
    enc = _Encoder(seg, W, H, fps, ctx.work_dir / "ffmpeg_post.log")
    try:
        a_idx, A = anchors[0], load(anchors[0])
        src_a = next_src() if src_it is not None else None
        for j in range(1, len(anchors)):
            enc.write(A)
            b_idx = anchors[j]
            B = load(b_idx)
            if b_idx - a_idx > 1:
                if src_it is not None:
                    mids = [next_src() for _ in range(b_idx - a_idx - 1)]
                    src_b = next_src()
                    for k, f in enumerate(range(a_idx + 1, b_idx)):
                        t = (f - a_idx) / (b_idx - a_idx)
                        enc.write(interpolate(A, B, src_a, src_b, mids[k], t))
                        ctx.progress(min(0.95, (f - f0) / max(f1 - f0, 1)), f"post chunk {i}: mc frame {f}")
                else:
                    for f in range(a_idx + 1, b_idx):
                        enc.write(blend(A, B, (f - a_idx) / (b_idx - a_idx)))
            elif src_it is not None:
                src_b = next_src()
            if src_it is not None:
                src_a = src_b
            ctx.progress(min(0.95, (b_idx - f0) / max(f1 - f0, 1)), f"post chunk {i}: frame {b_idx}")
            a_idx, A = b_idx, B
        if nxt_first is None:
            enc.write(A)
    finally:
        try:
            enc.close()
        except MediaError as e:
            raise StepError(ErrorKind.TRANSIENT, str(e))
    want = f1 - f0
    if enc.n != want:
        raise StepError(ErrorKind.PERMANENT, f"片段 {i} 写出 {enc.n} 帧,期望 {want}")
    ctx.output("seg.json", canonical_json({"chunk": i, "first": f0, "last": f1 - 1, "frames": enc.n}).encode())
    ctx.progress(1.0, "post done")


# --------------------------------------------------------------------------- 质量校验
def second_diff_level(x: np.ndarray) -> float:
    """亮度序列的二阶差分绝对值的中位数:越大说明逐帧亮度抖动越厉害(用中位数,镜头切换的尖峰不影响它)。"""
    if len(x) < 3:
        return 0.0
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    return float(np.median(np.abs(d2)))


STRUCT_W = 192             # 结构保真度指标用的灰度缩略图宽度(比亮度分析更大,否则看不出人物的双影)


def _edge_map(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    gx = np.abs(g[:, 2:] - g[:, :-2])[1:-1]
    gy = np.abs(g[2:, :] - g[:-2, :])[:, 1:-1]
    return np.sqrt(gx * gx + gy * gy)


def structure_scores(out_frames: list[np.ndarray], src_frames: list[np.ndarray]) -> np.ndarray:
    """逐帧:成品与源帧的"边缘图"归一化互相关(-1..1)。风格化后边缘仍应大体在源边缘的位置;
    运动补间出现双影时,多出来的边缘会让该帧分数明显低于关键帧。"""
    out = []
    for a, b in zip(out_frames, src_frames):
        ea, eb = _edge_map(a), _edge_map(b)
        ea, eb = ea - ea.mean(), eb - eb.mean()
        d = float(np.sqrt((ea * ea).sum() * (eb * eb).sum()))
        out.append(float((ea * eb).sum() / d) if d > 1e-6 else 0.0)
    return np.asarray(out, dtype=np.float32)


@register("an_qc")
def an_qc(ctx: StepContext) -> None:
    """确定性质检。硬指标不过 → 永久失败(需要人看);软指标(闪烁、降级比例)只告警,除非 strict。"""
    p = ctx.params
    tl = _load_json(ctx.input_path("keyframes", "timeline.json"))
    N = len(tl["chunks"])
    W, H, fps, n = tl["width"], tl["height"], tl["fps"], tl["n_frames"]
    hard: list[str] = []
    soft: list[str] = []
    lw = STRUCT_W
    lh = _even(lw * H / W)
    lum: list[float] = []
    out_small: list[np.ndarray] = []
    frames_total = 0
    for i in range(N):
        seg = ctx.input_path(f"post_{i}", "seg.mp4")
        meta = _load_json(ctx.input_path(f"post_{i}", "seg.json"))
        try:
            vi = probe_video(seg)
            cnt = 0
            for buf in _read_raw_frames(seg, f"scale={lw}:{lh}:flags=area,format=gray", lw * lh, "gray"):
                fr = np.frombuffer(buf, dtype=np.uint8).reshape(lh, lw)
                lum.append(float(fr.mean()))
                out_small.append(fr)
                cnt += 1
        except MediaError as e:
            raise StepError(ErrorKind.PERMANENT, f"片段 {i} 无法解码: {e}")
        frames_total += cnt
        if cnt != meta["frames"]:
            hard.append(f"片段 {i} 解码得到 {cnt} 帧,登记的是 {meta['frames']} 帧")
        if (vi["width"], vi["height"]) != (W, H):
            hard.append(f"片段 {i} 分辨率 {vi['width']}x{vi['height']} != {W}x{H}")
        ctx.progress(0.8 * (i + 1) / N, f"qc chunk {i + 1}/{N}")
    if frames_total != n:
        hard.append(f"片段总帧数 {frames_total} != 标准素材帧数 {n}")
    out_lum = np.asarray(lum, dtype=np.float32)
    src_lum = np.asarray(tl["lum"], dtype=np.float32)
    black = float((out_lum < 2.0).mean()) if len(out_lum) else 1.0
    black_src = float((src_lum < 2.0).mean())
    if black > black_src + 0.05:
        hard.append(f"全黑帧占比 {black:.1%}(源 {black_src:.1%}):疑似生成失败/空帧")
    f_out, f_src = second_diff_level(out_lum), second_diff_level(src_lum)
    ratio = f_out / (f_src + 0.5)
    max_ratio = float(p.get("max_flicker_ratio", 3.0))
    if ratio > max_ratio:
        soft.append(f"亮度闪烁比 {ratio:.2f} 超过 {max_ratio}(成品 {f_out:.2f} / 源 {f_src:.2f};阈值为经验值)")
    # 结构保真度(能发现补间重影;亮度闪烁指标发现不了):对比成品与源的边缘图
    struct = None
    if not hard and len(out_small) == n:
        try:
            src_small = [np.frombuffer(b, dtype=np.uint8).reshape(lh, lw) for b in _read_raw_frames(
                ctx.input_path("ingest", "standard.mp4"), f"scale={lw}:{lh}:flags=area,format=gray", lw * lh, "gray")]
        except MediaError as e:
            raise StepError(ErrorKind.PERMANENT, f"标准化素材无法解码: {e}")
        if len(src_small) == n:
            sc = structure_scores(out_small, src_small)
            is_key = np.zeros(n, dtype=bool)
            is_key[list(tl["keyframes"])] = True
            if is_key.any() and (~is_key).any():
                k_mean, m_mean = float(sc[is_key].mean()), float(sc[~is_key].mean())
                s_ratio = m_mean / max(k_mean, 1e-6)
                struct = {"keyframes": round(k_mean, 3), "inbetween": round(m_mean, 3), "ratio": round(s_ratio, 3),
                          "min_ratio": float(p.get("min_structure_ratio", 0.85))}
                if s_ratio < struct["min_ratio"]:
                    soft.append(f"补间帧的结构保真度只有关键帧的 {s_ratio:.0%}(<{struct['min_ratio']:.0%}):疑似运动重影"
                                "(阈值为经验值)")
    # 降级与可能多付(来自工序 3 的用量记录)
    degraded, maybe_dup, billed, cost, maybe_extra, tokens, calls = 0, 0, 0, 0.0, 0.0, 0, 0
    for i in range(N):
        u = _load_json(ctx.input_path(f"stylize_{i}", "usage.json"))
        degraded += len(u["frames"]["degraded"])
        maybe_dup += len(u["frames"]["maybe_duplicated"])
        billed += u["billed"]
        calls += u["calls"]
        tokens += u["output_tokens"]
        cost += u["cost_cny_est"]
        maybe_extra += u["cost_maybe_extra_cny_est"]
    if degraded:
        soft.append(f"{degraded}/{tl['n_keyframes']} 个关键帧被降级为原图直通(审核拒绝)")
    if maybe_dup:
        soft.append(f"{maybe_dup} 帧可能被重复计费(响应丢失/进程中途死亡后重发)")
    # 对比图:源关键帧 | 风格化关键帧(最多 3 对)
    ctx.progress(0.85, "compare image")
    picks = [0, N // 2, N - 1] if N > 2 else list(range(N))
    rows = []
    for i in dict.fromkeys(picks):
        ch = tl["chunks"][i]
        k = tl["keyframes"][(ch["start"] + ch["end"]) // 2]
        with _open_zip(ctx, "keyframes", f"kf_{i}.zip") as zk, _open_zip(ctx, f"stylize_{i}", "styled.zip") as zs:
            a = Image.open(io.BytesIO(zk.read(_frame_name(k)))).convert("RGB")
            b = Image.open(io.BytesIO(zs.read(_frame_name(k)))).convert("RGB")
        a.thumbnail((480, 480))
        b.thumbnail((480, 480))
        row = Image.new("RGB", (a.width + b.width + 6, max(a.height, b.height)), (20, 20, 20))
        row.paste(a, (0, 0))
        row.paste(b, (a.width + 6, 0))
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height + 6 for r in rows)), (20, 20, 20))
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 6
    ctx.output("compare.png", _png_bytes(sheet))
    strict = bool(p.get("qc_strict", False))
    report = {"ok": not hard and (not soft or not strict), "hard_failures": hard, "warnings": soft,
              "frames": frames_total, "expected_frames": n, "size": [W, H],
              "flicker": {"out": round(f_out, 3), "src": round(f_src, 3), "ratio": round(ratio, 3), "max_ratio": max_ratio},
              "structure": struct,
              "black_frame_ratio": {"out": round(black, 4), "src": round(black_src, 4)},
              "degraded_keyframes": degraded, "maybe_duplicated_frames": maybe_dup,
              "usage": {"calls": calls, "billed_images": billed, "output_tokens": tokens,
                        "cost_cny_est": round(cost, 4), "cost_maybe_extra_cny_est": round(maybe_extra, 4)}}
    ctx.note("qc", report)
    ctx.output("qc.json", canonical_json(report).encode())
    ctx.progress(1.0, "qc done")
    if hard:
        raise StepError(ErrorKind.PERMANENT, "质检未通过: " + "; ".join(hard))
    if soft and strict:
        raise StepError(ErrorKind.PERMANENT, "质检未通过(strict): " + "; ".join(soft))


# --------------------------------------------------------------------------- 工序 5:合成与音轨对齐
@register("an_mux")
def an_mux(ctx: StepContext) -> None:
    tl = _load_json(ctx.input_path("keyframes", "timeline.json"))
    ing = _load_json(ctx.input_path("ingest", "ingest.json"))
    N = len(tl["chunks"])
    fps, n = tl["fps"], tl["n_frames"]
    work = ctx.work_dir / "mux"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(N):
        dst = work / f"seg_{i}.mp4"
        shutil.copyfile(ctx.input_path(f"post_{i}", "seg.mp4"), dst)
        lines.append(f"file 'seg_{i}.mp4'")      # 相对 list.txt 所在目录,避开路径里的非 ASCII 字符
    (work / "list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ctx.progress(0.1, "concat")
    video = work / "video.mp4"
    final = ctx.output_path("final.mp4")
    try:
        run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(work / "list.txt"), "-c", "copy", str(video)])
        ctx.progress(0.4, "audio")
        # 以视频时长为准:音轨补静音到足够长,再用 -t 截断到视频长度(音轨短不会把视频截短)。
        # 注意不能用 -shortest:视频是 copy 时,配上无限长的 apad/anullsrc,ffmpeg 永远不会结束(实测卡死)。
        dur = f"{n / fps:.6f}"
        if ing["has_audio"]:
            run_ffmpeg(["-i", str(video), "-i", str(ctx.input_path("ingest", "audio.m4a")),
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-af", "apad",
                        "-t", dur, "-movflags", "+faststart", str(final)],
                       lambda f: ctx.progress(0.4 + 0.4 * min(f / max(n, 1), 1.0), f"muxed {f}"))
        else:
            run_ffmpeg(["-i", str(video), "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
                        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-t", dur,
                        "-movflags", "+faststart", str(final)])
        ctx.progress(0.85, "verifying")
        got = count_decoded_frames(final)
        vi = probe_video(final)
    except MediaError as e:
        raise StepError(ErrorKind.TRANSIENT, str(e))
    expect_dur = n / fps
    drift = abs(vi["duration_s"] - expect_dur)
    problems = []
    if got != n:
        problems.append(f"成品帧数 {got} != {n}")
    if not vi["has_audio"]:
        problems.append("成品没有音轨")
    if drift > float(ctx.params.get("max_av_drift_s", 0.25)):
        problems.append(f"成品时长 {vi['duration_s']}s 与视频帧推算的 {expect_dur:.3f}s 相差 {drift:.2f}s,音画可能不同步")
    info = {"frames": got, "duration_s": vi["duration_s"], "expected_duration_s": round(expect_dur, 3),
            "av_drift_s": round(drift, 3), "audio": "source" if ing["has_audio"] else "silent",
            "width": vi["width"], "height": vi["height"], "fps": vi["fps"], "size_mb": vi["size_mb"]}
    ctx.note("mux", info)
    ctx.output("mux.json", canonical_json(info).encode())
    if problems:
        raise StepError(ErrorKind.PERMANENT, "合成校验未通过: " + "; ".join(problems))
    ctx.progress(1.0, "mux done")


# --------------------------------------------------------------------------- 工序 6:打包与元数据归档
@register("an_package")
def an_package(ctx: StepContext) -> None:
    """交付件:成品视频 + 缩略图 + 元数据(参数、素材、关键帧统计、质检、用量与费用估算)。确定性,可缓存。"""
    p = ctx.params
    N = int(p["chunks"])
    tl = _load_json(ctx.input_path("keyframes", "timeline.json"))
    ing = _load_json(ctx.input_path("ingest", "ingest.json"))
    qc = _load_json(ctx.input_path("qc", "qc.json"))
    mux = _load_json(ctx.input_path("mux", "mux.json"))
    shutil.copyfile(ctx.input_path("mux", "final.mp4"), ctx.output_path("final.mp4"))
    ctx.progress(0.4, "thumbnail")
    t = min(1.0, mux["duration_s"] / 2)
    try:
        run_ffmpeg(["-ss", f"{t:.3f}", "-i", str(ctx.input_path("mux", "final.mp4")), "-frames:v", "1",
                    "-vf", "scale=320:-2", "-q:v", "3", str(ctx.output_path("thumbnail.jpg"))])
    except MediaError as e:
        raise StepError(ErrorKind.TRANSIENT, str(e))
    usage = {"calls": 0, "billed_images": 0, "output_tokens": 0, "cost_cny_est": 0.0, "cost_maybe_extra_cny_est": 0.0,
             "cached_frames": 0, "resumed_frames": 0, "degraded_frames": [], "maybe_duplicated_frames": []}
    for i in range(N):
        u = _load_json(ctx.input_path(f"stylize_{i}", "usage.json"))
        usage["calls"] += u["calls"]
        usage["billed_images"] += u["billed"]
        usage["output_tokens"] += u["output_tokens"]
        usage["cost_cny_est"] = round(usage["cost_cny_est"] + u["cost_cny_est"], 4)
        usage["cost_maybe_extra_cny_est"] = round(usage["cost_maybe_extra_cny_est"] + u["cost_maybe_extra_cny_est"], 4)
        usage["cached_frames"] += u["frames"]["cached"]
        usage["resumed_frames"] += u["frames"]["resumed"]
        usage["degraded_frames"] += u["frames"]["degraded"]
        usage["maybe_duplicated_frames"] += u["frames"]["maybe_duplicated"]
    meta = {
        "pipeline": "anime-frame-style", "request": p.get("summary", {}),
        "source": {"name": ing["source"]["name"], "sha256": ing["source"]["sha256"],
                   "width": ing["source"]["width"], "height": ing["source"]["height"],
                   "fps": ing["source"]["fps"], "duration_s": ing["source"]["duration_s"]},
        "standard": ing["standard"],
        "keyframes": {"count": tl["n_keyframes"], "of_frames": tl["n_frames"], "cuts": tl["cuts"],
                      "duplicates_skipped": tl["duplicates_skipped"], "params": tl["params"]},
        "stylize": usage, "qc": {"ok": qc["ok"], "warnings": qc["warnings"], "flicker": qc["flicker"]},
        "output": mux, "warnings": ing["warnings"],
        "note": "耗时按步骤记录在任务的 attempts 中(GET /api/tasks/{id}/report);费用是按标价的估算,以平台账单为准",
    }
    ctx.output("metadata.json", canonical_json(meta).encode())
    ctx.progress(1.0, "packaged")
