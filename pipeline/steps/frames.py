"""序列帧处理流水线的"真实"步骤:  拆帧 → 抠背景 → 超分辨率 → 补帧 → 合成视频。

和 animation.py 里的 MOCK 步骤不同,这里每一步都真的处理像素(ffmpeg + numpy + Pillow),
因此产物 hash、检查点、断点续算、卡死检测都是在真实计算上被检验的。

三个"算法"是 GPU 模型的**轻量 CPU 替身**,目的是让引擎处理真实的长任务,而不是追求画质:
  - 抠背景:背景差分(时间中值背景 + 差异阈值),要求固定机位;替身对象:RVM/rembg/SAM 等分割模型
  - 超分辨率:Lanczos 2 倍放大;替身对象:Real-ESRGAN
  - 补帧:相邻帧(预乘 alpha)线性混合;替身对象:RIFE/FILM。快速运动时会有"重影",这是已知局限
替换成真实模型时只需改对应函数,步骤划分、分块、检查点、缓存与恢复逻辑都不用动。

数据流:  帧序列按块打包成 zip(STORED、固定时间戳、固定顺序 → 同样的帧得到同样的字节),
作为内容寻址产物在步骤之间传递,所以任何步骤都能在任何 worker 上重跑。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from ..media import MediaError, assess_video, count_decoded_frames, probe_video, run_ffmpeg
from ..models import ErrorKind, canonical_json
from ..stepkit import StepContext, StepError
from . import register

# 抠背景参数(像素值 0-255):差异 <= LO 视为背景,>= HI 视为前景,之间线性过渡
MATTE_LO, MATTE_HI = 14, 40
CKPT_EVERY = 4            # 每处理几帧写一次检查点
BG_SAMPLES = 25           # 估计背景中值图时最多抽多少帧


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
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def pack_zip(dest: Path, items: list[tuple[str, bytes]]) -> None:
    """确定性 zip:同样的内容 → 同样的字节(固定顺序、固定时间戳、不压缩)。"""
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as z:
        for name, data in items:
            zi = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            z.writestr(zi, data)


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=3)
    return buf.getvalue()


def _open_zip(ctx: StepContext, dep: str, name: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(ctx.input_path(dep, name))


def _rgba(z: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(z.read(name))).convert("RGBA"))


def _map_frames(ctx: StepContext, label: str, names: list[str],
                produce: Callable[[int], bytes], out_name: str) -> None:
    """逐帧处理的长步骤,带步骤内检查点(与 MOCK 的 _chunked 同一套约定)。

    - 每帧结果原子写入 work_dir(跨 attempt 保留);检查点只记"已完成数 + 各帧 sha";
    - 恢复时先**校验**已落盘文件的 sha,不信任检查点本身;校验失败则从该帧起重算;
    - 所有帧完成后打成确定性 zip 作为本步骤的产物。
    """
    work = ctx.work_dir / label
    work.mkdir(parents=True, exist_ok=True)
    total = len(names)
    shas: list[str] = []
    for i, sha in enumerate((ctx.checkpoint or {}).get("shas", [])):
        p = work / names[i] if i < total else None
        if p is not None and p.is_file() and _sha_file(p) == sha:
            shas.append(sha)
        else:
            break
    resumed = len(shas)
    ctx.note("resumed_from", resumed)
    ctx.progress(resumed / total, f"{label}: start at {resumed}/{total}")
    for i in range(resumed, total):
        data = produce(i)
        _atomic_write(work / names[i], data)
        shas.append(_sha_bytes(data))
        done = i + 1
        if done % CKPT_EVERY == 0 or done == total:
            ctx.save_checkpoint({"done": done, "shas": shas})
        ctx.progress(done / total, f"{label}: {done}/{total}")
    ctx.note("frames_computed", total - resumed)
    pack_zip(ctx.output_path(out_name), [(n, (work / n).read_bytes()) for n in names])


def _chunk_range(man: dict, i: int) -> tuple[int, int]:
    return man["bounds"][i], man["bounds"][i + 1]


# --------------------------------------------------------------------------- 1. 素材检测
@register("fp_probe")
def fp_probe(ctx: StepContext) -> None:
    """素材检测。不合适就在这里永久失败,避免后面白白消耗计算。"""
    src = Path(str(ctx.params.get("source", "")))
    ctx.progress(0.0, "checking source")
    if not src.is_file():
        raise StepError(ErrorKind.PERMANENT, f"素材文件不存在: {src}")
    actual = _sha_file(src)
    expect = ctx.params.get("source_sha256")
    if expect and actual != expect:
        raise StepError(ErrorKind.PERMANENT, "素材文件在提交后被修改(sha256 与提交时不一致),请重新提交")
    ctx.progress(0.3, "assessing")
    try:
        rep = assess_video(src)
    except MediaError as e:
        raise StepError(ErrorKind.PERMANENT, f"素材无法读取: {e}")
    if not rep["ok"]:
        raise StepError(ErrorKind.PERMANENT, "素材不适合本流水线: " + "; ".join(rep["reasons"]))
    ctx.progress(1.0, "source ok")
    meta = {"name": src.name, "sha256": actual, "info": rep["info"], "metrics": rep["metrics"],
            "warnings": rep["warnings"]}
    ctx.output("meta.json", canonical_json(meta).encode())


# --------------------------------------------------------------------------- 2. 拆帧
@register("fp_extract")
def fp_extract(ctx: StepContext) -> None:
    """ffmpeg 拆成 PNG 序列,再按块打包。单次调用很快,不设步骤内检查点。"""
    src = Path(str(ctx.params["source"]))
    chunks = int(ctx.params["chunks"])
    info = _load_json(ctx.input_path("probe", "meta.json"))["info"]
    tmp = ctx.work_dir / "extract"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    est = max(info["frames_est"], 1)
    ctx.progress(0.0, "ffmpeg decoding")
    try:
        run_ffmpeg(["-i", str(src), "-an", "-fps_mode", "passthrough", str(tmp / "%05d.png")],
                   lambda f: ctx.progress(min(f / est, 0.9) * 0.8, f"decoded {f}"))
    except MediaError as e:
        raise StepError(ErrorKind.TRANSIENT, str(e))
    files = sorted(tmp.glob("*.png"))
    n = len(files)
    if n < chunks * 2:
        raise StepError(ErrorKind.PERMANENT, f"只解出 {n} 帧,不足以分成 {chunks} 块")
    w, h = Image.open(files[0]).size
    bounds = [i * n // chunks for i in range(chunks + 1)]
    for i in range(chunks):
        s, e = bounds[i], bounds[i + 1]
        pack_zip(ctx.output_path(f"frames_{i}.zip"), [(f"{k:05d}.png", files[k].read_bytes()) for k in range(s, e)])
        ctx.progress(0.8 + 0.2 * (i + 1) / chunks, f"packed chunk {i + 1}/{chunks}")
    man = {"n": n, "chunks": chunks, "bounds": bounds, "width": w, "height": h, "fps": info["fps"]}
    ctx.output("frames.json", canonical_json(man).encode())
    shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- 3. 背景估计 + 抠背景
@register("fp_plate")
def fp_plate(ctx: StepContext) -> None:
    """背景中值图:对均匀抽样的若干帧取逐像素中值。运动主体在每个像素上只占少数帧,被中值滤掉。"""
    man = _load_json(ctx.input_path("extract_frames", "frames.json"))
    n, N = man["n"], man["chunks"]
    picks = sorted({int(round(x)) for x in np.linspace(0, n - 1, min(BG_SAMPLES, n))})
    zips = [_open_zip(ctx, "extract_frames", f"frames_{i}.zip") for i in range(N)]
    stack = []
    for j, k in enumerate(picks):
        ci = next(i for i in range(N) if man["bounds"][i] <= k < man["bounds"][i + 1])
        stack.append(np.asarray(Image.open(io.BytesIO(zips[ci].read(f"{k:05d}.png"))).convert("RGB")))
        ctx.progress((j + 1) / len(picks) * 0.9, f"plate: sampled {j + 1}/{len(picks)}")
    plate = np.median(np.stack(stack), axis=0).astype(np.uint8)
    Image.fromarray(plate, "RGB").save(ctx.output_path("plate.png"), format="PNG", compress_level=3)
    ctx.progress(1.0, "plate done")


def matte_frame(rgb: np.ndarray, plate: np.ndarray) -> bytes:
    """背景差分抠图:与背景图的最大通道差 → alpha(软边),中值滤波去噪点。返回 RGBA PNG 字节。"""
    diff = np.abs(rgb.astype(np.int16) - plate.astype(np.int16)).max(axis=2).astype(np.float32)
    a = np.clip((diff - MATTE_LO) / (MATTE_HI - MATTE_LO), 0.0, 1.0)
    m = Image.fromarray((a * 255).astype(np.uint8), "L").filter(ImageFilter.MedianFilter(5))
    m = m.filter(ImageFilter.GaussianBlur(0.8))
    out = np.dstack([rgb, np.asarray(m)])
    return _png_bytes(Image.fromarray(out, "RGBA"))


@register("fp_matte")
def fp_matte(ctx: StepContext) -> None:
    i = int(ctx.params["chunk"])
    man = _load_json(ctx.input_path("extract_frames", "frames.json"))
    s, e = _chunk_range(man, i)
    plate = np.asarray(Image.open(ctx.input_path("bg_plate", "plate.png")).convert("RGB"))
    z = _open_zip(ctx, "extract_frames", f"frames_{i}.zip")
    names = [f"{k:05d}.png" for k in range(s, e)]
    _map_frames(ctx, "matte", names,
                lambda idx: matte_frame(np.asarray(Image.open(io.BytesIO(z.read(names[idx]))).convert("RGB")), plate),
                "frames.zip")


# --------------------------------------------------------------------------- 4. 超分辨率
@register("fp_upscale")
def fp_upscale(ctx: StepContext) -> None:
    i = int(ctx.params["chunk"])
    scale = int(ctx.params.get("scale", 2))
    z = _open_zip(ctx, f"matte_{i}", "frames.zip")
    names = sorted(z.namelist())

    def up(idx: int) -> bytes:
        im = Image.open(io.BytesIO(z.read(names[idx]))).convert("RGBA")
        # Pillow 对 RGBA 缩放会先转预乘 alpha,边缘不会出现黑边/白边
        return _png_bytes(im.resize((im.width * scale, im.height * scale), Image.LANCZOS))
    _map_frames(ctx, "upscale", names, up, "frames.zip")


# --------------------------------------------------------------------------- 5. 补帧
def blend_frames(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """两帧 RGBA 的中间帧:按预乘 alpha 平均,避免半透明边缘出现颜色污染。"""
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    aa, ab = fa[..., 3:4] / 255.0, fb[..., 3:4] / 255.0
    alpha = (aa + ab) / 2.0
    pre = (fa[..., :3] * aa + fb[..., :3] * ab) / 2.0
    rgb = np.where(alpha > 1e-6, pre / np.maximum(alpha, 1e-6), 0.0)
    out = np.concatenate([np.clip(rgb, 0, 255), np.clip(alpha * 255.0, 0, 255)], axis=2)
    return np.rint(out).astype(np.uint8)


@register("fp_interp")
def fp_interp(ctx: StepContext) -> None:
    """帧率翻倍:输出帧编号 2k 为原帧,2k+1 为 k 与 k+1 的中间帧。

    块边界处需要"下一块的第一帧",所以 interp_i 同时依赖 upscale_i 与 upscale_{i+1}。
    总输出帧数 = 2n - 1(最后一帧后面没有可混合的下一帧)。
    """
    i = int(ctx.params["chunk"])
    N = int(ctx.params["chunks"])
    man = _load_json(ctx.input_path("extract_frames", "frames.json"))
    s, e = _chunk_range(man, i)
    n = man["n"]
    z = _open_zip(ctx, f"upscale_{i}", "frames.zip")
    znext = _open_zip(ctx, f"upscale_{i + 1}", "frames.zip") if i + 1 < N else None
    here = {k: f"{k:05d}.png" for k in range(s, e)}

    def get(k: int) -> np.ndarray:
        if k in here:
            return _rgba(z, here[k])
        assert znext is not None
        return _rgba(znext, f"{k:05d}.png")

    plan: list[tuple[str, int]] = []      # (kind, k)
    for k in range(s, e):
        plan.append(("orig", k))
        if k + 1 < n:
            plan.append(("mid", k))
    names = [f"{(2 * k if kind == 'orig' else 2 * k + 1):05d}.png" for kind, k in plan]

    def make(idx: int) -> bytes:
        kind, k = plan[idx]
        if kind == "orig":
            return z.read(here[k])
        return _png_bytes(Image.fromarray(blend_frames(get(k), get(k + 1)), "RGBA"))
    _map_frames(ctx, "interp", names, make, "frames.zip")


# --------------------------------------------------------------------------- 6. 合成视频
@register("fp_assemble")
def fp_assemble(ctx: StepContext) -> None:
    """把补帧后的 RGBA 序列合成两个成品:带深色底的 H.264 预览(final.mp4)和带透明通道的 VP9(alpha.webm)。"""
    N = int(ctx.params["chunks"])
    man = _load_json(ctx.input_path("extract_frames", "frames.json"))
    scale = int(ctx.params.get("scale", 2))
    W, H = man["width"] * scale, man["height"] * scale
    fps2 = man["fps"] * 2
    seq = ctx.work_dir / "seq"
    shutil.rmtree(seq, ignore_errors=True)
    seq.mkdir(parents=True)

    total = 2 * man["n"] - 1
    cover: list[float] = []
    cnt = 0
    for i in range(N):
        with _open_zip(ctx, f"interp_{i}", "frames.zip") as z:
            for name in sorted(z.namelist()):
                data = z.read(name)
                (seq / name).write_bytes(data)
                alpha = np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))[..., 3]
                cover.append(round(float((alpha > 127).mean()), 5))
                cnt += 1
        ctx.progress(0.2 * (i + 1) / N, f"unpacked chunk {i + 1}/{N}")
    if cnt != total or sorted(p.name for p in seq.glob("*.png"))[-1] != f"{total - 1:05d}.png":
        raise StepError(ErrorKind.PERMANENT, f"帧数不连续: 得到 {cnt},期望 {total}")

    def prog(lo: float, hi: float):
        return lambda f: ctx.progress(lo + (hi - lo) * min(f / total, 1.0), f"encoding {f}/{total}")
    try:
        mp4 = ctx.output_path("final.mp4")
        run_ffmpeg(["-framerate", f"{fps2:g}", "-i", str(seq / "%05d.png"),
                    "-f", "lavfi", "-i", f"color=c=0x202020:s={W}x{H}:r={fps2:g}",
                    "-filter_complex", "[1:v][0:v]overlay=shortest=1,format=yuv420p",
                    "-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-movflags", "+faststart", str(mp4)],
                   prog(0.2, 0.55))
        webm = ctx.output_path("alpha.webm")
        run_ffmpeg(["-framerate", f"{fps2:g}", "-i", str(seq / "%05d.png"),
                    "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
                    "-b:v", "0", "-crf", "30", "-deadline", "good", "-cpu-used", "5", str(webm)],
                   prog(0.55, 1.0))
    except MediaError as e:
        raise StepError(ErrorKind.TRANSIENT, str(e))
    ctx.output("stats.json", canonical_json({"frames": total, "width": W, "height": H, "fps": fps2,
                                             "alpha_coverage": cover}).encode())
    shutil.rmtree(seq, ignore_errors=True)


# --------------------------------------------------------------------------- 7. 质检
@register("fp_qc")
def fp_qc(ctx: StepContext) -> None:
    """确定性质检:能解码、帧数/分辨率对、alpha 覆盖率合理。不通过就永久失败(需要人看,不应自动重试)。"""
    man = _load_json(ctx.input_path("extract_frames", "frames.json"))
    stats = _load_json(ctx.input_path("assemble", "stats.json"))
    scale = int(ctx.params.get("scale", 2))
    want_frames = 2 * man["n"] - 1
    want_w, want_h = man["width"] * scale, man["height"] * scale
    problems: list[str] = []

    ctx.progress(0.1, "decode mp4")
    mp4 = ctx.input_path("assemble", "final.mp4")
    try:
        got = count_decoded_frames(mp4)
        vi = probe_video(mp4)
        ctx.progress(0.5, "decode webm")
        got_a = count_decoded_frames(ctx.input_path("assemble", "alpha.webm"), decoder="libvpx-vp9")
    except MediaError as e:
        raise StepError(ErrorKind.PERMANENT, f"成品无法解码: {e}")
    if got != want_frames:
        problems.append(f"final.mp4 帧数 {got} != 期望 {want_frames}")
    if got_a != want_frames:
        problems.append(f"alpha.webm 帧数 {got_a} != 期望 {want_frames}")
    if (vi["width"], vi["height"]) != (want_w, want_h):
        problems.append(f"分辨率 {vi['width']}x{vi['height']} != 期望 {want_w}x{want_h}")
    cover = stats["alpha_coverage"]
    mean_c = sum(cover) / len(cover)
    empty = sum(1 for c in cover if c < 0.001) / len(cover)
    if not (0.005 <= mean_c <= 0.85):
        problems.append(f"主体覆盖率均值 {mean_c:.1%} 不在 0.5%-85% 内(抠图可能失败)")
    if empty > 0.10:
        problems.append(f"{empty:.0%} 的帧主体为空(>10%)")
    if ctx.params.get("force_fail"):
        problems.append("强制失败(测试用)")
    ctx.progress(1.0, "qc done")
    if problems:
        raise StepError(ErrorKind.PERMANENT, "质检未通过: " + "; ".join(problems))
    ctx.output("report.json", canonical_json({"ok": True, "frames": got, "alpha_frames": got_a, "size": [vi["width"], vi["height"]],
                                              "alpha_coverage_mean": round(mean_c, 4), "empty_frame_ratio": round(empty, 4)}).encode())
