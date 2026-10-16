"""动画素材"角色替换"流水线的 MOCK 步骤。

计算是假的(sleep + 确定性字节),但**产物、哈希、检查点、故障都是真的**。
确定性:输出内容只取决于 (步骤类型, 参数, 输入产物 hash),因此重算得到相同 hash,
便于"混沌测试结果 == 无故障基线"的断言。唯一例外是 gen_reference(nondeterministic=True),
用来模拟生成式模型的随机性,验证"必须固化产物而不是靠重算复用"。
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

from ..models import ErrorKind, canonical_json
from ..stepkit import StepContext, StepError
from . import register


def det_bytes(*parts, size: int = 256) -> bytes:
    """由 parts 确定性生成 size 字节内容。"""
    seed = hashlib.sha256(canonical_json(list(parts)).encode()).digest()
    out = bytearray()
    block = seed
    while len(out) < size:
        block = hashlib.sha256(block).digest()
        out += block
    return bytes(out[:size])


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".{uuid.uuid4().hex[:6]}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --------------------------------------------------------------------------- 轻量 CPU 步骤
@register("probe")
def probe(ctx: StepContext) -> None:
    """探测输入素材。坏输入在这里永久失败,避免后面白白消耗 GPU。"""
    src = str(ctx.params.get("source", ""))
    ctx.progress(0.0, "probing")
    if src.startswith("bad:") or not src:
        raise StepError(ErrorKind.PERMANENT, f"输入素材损坏或为空: {src!r}")
    ctx.simulate(ctx.params.get("duration", 0.3), ticks=2, msg="probing")
    meta = {"source": src, "frames": int(ctx.params.get("frames", 24)), "fps": int(ctx.params.get("fps", 24))}
    ctx.output("meta.json", canonical_json(meta).encode())


@register("extract_frames")
def extract_frames(ctx: StepContext) -> None:
    meta = json.loads(ctx.input_path("probe", "meta.json").read_text(encoding="utf-8"))
    ctx.simulate(ctx.params.get("duration", 0.5), msg="splitting frames")
    manifest = {"frames": meta["frames"], "digest": ctx.input_digest()}
    ctx.output("frames.json", canonical_json(manifest).encode())


@register("extract_audio")
def extract_audio(ctx: StepContext) -> None:
    ctx.simulate(ctx.params.get("duration", 0.4), msg="extracting audio")
    ctx.output("audio.bin", det_bytes("audio", ctx.params, ctx.input_digest()))


@register("assemble")
def assemble(ctx: StepContext) -> None:
    ctx.simulate(ctx.params.get("duration", 0.3), msg="assembling video")
    buf = bytearray()
    for dep in sorted(ctx.inputs):
        for name in sorted(ctx.inputs[dep]):
            buf += ctx.input_path(dep, name).read_bytes()
    ctx.output("video.bin", det_bytes("assemble", ctx.input_digest(), len(buf)) + bytes(buf[:64]))


@register("mux")
def mux(ctx: StepContext) -> None:
    ctx.simulate(ctx.params.get("duration", 0.2), msg="muxing")
    ctx.output("final.bin", det_bytes("mux", ctx.input_digest()))


@register("quality_check")
def quality_check(ctx: StepContext) -> None:
    ctx.simulate(ctx.params.get("duration", 0.2), msg="qc")
    if ctx.params.get("force_fail"):
        raise StepError(ErrorKind.PERMANENT, "质检未通过: 检测到角色边缘闪烁")
    report = {"ok": True, "digest": ctx.input_digest()}
    ctx.output("report.json", canonical_json(report).encode())


@register("publish")
def publish(ctx: StepContext) -> None:
    """有外部副作用的步骤:用幂等键保证"至少一次执行"不会产生重复发布。

    外部"素材库"用 data/published/<key>.json 模拟,O_EXCL 创建 = 对端的幂等去重。
    """
    ctx.progress(0.0, "publishing")
    # 默认读 mux/final.bin(角色替换模板);序列帧模板通过参数指向 assemble/final.mp4
    final_sha = ctx.input_sha(ctx.params.get("final_dep", "mux"), ctx.params.get("final_name", "final.bin"))
    target = ctx.params.get("target", "asset-library")
    pub_key = _sha(f"{final_sha}|{target}".encode())
    reg_dir = ctx.data_dir / "published"
    reg_dir.mkdir(parents=True, exist_ok=True)
    reg = reg_dir / f"{pub_key}.json"
    ctx.sleep(ctx.params.get("duration", 0.2) / 2)
    ctx.progress(0.5, "calling asset library")
    try:
        fd = os.open(str(reg), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"key": pub_key, "target": target, "final": final_sha, "by_task": ctx.task_id}, f)
        ctx.note("publish", "created")
    except FileExistsError:
        ctx.note("publish", "deduplicated")      # 之前的尝试已经发布过,不重复发布
    ctx.sleep(ctx.params.get("duration", 0.2) / 2)
    ctx.progress(1.0, "published")
    ctx.output("receipt.json", canonical_json({"publish_key": pub_key, "target": target}).encode())


# --------------------------------------------------------------------------- GPU 步骤
@register("gen_reference")
def gen_reference(ctx: StepContext) -> None:
    """按描述生成新角色参考图。nondeterministic=True 时每次执行结果都不同(模拟生成式模型)。"""
    if ctx.params.get("oom_until_attempt") and ctx.attempt <= ctx.params["oom_until_attempt"]:
        ctx.progress(0.0, "loading model")
        raise StepError(ErrorKind.RESOURCE, "CUDA out of memory (mock)")
    ctx.simulate(ctx.params.get("duration", 0.8), msg="generating reference")
    nonce = uuid.uuid4().hex if ctx.params.get("nondeterministic") else ""
    ctx.output("ref.bin", det_bytes("ref", ctx.params.get("prompt"), ctx.params.get("seed", 0), nonce))


def _chunked(ctx: StepContext, label: str, out_name: str) -> None:
    """按"帧"逐个处理的长步骤,带步骤内检查点。

    - 每帧结果原子写入 work_dir(跨 attempt 保留),检查点只记录"已完成帧数 + 各帧 sha"。
    - 恢复时先**校验**已落盘帧的 sha,不信任检查点本身;校验失败则从该帧起重算。
    """
    frames = int(ctx.params.get("frames", 6))
    unit = float(ctx.params.get("unit_s", 0.05))
    digest = ctx.input_digest()
    work = ctx.work_dir
    work.mkdir(parents=True, exist_ok=True)

    shas: list[str] = []
    ck = ctx.checkpoint or {}
    for i, sha in enumerate(ck.get("shas", [])):
        p = work / f"{label}_{i}.bin"
        if p.is_file() and _sha(p.read_bytes()) == sha:
            shas.append(sha)
        else:
            break
    resumed_from = len(shas)
    ctx.note("resumed_from", resumed_from)
    ctx.progress(resumed_from / frames, f"{label} start at frame {resumed_from}")

    for i in range(resumed_from, frames):
        if ctx.params.get("oom_until_attempt") and ctx.attempt <= ctx.params["oom_until_attempt"]:
            raise StepError(ErrorKind.RESOURCE, "CUDA out of memory (mock)")
        ctx.sleep(unit)
        data = det_bytes(label, ctx.params.get("chunk"), digest, i, size=128)
        _atomic_write(work / f"{label}_{i}.bin", data)
        shas.append(_sha(data))
        ctx.save_checkpoint({"done": len(shas), "shas": shas})
        ctx.progress((i + 1) / frames, f"{label} frame {i + 1}/{frames}")
    ctx.note("frames_computed", frames - resumed_from)

    buf = b"".join((work / f"{label}_{i}.bin").read_bytes() for i in range(frames))
    ctx.output(out_name, buf)


@register("segment")
def segment(ctx: StepContext) -> None:
    """抠出原角色(GPU,分块)。"""
    _chunked(ctx, "seg", "mask.bin")


@register("swap")
def swap(ctx: StepContext) -> None:
    """用新角色参考图替换(GPU,分块)。"""
    _chunked(ctx, "swap", "swapped.bin")
