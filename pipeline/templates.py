"""动画素材"角色替换"任务模板:把业务参数展开为静态 DAG。

                         ┌─► extract_audio ───────────────────────────┐
probe ─► extract_frames ─┼─► segment_0..N-1 (gpu) ─┐                  ▼
   │                     │                         ├─► swap_0..N-1 ─► assemble ─► mux ─► quality_check ─► [human_approval] ─► publish
   └─► gen_reference (gpu, 生成式,结果必须固化) ───┘
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import RetryPolicy, StepSpec, TaskSpec, validate_spec


def animation_swap_task(
    name: str = "character-swap", *,
    source: str = "demo.mp4",
    frames: int = 24,
    chunks: int = 4,
    prompt: str = "red-haired girl",
    target: str = "asset-library",
    require_approval: bool = False,
    qc_fail: bool = False,
    nondeterministic_ref: bool = False,
    unit_s: float = 0.05,
    retry: RetryPolicy | None = None,
    timeout_s: float = 600.0,
    stall_timeout_s: float = 30.0,
    chaos: dict[str, dict[str, Any]] | None = None,
    extra_params: dict[str, dict[str, Any]] | None = None,
) -> TaskSpec:
    """chaos / extra_params: {step_key: {...}},分别用于故障注入与覆盖步骤参数。"""
    if chunks < 1 or frames < chunks:
        raise ValueError("需要 chunks>=1 且 frames>=chunks")
    retry = retry or RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=30.0, jitter=0.2)
    chaos = chaos or {}
    extra = extra_params or {}
    per_chunk = frames // chunks

    def mk(key: str, type_: str, deps: list[str], params: dict, resource: str = "cpu",
           cacheable: bool = True) -> StepSpec:
        p = {**params, **extra.get(key, {})}
        return StepSpec(key=key, type=type_, deps=deps, params=p, resource=resource, retry=retry,
                        timeout_s=timeout_s, stall_timeout_s=stall_timeout_s,
                        cacheable=cacheable, chaos=chaos.get(key, {}))

    steps = [
        mk("probe", "probe", [], {"source": source, "frames": frames}),
        mk("extract_frames", "extract_frames", ["probe"], {}),
        mk("extract_audio", "extract_audio", ["probe"], {}),
        mk("gen_reference", "gen_reference", ["probe"],
           {"prompt": prompt, "seed": 0, "nondeterministic": nondeterministic_ref}, "gpu"),
    ]
    for i in range(chunks):
        steps.append(mk(f"segment_{i}", "segment", ["extract_frames"],
                        {"chunk": i, "frames": per_chunk, "unit_s": unit_s}, "gpu"))
    for i in range(chunks):
        steps.append(mk(f"swap_{i}", "swap", [f"segment_{i}", "gen_reference"],
                        {"chunk": i, "frames": per_chunk, "unit_s": unit_s}, "gpu"))
    steps.append(mk("assemble", "assemble", [f"swap_{i}" for i in range(chunks)], {}))
    steps.append(mk("mux", "mux", ["assemble", "extract_audio"], {}))
    steps.append(mk("quality_check", "quality_check", ["mux"], {"force_fail": qc_fail}))
    pub_deps = ["mux", "quality_check"]
    if require_approval:
        steps.append(mk("human_approval", "human_approval", ["quality_check"], {}, "none"))
        pub_deps.append("human_approval")
    # publish 有外部副作用:不走步骤缓存,靠对端幂等键去重
    steps.append(mk("publish", "publish", pub_deps, {"target": target}, cacheable=False))

    spec = TaskSpec(name=name, steps=steps,
                    meta={"template": "animation_swap", "frames": frames, "chunks": chunks})
    validate_spec(spec)
    return spec


# 故障注入预设(CLI 与 API 共用):每个预设只作用于一个步骤,首次尝试触发
FRAME_INJECT_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "transient": {"matte_0": {"fail": {"kind": "TRANSIENT", "until_attempt": 2, "at": 0.3}}},
    "crash": {"matte_1": {"crash": {"at": 0.5, "on_attempts": [1]}}},
    "hang": {"upscale_0": {"hang": {"at": 0.5, "on_attempts": [1]}}},
}


def frame_pipeline_task(
    name: str = "frame-pipeline", *,
    source: str,
    chunks: int = 4,
    scale: int = 2,
    target: str = "asset-library",
    require_approval: bool = False,
    qc_fail: bool = False,
    retry: RetryPolicy | None = None,
    timeout_s: float = 1800.0,
    stall_timeout_s: float = 60.0,
    chaos: dict[str, dict[str, Any]] | None = None,
    extra_params: dict[str, dict[str, Any]] | None = None,
) -> TaskSpec:
    """序列帧处理任务(真实处理):拆帧 → 抠背景 → 超分辨率 → 补帧 → 合成视频。

    probe ─► extract_frames ─► bg_plate ─► matte_i ─► upscale_i ─┬─► interp_i ─┐
                  │                                  upscale_{i+1}┘  (块边界)    ├─► assemble ─► quality_check ─► [human_approval] ─► publish
                  └──────────────────────────────────────────────────────────────┘

    - 分块数 chunks 在提交时固定;每块多少帧由 fp_extract 实际解出的帧数决定;
    - 提交时计算素材文件的 sha256 写入 probe 参数:素材被换掉/改动 → 缓存 key 变化,不会误用旧结果;
    - 全部是 CPU 步骤(resource="cpu"),不依赖 GPU 与第三方 API。
    """
    from .steps.frames import _sha_file       # 延迟导入:避免模板模块在导入时加载 numpy/Pillow
    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise ValueError(f"素材文件不存在: {source}")
    if not (1 <= chunks <= 32):
        raise ValueError("chunks 需要在 1..32")
    if scale != 2:
        raise ValueError("目前只支持 scale=2")
    retry = retry or RetryPolicy(max_attempts=3, base_delay=1.0, max_delay=30.0, jitter=0.2)
    chaos = chaos or {}
    extra = extra_params or {}

    def mk(key: str, type_: str, deps: list[str], params: dict, cacheable: bool = True,
           resource: str = "cpu") -> StepSpec:
        p = {**params, **extra.get(key, {})}
        return StepSpec(key=key, type=type_, deps=deps, params=p, resource=resource, retry=retry,
                        timeout_s=timeout_s, stall_timeout_s=stall_timeout_s,
                        cacheable=cacheable, chaos=chaos.get(key, {}))

    steps = [
        mk("probe", "fp_probe", [], {"source": str(src), "source_sha256": _sha_file(src)}),
        mk("extract_frames", "fp_extract", ["probe"], {"source": str(src), "chunks": chunks}),
        mk("bg_plate", "fp_plate", ["extract_frames"], {}),
    ]
    for i in range(chunks):
        steps.append(mk(f"matte_{i}", "fp_matte", ["extract_frames", "bg_plate"], {"chunk": i}))
    for i in range(chunks):
        steps.append(mk(f"upscale_{i}", "fp_upscale", [f"matte_{i}"], {"chunk": i, "scale": scale}))
    for i in range(chunks):
        deps = ["extract_frames", f"upscale_{i}"] + ([f"upscale_{i + 1}"] if i + 1 < chunks else [])
        steps.append(mk(f"interp_{i}", "fp_interp", deps, {"chunk": i, "chunks": chunks}))
    steps.append(mk("assemble", "fp_assemble", ["extract_frames"] + [f"interp_{i}" for i in range(chunks)],
                    {"chunks": chunks, "scale": scale}))
    steps.append(mk("quality_check", "fp_qc", ["extract_frames", "assemble"],
                    {"scale": scale, "force_fail": qc_fail}))
    pub_deps = ["assemble", "quality_check"]
    if require_approval:
        steps.append(mk("human_approval", "human_approval", ["quality_check"], {}, resource="none"))
        pub_deps.append("human_approval")
    steps.append(mk("publish", "publish", pub_deps,
                    {"target": target, "final_dep": "assemble", "final_name": "final.mp4"}, cacheable=False))

    spec = TaskSpec(name=name, steps=steps,
                    meta={"template": "frame_pipeline", "source": src.name, "chunks": chunks, "scale": scale})
    validate_spec(spec)
    return spec
