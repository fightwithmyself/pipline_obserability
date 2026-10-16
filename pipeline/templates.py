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


def kling_video_task(
    name: str = "kling-video", *,
    prompt: str,
    mode: str = "std",
    duration: int = 3,
    audio: bool = False,
    aspect_ratio: str = "16:9",
    poll_interval_s: float = 15.0,
    max_wait_s: float = 900.0,
    retry: RetryPolicy | None = None,
    chaos: dict[str, dict[str, Any]] | None = None,
) -> TaskSpec:
    """单步任务:文生视频(可灵 omni,百炼托管)。默认参数是最便宜的组合:720P、无声、3 秒。

    费用按"生成秒数 × 档位单价"计,所以 duration 在提交时就校验;提交的是 prompt 与参数,
    **密钥不进任务参数**(只从 worker 的环境变量读取)。
    cacheable=False:相同参数再提交一次就应该再生成一次(也再付一次钱),不能被缓存悄悄吞掉
    —— 这与"确定性本地计算可以缓存"相反。
    """
    if not prompt or len(prompt) > 2500:
        raise ValueError("prompt 不能为空,且不超过 2500 字符")
    if mode not in ("std", "pro", "4k"):
        raise ValueError("mode 只能是 std(720P) / pro(1080P) / 4k")
    if not (3 <= duration <= 15):
        raise ValueError("duration 需要在 3..15 秒")
    retry = retry or RetryPolicy(max_attempts=3, base_delay=2.0, max_delay=30.0, jitter=0.2)
    step = StepSpec(
        key="generate", type="kling_video", deps=[], resource="cpu", retry=retry,
        params={"prompt": prompt, "mode": mode, "duration": duration, "audio": audio,
                "aspect_ratio": aspect_ratio, "poll_interval_s": poll_interval_s, "max_wait_s": max_wait_s},
        timeout_s=max_wait_s + 300, stall_timeout_s=max(60.0, poll_interval_s * 4),
        cacheable=False, chaos=(chaos or {}).get("generate", {}))
    spec = TaskSpec(name=name, steps=[step], meta={"template": "kling_video", "mode": mode, "duration": duration})
    validate_spec(spec)
    return spec


REVIEW_PROMPT = (
    "请对比图 1(原视频帧)和图 2(风格重绘后的帧),只输出一个 JSON 对象,不要其他文字。字段:"
    '{"is_japanese_anime_style": 图2是否呈现日系动漫风格(true/false),'
    '"character_preserved": 图2是否保留了图1中的主要人物与动作(true/false),'
    '"score": 整体风格转换质量 1-5 的整数,'
    '"issues": 发现的问题字符串数组(没有则为空数组),'
    '"summary": 一句话中文总结}'
)


def style_transform_task(
    name: str = "style-transform", *,
    source: str,
    style: int = 0,
    min_len: int = 540,
    video_fps: int = 15,
    use_sr: bool = False,
    animate_emotion: bool = True,
    review_model: str = "qwen3.8-omni-flash",
    poll_interval_s: float = 10.0,
    max_wait_s: float = 900.0,
    retry: RetryPolicy | None = None,
    chaos: dict[str, dict[str, Any]] | None = None,
) -> TaskSpec:
    """视频风格重绘任务链(百炼 video-style-transform):

        probe ─► upload ─► style ─► quality_check ─► review
        素材检测   临时存储   风格重绘(按秒计费)  结构复核+对比图  多模态评审(按 token 计费)

    - 素材文件的 sha256 在提交时写进 probe 参数:提交后素材被改动会被拒绝,不会拿错的文件去花钱;
    - upload / style / review 都是 cacheable=False:对外部世界有副作用或有费用,不能被缓存悄悄吞掉
      (oss:// 临时地址 48 小时过期,缓存了反而会让下游拿到失效地址);
    - 用量(秒或 token)写在各步骤的 attempt.notes["usage"],在 Dashboard 步骤详情中查看。
    """
    from .steps.vst import ALLOWED_EXT, STYLE_NAMES, _sha_file
    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise ValueError(f"素材文件不存在: {source}")
    if src.suffix.lower() not in ALLOWED_EXT:
        raise ValueError(f"素材格式需要是 {sorted(ALLOWED_EXT)} 之一")
    if style not in STYLE_NAMES:
        raise ValueError(f"style 需要在 {sorted(STYLE_NAMES)} 内")
    if min_len not in (540, 720):
        raise ValueError("min_len 只能是 540 或 720(影响计费:720P 0.5 元/秒,540P 0.2 元/秒)")
    if not 15 <= video_fps <= 25:
        raise ValueError("video_fps 需要在 15..25")
    retry = retry or RetryPolicy(max_attempts=3, base_delay=2.0, max_delay=30.0, jitter=0.2)
    chaos = chaos or {}

    def mk(key: str, type_: str, deps: list[str], params: dict, *, cacheable: bool, timeout_s: float = 300.0,
           stall: float = 60.0) -> StepSpec:
        return StepSpec(key=key, type=type_, deps=deps, params=params, resource="cpu", retry=retry,
                        timeout_s=timeout_s, stall_timeout_s=stall, cacheable=cacheable, chaos=chaos.get(key, {}))

    style_params = {"style": style, "video_fps": video_fps, "min_len": min_len, "use_SR": use_sr,
                    "animate_emotion": animate_emotion, "poll_interval_s": poll_interval_s, "max_wait_s": max_wait_s}
    steps = [
        mk("probe", "vst_probe", [], {"source": str(src), "source_sha256": _sha_file(src)}, cacheable=True),
        mk("upload", "vst_upload", ["probe"], {"source": str(src)}, cacheable=False, timeout_s=600.0),
        mk("style", "vst_style", ["upload"], style_params, cacheable=False, timeout_s=max_wait_s + 300,
           stall=max(60.0, poll_interval_s * 4)),
        mk("quality_check", "vst_qc", ["probe", "style"],
           {"source": str(src), "min_len": min_len, "use_SR": use_sr}, cacheable=True),
        mk("review", "vst_review", ["quality_check"], {"model": review_model, "prompt": REVIEW_PROMPT,
                                                       "thinking": False}, cacheable=False),
    ]
    spec = TaskSpec(name=name, steps=steps, meta={
        "template": "style_transform", "source": src.name, "style": style, "style_name": STYLE_NAMES[style],
        "min_len": min_len})
    validate_spec(spec)
    return spec


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
