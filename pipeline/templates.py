"""任务模板:把业务参数展开为静态 DAG。

两个模板:
  - frame_style_task:**本项目的主流水线**(6 道工序,豆包逐帧风格化),定义在本文件末尾;
  - animation_swap_task:全 MOCK 的"角色替换"DAG(sleep + 确定性字节)。它不是业务功能,
    而是**引擎测试夹具**:故障矩阵测试(崩溃/卡死/租约丢失/并发抢占/缓存)需要一个步骤多、
    耗时可控、产物确定的 DAG,用真实 ffmpeg 流水线跑这些测试既慢又不稳定。

animation_swap_task 的 DAG:

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


# ============================================================================ 主流水线:豆包逐帧风格化
#
#   ingest ─► keyframes ─► stylize_0..N-1 ─► poststats ─► post_0..N-1 ─► qc ─► mux ─► [human_approval] ─► package ─► publish
#      │         ▲  └──────────────────────────────────────────▲              ▲        ▲
#      └─────────┴───── (reference 可选:锚点/人物参考图) ──────┘              │        └ ingest 的音轨
#                                                                             └ post_i 依赖 stylize_i 与 stylize_{i+1}(块边界补间)
#
# 6 道工序对应:ingest=1 keyframes=2 stylize=3 post*=4 mux=5 package=6,另有质量校验 qc 与可选的人工审批。

DEFAULT_PRICE_CNY = 0.12          # 方舟 Seedream 5.0 flash 的标价(元/张);以控制台为准,可用 price_cny 覆盖
MEDIA_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def frame_style_inject_presets(chunks: int) -> dict[str, dict[str, dict[str, Any]]]:
    """故障注入预设(MockDoubao + 引擎级):每个预设只作用于一个步骤。

    "__first__" 代表"该块的第一个关键帧":提交时还不知道关键帧序号,由步骤在运行时解析。
    """
    return {
        "transient": {"stylize_0": {"doubao": {"frames": {"__first__": {"kind": "transient", "first_calls": 2}}}}},
        "crash": {"stylize_0": {"crash": {"at": 0.5, "on_attempts": [1]}}},
        "moderation": {"stylize_0": {"doubao": {"frames": {"__first__": {"kind": "moderation"}}}}},
        "response_lost": {"stylize_0": {"doubao": {"frames": {"__first__": {"kind": "timeout_after_success",
                                                                          "first_calls": 1}}}}},
        # 第一次尝试里这一帧每次调用都是 429(用完帧级重试机会);引擎重试后恢复正常
        "rate_limit": {"stylize_0": {"doubao": {"frames": {"__first__": {"kind": "rate_limit", "until_attempt": 1}}}}},
    }


def frame_style_task(
    name: str = "anime-style", *,
    source: str,
    prompt: str,
    provider: str = "mock",
    model: str | None = None,
    reference: str | None = None,
    chunks: int = 4,
    stride: int = 6,
    max_gap: int = 24,
    dup_thr: float = 0.8,
    cut_thr: float = 18.0,
    target_height: int = 1080,
    fps: int = 24,
    size: str = "2K",
    watermark: bool = False,
    price_cny: float | None = None,
    budget_cny: float = 50.0,
    concurrency: int = 4,
    frame_max_attempts: int = 3,
    on_reject: str = "passthrough",
    max_degraded_ratio: float = 0.1,
    frame_overrides: dict[str, dict[str, str]] | None = None,
    rpm: float = 300.0,
    backoff_base_s: float = 2.0,
    color_strength: float = 0.6,
    denoise: float = 0.3,
    max_flicker_ratio: float = 3.0,
    qc_strict: bool = False,
    require_approval: bool = False,
    target: str = "asset-library",
    retry: RetryPolicy | None = None,
    stylize_timeout_s: float = 3600.0,
    stall_timeout_s: float = 180.0,
    chaos: dict[str, dict[str, Any]] | None = None,
) -> TaskSpec:
    """提交一个"素材 → 逐帧风格化 → 成品视频"任务。

    - 素材与参考图的 sha256 在提交时写进参数:提交后文件被改动会被拒绝,不会拿错的文件去花钱;
    - provider="mock" 用离线替身(测试/演示,不花钱);provider="ark" 调用豆包,密钥只从环境变量
      ARK_API_KEY 读取,**不会**进入任务参数或数据库;
    - 工序 3(stylize)用 resource="gpu":在没有本地 GPU 的情况下,它代表"外部算力并发槽位"——
      只有声明了 gpu 能力的 worker 才会领取,用来限制同时发起的接口调用批次;
    - publish 有外部副作用,不走步骤缓存,靠对端幂等键去重。
    """
    from .steps.anime import _sha_file
    src = Path(source).expanduser().resolve()
    if not src.is_file():
        raise ValueError(f"素材文件不存在: {source}")
    if src.suffix.lower() not in MEDIA_EXTS:
        raise ValueError(f"素材格式需要是 {sorted(MEDIA_EXTS)} 之一")
    if not prompt or len(prompt) > 2000:
        raise ValueError("prompt 不能为空,且不超过 2000 字符")
    if provider not in ("mock", "ark", "ark_plan"):
        raise ValueError("provider 只能是 mock、ark 或 ark_plan")
    if provider != "mock" and not model:
        raise ValueError(f"provider={provider} 需要指定 model(方舟控制台里的模型 ID 或 Endpoint ID)")
    if not (1 <= chunks <= 32):
        raise ValueError("chunks 需要在 1..32")
    if stride < 1 or max_gap < stride:
        raise ValueError("需要 1 <= stride <= max_gap")
    if on_reject not in ("passthrough", "fail"):
        raise ValueError("on_reject 只能是 passthrough 或 fail")
    if not (1 <= concurrency <= 16):
        raise ValueError("concurrency 需要在 1..16")
    if budget_cny <= 0:
        raise ValueError("必须设置预算上限 budget_cny > 0(用于提交前预估与运行中的保护线)")
    price = DEFAULT_PRICE_CNY if price_cny is None else float(price_cny)
    retry = retry or RetryPolicy(max_attempts=3, base_delay=2.0, max_delay=60.0, jitter=0.2)
    chaos = chaos or {}
    ref_info = None
    if reference:
        rp = Path(reference).expanduser().resolve()
        if not rp.is_file():
            raise ValueError(f"参考图不存在: {reference}")
        ref_info = {"path": str(rp), "sha256": _sha_file(rp)}

    def mk(key: str, type_: str, deps: list[str], params: dict, *, cacheable: bool = True, resource: str = "cpu",
           timeout_s: float = 1800.0, stall: float | None = None) -> StepSpec:
        return StepSpec(key=key, type=type_, deps=deps, params=params, resource=resource, retry=retry,
                        timeout_s=timeout_s, stall_timeout_s=stall or stall_timeout_s,
                        cacheable=cacheable, chaos=chaos.get(key, {}))

    summary = {"prompt": prompt, "provider": provider, "model": model, "size": size, "watermark": watermark,
               "price_cny": price, "budget_cny": budget_cny, "has_reference": bool(ref_info),
               "frame_overrides": sorted((frame_overrides or {}).keys())}
    steps = [
        mk("ingest", "an_ingest", [], {"source": str(src), "source_sha256": _sha_file(src),
                                       "target_height": target_height, "fps": fps}),
        mk("keyframes", "an_keyframes", ["ingest"],
           {"chunks": chunks, "stride": stride, "max_gap": max_gap, "dup_thr": dup_thr, "cut_thr": cut_thr,
            "price_cny": price, "budget_cny": budget_cny}),
    ]
    style_deps_extra: list[str] = []
    if ref_info:
        steps.append(mk("reference", "an_reference", [], ref_info))
        style_deps_extra = ["reference"]
    for i in range(chunks):
        steps.append(mk(
            f"stylize_{i}", "an_stylize", ["keyframes", *style_deps_extra],
            {"chunk": i, "prompt": prompt, "provider": provider, "model": model, "size": size,
             "watermark": watermark, "price_cny": price, "budget_cny": budget_cny, "concurrency": concurrency,
             "frame_max_attempts": frame_max_attempts, "on_reject": on_reject,
             "max_degraded_ratio": max_degraded_ratio, "frame_overrides": frame_overrides or {},
             "rpm": rpm, "backoff_base_s": backoff_base_s},
            resource="gpu", timeout_s=stylize_timeout_s))
    styl = [f"stylize_{i}" for i in range(chunks)]
    steps.append(mk("poststats", "an_poststats", ["keyframes", *styl], {}))
    for i in range(chunks):
        deps = ["keyframes", "poststats", f"stylize_{i}"] + ([f"stylize_{i + 1}"] if i + 1 < chunks else [])
        steps.append(mk(f"post_{i}", "an_post", deps,
                        {"chunk": i, "color_strength": color_strength, "denoise": denoise}))
    posts = [f"post_{i}" for i in range(chunks)]
    steps.append(mk("qc", "an_qc", ["ingest", "keyframes", *styl, *posts],
                    {"max_flicker_ratio": max_flicker_ratio, "qc_strict": qc_strict}))
    steps.append(mk("mux", "an_mux", ["ingest", "keyframes", *posts, "qc"], {}))
    pkg_deps = ["ingest", "keyframes", *styl, "qc", "mux"]
    if require_approval:
        steps.append(mk("human_approval", "human_approval", ["mux"], {}, resource="none"))
        pkg_deps.append("human_approval")
    steps.append(mk("package", "an_package", pkg_deps, {"chunks": chunks, "summary": summary}))
    steps.append(mk("publish", "publish", ["package"],
                    {"target": target, "final_dep": "package", "final_name": "final.mp4"}, cacheable=False))
    spec = TaskSpec(name=name, steps=steps,
                    meta={"template": "frame_style", "source": src.name, "chunks": chunks, "provider": provider,
                          "model": model, "prompt": prompt, "budget_cny": budget_cny, "price_cny": price})
    validate_spec(spec)
    return spec
