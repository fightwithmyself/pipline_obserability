"""第三方视频生成 API 步骤:可灵 kling/kling-v3-omni-video-generation(阿里云百炼托管,异步接口)。

异步提交/轮询/断点沿用的通用逻辑见 dashscope.py(提交一次、task_id 进检查点、逐轮观测、永久错误不重试)。

关于"token":这个模型**不按 token 计费,按生成视频的秒数计费**,响应里只有 usage.duration(秒)等字段,
没有 token 数。所以观测记录里的计费用量是"秒",费用是按文档标价估算的,真实金额以账单为准。
"""
from __future__ import annotations

from ..models import ErrorKind
from ..stepkit import StepContext, StepError
from . import register
from .dashscope import download, submit_and_poll

# 文档标价(元/秒,北京地域,omni、无参考视频),取自 help.aliyun.com 模型页;有活动优惠时实际更低。
PRICE_CNY_PER_S = {
    ("std", True): 0.9, ("pro", True): 1.2, ("4k", True): 3.0,
    ("std", False): 0.6, ("pro", False): 0.8, ("4k", False): 3.0,
}


def estimate_cost_cny(usage: dict, mode: str) -> float | None:
    price = PRICE_CNY_PER_S.get((mode, bool(usage.get("audio"))))
    if price is None or not isinstance(usage.get("duration"), (int, float)):
        return None
    return round(price * usage["duration"], 4)


def _request_body(p: dict) -> dict:
    prm = {"mode": p.get("mode", "std"), "aspect_ratio": p.get("aspect_ratio", "16:9"),
           "duration": int(p.get("duration", 3)), "audio": bool(p.get("audio", False)),
           "watermark": bool(p.get("watermark", False))}
    return {"model": p.get("model", "kling/kling-v3-omni-video-generation"),
            "input": {"prompt": p["prompt"]}, "parameters": prm}


@register("kling_video")
def kling_video(ctx: StepContext) -> None:
    p = ctx.params
    mode = p.get("mode", "std")
    ctx.note("model", p.get("model", "kling/kling-v3-omni-video-generation"))
    ctx.note("billing", "按秒计费(usage.duration),无 token 数;费用为文档标价估算")

    final = submit_and_poll(ctx, _request_body(p), poll_s=float(p.get("poll_interval_s", 15.0)),
                            max_wait_s=float(p.get("max_wait_s", 900.0)))

    usage = final.get("usage") or {}
    ctx.note("usage", usage)
    ctx.note("cost_cny_est", estimate_cost_cny(usage, mode))
    url = (final.get("output") or {}).get("video_url")
    if not url:
        raise StepError(ErrorKind.PERMANENT, "任务成功但响应里没有 video_url")
    ctx.progress(0.92, "下载视频")
    dest = ctx.output_path("video.mp4")
    sha, size = download(url, dest)
    ctx.note("video_bytes", size)
    ctx.note("video_sha256", sha)
    try:                                           # 用 ffmpeg 独立复核:真的是视频、分辨率/时长与 usage 一致
        from ..media import probe_video
        info = probe_video(dest)
        ctx.note("video_info", {k: info[k] for k in ("width", "height", "fps", "duration_s", "has_audio")})
    except Exception as e:                         # 复核失败不推翻已成功的结果,只记录
        ctx.note("video_info_error", f"{type(e).__name__}: {e}")
    ctx.progress(1.0, "完成")
