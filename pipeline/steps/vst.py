"""视频风格重绘任务链(百炼 video-style-transform + 通义多模态评审):

    vst_probe ─► vst_upload ─► vst_style ─► vst_qc ─► vst_review
    素材检测       传到临时存储    风格重绘(付费)   结构复核+对比图    多模态模型评审(有 token)

每一步的输入、输出、用量都落在步骤详情里(不另建 LLM 调用追踪表):
  - 输入/输出:引擎里的内容寻址产物(JSON / 图片 / 视频),Dashboard 点开步骤即可预览
  - 用量:写在 attempt.notes["usage"]。风格重绘按**输出秒数**计费(没有 token);评审步骤是真实的 **token** 用量
  - 外部任务的每一轮轮询:attempt.notes["poll_001"…]

密钥只从环境变量 DASHSCOPE_API_KEY 读取。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from ..models import ErrorKind
from ..stepkit import StepContext, StepError
from . import register
from .dashscope import api_key, download, submit_and_poll, upload_temp_file

STYLE_MODEL = "video-style-transform"
STYLE_NAMES = {0: "日式漫画", 1: "美式漫画", 2: "清新漫画", 3: "3D卡通", 4: "国风卡通", 5: "纸艺风格",
               6: "简易插画", 7: "国风水墨"}
PRICE_CNY_PER_S = {720: 0.5, 540: 0.2}                 # 文档标价;百炼另有 600 秒免费额度
INPUT_LIMITS = {"max_duration_s": 30.0, "max_size_mb": 100.0, "min_side": 256, "max_side": 4096, "max_ratio": 1.8}
ALLOWED_EXT = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".ts", ".mpg", ".mxf"}
COMPAT_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _json_out(ctx: StepContext, name: str, obj) -> None:
    ctx.output(name, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))


# --------------------------------------------------------------------------- 1. 素材检测
@register("vst_probe")
def vst_probe(ctx: StepContext) -> None:
    """检查素材是否满足风格重绘接口的输入要求。不满足就在这里永久失败,不去上传、不花钱。"""
    from ..media import MediaError, probe_video
    p = ctx.params
    src = Path(p["source"])
    ctx.progress(0.1, "读取素材")
    if not src.is_file():
        raise StepError(ErrorKind.PERMANENT, f"素材文件不存在: {src}")
    if _sha_file(src) != p["source_sha256"]:
        raise StepError(ErrorKind.PERMANENT, "素材文件在提交后被修改(sha256 与提交时不一致),请重新提交")
    try:
        info = probe_video(src)
    except MediaError as e:
        raise StepError(ErrorKind.PERMANENT, f"素材不是可解码的视频: {e}") from None
    L = INPUT_LIMITS
    w, h = info["width"], info["height"]
    ratio = round(max(w, h) / max(1, min(w, h)), 4)
    checks = [
        {"name": "格式", "value": src.suffix.lower(), "limit": sorted(ALLOWED_EXT), "ok": src.suffix.lower() in ALLOWED_EXT},
        {"name": "时长(秒)", "value": info["duration_s"], "limit": f"≤{L['max_duration_s']}", "ok": info["duration_s"] <= L["max_duration_s"]},
        {"name": "大小(MB)", "value": info["size_mb"], "limit": f"≤{L['max_size_mb']}", "ok": info["size_mb"] <= L["max_size_mb"]},
        {"name": "短边(px)", "value": min(w, h), "limit": f"≥{L['min_side']}", "ok": min(w, h) >= L["min_side"]},
        {"name": "长边(px)", "value": max(w, h), "limit": f"≤{L['max_side']}", "ok": max(w, h) <= L["max_side"]},
        {"name": "长短边比", "value": ratio, "limit": f"≤{L['max_ratio']}", "ok": ratio <= L["max_ratio"] + 1e-6},
    ]
    bad = [c for c in checks if not c["ok"]]
    _json_out(ctx, "probe.json", {"source": src.name, "sha256": p["source_sha256"], "info": info,
                                  "checks": checks, "accepted": not bad})
    ctx.note("usage", {"billing": "本地计算,无外部用量"})
    # 失败的步骤不会登记产物(只有成功才入库),所以检测报告同时写进 notes:失败时点开步骤也能看到哪几条不满足
    ctx.note("report", {"accepted": not bad, "checks": checks})
    if bad:
        raise StepError(ErrorKind.PERMANENT, "素材不满足风格重绘接口要求: " +
                        ";".join(f"{c['name']}={c['value']}(要求{c['limit']})" for c in bad))
    ctx.progress(1.0, "素材合格")


# --------------------------------------------------------------------------- 2. 上传到临时存储
@register("vst_upload")
def vst_upload(ctx: StepContext) -> None:
    """把本地视频传到百炼的免费临时存储,得到 48 小时有效的 oss:// 地址(接口只接受公网/临时 URL)。"""
    src = Path(ctx.params["source"])
    ctx.progress(0.1, f"上传 {src.name}")
    url, detail = upload_temp_file(src, STYLE_MODEL)
    _json_out(ctx, "upload.json", {"oss_url": url, "file": src.name, "uploaded_at": time.time(), **detail})
    ctx.note("usage", {"billing": "免费临时存储(48 小时)", **{k: detail[k] for k in ("bytes", "upload_s")}})
    ctx.progress(1.0, "上传完成")


# --------------------------------------------------------------------------- 3. 风格重绘(付费)
@register("vst_style")
def vst_style(ctx: StepContext) -> None:
    p = ctx.params
    up = json.loads(ctx.input_path("upload", "upload.json").read_text(encoding="utf-8"))
    params = {"style": int(p.get("style", 0)), "video_fps": int(p.get("video_fps", 15)),
              "min_len": int(p.get("min_len", 540)), "use_SR": bool(p.get("use_SR", False)),
              "animate_emotion": bool(p.get("animate_emotion", True))}
    body = {"model": STYLE_MODEL, "input": {"video_url": up["oss_url"]}, "parameters": params}
    headers = {"X-DashScope-OssResourceResolve": "enable"} if up["oss_url"].startswith("oss://") else None
    ctx.note("request", body)
    ctx.note("style_name", STYLE_NAMES.get(params["style"]))

    final = submit_and_poll(ctx, body, poll_s=float(p.get("poll_interval_s", 10.0)),
                            max_wait_s=float(p.get("max_wait_s", 900.0)), submit_headers=headers)

    usage = final.get("usage") or {}
    price = PRICE_CNY_PER_S.get(int(usage.get("SR", params["min_len"])))
    dur = usage.get("duration")
    ctx.note("usage", {**usage, "billing": "按输出秒数计费,无 token", "unit_price_cny_per_s": price})
    ctx.note("cost_cny_est", round(price * dur, 4) if price and isinstance(dur, (int, float)) else None)
    ctx.note("cost_note", "按文档标价估算;该模型另有 600 秒免费额度,额度内实际不扣费")
    url = (final.get("output") or {}).get("output_video_url")
    if not url:
        raise StepError(ErrorKind.PERMANENT, "任务成功但响应里没有 output_video_url")
    ctx.progress(0.92, "下载结果视频")
    sha, size = download(url, ctx.output_path("styled.mp4"))
    ctx.note("video_bytes", size)
    try:       # 独立量一遍结果视频,和接口上报的计费时长并排记录(两者可能不一致,实测过 4.64s vs 7.47s)
        from ..media import probe_video
        info = probe_video(ctx.out_dir / "styled.mp4")
        ctx.note("video_info", {k: info[k] for k in ("width", "height", "fps", "duration_s", "frames_est")})
        if isinstance(dur, (int, float)) and abs(info["duration_s"] - dur) > 0.6:
            ctx.note("usage_vs_actual_duration", {
                "usage_duration_s": dur, "actual_video_s": info["duration_s"],
                "warning": "接口上报的计费时长与视频实际时长不一致;费用以账单为准"})
    except Exception as e:
        ctx.note("video_info_error", f"{type(e).__name__}: {e}")
    ctx.progress(1.0, "完成")


# --------------------------------------------------------------------------- 4. 结构复核 + 对比图
def _frame_at(video: Path, t: float, dest: Path) -> None:
    from ..media import ffmpeg_exe
    r = subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(video),
                        "-frames:v", "1", str(dest)], capture_output=True)
    if r.returncode != 0 or not dest.is_file():
        raise StepError(ErrorKind.PERMANENT, f"无法从 {video.name} 抽取 {t:.2f}s 处的画面: "
                                             f"{r.stderr.decode('utf-8', 'replace')[:160]}")


@register("vst_qc")
def vst_qc(ctx: StepContext) -> None:
    """独立复核结果视频(不信任外部接口自报的 usage),并生成「原片 | 重绘」对比图供人和评审模型使用。"""
    from PIL import Image
    from ..media import MediaError, probe_video
    p = ctx.params
    out_path = ctx.input_path("style", "styled.mp4")
    src = Path(p["source"])
    min_len, use_sr = int(p.get("min_len", 540)), bool(p.get("use_SR", False))
    try:
        oi, si = probe_video(out_path), probe_video(src)
    except MediaError as e:
        raise StepError(ErrorKind.PERMANENT, f"结果视频无法解码: {e}") from None
    short = min(oi["width"], oi["height"])
    checks = [
        {"name": "结果可解码", "value": f"{oi['width']}x{oi['height']} {oi['codec']}", "ok": True},
        {"name": "时长与原片一致(±0.6s)", "value": f"{oi['duration_s']}s vs {si['duration_s']}s",
         "ok": abs(oi["duration_s"] - si["duration_s"]) <= 0.6},
        # 真实输出 960x536(目标 540):编码会把尺寸取整到 8 的倍数,所以容差给 8px。第一版我写成 ±2,把合格的结果误判为失败
        {"name": f"短边≈{min_len}px(±8)", "value": short,
         "ok": short >= min_len - 8 if use_sr else abs(short - min_len) <= 8},
        {"name": "帧率在 15~25", "value": oi["fps"], "ok": 14.5 <= oi["fps"] <= 25.5},
    ]
    ctx.progress(0.3, "抽取对比画面")
    tmp = ctx.out_dir / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    rows, diffs = [], []
    for j, frac in enumerate((0.2, 0.5, 0.8)):
        t_out = max(0.0, oi["duration_s"] * frac - 0.05)
        t_src = max(0.0, si["duration_s"] * frac - 0.05)
        a, b = tmp / f"s{j}.png", tmp / f"o{j}.png"
        _frame_at(src, t_src, a)
        _frame_at(out_path, t_out, b)
        ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
        ia_s = ia.resize(ib.size)
        diffs.append(_mean_abs_diff(ia_s, ib))
        rows.append((ia, ib))
    cw = 480
    sheet = Image.new("RGB", (cw * 2, sum(int(cw * ib.height / ib.width) for _, ib in rows)))
    y = 0
    for ia, ib in rows:
        h = int(cw * ib.height / ib.width)
        sheet.paste(ia.resize((cw, h)), (0, y))
        sheet.paste(ib.resize((cw, h)), (cw, y))
        y += h
    sheet.save(ctx.output_path("compare.png"))
    ia, ib = rows[1]                                                    # 中间帧交给评审模型:缩小并转 JPEG 省 token
    for im, name in ((ia, "src.jpg"), (ib, "out.jpg")):
        im = im.copy()
        im.thumbnail((896, 896))
        im.save(ctx.output_path(name), format="JPEG", quality=85)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    mean_diff = round(sum(diffs) / len(diffs), 2)
    warnings = []
    if mean_diff < 6:
        warnings.append(f"重绘前后画面平均差异仅 {mean_diff}/255,风格变化可能不明显(阈值 6 是我手工设的,未标定)")
    _json_out(ctx, "qc.json", {"checks": checks, "warnings": warnings, "mean_abs_diff_src_vs_out": mean_diff,
                               "source_info": si, "output_info": oi, "passed": all(c["ok"] for c in checks)})
    ctx.note("usage", {"billing": "本地计算,无外部用量"})
    bad = [c for c in checks if not c["ok"]]
    if bad:
        raise StepError(ErrorKind.PERMANENT, "结果视频未通过复核: " + ";".join(f"{c['name']}={c['value']}" for c in bad))
    ctx.progress(1.0, "复核通过")


def _mean_abs_diff(a, b) -> float:
    import numpy as np
    return float(np.abs(np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)).mean())


# --------------------------------------------------------------------------- 5. 多模态评审(真实 token)
def _extract_json(text: str) -> dict:
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        raise ValueError("找不到 JSON 对象")
    return json.loads(text[s:e + 1])


@register("vst_review")
def vst_review(ctx: StepContext) -> None:
    """让多模态模型对比「原片帧」与「重绘帧」,判断是否为日系动漫风格、角色是否保留。

    这是链路里唯一按 token 计费的步骤:prompt/completion/图片 token 都写进步骤详情的 usage。
    评审是**参考意见**:它给出结论但不会让任务失败(风格好不好不是工程上的成败)。
    """
    import openai
    p = ctx.params
    model = p.get("model", "qwen3.8-omni-flash")

    def data_url(dep_file: str) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(ctx.input_path("quality_check", dep_file).read_bytes()).decode()

    messages = [{"role": "user", "content": [
        {"type": "text", "text": "图 1 是原视频的一帧。"},
        {"type": "image_url", "image_url": {"url": data_url("src.jpg")}},
        {"type": "text", "text": "图 2 是经过风格重绘后的同一时刻画面。"},
        {"type": "image_url", "image_url": {"url": data_url("out.jpg")}},
        {"type": "text", "text": p["prompt"]},
    ]}]
    client = openai.OpenAI(base_url=os.environ.get("DASHSCOPE_COMPAT_BASE_URL", COMPAT_BASE), api_key=api_key(),
                           timeout=90, max_retries=0)
    ctx.progress(0.2, f"调用 {model}")
    t0 = time.time()
    try:
        res = client.chat.completions.create(model=model, messages=messages,
                                             extra_body={"enable_thinking": bool(p.get("thinking", False))})
    except openai.APIStatusError as e:
        kind = ErrorKind.TRANSIENT if (e.status_code in (408, 429) or e.status_code >= 500) else ErrorKind.PERMANENT
        raise StepError(kind, f"HTTP {e.status_code}: {str(e)[:200]}") from None
    except openai.APIError as e:
        raise StepError(ErrorKind.TRANSIENT, f"{type(e).__name__}: {e}") from None
    latency = round(time.time() - t0, 2)
    u = res.usage
    ptd = getattr(u, "prompt_tokens_details", None)
    ctd = getattr(u, "completion_tokens_details", None)
    ctx.note("usage", {"model": model, "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                       "total_tokens": u.total_tokens,
                       "image_tokens": getattr(ptd, "image_tokens", None),
                       "reasoning_tokens": getattr(ctd, "reasoning_tokens", None), "latency_s": latency,
                       "billing": "按 token 计费(单价以百炼控制台为准,未估算金额)"})
    text = res.choices[0].message.content or ""
    try:
        verdict = _extract_json(text)
        for k in ("is_japanese_anime_style", "character_preserved", "score", "summary"):
            if k not in verdict:
                raise ValueError(f"缺少字段 {k}")
    except (ValueError, json.JSONDecodeError) as e:
        raise StepError(ErrorKind.TRANSIENT, f"评审模型输出不符合约定格式: {e};原文: {text[:200]!r}") from None
    _json_out(ctx, "review.json", {"model": model, "verdict": verdict, "raw": text})
    ctx.progress(1.0, "评审完成")
