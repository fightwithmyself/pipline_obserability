"""素材检测与 ffmpeg 封装(序列帧处理流水线使用)。

素材检测(assess_video)同时被两处使用:
  1. 命令行 `python -m pipeline.media assess/scan`:提交任务前先筛素材;
  2. 流水线的 fp_probe 步骤:素材不合适就在最前面永久失败,不浪费后面的计算。
两处共用同一个函数,所以"命令行说合适"与"流水线说合适"不会不一致。

判定依据(都是可测量、可解释的量,不用模型):
  - 容器/编码可解码、有视频流;
  - 时长、帧数、分辨率、帧率在流水线能处理的范围内;
  - 固定机位:在多数抽样帧里都与"背景中值图"一致的像素占比(static_bg_ratio)足够高。
    机位移动、镜头切换、整体光照变化都会让它变低;
  - 有运动主体:相对"背景中值图"差异明显的像素占比(fg_ratio)在合理范围。
    太小 = 没有主体(或主体静止被当作背景);太大 = 背景不稳定,背景差分抠图不可靠。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

# 判定阈值。集中放在这里,便于解释与调整;README 里的"素材要求"表与此一一对应。
LIMITS = {
    "min_duration_s": 2.0,
    "max_duration_s": 15.0,
    "min_frames": 12,
    "max_frames": 360,
    "max_width": 1280,
    "max_height": 720,
    "min_side": 96,
    "min_fps": 8.0,
    "max_fps": 30.0,
    "warn_width": 640,            # 超过只告警:CPU 处理会明显变慢
    "min_static_bg_ratio": 0.70,  # 固定机位:至少 70% 的像素在 60% 以上的帧里与背景中值图一致
    "min_fg_ratio": 0.01,         # 有运动主体:平均前景占比至少 1%
    "max_fg_ratio": 0.50,         # 且不超过 50%,否则背景不稳定
}
SAMPLE_W, SAMPLE_H = 160, 96      # 统计用的缩略尺寸(只用于判断,不用于生成成品)
SAMPLE_N = 24                     # 每段素材均匀抽样帧数
BG_VISIBLE_FRAC = 0.6             # 像素在至少这么多比例的帧里与背景中值图一致,才算"背景稳定"
FG_DIFF_THR = 28                  # 与背景中值图的差异超过它视为前景


class MediaError(Exception):
    pass


def ffmpeg_exe() -> str:
    """优先级:环境变量 FFMPEG_BIN > PATH 里的 ffmpeg > imageio-ffmpeg 自带的二进制。"""
    env = os.environ.get("FFMPEG_BIN")
    if env and Path(env).is_file():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise MediaError("找不到 ffmpeg:请安装 ffmpeg,或 pip install imageio-ffmpeg,或设置 FFMPEG_BIN") from e


def _run_text(args: list[str]) -> tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       stdin=subprocess.DEVNULL)
    return p.returncode, (p.stderr or "") + (p.stdout or "")


def probe_video(path: str | Path) -> dict:
    """读取视频基本信息(解析 ffmpeg -i 的输出,不依赖 ffprobe)。"""
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"文件不存在: {path}")
    _, out = _run_text([ffmpeg_exe(), "-hide_banner", "-i", str(path)])
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    v = re.search(r"Video:\s*(\w+).*?[ ,](\d{2,5})x(\d{2,5})[ ,\[]", out)
    if not v:
        raise MediaError("无法识别为视频(没有视频流,或文件已损坏)")
    fps_m = re.search(r"(\d+(?:\.\d+)?)\s*fps", out)
    duration = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))) if m else 0.0
    fps = float(fps_m.group(1)) if fps_m else 0.0
    return {
        "codec": v.group(1), "width": int(v.group(2)), "height": int(v.group(3)),
        "fps": fps, "duration_s": round(duration, 3),
        "frames_est": int(round(duration * fps)) if fps else 0,
        "has_audio": "Audio:" in out,
        "size_mb": round(path.stat().st_size / 1048576, 2),
    }


def run_ffmpeg(args: list[str], on_frame: Callable[[int], None] | None = None) -> None:
    """运行 ffmpeg;通过 -progress 解析已处理帧数回调给调用方(用于上报进度/检测卡死)。"""
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
           "-progress", "pipe:1", "-nostats", *args]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)
    tail: list[str] = []
    assert p.stdout is not None
    for line in p.stdout:
        line = line.strip()
        if line.startswith("frame="):
            try:
                if on_frame:
                    on_frame(int(line.split("=", 1)[1]))
            except ValueError:
                pass
        elif "=" not in line or line.startswith(("error", "Error")):
            tail.append(line)
    rc = p.wait()
    if rc != 0:
        raise MediaError(f"ffmpeg 失败(退出码 {rc}): " + " | ".join(tail[-5:]))


def count_decoded_frames(path: str | Path, decoder: str | None = None) -> int:
    """完整解码一遍并数帧。decoder 用于强制指定解码器(例如 libvpx-vp9 才能读出 VP9 的 alpha)。"""
    n = {"v": 0}
    pre = ["-c:v", decoder] if decoder else []
    run_ffmpeg([*pre, "-i", str(path), "-map", "0:v:0", "-f", "null", "-"], lambda f: n.__setitem__("v", f))
    return n["v"]


# --------------------------------------------------------------------------- 抽样与统计
def sample_frames(path: str | Path, n: int, duration: float, start: float = 0.0) -> np.ndarray:
    """在 [start, start+duration) 内均匀抽 n 帧,缩成统计用小图,返回 (k, H, W, 3) uint8。"""
    fps = n / max(duration, 0.1)
    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{start:.3f}",
           "-t", f"{duration:.3f}", "-i", str(path), "-an",
           "-vf", f"fps={fps:.5f},scale={SAMPLE_W}:{SAMPLE_H}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL)
    if p.returncode != 0:
        raise MediaError("抽样解码失败: " + p.stderr.decode("utf-8", "replace")[-200:])
    size = SAMPLE_W * SAMPLE_H * 3
    k = len(p.stdout) // size
    if k == 0:
        raise MediaError("抽样得到 0 帧")
    return np.frombuffer(p.stdout[: k * size], dtype=np.uint8).reshape(k, SAMPLE_H, SAMPLE_W, 3)


def window_metrics(samples: np.ndarray) -> dict:
    """由抽样帧计算两个核心指标。assess 与 scan 共用,保证口径一致。"""
    s = samples.astype(np.float32)
    plate = np.median(s, axis=0)
    diff = np.abs(s - plate[None]).max(axis=3)                    # (k, H, W)
    is_fg = diff > FG_DIFF_THR
    fg_ratio = is_fg.mean(axis=(1, 2))
    # 固定机位:每个像素在大多数帧里都应与"背景中值图"一致(主体只会在某个像素上短暂停留)。
    # 机位平移/镜头切换/整体变光时,背景在大多数帧里都对不上中值图,这个占比会塌到接近 0。
    # (不能用"逐像素时间标准差":主体横穿画面会让每个像素都被覆盖一次,标准差变大,误伤固定机位。)
    static_bg_ratio = float(((~is_fg).mean(axis=0) >= BG_VISIBLE_FRAC).mean())
    return {
        "samples": int(len(samples)),
        "static_bg_ratio": round(static_bg_ratio, 4),
        "fg_ratio_mean": round(float(fg_ratio.mean()), 4),
        "fg_ratio_min": round(float(fg_ratio.min()), 4),
        "fg_ratio_max": round(float(fg_ratio.max()), 4),
    }


def judge(info: dict, m: dict | None) -> tuple[list[str], list[str]]:
    """把指标与阈值比较,返回 (不合格原因, 告警)。每条原因都说明数值与建议,便于直接换素材。"""
    L = LIMITS
    why: list[str] = []
    warn: list[str] = []
    d, w, h, fps = info["duration_s"], info["width"], info["height"], info["fps"]
    if not (L["min_duration_s"] <= d <= L["max_duration_s"]):
        why.append(f"时长 {d:.1f}s 不在 {L['min_duration_s']:.0f}-{L['max_duration_s']:.0f}s 范围内")
    if info["frames_est"] > L["max_frames"]:
        why.append(f"约 {info['frames_est']} 帧,超过上限 {L['max_frames']}(请裁短或降帧率)")
    if info["frames_est"] < L["min_frames"]:
        why.append(f"约 {info['frames_est']} 帧,少于下限 {L['min_frames']}")
    if w > L["max_width"] or h > L["max_height"]:
        why.append(f"分辨率 {w}x{h} 超过 {L['max_width']}x{L['max_height']}")
    if min(w, h) < L["min_side"]:
        why.append(f"分辨率 {w}x{h} 太小(短边需 >= {L['min_side']})")
    if not (L["min_fps"] <= fps <= L["max_fps"]):
        why.append(f"帧率 {fps:g} 不在 {L['min_fps']:g}-{L['max_fps']:g} 范围内(补帧后会翻倍)")
    if w > L["warn_width"] and w <= L["max_width"]:
        warn.append(f"宽度 {w} > {L['warn_width']}:纯 CPU 处理会较慢")
    if m is not None:
        if m["static_bg_ratio"] < L["min_static_bg_ratio"]:
            why.append(f"背景不固定(静止像素占比 {m['static_bg_ratio']:.0%} < {L['min_static_bg_ratio']:.0%}):"
                       "机位移动、镜头切换或光照变化,背景差分抠图不可靠")
        if m["fg_ratio_mean"] < L["min_fg_ratio"]:
            why.append(f"几乎没有运动主体(前景占比 {m['fg_ratio_mean']:.1%} < {L['min_fg_ratio']:.0%})")
        elif m["fg_ratio_mean"] > L["max_fg_ratio"]:
            why.append(f"前景占比过大({m['fg_ratio_mean']:.0%} > {L['max_fg_ratio']:.0%}),背景不稳定")
        if m["fg_ratio_min"] < 0.001 and m["fg_ratio_mean"] >= L["min_fg_ratio"]:
            warn.append("部分抽样帧里主体不在画面内或静止,这些帧抠图结果可能为空")
    return why, warn


def assess_video(path: str | Path) -> dict:
    """素材是否适合序列帧流水线(拆帧→抠背景→超分→补帧→合成)。"""
    path = Path(path)
    info = probe_video(path)
    metrics = None
    pre_why, _ = judge(info, None)
    # 基本规格都不合格就不必再抽样解码
    if not any(("分辨率" in x or "时长" in x) for x in pre_why) and info["duration_s"] > 0:
        metrics = window_metrics(sample_frames(path, SAMPLE_N, info["duration_s"]))
    why, warn = judge(info, metrics)
    return {"ok": not why, "reasons": why, "warnings": warn, "info": info, "metrics": metrics}


def scan_windows(path: str | Path, win_s: float = 8.0, step_s: float = 2.0, sample_fps: float = 3.0) -> list[dict]:
    """在长视频里滑窗找"最像合格素材"的片段(用于挑选/剪辑素材)。

    只解码一遍:按 sample_fps 抽全片小图,再在内存里对每个窗口算 window_metrics。
    打分:先按是否过线,再按 static_bg_ratio 降序。
    """
    info = probe_video(path)
    allf = sample_frames(path, int(info["duration_s"] * sample_fps), info["duration_s"])
    per = max(int(win_s * sample_fps), 8)
    stride = max(int(step_s * sample_fps), 1)
    out = []
    for i in range(0, len(allf) - per + 1, stride):
        m = window_metrics(allf[i: i + per])
        start = i / sample_fps
        why, _ = judge({**info, "duration_s": win_s, "frames_est": int(win_s * info["fps"])}, m)
        out.append({"start_s": round(start, 2), "end_s": round(start + win_s, 2), "ok": not why,
                    "metrics": m, "reasons": why})
    out.sort(key=lambda r: (not r["ok"], -r["metrics"]["static_bg_ratio"]))
    return out


def cut_clip(src: str | Path, dst: str | Path, start_s: float, dur_s: float) -> None:
    run_ffmpeg(["-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", str(src), "-an",
                "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)])


# --------------------------------------------------------------------------- 命令行
def _print_report(path: Path, r: dict) -> None:
    i, m = r["info"], r["metrics"]
    print(f"\n== {path.name}  {'✔ 合适' if r['ok'] else '✘ 不合适'}")
    print(f"   {i['width']}x{i['height']} {i['fps']:g}fps {i['duration_s']:.1f}s ~{i['frames_est']}帧 "
          f"{i['codec']} {i['size_mb']}MB 音轨={'有' if i['has_audio'] else '无'}")
    if m:
        print(f"   静止背景占比={m['static_bg_ratio']:.0%}  前景占比 均值={m['fg_ratio_mean']:.1%} "
              f"(最小 {m['fg_ratio_min']:.1%} / 最大 {m['fg_ratio_max']:.1%})  抽样帧={m['samples']}")
    for x in r["reasons"]:
        print(f"   ✘ {x}")
    for x in r["warnings"]:
        print(f"   ! {x}")


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="pipeline.media", description="序列帧流水线素材检测")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("assess", help="检测素材是否合适")
    a.add_argument("files", nargs="+")
    a.add_argument("--json", action="store_true")
    s = sub.add_parser("scan", help="在长视频里找合适的片段")
    s.add_argument("file")
    s.add_argument("--win", type=float, default=8.0)
    s.add_argument("--step", type=float, default=2.0)
    s.add_argument("--top", type=int, default=5)
    c = sub.add_parser("cut", help="剪出片段(重编码 H.264,去音轨)")
    c.add_argument("src")
    c.add_argument("dst")
    c.add_argument("--start", type=float, required=True)
    c.add_argument("--dur", type=float, default=8.0)
    args = ap.parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:
            pass
    if args.cmd == "assess":
        bad = 0
        for f in args.files:
            try:
                r = assess_video(f)
            except MediaError as e:
                r = {"ok": False, "reasons": [str(e)], "warnings": [], "info": None, "metrics": None}
            if args.json:
                print(json.dumps(r, ensure_ascii=False))
            elif r["info"] is None:
                print(f"\n== {Path(f).name}  ✘ 不合适\n   ✘ {r['reasons'][0]}")
            else:
                _print_report(Path(f), r)
            bad += 0 if r["ok"] else 1
        return 1 if bad else 0
    if args.cmd == "scan":
        res = scan_windows(args.file, args.win, args.step)
        print(f"{Path(args.file).name}: 共 {len(res)} 个窗口,合格 {sum(r['ok'] for r in res)} 个。前 {args.top} 名:")
        for r in res[: args.top]:
            m = r["metrics"]
            print(f"  [{r['start_s']:7.2f}s - {r['end_s']:7.2f}s] {'✔' if r['ok'] else '✘'} 静止背景={m['static_bg_ratio']:.0%} "
                  f"前景均值={m['fg_ratio_mean']:.1%} 最小={m['fg_ratio_min']:.1%}" + ("" if r["ok"] else f"  ({r['reasons'][0][:30]}…)"))
        return 0
    cut_clip(args.src, args.dst, args.start, args.dur)
    print("已写入", args.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
