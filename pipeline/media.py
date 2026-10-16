"""ffmpeg 封装与素材探测(不依赖 ffprobe,解析 `ffmpeg -i` 的输出)。

命令行:  python -m pipeline.media probe <文件...>
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable


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
    """读取视频基本信息。"""
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


_PROGRESS_LINE = re.compile(r"^[A-Za-z_0-9]+=")


def run_ffmpeg(args: list[str], on_frame: Callable[[int], None] | None = None,
               collect: list[str] | None = None) -> None:
    """运行 ffmpeg;通过 -progress 解析已处理帧数回调给调用方(用于上报进度/检测卡死)。

    collect: 若提供,ffmpeg 输出的所有非进度行(警告/解码错误)会追加进去,调用方据此统计坏帧。
    """
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
        elif line and not _PROGRESS_LINE.match(line):
            tail.append(line)
            if collect is not None:
                collect.append(line)
    rc = p.wait()
    if rc != 0:
        raise MediaError(f"ffmpeg 失败(退出码 {rc}): " + " | ".join(tail[-5:]))


def count_decoded_frames(path: str | Path, decoder: str | None = None) -> int:
    """完整解码一遍并数帧。"""
    n = {"v": 0}
    pre = ["-c:v", decoder] if decoder else []
    run_ffmpeg([*pre, "-i", str(path), "-map", "0:v:0", "-f", "null", "-"], lambda f: n.__setitem__("v", f))
    return n["v"]


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="pipeline.media", description="素材探测")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("probe", help="打印视频基本信息")
    a.add_argument("files", nargs="+")
    args = ap.parse_args(argv)
    bad = 0
    for f in args.files:
        try:
            print(f, json.dumps(probe_video(f), ensure_ascii=False))
        except MediaError as e:
            print(f, "ERROR", e)
            bad += 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
