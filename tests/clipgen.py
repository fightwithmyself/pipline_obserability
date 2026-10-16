"""测试用合成素材生成器:固定机位 + 有纹理的静态背景 + 一个运动的"角色"。

用途:
  - 作为素材检测的**正例**(应判合适)与各种**反例**(机位平移、没有主体、规格超限…);
  - 作为流水线端到端测试的输入(小分辨率,几秒钟就能跑完)。
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline.media import run_ffmpeg


def _background(w: int, h: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([60 + 90 * xx / w, 90 + 60 * yy / h, 140 - 50 * xx / w], axis=2)
    tex = rng.normal(0, 6, size=(h, w, 1))                       # 静态纹理,模拟真实背景细节
    checker = (((xx // 16) + (yy // 16)) % 2)[..., None] * 18
    return np.clip(base + tex + checker, 0, 255).astype(np.uint8)


def draw_frame(bg: np.ndarray, t: float, w: int, h: int, *, pan_after: float | None = None,
               subject: bool = True) -> Image.Image:
    """pan_after:从该时刻起背景整体平移(模拟机位移动);None 表示全程固定机位。"""
    shift = 0 if pan_after is None else max(0, int((t - pan_after) * 60))
    arr = np.roll(bg, shift, axis=1) if shift else bg
    im = Image.fromarray(arr, "RGB")
    if not subject:
        return im
    d = ImageDraw.Draw(im)
    cx = int(w * 0.15 + (w * 0.7) * ((t * 0.5) % 1.0))           # 角色从左向右走
    cy = int(h * 0.55 + 6 * np.sin(t * 6))
    r = max(h // 6, 8)
    d.ellipse([cx - r, cy - r * 2, cx + r, cy], fill=(235, 190, 150))          # 头
    d.rectangle([cx - r, cy, cx + r, cy + r * 2], fill=(200, 40, 50))          # 身体
    d.rectangle([cx - r, cy + r * 2, cx - 2, cy + r * 3], fill=(30, 30, 90))   # 腿
    d.rectangle([cx + 2, cy + r * 3 - 4, cx + r, cy + r * 3 + 4], fill=(30, 30, 90))
    return im


def make_clip(dest: str | Path, *, w: int = 320, h: int = 180, fps: int = 12, secs: float = 5.0,
              pan_after: float | None = None, subject: bool = True) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="clipgen_"))
    try:
        bg = _background(w, h)
        n = int(round(fps * secs))
        for i in range(n):
            draw_frame(bg, i / fps, w, h, pan_after=pan_after, subject=subject).save(tmp / f"{i:05d}.png")
        run_ffmpeg(["-framerate", str(fps), "-i", str(tmp / "%05d.png"), "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", str(dest)])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return dest


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "samples/synthetic_walk_5s.mp4")
    print("written", make_clip(out))
