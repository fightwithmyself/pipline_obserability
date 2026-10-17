"""由素材与成品视频生成 README 用的对比图(左=素材,右=成品):GIF 预览、对比 mp4、关键帧对比图。

用法:  python scripts/make_readme_assets.py docs/assets/source_cba.mp4 docs/assets/result_seedream5pro.mp4 docs/assets
只依赖仓库已有的 ffmpeg 封装与 Pillow;不联网、不调用任何模型。
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import media  # noqa: E402

H = 400


def ff(*args: str) -> None:
    r = subprocess.run([media.ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *args], capture_output=True)
    if r.returncode:
        raise SystemExit(r.stderr.decode("utf-8", "replace")[-800:])


def main(src: str, res: str, out: str) -> None:
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stack = f"[0:v]scale=-2:{H},setsar=1[a];[1:v]scale=-2:{H},setsar=1[b];[a][b]hstack=inputs=2"
    ff("-i", src, "-i", res, "-filter_complex", stack, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
       str(out_dir / "compare_side_by_side.mp4"))
    ff("-i", src, "-i", res, "-t", "6", "-filter_complex",
       stack + ",fps=10,split[s0][s1];[s0]palettegen=max_colors=96[p];[s1][p]paletteuse=dither=bayer:bayer_scale=4",
       str(out_dir / "compare.gif"))
    # 关键帧对比:上排素材,下排成品
    ts = [0.5, 1.5, 2.5, 3.5, 4.5]
    with tempfile.TemporaryDirectory() as td:
        rows = []
        for tag, f in (("src", src), ("res", res)):
            imgs = []
            for i, t in enumerate(ts):
                p = Path(td) / f"{tag}{i}.png"
                ff("-ss", str(t), "-i", f, "-frames:v", "1", "-vf", f"scale=-2:{H // 2}", str(p))
                imgs.append(Image.open(p).convert("RGB"))
            rows.append(imgs)
    w = max(i.width for r in rows for i in r)
    sheet = Image.new("RGB", (w * len(ts), (H // 2) * 2 + 40), (15, 20, 25))
    d = ImageDraw.Draw(sheet)
    for ri, imgs in enumerate(rows):
        for ci, im in enumerate(imgs):
            sheet.paste(im, (ci * w, ri * (H // 2 + 20) + 20))
    d.text((6, 4), "source", fill=(200, 200, 200))
    d.text((6, H // 2 + 24), "Seedream 5.0 pro result", fill=(200, 200, 200))
    sheet.save(out_dir / "compare_frames.png", optimize=True)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
