"""真实模型单帧冒烟:密钥只从环境变量 ARK_API_KEY 读取。会产生一次真实调用(扣费/扣套餐额度)。

用法:  python scripts/smoke_real.py <provider: ark|ark_plan> <模型ID> <视频> [秒] [提示词]
输出:  data/smoke/<模型>_<秒>.png 与 *_sheet.png(源图|结果图),并打印耗时与完整 usage。
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.doubao import DoubaoError, make_client  # noqa: E402
from pipeline.media import ffmpeg_exe  # noqa: E402

DEFAULT_PROMPT = "日式动漫风格,赛璐璐上色,清晰的线稿,保持人物、动作与构图不变"


def main() -> int:
    provider, model, video = sys.argv[1:4]
    sec = sys.argv[4] if len(sys.argv) > 4 else "2.0"
    prompt = sys.argv[5] if len(sys.argv) > 5 else DEFAULT_PROMPT
    png = subprocess.run([ffmpeg_exe(), "-v", "error", "-ss", sec, "-i", video, "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    src = Image.open(io.BytesIO(png))
    print("源帧", src.size)
    c = make_client(provider, model=model, data_dir=Path("data"), chaos=None, attempt=1, sleep=time.sleep)
    t = time.time()
    try:
        r = c.edit(image=png, prompt=prompt, reference=None, frame_key=f"smoke_{sec}")
    except DoubaoError as e:
        print("失败:", type(e).__name__, getattr(e, "kind", ""), str(e)[:500])
        return 1
    print("耗时 %.1fs" % (time.time() - t), "usage:", json.dumps(r.usage, ensure_ascii=False),
          "request_id:", r.request_id, "meta:", r.meta)
    out = Image.open(io.BytesIO(r.image))
    print("结果", out.size)
    d = Path("data/smoke"); d.mkdir(parents=True, exist_ok=True)
    tag = f"{model}_{sec}"
    out.save(d / f"{tag}.png")
    h = 720
    a = src.convert("RGB").resize((int(src.width * h / src.height), h))
    b = out.convert("RGB").resize((int(out.width * h / out.height), h))
    sheet = Image.new("RGB", (a.width + b.width, h)); sheet.paste(a, (0, 0)); sheet.paste(b, (a.width, 0))
    sheet.save(d / f"{tag}_sheet.png")
    print("写入", d / f"{tag}_sheet.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
