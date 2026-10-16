"""导出一个任务的成品视频与"源帧 vs 结果帧"对比图,并打印用量。

用法:  python scripts/export_run.py <data_dir> <task_id> <输出前缀>
"""
from __future__ import annotations

import io
import json
import shutil
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.api import create_app  # noqa: E402


def main(data: str, tid: str, prefix: str) -> None:
    c = TestClient(create_app(data))
    task = c.get(f"/api/tasks/{tid}").json()
    print("status", task["status"])
    print("usage", json.dumps(c.get(f"/api/tasks/{tid}/usage").json()["total"], ensure_ascii=False))
    pk = next(s for s in task["steps"] if s["key"] == "package")
    sha = pk["outputs"]["final.mp4"]["sha256"]
    src = next(Path(data, "artifacts").rglob(sha))
    out = Path(prefix + ".mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, out)
    fs = c.get(f"/api/tasks/{tid}/frames").json()["frames"]
    rows, W = [], 300
    for f in (fs[2], fs[len(fs) // 2], fs[-3]):
        base = f"/api/tasks/{tid}/frames/{f['step_key']}/{f['item_key']}/image"
        a = Image.open(io.BytesIO(c.get(base, params={"kind": "src"}).content)).convert("RGB")
        b = Image.open(io.BytesIO(c.get(base, params={"kind": "out"}).content)).convert("RGB")
        h = int(W * a.height / a.width)
        rows.append((a.resize((W, h)), b.resize((W, h))))
    sheet = Image.new("RGB", (W * 2 * len(rows), max(r[0].height for r in rows)))
    for i, (a, b) in enumerate(rows):
        sheet.paste(a, (i * 2 * W, 0)); sheet.paste(b, (i * 2 * W + W, 0))
    sheet.save(prefix + "_compare.png")
    print("wrote", out, prefix + "_compare.png")


if __name__ == "__main__":
    main(*sys.argv[1:4])
