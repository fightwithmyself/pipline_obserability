"""探测:某个对话模型 ID 在给定 base/path 下能不能调用(1 次极小请求,max_tokens=8)。
用法: python scripts/probe_chat.py <plan|std> <model> [<model> ...]      密钥只读环境变量 ARK_API_KEY
"""
from __future__ import annotations

import os
import sys

import httpx

BASE = "https://ark.cn-beijing.volces.com"
PREFIX = {"plan": "/api/plan/v3", "std": "/api/v3"}


def main() -> int:
    mode, models = sys.argv[1], sys.argv[2:]
    key = os.environ["ARK_API_KEY"]
    for m in models:
        body = {"model": m, "messages": [{"role": "user", "content": "回复OK"}], "max_tokens": 8}
        try:
            r = httpx.post(BASE + PREFIX[mode] + "/chat/completions", json=body, timeout=60,
                           headers={"Authorization": f"Bearer {key}"})
            print(m, r.status_code, r.text[:300].replace("\n", " "))
        except Exception as e:  # noqa: BLE001
            print(m, "EXC", type(e).__name__, e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
