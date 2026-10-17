"""探测 doubao-seed-2.1-pro 的 function calling:是否返回 tool_calls、是否支持并行调用、usage 字段长什么样。
用法: python scripts/probe_tools.py [model]    密钥只读环境变量 ARK_API_KEY
"""
from __future__ import annotations

import json
import os
import sys

import httpx

URL = "https://ark.cn-beijing.volces.com/api/plan/v3/chat/completions"
TOOLS = [{"type": "function", "function": {
    "name": "stylize_chunk", "description": "对第 chunk 块关键帧调用图像模型做风格化",
    "parameters": {"type": "object", "properties": {"chunk": {"type": "integer"}}, "required": ["chunk"]}}}]


def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else "doubao-seed-2.1-pro"
    key = os.environ["ARK_API_KEY"]
    msgs = [{"role": "system", "content": "你是调度员,只能通过工具行动。"},
            {"role": "user", "content": "共有 3 块(0,1,2)需要风格化,请并行地为每一块各调用一次 stylize_chunk。"}]
    r = httpx.post(URL, json={"model": model, "messages": msgs, "tools": TOOLS, "max_tokens": 600}, timeout=90,
                   headers={"Authorization": f"Bearer {key}"})
    print(r.status_code)
    j = r.json()
    m = j["choices"][0]["message"]
    print("finish:", j["choices"][0]["finish_reason"])
    print("content:", m.get("content"))
    print("tool_calls:", json.dumps(m.get("tool_calls"), ensure_ascii=False))
    print("message keys:", sorted(m))
    print("usage:", json.dumps(j.get("usage"), ensure_ascii=False))
    # 第二轮:把工具结果回传,看是否需要回传 reasoning_content
    if m.get("tool_calls"):
        follow = msgs + [{"role": "assistant", "content": m.get("content") or "", "tool_calls": m["tool_calls"]}]
        for tc in m["tool_calls"]:
            follow.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"ok": True})})
        r2 = httpx.post(URL, json={"model": model, "messages": follow, "tools": TOOLS, "max_tokens": 400}, timeout=90,
                        headers={"Authorization": f"Bearer {key}"})
        print("round2:", r2.status_code, r2.text[:400].replace("\n", " "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
