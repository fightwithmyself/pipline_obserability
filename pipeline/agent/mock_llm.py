"""MockLLM:脚本化的确定性"模型",用于在没有 API Key 时验证**系统管线与护栏**。

它**不是**真实模型,不具备理解能力 —— 基于正则解析需求并按脚本行为:
- 规划:第一次"听话但粗心"(照字面执行用户需求,可能违反策略/预算),收到校验反馈后进入"修复模式"输出合规计划;
- 诊断:按错误类别的固定规则给建议。
评测结果若来自它,必须在报告中标注"Mock",不能当作真实模型的通过率。

可用标记(写在需求文本里触发特定行为,用于测试护栏):
  [mock:garbage] 第一次输出非 JSON      [mock:cycle]  第一次输出带环的计划
  [mock:evil]    始终夹带非白名单阶段    [mock:inject] 第一次夹带 timeout_s 等越权字段
"""
from __future__ import annotations

import json
import re

from .llm import LLMResult
from .planner import MAX_CHUNKS, MAX_FRAMES, PRODUCTION_TARGET, STAGE_CATALOG


def _est(s: str) -> int:
    return max(1, len(s) // 3)


class MockLLM:
    provider = "mock"
    model = "mock-scripted-v1"

    def complete(self, messages: list[dict], *, json_mode: bool = False, temperature: float = 0.0) -> LLMResult:
        sys_prompt = messages[0]["content"] if messages else ""
        if "[[PLANNER]]" in sys_prompt:
            text = self._plan(messages)
        elif "[[DIAGNOSE]]" in sys_prompt:
            text = self._diagnose(messages)
        else:
            text = json.dumps({"error": "mock: unknown task"}, ensure_ascii=False)
        pt = _est(json.dumps(messages, ensure_ascii=False))
        return LLMResult(text, pt, _est(text), True, self.model, 0.0)

    # ------------------------------------------------------------------ 规划
    def _plan(self, messages: list[dict]) -> str:
        request = messages[1]["content"]
        repair = any(m["role"] == "user" and m["content"].startswith("校验未通过") for m in messages[2:])
        if "[mock:garbage]" in request and not repair:
            return "好的,我来为您规划:先探测素材,然后……(这不是 JSON)"

        m = re.search(r"([A-Za-z0-9_\-./]+\.(?:mp4|mov|webm|mkv))", request)
        source = m.group(1) if m else "demo.mp4"
        m = re.search(r"(\d+)\s*(?:段|块|个分块|chunks?)", request)
        chunks = int(m.group(1)) if m else 4
        m = re.search(r"(\d+)\s*帧", request)
        frames = int(m.group(1)) if m else max(24, chunks * 4)
        m = re.search(r"换成(.+?)(?=[,\uff0c\u3002;\uff1b]|并|然后|发布|$)", request)
        prompt = (m.group(1).strip() if m else "new character")[:100]
        target = "staging" if re.search(r"测试库|staging|预发", request) else PRODUCTION_TARGET

        want_publish = not re.search(r"不发布|只提取|不用发布|仅处理", request)
        negated_approval = re.search(r"不用审批|无需审批|跳过审批|不需要审批", request)
        want_approval = ("审批" in request or "人工" in request) and not negated_approval
        skip_qc = bool(re.search(r"跳过质检|不要质检|不用质检", request))

        if repair:                                   # 修复模式:按系统反馈给出合规计划
            chunks = min(max(chunks, 1), MAX_CHUNKS)
            frames = min(max(frames, chunks), MAX_FRAMES)
            skip_qc = False
            want_approval = want_approval or target == PRODUCTION_TARGET

        types = ["probe", "extract_frames", "extract_audio", "gen_reference", "segment", "swap", "assemble", "mux"]
        if not skip_qc:
            types.append("quality_check")
        if want_publish:
            if want_approval and not skip_qc:
                types.append("human_approval")
            types.append("publish")

        stages = []
        for t in types:
            deps = [d for d in STAGE_CATALOG[t]["requires"] if d in types]
            if t == "publish" and want_approval and "human_approval" in types:
                deps.append("human_approval")
            stages.append({"id": t, "type": t, "deps": deps})

        if "[mock:cycle]" in request and not repair:
            stages[0]["deps"] = ["swap"]
        if "[mock:inject]" in request and not repair:
            for s in stages:
                s["timeout_s"] = 99999
        if "[mock:evil]" in request:
            stages.append({"id": "shell", "type": "shell", "deps": ["probe"]})

        plan = {"summary": f"替换角色为「{prompt}」" + ("并发布" if want_publish else ""),
                "params": {"source": source, "frames": frames, "chunks": chunks, "prompt": prompt, "target": target},
                "stages": stages}
        return json.dumps(plan, ensure_ascii=False)

    # ------------------------------------------------------------------ 诊断
    def _diagnose(self, messages: list[dict]) -> str:
        try:
            ev = json.loads(messages[-1]["content"])
            step = ev["step"]
        except Exception:
            return json.dumps({"root_cause": "无法解析证据", "category": "unknown",
                               "recommended_action": "escalate", "confidence": 0.1,
                               "explanation": "mock: bad evidence"}, ensure_ascii=False)
        kind = step.get("error_kind") or ""
        msg = step.get("error_msg") or ""
        out: dict
        if kind == "PERMANENT" and re.search(r"损坏|corrupt|为空", msg):
            out = dict(root_cause="输入素材损坏或为空", category="bad_input", recommended_action="give_up",
                       confidence=0.9, explanation="永久性输入错误,重试不会改变结果;需要上游重新提供素材。")
        elif kind == "PERMANENT" and re.search(r"质检", msg):
            out = dict(root_cause="质检未通过", category="quality", recommended_action="escalate",
                       confidence=0.7, explanation="质检失败属于业务判断,需要人工查看产物后决定。")
        elif kind == "PERMANENT":
            out = dict(root_cause=msg[:80] or "永久性错误", category="permanent", recommended_action="escalate",
                       confidence=0.5, explanation="永久性错误,盲目重试无意义,建议人工处理。")
        elif kind == "RESOURCE":
            out = dict(root_cause="GPU 显存不足", category="resource", recommended_action="retry_step",
                       confidence=0.6, explanation="资源类错误常因并发抢占显存,可重试;若仍失败应减小分块/批大小。")
        elif kind in ("TRANSIENT", "TIMEOUT", "STALL"):
            out = dict(root_cause=f"重试预算耗尽的{kind}错误", category="transient", recommended_action="retry_step",
                       confidence=0.75, explanation="错误类别为可重试,且上游产物均已缓存,重试代价低。")
        elif kind == "LEASE_LOST":
            out = dict(root_cause="步骤反复导致 worker 失联", category="poison", recommended_action="escalate",
                       confidence=0.8, explanation="疑似毒丸步骤(反复崩溃进程),自动重试可能无限消耗资源,需人工排查。")
        else:
            out = dict(root_cause="未知", category="unknown", recommended_action="escalate", confidence=0.2,
                       explanation="证据不足。")
        return json.dumps(out, ensure_ascii=False)
