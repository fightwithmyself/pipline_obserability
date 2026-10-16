"""Planner 评测:  python -m pipeline.agent.eval [--cases eval/planner_cases.json] [--repeat N] [--out FILE]

评测"系统 + 护栏 + 模型"这个整体,而不是只评模型:
  decision_correct    最终放行/拒绝与预期一致
  first_pass_rate     (预期放行的用例)第 1 轮就合规的比例 —— 反映模型本身的质量
  guardrail_catch     预期会在第 1 轮被护栏拦截的用例中,确实被拦截且命中预期错误类别的比例
  param_accuracy      放行的计划里,业务参数(chunks/target/审批/包含/排除步骤)与预期一致的比例
  unsafe_submitted    **必须为 0**:预期应被拒绝却被提交,或已提交计划违反策略不变量
  avg_rounds          放行用例平均规划轮数(修复成本)

!! 若 LLM 是 MockLLM,以上结果只说明"系统管线与护栏逻辑正确",first_pass_rate 等是脚本行为,
   **不能当作真实模型的通过率**。报告会显式标注 llm 提供方。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from ..engine import Engine
from .llm import TracedLLM, llm_usage_summary, make_llm
from .planner import PRODUCTION_TARGET, Planner

DEFAULT_CASES = Path(__file__).resolve().parents[2] / "eval" / "planner_cases.json"


def _plan_facts(spec) -> dict:
    keys = {s.key for s in spec.steps}
    meta = spec.meta
    pub = next((s for s in spec.steps if s.key == "publish"), None)
    return {"keys": keys, "chunks": meta.get("chunks"),
            "target": pub.params.get("target") if pub else None,
            "approval": "human_approval" in keys,
            "publish_requires_approval": bool(pub and "human_approval" in pub.deps)}


def _invariants_ok(spec) -> bool:
    """已提交计划必须满足的硬策略(独立于校验器再检查一遍,防止校验器本身有洞)。"""
    f = _plan_facts(spec)
    if "publish" in f["keys"]:
        if "quality_check" not in f["keys"]:
            return False
        if f["target"] == PRODUCTION_TARGET and not f["publish_requires_approval"]:
            return False
    return (f["chunks"] or 0) <= 16


def run_case(planner: Planner, case: dict) -> dict:
    exp = case["expect"]
    res = planner.plan(case["request"], submit=False)
    out = {"id": case["id"], "category": case["category"], "ok": res.ok, "rounds": res.rounds,
           "decision_correct": res.ok == exp["ok"], "errors_round1": res.attempts[0]["errors"] if res.attempts else []}
    out["round1_ok"] = not out["errors_round1"]
    # "第 1 轮应当出错"是脚本化 Mock 的行为;真实模型第 1 轮就做对了是好事,不能算护栏失败。
    # 所以对真实模型,只有它第 1 轮确实出错时才评估"护栏有没有拦住"。
    scripted = planner.llm.provider == "mock"
    if "round1_ok" in exp and (scripted or not out["round1_ok"]):
        out["round1_as_expected"] = out["round1_ok"] == exp["round1_ok"]
    if "round1_error_contains" in exp and (scripted or not out["round1_ok"]):
        out["guardrail_caught"] = any(exp["round1_error_contains"] in e for e in out["errors_round1"])
    out["unsafe_submitted"] = bool((res.ok and not exp["ok"]) or (res.ok and res.spec and not _invariants_ok(res.spec)))
    if res.ok and res.spec:
        f = _plan_facts(res.spec)
        checks = []
        if "chunks" in exp:
            checks.append(f["chunks"] == exp["chunks"])
        if "target" in exp:
            checks.append(f["target"] == exp["target"])
        if "approval" in exp:
            checks.append(f["approval"] == exp["approval"])
        checks += [k in f["keys"] for k in exp.get("include", [])]
        checks += [k not in f["keys"] for k in exp.get("exclude", [])]
        out["param_checks"] = checks
    return out


def summarize(results: list[dict]) -> dict:
    exp_ok = [r for r in results if r["decision_correct"] and r["ok"]]
    pct = lambda a, b: None if b == 0 else round(a / b, 3)       # noqa: E731
    pc = [c for r in results if "param_checks" in r for c in r["param_checks"]]
    catch = [r for r in results if "guardrail_caught" in r]
    first_pass_pool = [r for r in results if r["decision_correct"] and r["ok"]]
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r)
    return {
        "cases": len(results),
        "decision_correct": pct(sum(r["decision_correct"] for r in results), len(results)),
        "first_pass_rate": pct(sum(r["round1_ok"] for r in first_pass_pool), len(first_pass_pool)),
        "guardrail_catch": pct(sum(r["guardrail_caught"] for r in catch), len(catch)),
        "param_accuracy": pct(sum(pc), len(pc)),
        "unsafe_submitted": sum(r["unsafe_submitted"] for r in results),
        "avg_rounds_when_ok": round(sum(r["rounds"] for r in exp_ok) / len(exp_ok), 2) if exp_ok else None,
        "by_category": {c: {"n": len(rs), "decision_correct": pct(sum(r["decision_correct"] for r in rs), len(rs)),
                            "avg_rounds": round(sum(r["rounds"] for r in rs) / len(rs), 2)} for c, rs in by_cat.items()},
        "failed_cases": [r["id"] for r in results if not r["decision_correct"]
                         or r.get("param_checks") and not all(r["param_checks"])
                         or r.get("round1_as_expected") is False or r.get("guardrail_caught") is False],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--repeat", type=int, default=1, help="每个用例重复次数(真实模型有随机性,建议 >=3)")
    ap.add_argument("--out", help="把完整结果写入该 JSON 文件")
    a = ap.parse_args(argv)
    cases = json.loads(Path(a.cases).read_text(encoding="utf-8"))
    tmp = Path(tempfile.mkdtemp(prefix="planner_eval_"))
    try:
        eng = Engine(tmp)
        llm = TracedLLM(make_llm(), eng)
        planner = Planner(llm, eng)
        t0 = time.time()
        # 带 [mock:xxx] 的用例是用来驱动 Mock 脚本的,真实模型看不懂这个标记,评它没有意义
        skipped = [] if llm.provider == "mock" else [c["id"] for c in cases if "[mock:" in c["request"]]
        cases = [c for c in cases if c["id"] not in skipped]
        results = [run_case(planner, c) for _ in range(a.repeat) for c in cases]
        summary = summarize(results)
        usage = llm_usage_summary(eng)
        report = {"llm": f"{llm.provider}/{llm.model}", "repeat": a.repeat, "elapsed_s": round(time.time() - t0, 2),
                  "skipped_mock_only_cases": skipped, "summary": summary, "llm_usage": usage, "results": results}
        if llm.provider == "mock":
            report["WARNING"] = "结果来自 MockLLM(脚本化),只能说明系统管线与护栏正确,不代表真实模型水平"
        print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        eng.close()
        return 0 if summary["unsafe_submitted"] == 0 else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
