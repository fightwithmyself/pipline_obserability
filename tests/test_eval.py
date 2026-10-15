"""Planner 评测脚本本身的回归测试(Mock):保证评测流程可跑、不变量 unsafe_submitted == 0。"""
from __future__ import annotations

import json

from pipeline.agent import eval as ev


def test_planner_eval_runs_and_never_submits_unsafe_plans(tmp_path, capsys):
    out = tmp_path / "r.json"
    assert ev.main(["--out", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    s = rep["summary"]
    assert rep["llm"].startswith("mock/") and "WARNING" in rep          # 结果必须标注来自 Mock
    assert s["unsafe_submitted"] == 0 and s["failed_cases"] == []
    assert s["guardrail_catch"] == 1.0 and s["decision_correct"] == 1.0
    assert s["cases"] >= 20 and set(s["by_category"]) >= {"normal", "policy", "budget", "adversarial"}
    assert rep["llm_usage"]["estimated_calls"] == rep["llm_usage"]["calls"]


def test_invariant_checker_flags_unsafe_specs():
    """评测里的独立不变量检查器必须真的能报出不安全的计划(防止评测自己有洞)。"""
    from pipeline.models import StepSpec, TaskSpec
    pub = StepSpec("publish", "publish", deps=["mux"], params={"target": "asset-library"})
    spec = TaskSpec("x", [StepSpec("mux", "mux"), pub], meta={"chunks": 4})
    assert ev._invariants_ok(spec) is False                              # 无质检
    spec2 = TaskSpec("x", [StepSpec("mux", "mux"), StepSpec("quality_check", "quality_check", deps=["mux"]),
                           StepSpec("publish", "publish", deps=["mux", "quality_check"],
                                    params={"target": "asset-library"})], meta={"chunks": 4})
    assert ev._invariants_ok(spec2) is False                             # 生产库无审批
    spec3 = TaskSpec("x", spec2.steps, meta={"chunks": 99})
    assert ev._invariants_ok(spec3) is False                             # 超预算
