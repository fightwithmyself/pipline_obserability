"""动画素材"角色替换"任务模板:把业务参数展开为静态 DAG。

                         ┌─► extract_audio ───────────────────────────┐
probe ─► extract_frames ─┼─► segment_0..N-1 (gpu) ─┐                  ▼
   │                     │                         ├─► swap_0..N-1 ─► assemble ─► mux ─► quality_check ─► [human_approval] ─► publish
   └─► gen_reference (gpu, 生成式,结果必须固化) ───┘
"""
from __future__ import annotations

from typing import Any

from .models import RetryPolicy, StepSpec, TaskSpec, validate_spec


def animation_swap_task(
    name: str = "character-swap", *,
    source: str = "demo.mp4",
    frames: int = 24,
    chunks: int = 4,
    prompt: str = "red-haired girl",
    target: str = "asset-library",
    require_approval: bool = False,
    qc_fail: bool = False,
    nondeterministic_ref: bool = False,
    unit_s: float = 0.05,
    retry: RetryPolicy | None = None,
    timeout_s: float = 600.0,
    stall_timeout_s: float = 30.0,
    chaos: dict[str, dict[str, Any]] | None = None,
    extra_params: dict[str, dict[str, Any]] | None = None,
) -> TaskSpec:
    """chaos / extra_params: {step_key: {...}},分别用于故障注入与覆盖步骤参数。"""
    if chunks < 1 or frames < chunks:
        raise ValueError("需要 chunks>=1 且 frames>=chunks")
    retry = retry or RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=30.0, jitter=0.2)
    chaos = chaos or {}
    extra = extra_params or {}
    per_chunk = frames // chunks

    def mk(key: str, type_: str, deps: list[str], params: dict, resource: str = "cpu",
           cacheable: bool = True) -> StepSpec:
        p = {**params, **extra.get(key, {})}
        return StepSpec(key=key, type=type_, deps=deps, params=p, resource=resource, retry=retry,
                        timeout_s=timeout_s, stall_timeout_s=stall_timeout_s,
                        cacheable=cacheable, chaos=chaos.get(key, {}))

    steps = [
        mk("probe", "probe", [], {"source": source, "frames": frames}),
        mk("extract_frames", "extract_frames", ["probe"], {}),
        mk("extract_audio", "extract_audio", ["probe"], {}),
        mk("gen_reference", "gen_reference", ["probe"],
           {"prompt": prompt, "seed": 0, "nondeterministic": nondeterministic_ref}, "gpu"),
    ]
    for i in range(chunks):
        steps.append(mk(f"segment_{i}", "segment", ["extract_frames"],
                        {"chunk": i, "frames": per_chunk, "unit_s": unit_s}, "gpu"))
    for i in range(chunks):
        steps.append(mk(f"swap_{i}", "swap", [f"segment_{i}", "gen_reference"],
                        {"chunk": i, "frames": per_chunk, "unit_s": unit_s}, "gpu"))
    steps.append(mk("assemble", "assemble", [f"swap_{i}" for i in range(chunks)], {}))
    steps.append(mk("mux", "mux", ["assemble", "extract_audio"], {}))
    steps.append(mk("quality_check", "quality_check", ["mux"], {"force_fail": qc_fail}))
    pub_deps = ["mux", "quality_check"]
    if require_approval:
        steps.append(mk("human_approval", "human_approval", ["quality_check"], {}, "none"))
        pub_deps.append("human_approval")
    # publish 有外部副作用:不走步骤缓存,靠对端幂等键去重
    steps.append(mk("publish", "publish", pub_deps, {"target": target}, cacheable=False))

    spec = TaskSpec(name=name, steps=steps,
                    meta={"template": "animation_swap", "frames": frames, "chunks": chunks})
    validate_spec(spec)
    return spec
