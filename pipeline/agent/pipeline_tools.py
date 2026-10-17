"""流水线 Agent 的工具集:每个流水线阶段是一个 function-calling 工具。

- 参数用 Pydantic 校验(同时生成 OpenAI function-calling 的 JSON Schema);非法参数以错误结果回给模型,让它自己修正。
- 并行:模型在同一轮里发出多个 `stylize_chunk` / `postprocess_chunk` 调用,工具节点用线程池并发执行,
  每个调用对应引擎里一个独立任务,由 worker 并行处理(每块内部各帧还有并发度 `concurrency`)。
- 工具只做"提交阶段 + 等结果 + 回精简摘要";真正的执行、重试、恢复由引擎负责(见 stages.py)。
- Agent 没有"批准发布""取消任务""改预算"之类的工具。失败时能做的只有:重试(受处置手册约束)或升级给人。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError, model_validator

from ..engine import Engine
from . import runbook
from .stages import StageError, StageRunner, stage_group

SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"


class _NoArgs(BaseModel):
    pass


class _LoadSkill(BaseModel):
    name: str = Field(description="技能名,如 anime-style-pipeline", pattern=r"^[a-z0-9][a-z0-9\-]{0,60}$")


class _Keyframes(BaseModel):
    chunks: int = Field(4, ge=1, le=32, description="把关键帧分成几块并行风格化")
    stride: int = Field(6, ge=1, le=60, description="关键帧间隔(帧);越大越省钱,补间越多")
    max_gap: int = Field(24, ge=1, le=120, description="两个关键帧之间最大间隔(帧)")

    @model_validator(mode="after")
    def _gap(self):
        if self.max_gap < self.stride:
            raise ValueError("max_gap 不能小于 stride")
        return self


class _Chunk(BaseModel):
    chunk: int = Field(ge=0, description="块序号,从 0 开始")


class _Stylize(_Chunk):
    frame_prompts: dict[str, str] | None = Field(
        None, description="可选:{关键帧序号: 提示词},只在需要对个别帧换提示词(例如被审核拒绝)时使用")


class _Stage(BaseModel):
    stage_id: str = Field(description="阶段 id,如 stylize:0、post:1、qc")


class _Escalate(BaseModel):
    reason: str = Field(min_length=1, max_length=500, description="为什么无法继续、需要人做什么")


@dataclass
class PTool:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[["ToolContext", BaseModel], dict]
    group: str


class ToolContext:
    def __init__(self, data_dir: str | Path, thread_id: str, cfg: dict, runner: StageRunner):
        self.data_dir = Path(data_dir)
        self.thread_id = thread_id
        self.cfg = cfg
        self.runner = runner


def _load_skill(c: ToolContext, a: _LoadSkill) -> dict:
    p = SKILLS_DIR / a.name / "SKILL.md"
    if not p.is_file():
        avail = sorted(d.name for d in SKILLS_DIR.iterdir() if (d / "SKILL.md").is_file()) if SKILLS_DIR.is_dir() else []
        raise StageError(f"没有技能 {a.name!r},可用: {avail}")
    return {"status": "OK", "name": a.name, "content": p.read_text(encoding="utf-8")}


def _status(c: ToolContext, _a: _NoArgs) -> dict:
    st = c.runner.state()
    tot = {"calls": 0, "billed": 0, "output_tokens": 0, "reused_output_tokens": 0, "cost_cny_est": 0.0}
    for s in st.values():
        u = s.get("usage") or {}
        for k in tot:
            tot[k] = round(tot[k] + u.get(k, 0), 4) if isinstance(tot[k], float) else tot[k] + u.get(k, 0)
    return {"status": "OK", "budget_cny": c.cfg.get("budget_cny"), "seedream_total": tot,
            "stages": {k: {"status": v["status"], "task_id": v["task_id"], "retries": v["retries"]}
                       for k, v in sorted(st.items())}}


def _inspect(c: ToolContext, a: _Chunk) -> dict:
    row = c.runner.state().get(f"stylize:{a.chunk}")
    if row is None:
        raise StageError(f"stylize:{a.chunk} 还没有执行过")
    eng = Engine(c.data_dir)
    try:
        items = eng.list_items(row["task_id"], step_key=f"stylize_{a.chunk}")
    finally:
        eng.close()
    by: dict[str, int] = {}
    for it in items:
        by[it["status"]] = by.get(it["status"], 0) + 1
    odd = [{"idx": it.get("idx"), "status": it["status"], "error_kind": it.get("error_kind"),
            "error": (it.get("error") or "")[:120], "maybe_billed": it.get("maybe_billed", 0)}
           for it in items if it["status"] in ("FAILED", "DEGRADED") or it.get("maybe_billed")]
    return {"status": "OK", "chunk": a.chunk, "by_status": by, "abnormal_frames": odd[:20]}


def _retry(c: ToolContext, a: _Stage) -> dict:
    return c.runner.retry(a.stage_id)


def _escalate(c: ToolContext, a: _Escalate) -> dict:
    """升级给人:取最近一个失败阶段的手册步骤,随结果一起记录;图在这之后结束。"""
    st = c.runner.state()
    failed = [(k, v) for k, v in st.items() if v["status"] == "FAILED"]
    steps: list[str] = []
    stage = None
    if failed:
        stage, v = failed[-1]
        e = v.get("error") or {}
        steps = list(runbook.match(e.get("error_kind"), e.get("error_msg")).human_steps)
    return {"status": "ESCALATED", "reason": a.reason, "stage_id": stage, "human_steps": steps}


def build_tools() -> list[PTool]:
    def stage(name, desc, args, stage_of, group, params_of=None, overrides_of=None) -> PTool:
        def fn(c: ToolContext, a: BaseModel) -> dict:
            return c.runner.run(stage_of(a), params=params_of(a) if params_of else None,
                                frame_overrides=overrides_of(a) if overrides_of else None)
        return PTool(name, desc, args, fn, group)

    return [
        PTool("load_skill", "读取技能说明(流水线的阶段定义、决策规则、失败处置)。开始任务前必须先调用。",
              _LoadSkill, _load_skill, "plan"),
        stage("ingest_video", "阶段1 校验与标准化素材:探测、转 1080p/24fps、分离音轨。返回素材时长/帧率/尺寸与解码告警。",
              _NoArgs, lambda a: "ingest", "ingest"),
        stage("extract_keyframes",
              "阶段2 选择关键帧并分块。你决定 chunks/stride/max_gap(依据素材时长与预算)。返回关键帧数、预估费用、预算。",
              _Keyframes, lambda a: "keyframes", "keyframes",
              params_of=lambda a: a.model_dump()),
        stage("stylize_chunk",
              "阶段3 调用图像模型(Seedream)对第 chunk 块的每个关键帧做风格化/人物替换。会花钱。"
              "不同块相互独立:请在同一轮里并行调用多块。",
              _Stylize, lambda a: f"stylize:{a.chunk}", "stylize",
              overrides_of=lambda a: ({str(k): {"prompt": v} for k, v in (a.frame_prompts or {}).items()} or None)),
        stage("poststats", "阶段4a 汇总各块风格化结果的色彩统计(需要所有 stylize 块完成)。",
              _NoArgs, lambda a: "poststats", "poststats"),
        stage("postprocess_chunk", "阶段4b 对第 chunk 块做色彩对齐、去噪与按源视频运动补帧,编码成分段。各块可并行。",
              _Chunk, lambda a: f"post:{a.chunk}", "post"),
        stage("quality_check", "阶段5a 质检:帧数/尺寸/黑帧(硬检查),闪烁/结构保真度/降级比例(软告警)。",
              _NoArgs, lambda a: "qc", "qc"),
        stage("mux_audio", "阶段5b 拼接分段并合入源音轨,音画对齐。", _NoArgs, lambda a: "mux", "mux"),
        stage("package_and_publish", "阶段6 打包成品与元数据并发布(幂等)。只有质检通过后才应该调用。",
              _NoArgs, lambda a: "package", "package"),
        PTool("get_run_status", "查看各阶段状态、任务 id、重试次数、图像模型累计用量与预算。",
              _NoArgs, _status, "control"),
        PTool("inspect_frames", "查看某一块风格化的逐帧结果:各状态数量,以及失败/降级/可能多付的帧。",
              _Chunk, _inspect, "control"),
        PTool("retry_stage", "重试一个失败的阶段。只有处置手册允许(瞬时类错误)且未超过次数才会执行,否则返回拒绝原因。",
              _Stage, _retry, "control"),
        PTool("escalate_to_human", "无法自行恢复时调用:记录原因与人工恢复步骤并结束本次运行。认证/欠费/审核/质检失败等不要重试。",
              _Escalate, _escalate, "control"),
    ]


class PipelineToolRegistry:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self.tools = {t.name: t for t in build_tools()}

    def specs(self) -> list[dict]:
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.args.model_json_schema()}}
                for t in self.tools.values()]

    def group(self, name: str) -> str:
        t = self.tools.get(name)
        return t.group if t else "control"

    def call(self, name: str, args: dict) -> dict:
        """执行一个工具。参数/业务错误变成 {"status":"ERROR"} 回给模型;KeyboardInterrupt 等进程级中断不吞。"""
        t = self.tools.get(name)
        if t is None:
            return {"status": "ERROR", "error": f"未知工具 {name!r},可用: {sorted(self.tools)}"}
        if "__invalid_json__" in args:
            return {"status": "ERROR", "error": f"参数不是合法 JSON: {args['__invalid_json__']}"}
        try:
            parsed = t.args.model_validate(args)
        except ValidationError as e:
            return {"status": "ERROR", "error": "参数错误: " + "; ".join(
                f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())}
        try:
            return t.fn(self.ctx, parsed)
        except StageError as e:
            return {"status": "ERROR", "error": str(e)}
        except Exception as e:  # noqa: BLE001  未预期的错误也回给模型/记录,不让整个图崩掉
            return {"status": "ERROR", "error": f"{type(e).__name__}: {e}"}


def compact(result: dict, limit: int = 12000) -> str:
    """工具结果序列化成给模型看的文本;过长时改成带预览的合法 JSON(控制 token,且不破坏格式)。"""
    s = json.dumps(result, ensure_ascii=False, default=str)
    if len(s) <= limit:
        return s
    return json.dumps({"status": result.get("status", "OK"), "truncated": True, "preview": s[:limit]},
                      ensure_ascii=False)