"""Agent 工具集:LLM / MCP 客户端操作引擎的**唯一**入口。

- 每个工具有 Pydantic 参数 schema(同时生成 OpenAI function-calling 与 MCP 所需的 JSON Schema)。
- 读/写分级:默认只允许读;写工具需要调用方显式 allow_write。
- **刻意不提供 approve(人工审批)工具**:Agent 不能批准自己负责的产物,审批只能由人通过 API 完成。
- 工具只返回精简后的 JSON(控制 token 消耗),错误统一抛 ToolError(信息可直接回灌给模型)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from ..engine import Engine, EngineError


class ToolError(Exception):
    pass


class _TaskArgs(BaseModel):
    task_id: str = Field(description="任务 ID,形如 t_xxxxxxxxxxxx")


class _StepArgs(_TaskArgs):
    key: str = Field(description="步骤 key,如 segment_0")


class _ListArgs(BaseModel):
    status: str | None = Field(None, description="按任务状态过滤: PENDING/RUNNING/SUCCEEDED/FAILED/CANCELLED")
    limit: int = Field(20, ge=1, le=100)


class _EventsArgs(_TaskArgs):
    after: int = Field(0, ge=0, description="只返回 seq 大于该值的事件")
    limit: int = Field(50, ge=1, le=200)


class _NoArgs(BaseModel):
    pass


class _SubmitArgs(BaseModel):
    request: str = Field(min_length=1, max_length=2000, description="自然语言需求,如:把 demo.mp4 的主角换成红发女孩并发布")


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    fn: Callable[["ToolRegistry", BaseModel], Any]
    write: bool = False


def _trim_task(t: dict) -> dict:
    keep = ("key", "status", "attempt_no", "failures", "lost_count", "max_attempts", "error_kind", "cached", "resource")
    steps = []
    for s in t["steps"]:
        d = {k: s[k] for k in keep}
        if s["error_msg"]:
            d["error_msg"] = s["error_msg"][:200]
        if s["status"] == "RUNNING":
            d["progress"] = s["progress"]
        steps.append(d)
    return {"id": t["id"], "name": t["name"], "status": t["status"], "percent": t["percent"],
            "steps_total": t["steps_total"], "steps_succeeded": t["steps_succeeded"],
            "steps_by_status": t["steps_by_status"], "waiting_approval": t["waiting_approval"],
            "request": t["request"], "steps": steps}


class ToolRegistry:
    def __init__(self, engine: Engine, planner=None):
        self.engine = engine
        self.planner = planner
        self.tools: dict[str, Tool] = {}
        for t in self._builtin():
            self.tools[t.name] = t

    def _builtin(self) -> list[Tool]:
        e = lambda: self.engine   # noqa: E731
        return [
            Tool("get_task", "查询任务详情:任务状态、各步骤状态/重试次数/错误摘要。",
                 _TaskArgs, lambda r, a: _trim_task(e().get_task(a.task_id))),
            Tool("list_tasks", "列出最近的任务及其状态与进度。", _ListArgs,
                 lambda r, a: [{k: t[k] for k in ("id", "name", "status", "percent", "steps_succeeded", "steps_total",
                                                  "has_failed_steps", "waiting_approval")}
                               for t in e().list_tasks(a.limit, a.status)]),
            Tool("get_step_attempts", "查询某步骤的全部执行尝试:worker、耗时、结果、错误类别与信息。",
                 _StepArgs, lambda r, a: e().get_attempts(a.task_id, a.key)),
            Tool("get_events", "查询任务的追加式事件历史(状态变更、重试、缓存命中、Agent 动作等)。",
                 _EventsArgs, lambda r, a: e().get_events(a.task_id, a.after, a.limit)),
            Tool("get_metrics", "查询系统级指标:队列深度、重试/失败统计、浪费耗时比例、缓存命中。",
                 _NoArgs, lambda r, a: e().metrics()),
            Tool("list_workers", "查询 worker 的能力、存活状态。", _NoArgs, lambda r, a: e().list_workers()),
            # ---- 写工具
            Tool("submit_request", "把自然语言需求交给 Planner 生成计划(含护栏校验)并提交执行。",
                 _SubmitArgs, self._submit, write=True),
            Tool("retry_step", "重试一个 FAILED 步骤(失败预算重置,已成功的上游不会重跑)。",
                 _StepArgs, lambda r, a: (e().retry_step(a.task_id, a.key), {"ok": True})[1], write=True),
            Tool("resume_task", "恢复任务:重试其中所有 FAILED 步骤。",
                 _TaskArgs, lambda r, a: {"reset_steps": e().resume_task(a.task_id)}, write=True),
            Tool("cancel_task", "取消任务(运行中的步骤会被终止)。",
                 _TaskArgs, lambda r, a: (e().cancel_task(a.task_id, "agent"), {"ok": True})[1], write=True),
        ]

    def _submit(self, _r: "ToolRegistry", a: _SubmitArgs) -> dict:
        if self.planner is None:
            raise ToolError("未配置 Planner,无法提交自然语言需求")
        return self.planner.plan(a.request).to_dict()

    # ------------------------------------------------------------------ 对外
    def specs(self, include_write: bool = False) -> list[dict]:
        """OpenAI function-calling 格式(MCP 注册时也复用同一份 schema)。"""
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.args.model_json_schema()}}
                for t in self.tools.values() if include_write or not t.write]

    def call(self, name: str, arguments: dict | None = None, *, allow_write: bool = False) -> Any:
        tool = self.tools.get(name)
        if tool is None:
            raise ToolError(f"未知工具 {name!r},可用: {sorted(self.tools)}")
        if tool.write and not allow_write:
            raise ToolError(f"工具 {name!r} 是写操作,当前调用方未被授权")
        try:
            args = tool.args.model_validate(arguments or {})
        except ValidationError as e:
            raise ToolError(f"参数错误: {e.errors(include_url=False)}") from e
        try:
            return tool.fn(self, args)
        except EngineError as e:
            raise ToolError(f"{type(e).__name__}: {e}") from e
