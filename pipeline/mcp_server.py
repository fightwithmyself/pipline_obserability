"""MCP Server:把引擎的受控工具集暴露给任意 MCP 客户端(Cursor / Claude Desktop 等)。

    python -m pipeline.mcp_server --data data                 # 只读工具(默认)
    PIPELINE_MCP_WRITE=1 python -m pipeline.mcp_server --data data   # 额外开放 retry/resume/cancel/submit

安全:写工具默认关闭;**没有 approve 工具**(人工审批只能由人走 API)。
工具定义与 Agent 共用 `ToolRegistry`,参数 schema 来自同一份 Pydantic 模型。
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from .agent.llm import TracedLLM, make_llm
from .agent.planner import Planner
from .agent.tools import ToolError, ToolRegistry
from .engine import Engine


def build_server(data_dir: str | Path, allow_write: bool = False, with_planner: bool = True) -> MCPServer:
    engine = Engine(data_dir)
    planner = Planner(TracedLLM(make_llm(), engine), engine) if (allow_write and with_planner) else None
    registry = ToolRegistry(engine, planner)
    server = MCPServer("task-pipeline", instructions="查询与管理可观测任务执行管线中的动画素材处理任务。")

    def make_handler(name: str):
        # MCP 工具以关键字参数调用;统一转交 registry,保证校验/授权逻辑只有一份
        def handler(**kwargs: Any) -> str:
            try:
                return json.dumps(registry.call(name, kwargs, allow_write=allow_write), ensure_ascii=False, default=str)
            except ToolError as e:
                return json.dumps({"error": str(e)}, ensure_ascii=False)
        handler.__name__ = name
        return handler

    for tool in registry.tools.values():
        if tool.write and not allow_write:
            continue
        # 用工具自己的 Pydantic 模型作为参数 schema:通过带类型注解的包装函数注册
        server.add_tool(_typed_wrapper(tool, make_handler(tool.name)), name=tool.name, description=tool.description)
    return server


def _typed_wrapper(tool, handler):
    """为 MCPServer 生成与 Pydantic 模型同签名的函数,使其推导出一致的 JSON Schema。"""
    import inspect
    sig_params = []
    annotations: dict[str, Any] = {}
    for fname, field in tool.args.model_fields.items():
        default = inspect.Parameter.empty if field.is_required() else field.default
        sig_params.append(inspect.Parameter(fname, inspect.Parameter.KEYWORD_ONLY, default=default,
                                            annotation=field.annotation))
        annotations[fname] = field.annotation

    def wrapper(**kwargs: Any) -> str:
        return handler(**kwargs)

    wrapper.__signature__ = inspect.Signature(sig_params, return_annotation=str)   # type: ignore[attr-defined]
    wrapper.__annotations__ = {**annotations, "return": str}
    wrapper.__name__ = tool.name
    wrapper.__doc__ = tool.description
    return wrapper


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.environ.get("PIPELINE_DATA", "data"))
    a = p.parse_args()
    allow_write = os.environ.get("PIPELINE_MCP_WRITE") == "1"
    build_server(a.data, allow_write).run("stdio")


if __name__ == "__main__":
    main()
