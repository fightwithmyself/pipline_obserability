"""步骤注册表。步骤函数签名:fn(ctx: StepContext) -> None,产物通过 ctx.output(...) 登记。"""
from __future__ import annotations

from typing import Callable

REGISTRY: dict[str, Callable] = {}


def register(name: str):
    def deco(fn: Callable) -> Callable:
        REGISTRY[name] = fn
        return fn
    return deco


def known_step_types() -> set[str]:
    from . import animation, frames, kling, vst  # noqa: F401  触发注册
    return set(REGISTRY) | {"human_approval"}
