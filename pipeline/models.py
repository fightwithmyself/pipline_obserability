"""状态枚举、合法转换表、任务/步骤规格。纯数据,不依赖数据库。"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


# --------------------------------------------------------------------------- 状态
class StepStatus:
    PENDING = "PENDING"                    # 等待依赖 / 等待被认领
    RUNNING = "RUNNING"                    # 已被某 worker 认领(持有租约)
    RETRY_WAIT = "RETRY_WAIT"              # 失败后退避等待重试
    AWAITING_APPROVAL = "AWAITING_APPROVAL"  # 等待人工审批(不占 worker)
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"                      # 永久失败或重试预算耗尽(可人工重试)
    BLOCKED = "BLOCKED"                    # 上游失败,无法继续
    CANCELLED = "CANCELLED"


class TaskStatus:
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"                      # 无可运行步骤且存在失败步骤;可 resume
    CANCELLED = "CANCELLED"


ACTIVE_STEP = {
    StepStatus.PENDING,
    StepStatus.RUNNING,
    StepStatus.RETRY_WAIT,
    StepStatus.AWAITING_APPROVAL,
}
TERMINAL_STEP = {
    StepStatus.SUCCEEDED,
    StepStatus.FAILED,
    StepStatus.BLOCKED,
    StepStatus.CANCELLED,
}

S = StepStatus
ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    S.PENDING: {S.RUNNING, S.SUCCEEDED, S.BLOCKED, S.CANCELLED, S.AWAITING_APPROVAL},
    S.RETRY_WAIT: {S.RUNNING, S.SUCCEEDED, S.CANCELLED},
    S.RUNNING: {S.SUCCEEDED, S.RETRY_WAIT, S.FAILED, S.CANCELLED},
    S.AWAITING_APPROVAL: {S.SUCCEEDED, S.FAILED, S.CANCELLED},
    S.FAILED: {S.PENDING},
    S.BLOCKED: {S.PENDING},
    S.SUCCEEDED: set(),
    S.CANCELLED: set(),
}


class ErrorKind:
    TRANSIENT = "TRANSIENT"      # 瞬时错误:退避重试
    RESOURCE = "RESOURCE"        # 资源不足(如 GPU OOM):重试,步骤可据 attempt 降配
    PERMANENT = "PERMANENT"      # 永久错误:不重试
    LEASE_LOST = "LEASE_LOST"    # worker 失联/崩溃:不计入失败预算,立即重试
    TIMEOUT = "TIMEOUT"          # 总时限超出
    STALL = "STALL"              # 长时间无进展
    CANCELLED = "CANCELLED"


ALL_ERROR_KINDS = {
    ErrorKind.TRANSIENT, ErrorKind.RESOURCE, ErrorKind.PERMANENT, ErrorKind.LEASE_LOST,
    ErrorKind.TIMEOUT, ErrorKind.STALL, ErrorKind.CANCELLED,
}

MAX_LOST = 5          # 单个步骤因"租约丢失"可被回收的最大次数(防止毒丸步骤无限重跑)
RESOURCES = {"cpu", "gpu", "none"}   # none = 由引擎自己处理(如人工审批)


class SpecError(ValueError):
    """任务规格不合法(重复 key、未知依赖、成环……)。"""


# --------------------------------------------------------------------------- 规格
@dataclass
class RetryPolicy:
    max_attempts: int = 3        # 计入预算的失败次数上限
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.2


@dataclass
class StepSpec:
    key: str
    type: str
    deps: list[str] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    version: str = "1"                  # 步骤逻辑版本;变化会使缓存失效
    resource: str = "cpu"
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout_s: float = 600.0            # 单次尝试总时限
    stall_timeout_s: float = 30.0       # 单次尝试无进展时限
    cacheable: bool = True
    chaos: dict[str, Any] = field(default_factory=dict)   # 故障注入,不参与缓存 key

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "StepSpec":
        d = dict(d)
        d["retry"] = RetryPolicy(**d.get("retry", {}))
        return StepSpec(**d)


@dataclass
class TaskSpec:
    name: str
    steps: list[StepSpec]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "meta": self.meta, "steps": [s.to_dict() for s in self.steps]}

    @staticmethod
    def from_dict(d: dict) -> "TaskSpec":
        return TaskSpec(d["name"], [StepSpec.from_dict(s) for s in d["steps"]], d.get("meta", {}))


def validate_spec(spec: TaskSpec) -> None:
    """DAG 合法性检查:非空、key 唯一、依赖存在、无自依赖、无环。"""
    if not spec.steps:
        raise SpecError("任务至少需要一个步骤")
    keys = [s.key for s in spec.steps]
    if len(set(keys)) != len(keys):
        dup = sorted({k for k in keys if keys.count(k) > 1})
        raise SpecError(f"步骤 key 重复: {dup}")
    keyset = set(keys)
    for s in spec.steps:
        if s.resource not in RESOURCES:
            raise SpecError(f"步骤 {s.key} 的 resource 非法: {s.resource}")
        if s.retry.max_attempts < 1:
            raise SpecError(f"步骤 {s.key} 的 max_attempts 必须 >= 1")
        for d in s.deps:
            if d == s.key:
                raise SpecError(f"步骤 {s.key} 依赖自身")
            if d not in keyset:
                raise SpecError(f"步骤 {s.key} 依赖了不存在的步骤 {d}")
    # Kahn 拓扑排序检测环
    indeg = {s.key: len(set(s.deps)) for s in spec.steps}
    children: dict[str, list[str]] = {k: [] for k in keys}
    for s in spec.steps:
        for d in set(s.deps):
            children[d].append(s.key)
    queue = [k for k, n in indeg.items() if n == 0]
    seen = 0
    while queue:
        k = queue.pop()
        seen += 1
        for c in children[k]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if seen != len(keys):
        cyc = sorted(k for k, n in indeg.items() if n > 0)
        raise SpecError(f"依赖关系存在环: {cyc}")


# --------------------------------------------------------------------------- 工具
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_cache_key(step_type: str, version: str, params: dict, inputs: list[tuple[str, str, str]]) -> str:
    """cache_key = sha256(类型, 版本, 规范化参数, 输入产物 hash 列表)。chaos 不参与。"""
    payload = canonical_json({"type": step_type, "version": version, "params": params,
                              "inputs": sorted(inputs)})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
