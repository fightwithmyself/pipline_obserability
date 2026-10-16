"""执行引擎:任务/步骤状态的唯一写入口。

正确性要点(详见 docs/DESIGN.md):
1. 所有写操作都在 `BEGIN IMMEDIATE` 事务内完成;状态变更与事件追加在同一事务。
2. 状态只能通过 `_transition` 变更,并校验合法转换表。
3. `attempt_no` 是 fencing token:complete/fail/heartbeat 都必须带它且步骤仍处于 RUNNING,
   租约被回收后旧 worker(僵尸)的任何写入都会被拒绝。
4. "可运行"不落库,是派生谓词:状态 ∈ {PENDING, RETRY_WAIT} ∧ 到点 ∧ 所有依赖 SUCCEEDED。
5. 任务状态由步骤状态派生,并在同一事务内物化到 tasks 表。
"""
from __future__ import annotations

import json
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .artifacts import ArtifactStore
from .db import connect, init_schema, tx
from .models import (
    ACTIVE_STEP, ALLOWED_TRANSITIONS, MAX_LOST, ErrorKind, StepSpec, StepStatus as S,
    TaskSpec, TaskStatus as T, compute_cache_key, validate_spec,
)


class EngineError(Exception):
    pass


class IllegalTransition(EngineError):
    pass


class NotFound(EngineError):
    pass


class InvalidAction(EngineError):
    pass


@dataclass
class Claim:
    """worker 认领到的一次执行。"""
    step_id: int
    task_id: str
    key: str
    type: str
    params: dict
    chaos: dict
    attempt_no: int
    checkpoint: dict | None
    inputs: dict          # {dep_key: {output_name: {"sha256","size","path"}}}
    timeout_s: float
    stall_timeout_s: float
    cache_key: str
    lease_ttl: float


def _j(v: Any) -> str | None:
    return None if v is None else json.dumps(v, ensure_ascii=False)


def _unj(s: str | None) -> Any:
    return None if s is None else json.loads(s)


class Engine:
    def __init__(self, data_dir: str | Path, clock: Callable[[], float] = time.time):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.store = ArtifactStore(self.data_dir / "artifacts")
        self.conn = connect(self.data_dir / "pipeline.db")
        init_schema(self.conn)
        self.clock = clock
        self.rng = random.Random()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ 内部:事件与状态
    def _emit(self, task_id: str, step_key: str | None, kind: str, frm: str | None = None,
              to: str | None = None, attempt_no: int | None = None, detail: dict | None = None) -> None:
        self.conn.execute(
            "INSERT INTO events(ts, task_id, step_key, kind, from_status, to_status, attempt_no, detail)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (self.clock(), task_id, step_key, kind, frm, to, attempt_no, _j(detail)),
        )

    def _transition(self, step_id: int, to: str, detail: dict | None = None, **fields: Any) -> None:
        """步骤状态变更的唯一入口:校验转换、更新字段、追加事件。必须在事务内调用。"""
        row = self.conn.execute(
            "SELECT task_id, key, status, attempt_no FROM steps WHERE id=?", (step_id,)
        ).fetchone()
        if row is None:
            raise NotFound(f"step {step_id}")
        frm = row["status"]
        if to not in ALLOWED_TRANSITIONS[frm]:
            raise IllegalTransition(f"step {row['key']}: {frm} -> {to} 不合法")
        cols = ["status=?"]
        vals: list[Any] = [to]
        for c, v in fields.items():
            cols.append(f"{c}=?")
            vals.append(v)
        vals.append(step_id)
        self.conn.execute(f"UPDATE steps SET {', '.join(cols)} WHERE id=?", vals)
        self._emit(row["task_id"], row["key"], "STEP_STATUS", frm, to,
                   fields.get("attempt_no", row["attempt_no"]), detail)

    def _refresh_task(self, task_id: str) -> None:
        """由步骤状态派生任务状态,同事务物化。"""
        task = self.conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
        if task is None or task["status"] == T.CANCELLED:
            return
        sts = [r["status"] for r in self.conn.execute(
            "SELECT status FROM steps WHERE task_id=?", (task_id,))]
        active = any(s in ACTIVE_STEP for s in sts)
        if active:
            untouched = all(s == S.PENDING for s in sts)
            new = T.PENDING if untouched else T.RUNNING
        elif any(s == S.FAILED for s in sts):
            new = T.FAILED
        elif all(s == S.SUCCEEDED for s in sts):
            new = T.SUCCEEDED
        else:
            new = T.CANCELLED
        old = task["status"]
        now = self.clock()
        if new != old:
            finished = now if new in (T.SUCCEEDED, T.FAILED, T.CANCELLED) else None
            self.conn.execute("UPDATE tasks SET status=?, updated_at=?, finished_at=? WHERE id=?",
                              (new, now, finished, task_id))
            self._emit(task_id, None, "TASK_STATUS", old, new)
        else:
            self.conn.execute("UPDATE tasks SET updated_at=? WHERE id=?", (now, task_id))

    def _recompute_blocked(self, task_id: str) -> None:
        """失败步骤的下游置 BLOCKED;失败消除后解除。按固定点迭代,结果与调用顺序无关。"""
        rows = self.conn.execute(
            "SELECT id, status FROM steps WHERE task_id=?", (task_id,)).fetchall()
        status = {r["id"]: r["status"] for r in rows}
        deps: dict[int, list[int]] = {i: [] for i in status}
        for d in self.conn.execute(
                "SELECT d.step_id, d.dep_id FROM step_deps d JOIN steps s ON s.id=d.step_id WHERE s.task_id=?",
                (task_id,)):
            deps[d["step_id"]].append(d["dep_id"])
        failed = {i for i, s in status.items() if s == S.FAILED}
        blocked: set[int] = set()
        changed = True
        while changed:
            changed = False
            for i, s in status.items():
                if s in (S.PENDING, S.BLOCKED) and i not in blocked:
                    if any(d in failed or d in blocked for d in deps[i]):
                        blocked.add(i)
                        changed = True
        for i, s in status.items():
            if s == S.PENDING and i in blocked:
                self._transition(i, S.BLOCKED, {"reason": "upstream failed"})
            elif s == S.BLOCKED and i not in blocked:
                self._transition(i, S.PENDING, {"reason": "upstream recovered"})

    def _finish_attempt(self, step_id: int, attempt_no: int, outcome: str, kind: str | None,
                        msg: str | None, notes: dict | None, finished_at: float) -> None:
        self.conn.execute(
            "UPDATE attempts SET finished_at=?, outcome=?, error_kind=?, error_msg=?,"
            " notes=COALESCE(?, notes) WHERE step_id=? AND attempt_no=?",
            (finished_at, outcome, kind, msg, _j(notes), step_id, attempt_no))

    def _backoff(self, row, failures: int) -> float:
        d = min(row["max_delay"], row["base_delay"] * (2 ** max(0, failures - 1)))
        j = row["jitter"]
        return max(0.0, d * (1 + self.rng.uniform(-j, j)))

    def _fail_locked(self, row, kind: str, msg: str, now: float, notes: dict | None = None,
                     died_at: float | None = None) -> str:
        """处理一次失败(含租约丢失)。返回新状态。row 为 steps 行。"""
        step_id, attempt_no = row["id"], row["attempt_no"]
        lost = kind == ErrorKind.LEASE_LOST
        failures = row["failures"] + (0 if lost else 1)
        lost_count = row["lost_count"] + (1 if lost else 0)
        self._finish_attempt(step_id, attempt_no, "LOST" if lost else "FAILED", kind, msg, notes,
                             died_at if died_at is not None else now)
        permanent = kind == ErrorKind.PERMANENT
        exhausted = failures >= row["max_attempts"] or lost_count > MAX_LOST
        common = dict(failures=failures, lost_count=lost_count, error_kind=kind, error_msg=msg,
                      lease_owner=None, lease_expires_at=None)
        if permanent or exhausted:
            reason = "permanent error" if permanent else (
                "lost budget exhausted" if lost_count > MAX_LOST else "retry budget exhausted")
            self._transition(step_id, S.FAILED, {"kind": kind, "msg": msg, "reason": reason},
                             finished_at=now, **common)
            self._recompute_blocked(row["task_id"])
            new = S.FAILED
        else:
            delay = 0.0 if lost else self._backoff(row, failures)
            self._transition(step_id, S.RETRY_WAIT, {"kind": kind, "msg": msg, "retry_in": round(delay, 3)},
                             next_run_at=now + delay, **common)
            new = S.RETRY_WAIT
        self._refresh_task(row["task_id"])
        return new

    def _reclaim_expired(self, now: float) -> int:
        rows = self.conn.execute(
            "SELECT * FROM steps WHERE status=? AND lease_expires_at < ?", (S.RUNNING, now)).fetchall()
        for r in rows:
            self._fail_locked(
                r, ErrorKind.LEASE_LOST,
                f"lease expired (owner={r['lease_owner']}, attempt={r['attempt_no']})", now,
                died_at=r["last_progress_at"] or r["started_at"])
        return len(rows)

    def _deps_ready_sql(self) -> str:
        return ("NOT EXISTS (SELECT 1 FROM step_deps d JOIN steps p ON p.id=d.dep_id "
                "WHERE d.step_id=s.id AND p.status != 'SUCCEEDED')")

    def _promote_approvals(self) -> None:
        rows = self.conn.execute(
            f"SELECT s.id FROM steps s WHERE s.type='human_approval' AND s.status=? AND "
            f"{self._deps_ready_sql()}", (S.PENDING,)).fetchall()
        for r in rows:
            self._transition(r["id"], S.AWAITING_APPROVAL, {"reason": "deps done, waiting for human"})
            tid = self.conn.execute("SELECT task_id FROM steps WHERE id=?", (r["id"],)).fetchone()["task_id"]
            self._refresh_task(tid)

    def _dep_outputs(self, step_id: int) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for r in self.conn.execute(
                "SELECT p.key, p.outputs FROM step_deps d JOIN steps p ON p.id=d.dep_id WHERE d.step_id=?",
                (step_id,)):
            out[r["key"]] = _unj(r["outputs"]) or {}
        return out

    def _cache_lookup(self, task_id: str, key: str, cache_key: str) -> dict | None:
        row = self.conn.execute("SELECT outputs FROM step_cache WHERE cache_key=?", (cache_key,)).fetchone()
        if row is None:
            return None
        outputs = json.loads(row["outputs"])
        bad = [n for n, o in outputs.items() if not self.store.verify(o["sha256"])]
        if bad:
            # 产物缺失/被篡改:视为未命中,删除缓存条目,重新计算
            self.conn.execute("DELETE FROM step_cache WHERE cache_key=?", (cache_key,))
            self._emit(task_id, key, "CACHE_INVALID", detail={"cache_key": cache_key, "bad_outputs": bad})
            return None
        self.conn.execute("UPDATE step_cache SET hits=hits+1 WHERE cache_key=?", (cache_key,))
        return outputs

    # ------------------------------------------------------------------ 提交
    def submit(self, spec: TaskSpec, idempotency_key: str | None = None, request: str | None = None,
               plan_id: int | None = None) -> tuple[str, bool]:
        """提交任务。返回 (task_id, created)。同一 idempotency_key 重复提交返回已有任务。"""
        validate_spec(spec)
        now = self.clock()
        with tx(self.conn):
            if idempotency_key:
                ex = self.conn.execute("SELECT id FROM tasks WHERE idempotency_key=?",
                                       (idempotency_key,)).fetchone()
                if ex:
                    return ex["id"], False
            task_id = "t_" + uuid.uuid4().hex[:12]
            self.conn.execute(
                "INSERT INTO tasks(id,name,status,idempotency_key,spec,request,plan_id,created_at,updated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (task_id, spec.name, T.PENDING, idempotency_key, json.dumps(spec.to_dict(), ensure_ascii=False),
                 request, plan_id, now, now))
            ids: dict[str, int] = {}
            for s in spec.steps:
                resource = "none" if s.type == "human_approval" else s.resource
                cur = self.conn.execute(
                    "INSERT INTO steps(task_id,key,type,version,params,chaos,resource,status,cacheable,"
                    "max_attempts,base_delay,max_delay,jitter,timeout_s,stall_timeout_s,next_run_at,created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (task_id, s.key, s.type, s.version, json.dumps(s.params, ensure_ascii=False),
                     json.dumps(s.chaos), resource, S.PENDING, int(s.cacheable),
                     s.retry.max_attempts, s.retry.base_delay, s.retry.max_delay, s.retry.jitter,
                     s.timeout_s, s.stall_timeout_s, now, now))
                ids[s.key] = cur.lastrowid
            for s in spec.steps:
                for d in set(s.deps):
                    self.conn.execute("INSERT INTO step_deps(step_id, dep_id) VALUES (?,?)",
                                      (ids[s.key], ids[d]))
            self._emit(task_id, None, "TASK_CREATED", to=T.PENDING,
                       detail={"name": spec.name, "steps": len(spec.steps), "request": request})
        return task_id, True

    # ------------------------------------------------------------------ worker 侧
    def register_worker(self, worker_id: str, resources: Iterable[str], pid: int | None = None) -> None:
        now = self.clock()
        with tx(self.conn):
            self.conn.execute(
                "INSERT INTO workers(id,resources,pid,started_at,last_seen) VALUES (?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET resources=excluded.resources, pid=excluded.pid,"
                " started_at=excluded.started_at, last_seen=excluded.last_seen, current_step=NULL",
                (worker_id, json.dumps(sorted(resources)), pid, now, now))

    def touch_worker(self, worker_id: str, current_step: int | None = None) -> None:
        with tx(self.conn):
            self.conn.execute("UPDATE workers SET last_seen=?, current_step=? WHERE id=?",
                              (self.clock(), current_step, worker_id))

    def sweep(self) -> int:
        """回收过期租约 + 推进审批步骤。任何进程都可调用。返回回收数。"""
        with tx(self.conn):
            n = self._reclaim_expired(self.clock())
            self._promote_approvals()
        return n

    def claim(self, worker_id: str, resources: Iterable[str] = ("cpu", "gpu"),
              lease_ttl: float = 10.0) -> Claim | None:
        """原子认领一个可运行步骤。认领前先回收过期租约,并尝试命中缓存(命中则不占用 worker)。"""
        res = list(resources)
        marks = ",".join("?" * len(res))
        with tx(self.conn):
            now = self.clock()
            self._reclaim_expired(now)
            self._promote_approvals()
            while True:
                row = self.conn.execute(
                    f"SELECT s.* FROM steps s WHERE s.status IN (?,?) AND s.next_run_at <= ? "
                    f"AND s.resource IN ({marks}) AND {self._deps_ready_sql()} "
                    f"ORDER BY s.next_run_at, s.id LIMIT 1",
                    (S.PENDING, S.RETRY_WAIT, now, *res)).fetchone()
                if row is None:
                    return None
                dep_out = self._dep_outputs(row["id"])
                inputs = [(dk, n, o["sha256"]) for dk, outs in dep_out.items() for n, o in outs.items()]
                params = json.loads(row["params"])
                cache_key = compute_cache_key(row["type"], row["version"], params, inputs)
                if row["cacheable"]:
                    hit = self._cache_lookup(row["task_id"], row["key"], cache_key)
                    if hit is not None:
                        self._transition(row["id"], S.SUCCEEDED, {"cached": True, "cache_key": cache_key},
                                         outputs=_j(hit), cached=1, cache_key=cache_key, progress=1.0,
                                         started_at=now, finished_at=now, lease_owner=None,
                                         lease_expires_at=None)
                        self._refresh_task(row["task_id"])
                        continue
                attempt_no = row["attempt_no"] + 1
                self._transition(
                    row["id"], S.RUNNING, {"worker": worker_id}, attempt_no=attempt_no,
                    lease_owner=worker_id, lease_expires_at=now + lease_ttl, last_progress_at=now,
                    progress=0.0, progress_msg=None, cache_key=cache_key, cached=0,
                    started_at=row["started_at"] or now, error_kind=None, error_msg=None)
                self.conn.execute(
                    "INSERT INTO attempts(step_id,attempt_no,worker_id,started_at,outcome) VALUES (?,?,?,?,?)",
                    (row["id"], attempt_no, worker_id, now, "RUNNING"))
                self._refresh_task(row["task_id"])
                inputs_resolved = {
                    dk: {n: {"sha256": o["sha256"], "size": o["size"], "path": str(self.store.path(o["sha256"]))}
                         for n, o in outs.items()}
                    for dk, outs in dep_out.items()}
                return Claim(
                    step_id=row["id"], task_id=row["task_id"], key=row["key"], type=row["type"],
                    params=params, chaos=json.loads(row["chaos"]), attempt_no=attempt_no,
                    checkpoint=_unj(row["checkpoint"]), inputs=inputs_resolved,
                    timeout_s=row["timeout_s"], stall_timeout_s=row["stall_timeout_s"],
                    cache_key=cache_key, lease_ttl=lease_ttl)

    def set_child_pid(self, step_id: int, attempt_no: int, pid: int) -> None:
        with tx(self.conn):
            self.conn.execute("UPDATE attempts SET child_pid=? WHERE step_id=? AND attempt_no=?",
                              (pid, step_id, attempt_no))

    def heartbeat(self, step_id: int, attempt_no: int, lease_ttl: float, progress: float | None = None,
                  msg: str | None = None, last_progress_at: float | None = None) -> bool:
        """续租并落盘最新进度。返回 False 表示租约已失效(被回收/取消),调用方必须停止执行。"""
        now = self.clock()
        with tx(self.conn):
            cur = self.conn.execute(
                "UPDATE steps SET lease_expires_at=?, progress=COALESCE(?,progress),"
                " progress_msg=COALESCE(?,progress_msg), last_progress_at=COALESCE(?,last_progress_at)"
                " WHERE id=? AND attempt_no=? AND status=?",
                (now + lease_ttl, progress, msg, last_progress_at, step_id, attempt_no, S.RUNNING))
            return cur.rowcount == 1

    def save_checkpoint(self, step_id: int, attempt_no: int, data: dict, lease_ttl: float | None = None) -> bool:
        now = self.clock()
        with tx(self.conn):
            if lease_ttl is None:
                cur = self.conn.execute(
                    "UPDATE steps SET checkpoint=?, last_progress_at=? WHERE id=? AND attempt_no=? AND status=?",
                    (_j(data), now, step_id, attempt_no, S.RUNNING))
            else:
                cur = self.conn.execute(
                    "UPDATE steps SET checkpoint=?, last_progress_at=?, lease_expires_at=?"
                    " WHERE id=? AND attempt_no=? AND status=?",
                    (_j(data), now, now + lease_ttl, step_id, attempt_no, S.RUNNING))
            return cur.rowcount == 1

    def upsert_item(self, step_id: int, attempt_no: int, item_key: str, status: str, data: dict | None) -> bool:
        """写入/更新一个工作项(如一个关键帧)。受 fencing 保护:旧 attempt 的更新被忽略。

        data 做浅合并:后续消息只需带变化的字段。这张表是派生的观测视图,不参与恢复决策。
        """
        now = self.clock()
        with tx(self.conn):
            row = self.conn.execute("SELECT task_id,key,attempt_no,status FROM steps WHERE id=?",
                                    (step_id,)).fetchone()
            if row is None or row["status"] != S.RUNNING or row["attempt_no"] != attempt_no:
                return False
            old = self.conn.execute("SELECT data FROM items WHERE task_id=? AND step_key=? AND item_key=?",
                                    (row["task_id"], row["key"], item_key)).fetchone()
            merged = {**(_unj(old["data"]) if old and old["data"] else {}), **(data or {})}
            self.conn.execute(
                "INSERT INTO items(task_id,step_key,item_key,status,data,updated_at) VALUES (?,?,?,?,?,?)"
                " ON CONFLICT(task_id,step_key,item_key) DO UPDATE SET status=excluded.status,"
                " data=excluded.data, updated_at=excluded.updated_at",
                (row["task_id"], row["key"], item_key, status, _j(merged), now))
            return True

    def list_items(self, task_id: str, step_key: str | None = None, prefix: str | None = None) -> list[dict]:
        q = "SELECT * FROM items WHERE task_id=?"
        args: list[Any] = [task_id]
        if step_key:
            q += " AND step_key=?"
            args.append(step_key)
        if prefix:
            q += " AND step_key LIKE ?"
            args.append(prefix + "%")
        q += " ORDER BY item_key, step_key"
        return [{"step_key": r["step_key"], "item_key": r["item_key"], "status": r["status"],
                 "updated_at": r["updated_at"], **(_unj(r["data"]) or {})}
                for r in self.conn.execute(q, args).fetchall()]

    def complete(self, step_id: int, attempt_no: int, outputs: dict[str, dict],
                 notes: dict | None = None) -> bool:
        """提交成功结果。fencing:attempt_no 不匹配或不再 RUNNING → 拒绝并留痕。"""
        now = self.clock()
        with tx(self.conn):
            row = self.conn.execute("SELECT * FROM steps WHERE id=?", (step_id,)).fetchone()
            if row is None:
                raise NotFound(f"step {step_id}")
            if row["status"] != S.RUNNING or row["attempt_no"] != attempt_no:
                self._emit(row["task_id"], row["key"], "COMMIT_REJECTED", attempt_no=attempt_no,
                           detail={"reason": "fenced", "current_status": row["status"],
                                   "current_attempt": row["attempt_no"]})
                return False
            for o in outputs.values():
                self.conn.execute("INSERT OR IGNORE INTO artifacts(sha256,size,created_at,first_step)"
                                  " VALUES (?,?,?,?)", (o["sha256"], o["size"], now, step_id))
            if row["cacheable"] and row["cache_key"]:
                self.conn.execute("INSERT OR REPLACE INTO step_cache(cache_key,outputs,created_at,hits)"
                                  " VALUES (?,?,?,COALESCE((SELECT hits FROM step_cache WHERE cache_key=?),0))",
                                  (row["cache_key"], _j(outputs), now, row["cache_key"]))
            self._finish_attempt(step_id, attempt_no, "SUCCEEDED", None, None, notes, now)
            self._transition(step_id, S.SUCCEEDED, {"attempt": attempt_no}, outputs=_j(outputs), progress=1.0,
                             finished_at=now, lease_owner=None, lease_expires_at=None,
                             error_kind=None, error_msg=None)
            self._refresh_task(row["task_id"])
            return True

    def fail(self, step_id: int, attempt_no: int, kind: str, msg: str, notes: dict | None = None) -> bool:
        """登记一次失败。同样受 fencing 保护。"""
        now = self.clock()
        with tx(self.conn):
            row = self.conn.execute("SELECT * FROM steps WHERE id=?", (step_id,)).fetchone()
            if row is None:
                raise NotFound(f"step {step_id}")
            if row["status"] != S.RUNNING or row["attempt_no"] != attempt_no:
                self._emit(row["task_id"], row["key"], "FAIL_REJECTED", attempt_no=attempt_no,
                           detail={"reason": "fenced", "kind": kind})
                return False
            self._fail_locked(row, kind, msg, now, notes)
            return True

    # ------------------------------------------------------------------ 控制面
    def cancel_task(self, task_id: str, reason: str = "user") -> None:
        now = self.clock()
        with tx(self.conn):
            task = self.conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
            if task is None:
                raise NotFound(task_id)
            if task["status"] in (T.SUCCEEDED, T.CANCELLED):
                raise InvalidAction(f"任务已是终态 {task['status']}")
            for r in self.conn.execute("SELECT id, status, attempt_no FROM steps WHERE task_id=?",
                                       (task_id,)).fetchall():
                if r["status"] in ACTIVE_STEP:
                    if r["status"] == S.RUNNING:
                        self._finish_attempt(r["id"], r["attempt_no"], "CANCELLED", ErrorKind.CANCELLED,
                                             reason, None, now)
                    self._transition(r["id"], S.CANCELLED, {"reason": reason}, finished_at=now,
                                     lease_owner=None, lease_expires_at=None)
            old = task["status"]
            self.conn.execute("UPDATE tasks SET status=?, updated_at=?, finished_at=? WHERE id=?",
                              (T.CANCELLED, now, now, task_id))
            self._emit(task_id, None, "TASK_STATUS", old, T.CANCELLED, detail={"reason": reason})

    def _reset_failed_step(self, row, now: float) -> None:
        self._transition(row["id"], S.PENDING, {"reason": "manual retry"}, failures=0, lost_count=0,
                         next_run_at=now, error_kind=None, error_msg=None, finished_at=None)

    def retry_step(self, task_id: str, key: str) -> None:
        """人工重试某个失败步骤:预算重置,历史保留;已成功的上游不重跑。"""
        now = self.clock()
        with tx(self.conn):
            row = self.conn.execute("SELECT * FROM steps WHERE task_id=? AND key=?", (task_id, key)).fetchone()
            if row is None:
                raise NotFound(f"{task_id}/{key}")
            task = self.conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
            if task["status"] == T.CANCELLED:
                raise InvalidAction("任务已取消")
            if row["status"] != S.FAILED:
                raise InvalidAction(f"只有 FAILED 步骤可重试,当前 {row['status']}")
            self._emit(task_id, key, "ACTION", detail={"action": "retry_step"})
            self._reset_failed_step(row, now)
            self._recompute_blocked(task_id)
            self._refresh_task(task_id)

    def resume_task(self, task_id: str) -> int:
        """重试任务内所有失败步骤。返回重置的步骤数。"""
        now = self.clock()
        with tx(self.conn):
            task = self.conn.execute("SELECT status FROM tasks WHERE id=?", (task_id,)).fetchone()
            if task is None:
                raise NotFound(task_id)
            if task["status"] == T.CANCELLED:
                raise InvalidAction("任务已取消")
            rows = self.conn.execute("SELECT * FROM steps WHERE task_id=? AND status=?",
                                     (task_id, S.FAILED)).fetchall()
            if not rows:
                raise InvalidAction("没有失败的步骤可恢复")
            self._emit(task_id, None, "ACTION", detail={"action": "resume_task", "steps": len(rows)})
            for r in rows:
                self._reset_failed_step(r, now)
            self._recompute_blocked(task_id)
            self._refresh_task(task_id)
            return len(rows)

    def approve(self, task_id: str, key: str, decision: str, who: str = "anonymous", comment: str = "") -> None:
        """人工审批。decision: approve | reject。"""
        if decision not in ("approve", "reject"):
            raise InvalidAction("decision 必须是 approve 或 reject")
        now = self.clock()
        with tx(self.conn):
            row = self.conn.execute("SELECT * FROM steps WHERE task_id=? AND key=?", (task_id, key)).fetchone()
            if row is None:
                raise NotFound(f"{task_id}/{key}")
            if row["status"] != S.AWAITING_APPROVAL:
                raise InvalidAction(f"步骤不在等待审批状态,当前 {row['status']}")
            self.conn.execute("INSERT INTO approvals(task_id,step_key,decision,who,comment,ts) VALUES (?,?,?,?,?,?)",
                              (task_id, key, decision, who, comment, now))
            if decision == "approve":
                self._transition(row["id"], S.SUCCEEDED, {"by": who, "comment": comment}, outputs=_j({}),
                                 progress=1.0, finished_at=now, started_at=row["started_at"] or now)
            else:
                self._transition(row["id"], S.FAILED, {"by": who, "comment": comment},
                                 error_kind=ErrorKind.PERMANENT, error_msg=f"rejected by {who}: {comment}",
                                 finished_at=now)
                self._recompute_blocked(task_id)
            self._refresh_task(task_id)

    def add_event(self, task_id: str, kind: str, step_key: str | None = None, detail: dict | None = None) -> None:
        """追加一条非状态变更类事件(如 Agent 的诊断/动作记录),进入同一条可查询的历史。"""
        with tx(self.conn):
            if self.conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone() is None:
                raise NotFound(task_id)
            self._emit(task_id, step_key, kind, detail=detail)

    # ------------------------------------------------------------------ 查询
    def _step_view(self, r, dep_keys: list[str]) -> dict:
        return {
            "key": r["key"], "type": r["type"], "status": r["status"], "resource": r["resource"],
            "deps": dep_keys, "attempt_no": r["attempt_no"], "failures": r["failures"],
            "lost_count": r["lost_count"], "max_attempts": r["max_attempts"],
            "progress": r["progress"], "progress_msg": r["progress_msg"], "cached": bool(r["cached"]),
            "error_kind": r["error_kind"], "error_msg": r["error_msg"], "next_run_at": r["next_run_at"],
            "lease_owner": r["lease_owner"], "lease_expires_at": r["lease_expires_at"],
            "started_at": r["started_at"], "finished_at": r["finished_at"],
            "cache_key": r["cache_key"], "checkpoint": _unj(r["checkpoint"]), "outputs": _unj(r["outputs"]),
            "params": json.loads(r["params"]),
        }

    def get_task(self, task_id: str) -> dict:
        t = self.conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if t is None:
            raise NotFound(task_id)
        deps: dict[int, list[str]] = {}
        for d in self.conn.execute(
                "SELECT d.step_id, p.key FROM step_deps d JOIN steps p ON p.id=d.dep_id "
                "JOIN steps s ON s.id=d.step_id WHERE s.task_id=?", (task_id,)):
            deps.setdefault(d["step_id"], []).append(d["key"])
        rows = self.conn.execute("SELECT * FROM steps WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
        steps = [self._step_view(r, sorted(deps.get(r["id"], []))) for r in rows]
        total = len(steps) or 1
        done = sum(1 for s in steps if s["status"] == S.SUCCEEDED)
        partial = sum(s["progress"] for s in steps if s["status"] == S.RUNNING)
        by_status: dict[str, int] = {}
        for s in steps:
            by_status[s["status"]] = by_status.get(s["status"], 0) + 1
        return {
            "id": t["id"], "name": t["name"], "status": t["status"], "request": t["request"],
            "created_at": t["created_at"], "updated_at": t["updated_at"], "finished_at": t["finished_at"],
            "percent": round((done + partial) / total * 100, 1),
            "steps_total": len(steps), "steps_succeeded": done,
            "steps_by_status": by_status,
            "running_steps": [s["key"] for s in steps if s["status"] == S.RUNNING],
            "cached_steps": sum(1 for s in steps if s["cached"]),
            "has_failed_steps": by_status.get(S.FAILED, 0) > 0,
            "waiting_approval": [s["key"] for s in steps if s["status"] == S.AWAITING_APPROVAL],
            "steps": steps,
        }

    def list_tasks(self, limit: int = 50, status: str | None = None) -> list[dict]:
        q = "SELECT id FROM tasks"
        args: list[Any] = []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        out = []
        for r in self.conn.execute(q, args).fetchall():
            d = self.get_task(r["id"])
            d.pop("steps")
            out.append(d)
        return out

    def get_events(self, task_id: str | None = None, after_seq: int = 0, limit: int = 1000) -> list[dict]:
        q = "SELECT * FROM events WHERE seq > ?"
        args: list[Any] = [after_seq]
        if task_id:
            q += " AND task_id=?"
            args.append(task_id)
        q += " ORDER BY seq LIMIT ?"
        args.append(limit)
        return [{"seq": r["seq"], "ts": r["ts"], "task_id": r["task_id"], "step_key": r["step_key"],
                 "kind": r["kind"], "from": r["from_status"], "to": r["to_status"],
                 "attempt_no": r["attempt_no"], "detail": _unj(r["detail"])}
                for r in self.conn.execute(q, args)]

    def get_attempts(self, task_id: str, key: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT a.* FROM attempts a JOIN steps s ON s.id=a.step_id WHERE s.task_id=? AND s.key=?"
            " ORDER BY a.attempt_no", (task_id, key)).fetchall()
        if not rows and self.conn.execute("SELECT 1 FROM steps WHERE task_id=? AND key=?",
                                          (task_id, key)).fetchone() is None:
            raise NotFound(f"{task_id}/{key}")
        return [{"attempt_no": r["attempt_no"], "worker_id": r["worker_id"], "child_pid": r["child_pid"],
                 "started_at": r["started_at"], "finished_at": r["finished_at"], "outcome": r["outcome"],
                 "error_kind": r["error_kind"], "error_msg": r["error_msg"], "notes": _unj(r["notes"]),
                 "duration": (r["finished_at"] - r["started_at"]) if r["finished_at"] else None}
                for r in rows]

    def list_workers(self, alive_within: float = 15.0) -> list[dict]:
        now = self.clock()
        return [{"id": r["id"], "resources": json.loads(r["resources"]), "pid": r["pid"],
                 "last_seen": r["last_seen"], "alive": now - r["last_seen"] <= alive_within,
                 "current_step": r["current_step"]}
                for r in self.conn.execute("SELECT * FROM workers ORDER BY id")]

    def metrics(self) -> dict:
        c = self.conn
        steps = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM steps GROUP BY status")}
        tasks = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM tasks GROUP BY status")}
        att = {r["outcome"]: r["n"] for r in c.execute("SELECT outcome, COUNT(*) n FROM attempts GROUP BY outcome")}
        errs = {r["error_kind"]: r["n"] for r in c.execute(
            "SELECT error_kind, COUNT(*) n FROM attempts WHERE error_kind IS NOT NULL GROUP BY error_kind")}
        eff = c.execute("SELECT COALESCE(SUM(finished_at-started_at),0) v FROM attempts WHERE outcome='SUCCEEDED'"
                        ).fetchone()["v"]
        waste = c.execute("SELECT COALESCE(SUM(finished_at-started_at),0) v FROM attempts "
                          "WHERE outcome IN ('FAILED','LOST') AND finished_at IS NOT NULL").fetchone()["v"]
        cached = c.execute("SELECT COUNT(*) n FROM steps WHERE cached=1").fetchone()["n"]
        done = steps.get(S.SUCCEEDED, 0)
        return {
            "tasks_by_status": tasks, "steps_by_status": steps, "attempts_by_outcome": att,
            "errors_by_kind": errs, "effective_seconds": round(eff, 3), "wasted_seconds": round(waste, 3),
            "waste_ratio": round(waste / eff, 3) if eff else None,
            "cache_hit_steps": cached, "cache_hit_ratio": round(cached / done, 3) if done else None,
            "queue_depth": sum(steps.get(k, 0) for k in (S.PENDING, S.RETRY_WAIT)),
        }
