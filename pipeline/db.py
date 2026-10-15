"""SQLite 连接、事务封装、表结构。

约定:
- 连接使用 autocommit 模式(isolation_level=None),写操作统一用 `tx()` 显式开 `BEGIN IMMEDIATE`,
  这样"读-判断-写"在同一把写锁下完成,多进程 worker 抢占同一步骤时天然串行。
- WAL 模式:读(API/Dashboard)不阻塞写。
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    status          TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    spec            TEXT NOT NULL,          -- 完整任务规格(JSON),恢复只依赖它
    request         TEXT,                   -- 用户自然语言需求(Agent 场景)
    plan_id         INTEGER,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    finished_at     REAL
);

CREATE TABLE IF NOT EXISTS steps (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id          TEXT NOT NULL REFERENCES tasks(id),
    key              TEXT NOT NULL,
    type             TEXT NOT NULL,
    version          TEXT NOT NULL,
    params           TEXT NOT NULL,
    chaos            TEXT NOT NULL DEFAULT '{}',
    resource         TEXT NOT NULL,
    status           TEXT NOT NULL,
    cacheable        INTEGER NOT NULL DEFAULT 1,
    -- 重试策略
    max_attempts     INTEGER NOT NULL,
    base_delay       REAL NOT NULL,
    max_delay        REAL NOT NULL,
    jitter           REAL NOT NULL,
    timeout_s        REAL NOT NULL,
    stall_timeout_s  REAL NOT NULL,
    -- 执行状态
    attempt_no       INTEGER NOT NULL DEFAULT 0,   -- 已开始的尝试次数,同时是 fencing token
    failures         INTEGER NOT NULL DEFAULT 0,   -- 计入预算的失败次数
    lost_count       INTEGER NOT NULL DEFAULT 0,   -- 租约丢失次数(单独预算)
    next_run_at      REAL NOT NULL,
    lease_owner      TEXT,
    lease_expires_at REAL,
    last_progress_at REAL,
    progress         REAL NOT NULL DEFAULT 0,
    progress_msg     TEXT,
    checkpoint       TEXT,
    cache_key        TEXT,
    cached           INTEGER NOT NULL DEFAULT 0,
    outputs          TEXT,                         -- {name: {sha256, size}}
    error_kind       TEXT,
    error_msg        TEXT,
    created_at       REAL NOT NULL,
    started_at       REAL,
    finished_at      REAL,
    UNIQUE(task_id, key)
);
CREATE INDEX IF NOT EXISTS idx_steps_status ON steps(status, next_run_at);
CREATE INDEX IF NOT EXISTS idx_steps_task ON steps(task_id);

CREATE TABLE IF NOT EXISTS step_deps (
    step_id INTEGER NOT NULL REFERENCES steps(id),
    dep_id  INTEGER NOT NULL REFERENCES steps(id),
    PRIMARY KEY (step_id, dep_id)
);

CREATE TABLE IF NOT EXISTS attempts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    step_id     INTEGER NOT NULL REFERENCES steps(id),
    attempt_no  INTEGER NOT NULL,
    worker_id   TEXT,
    child_pid   INTEGER,
    started_at  REAL NOT NULL,
    finished_at REAL,
    outcome     TEXT NOT NULL,                 -- RUNNING/SUCCEEDED/FAILED/LOST/CANCELLED
    error_kind  TEXT,
    error_msg   TEXT,
    notes       TEXT,                          -- 步骤上报的诊断信息(如 resumed_from)
    UNIQUE(step_id, attempt_no)
);

CREATE TABLE IF NOT EXISTS artifacts (
    sha256     TEXT PRIMARY KEY,
    size       INTEGER NOT NULL,
    created_at REAL NOT NULL,
    first_step INTEGER
);

CREATE TABLE IF NOT EXISTS step_cache (
    cache_key  TEXT PRIMARY KEY,
    outputs    TEXT NOT NULL,
    created_at REAL NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL NOT NULL,
    task_id     TEXT NOT NULL,
    step_key    TEXT,
    kind        TEXT NOT NULL,
    from_status TEXT,
    to_status   TEXT,
    attempt_no  INTEGER,
    detail      TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_task ON events(task_id, seq);

CREATE TABLE IF NOT EXISTS workers (
    id           TEXT PRIMARY KEY,
    resources    TEXT NOT NULL,
    pid          INTEGER,
    started_at   REAL NOT NULL,
    last_seen    REAL NOT NULL,
    current_step INTEGER
);

CREATE TABLE IF NOT EXISTS approvals (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id  TEXT NOT NULL,
    step_key TEXT NOT NULL,
    decision TEXT NOT NULL,
    who      TEXT,
    comment  TEXT,
    ts       REAL NOT NULL
);

-- ---------------------------------------------------------------- Agent 层
-- 每次 LLM 调用的追踪(类似 LangSmith/Langfuse 的最小自建版)
CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL NOT NULL,
    task_id           TEXT,
    plan_id           INTEGER,
    purpose           TEXT NOT NULL,          -- plan / plan_repair / diagnose ...
    provider          TEXT NOT NULL,          -- mock / openai-compat
    model             TEXT NOT NULL,
    messages          TEXT NOT NULL,          -- 请求 messages(JSON)
    output            TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    tokens_estimated  INTEGER NOT NULL DEFAULT 0,   -- 1 = 非供应商返回的真实用量
    latency_s         REAL,
    cost_usd          REAL,
    status            TEXT NOT NULL,          -- ok / error
    error             TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_task ON llm_calls(task_id);

-- Planner 的每次规划:LLM 原始输出、校验错误、最终规格。恢复任务时不再询问 LLM,以此为准。
CREATE TABLE IF NOT EXISTS plans (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       REAL NOT NULL,
    request  TEXT NOT NULL,
    status   TEXT NOT NULL,                   -- PLANNING / OK / REJECTED
    attempts TEXT,                            -- [{output, errors}]
    spec     TEXT,
    task_id  TEXT
);

-- Diagnosis Agent 运行索引(图状态本身在 LangGraph checkpointer 里)
CREATE TABLE IF NOT EXISTS diagnoses (
    thread_id  TEXT PRIMARY KEY,
    task_id    TEXT NOT NULL,
    step_key   TEXT NOT NULL,
    status     TEXT NOT NULL,                 -- AWAITING_HUMAN / DONE
    proposal   TEXT,
    decision   TEXT,
    result     TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")   # WAL+NORMAL: 进程崩溃不丢已提交事务;整机掉电可能丢最后几笔
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


@contextmanager
def tx(conn: sqlite3.Connection):
    """写事务。BEGIN IMMEDIATE 立即拿写锁,保证事务内"读-判断-写"不被其他进程插入。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")
