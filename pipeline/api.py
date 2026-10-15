"""查询/控制接口(FastAPI)。核心逻辑全在 Engine 里,这里只做协议转换与错误映射。

只读接口给 Dashboard 和上游系统轮询;写接口(提交/取消/恢复/重试/审批)都是对 Engine 单入口的薄封装。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .engine import Engine, EngineError, InvalidAction, NotFound
from .models import RetryPolicy, SpecError, TaskSpec
from .templates import animation_swap_task

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class AnimationTaskRequest(BaseModel):
    name: str = "character-swap"
    source: str = "demo.mp4"
    frames: int = Field(24, ge=1, le=100000)
    chunks: int = Field(4, ge=1, le=64)
    prompt: str = "red-haired girl"
    target: str = "asset-library"
    require_approval: bool = False
    qc_fail: bool = False
    nondeterministic_ref: bool = False
    unit_s: float = Field(0.5, gt=0, le=60)
    stall_timeout_s: float = Field(30.0, gt=0)
    max_attempts: int = Field(4, ge=1, le=20)
    base_delay: float = Field(1.0, ge=0)
    chaos: dict[str, dict[str, Any]] | None = None
    request: str | None = None


class ApprovalRequest(BaseModel):
    decision: str
    who: str = "dashboard"
    comment: str = ""


def create_app(data_dir: str | Path) -> FastAPI:
    app = FastAPI(title="Observable Task Pipeline", version="0.1")
    data_dir = Path(data_dir)
    Engine(data_dir).close()          # 确保库和表存在

    def get_engine() -> Iterator[Engine]:
        e = Engine(data_dir)
        try:
            yield e
        finally:
            e.close()

    @app.exception_handler(NotFound)
    async def _nf(_, exc: NotFound):
        return JSONResponse({"detail": f"not found: {exc}"}, status_code=404)

    @app.exception_handler(InvalidAction)
    async def _ia(_, exc: InvalidAction):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(SpecError)
    async def _se(_, exc: SpecError):
        return JSONResponse({"detail": f"invalid spec: {exc}"}, status_code=422)

    # ------------------------------------------------------------ 提交
    @app.post("/api/tasks/animation", status_code=201)
    def submit_animation(req: AnimationTaskRequest, idempotency_key: str | None = Header(default=None),
                         eng: Engine = Depends(get_engine)):
        try:
            spec = animation_swap_task(
                req.name, source=req.source, frames=req.frames, chunks=req.chunks, prompt=req.prompt,
                target=req.target, require_approval=req.require_approval, qc_fail=req.qc_fail,
                nondeterministic_ref=req.nondeterministic_ref, unit_s=req.unit_s,
                stall_timeout_s=req.stall_timeout_s, chaos=req.chaos,
                retry=RetryPolicy(max_attempts=req.max_attempts, base_delay=req.base_delay, max_delay=30, jitter=0.2))
        except ValueError as e:
            raise HTTPException(422, str(e))
        tid, created = eng.submit(spec, idempotency_key=idempotency_key, request=req.request)
        return {"id": tid, "created": created}

    @app.post("/api/tasks", status_code=201)
    def submit_spec(spec: dict = Body(...), idempotency_key: str | None = Header(default=None),
                    eng: Engine = Depends(get_engine)):
        """提交通用任务规格(TaskSpec 的 JSON 形式)。"""
        try:
            ts = TaskSpec.from_dict(spec)
        except (KeyError, TypeError) as e:
            raise HTTPException(422, f"规格格式错误: {e!r}")
        tid, created = eng.submit(ts, idempotency_key=idempotency_key)
        return {"id": tid, "created": created}

    # ------------------------------------------------------------ 查询
    @app.get("/api/tasks")
    def list_tasks(status: str | None = None, limit: int = Query(50, ge=1, le=500),
                   eng: Engine = Depends(get_engine)):
        return eng.list_tasks(limit, status)

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str, eng: Engine = Depends(get_engine)):
        return eng.get_task(task_id)

    @app.get("/api/tasks/{task_id}/events")
    def task_events(task_id: str, after: int = 0, limit: int = Query(1000, ge=1, le=5000),
                    eng: Engine = Depends(get_engine)):
        eng.get_task(task_id)          # 404 检查
        return eng.get_events(task_id, after, limit)

    @app.get("/api/events")
    def all_events(after: int = 0, limit: int = Query(1000, ge=1, le=5000), eng: Engine = Depends(get_engine)):
        return eng.get_events(None, after, limit)

    @app.get("/api/tasks/{task_id}/steps/{key}/attempts")
    def step_attempts(task_id: str, key: str, eng: Engine = Depends(get_engine)):
        return eng.get_attempts(task_id, key)

    @app.get("/api/workers")
    def workers(eng: Engine = Depends(get_engine)):
        return eng.list_workers()

    @app.get("/api/metrics")
    def metrics(eng: Engine = Depends(get_engine)):
        return eng.metrics()

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    # ------------------------------------------------------------ 控制
    @app.post("/api/tasks/{task_id}/cancel")
    def cancel(task_id: str, eng: Engine = Depends(get_engine)):
        eng.cancel_task(task_id, "api")
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/resume")
    def resume(task_id: str, eng: Engine = Depends(get_engine)):
        return {"reset_steps": eng.resume_task(task_id)}

    @app.post("/api/tasks/{task_id}/steps/{key}/retry")
    def retry(task_id: str, key: str, eng: Engine = Depends(get_engine)):
        eng.retry_step(task_id, key)
        return {"ok": True}

    @app.post("/api/tasks/{task_id}/steps/{key}/approve")
    def approve(task_id: str, key: str, body: ApprovalRequest, eng: Engine = Depends(get_engine)):
        eng.approve(task_id, key, body.decision, body.who, body.comment)
        return {"ok": True}

    # ------------------------------------------------------------ Dashboard
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    return app
