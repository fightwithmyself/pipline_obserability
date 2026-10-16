"""查询/控制接口(FastAPI)。核心逻辑全在 Engine 里,这里只做协议转换与错误映射。

只读接口给 Dashboard 和上游系统轮询;写接口(提交/取消/恢复/重试/审批)都是对 Engine 单入口的薄封装。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Literal

from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from .agent.diagnosis import DiagnosisAgent
from .agent.llm import TracedLLM, list_llm_calls, llm_usage_summary, make_llm
from .agent.planner import Planner, get_plan, list_plans
from .engine import Engine, EngineError, InvalidAction, NotFound
from .models import RetryPolicy, SpecError, TaskSpec
from .templates import (MEDIA_EXTS, animation_swap_task, frame_style_inject_presets, frame_style_task)

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


class StyleTaskRequest(BaseModel):
    """主流水线:素材 → 逐帧风格化 → 成品视频。provider=mock 离线演示;provider=ark 调用豆包(密钥只在环境变量)。"""
    name: str = "anime-style"
    source: str                                   # 素材目录下的文件名,不接受路径
    prompt: str = Field(min_length=1, max_length=2000)
    provider: Literal["mock", "ark"] = "mock"
    model: str | None = None
    reference: str | None = None                  # 素材目录下的参考图文件名(可选)
    chunks: int = Field(4, ge=1, le=16)
    stride: int = Field(6, ge=1, le=240)
    max_gap: int = Field(24, ge=1, le=480)
    budget_cny: float = Field(20.0, gt=0, le=10000)
    price_cny: float | None = Field(None, ge=0)
    concurrency: int = Field(4, ge=1, le=16)
    on_reject: Literal["passthrough", "fail"] = "passthrough"
    target_height: int = Field(1080, ge=64, le=2160)
    require_approval: bool = False
    inject: Literal["transient", "crash", "moderation", "response_lost", "rate_limit"] | None = None
    max_attempts: int = Field(3, ge=1, le=20)
    base_delay: float = Field(2.0, ge=0)


class ApprovalRequest(BaseModel):
    decision: str
    who: str = "dashboard"
    comment: str = ""


class PlanRequest(BaseModel):
    request: str = Field(min_length=1, max_length=2000)


class DecisionRequest(BaseModel):
    approve: bool
    comment: str = ""
    who: str = "dashboard"


def create_app(data_dir: str | Path, llm_factory: Callable[[], Any] | None = None) -> FastAPI:
    """llm_factory 默认读环境变量(见 agent/llm.py),没有 Key 时是 MockLLM;测试可注入。"""
    app = FastAPI(title="Observable Task Pipeline", version="0.1")
    data_dir = Path(data_dir)
    Engine(data_dir).close()          # 确保库和表存在
    llm_factory = llm_factory or make_llm

    def get_engine() -> Iterator[Engine]:
        e = Engine(data_dir)
        try:
            yield e
        finally:
            e.close()

    def get_diag(eng: Engine = Depends(get_engine)) -> Iterator[DiagnosisAgent]:
        a = DiagnosisAgent(data_dir, eng, TracedLLM(llm_factory(), eng))
        try:
            yield a
        finally:
            a.close()

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

    # ------------------------------------------------------------ 序列帧流水线(真实处理)
    def media_dir() -> Path:
        return Path(os.environ.get("PIPELINE_MEDIA_DIR", str(WEB_DIR.parent / "samples")))

    def media_path(name: str) -> Path:
        # 只接受素材目录下的"纯文件名":拒绝路径分隔符/上级目录/隐藏与下划线开头的临时文件
        if not name or Path(name).name != name or name.startswith((".", "_")):
            raise HTTPException(422, "source 只能是素材目录下的文件名")
        p = media_dir() / name
        if not p.is_file():
            raise HTTPException(404, f"素材不存在: {name}")
        return p

    @app.get("/api/media")
    def list_media():
        d = media_dir()
        if not d.is_dir():
            return []
        return [{"name": p.name, "size_mb": round(p.stat().st_size / 1048576, 2)}
                for p in sorted(d.iterdir())
                if p.is_file() and p.suffix.lower() in MEDIA_EXTS and not p.name.startswith(("_", "."))]

    @app.post("/api/tasks/style", status_code=201)
    def submit_style(req: StyleTaskRequest, idempotency_key: str | None = Header(default=None),
                     eng: Engine = Depends(get_engine)):
        if req.inject and req.provider != "mock" and req.inject != "crash":
            raise HTTPException(422, "除 crash 外,故障注入只对 provider=mock 有效")
        try:
            spec = frame_style_task(
                req.name, source=str(media_path(req.source)), prompt=req.prompt, provider=req.provider,
                model=req.model, reference=str(media_path(req.reference)) if req.reference else None,
                chunks=req.chunks, stride=req.stride, max_gap=req.max_gap, budget_cny=req.budget_cny,
                price_cny=req.price_cny, concurrency=req.concurrency, on_reject=req.on_reject,
                target_height=req.target_height, require_approval=req.require_approval,
                chaos=frame_style_inject_presets(req.chunks)[req.inject] if req.inject else None,
                retry=RetryPolicy(max_attempts=req.max_attempts, base_delay=req.base_delay, max_delay=60, jitter=0.2))
        except ValueError as e:
            raise HTTPException(422, str(e))
        tid, created = eng.submit(spec, idempotency_key=idempotency_key)
        return {"id": tid, "created": created}

    # ------------------------------------------------------------ 逐帧账本(工序 3 的观测视图)
    def _stylize_step(eng: Engine, task_id: str, step: str):
        import re
        if not re.fullmatch(r"stylize_\d+", step):
            raise HTTPException(422, "step 只能是 stylize_N")
        row = eng.conn.execute("SELECT id, status, outputs, params FROM steps WHERE task_id=? AND key=?",
                               (task_id, step)).fetchone()
        if row is None:
            raise NotFound(f"{task_id}/{step}")
        return row

    @app.get("/api/tasks/{task_id}/frames")
    def task_frames(task_id: str, eng: Engine = Depends(get_engine)):
        """每个关键帧一行:状态、调用次数、用量、是否缓存/降级/可能多付。帧网格用它。"""
        eng.get_task(task_id)
        items = eng.list_items(task_id, prefix="stylize_")
        by_status: dict[str, int] = {}
        for it in items:
            by_status[it["status"]] = by_status.get(it["status"], 0) + 1
        items.sort(key=lambda x: x.get("idx", 0))
        return {"total": len(items), "by_status": by_status, "frames": items}

    @app.get("/api/tasks/{task_id}/frames/{step}/{item}/image")
    def frame_image(task_id: str, step: str, item: str, kind: Literal["src", "out"] = "out",
                    w: int = Query(0, ge=0, le=1920), eng: Engine = Depends(get_engine)):
        """取某个关键帧的源图或风格化结果(w>0 时缩成 JPEG 缩略图)。运行中的块直接从工作目录读,不用等步骤结束。"""
        import io
        import re
        import zipfile
        if not re.fullmatch(r"\d{6}", item):
            raise HTTPException(422, "item 必须是 6 位帧号")
        row = _stylize_step(eng, task_id, step)
        chunk = int(step.split("_")[1])
        data: bytes | None = None
        if kind == "out":
            wp = eng.data_dir / "work" / str(row["id"]) / "styled" / f"{item}.png"
            if wp.is_file():
                data = wp.read_bytes()
            else:
                outs = json.loads(row["outputs"] or "{}")
                if "styled.zip" in outs:
                    with zipfile.ZipFile(eng.store.path(outs["styled.zip"]["sha256"])) as z:
                        data = z.read(f"{item}.png") if f"{item}.png" in z.namelist() else None
        else:
            k = eng.conn.execute("SELECT outputs FROM steps WHERE task_id=? AND key='keyframes'", (task_id,)).fetchone()
            outs = json.loads(k["outputs"] or "{}") if k else {}
            if f"kf_{chunk}.zip" in outs:
                with zipfile.ZipFile(eng.store.path(outs[f"kf_{chunk}.zip"]["sha256"])) as z:
                    data = z.read(f"{item}.png") if f"{item}.png" in z.namelist() else None
        if data is None:
            raise HTTPException(404, "这一帧的图片还不存在")
        if w:
            from PIL import Image
            im = Image.open(io.BytesIO(data)).convert("RGB")
            im.thumbnail((w, w))
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=85)
            return Response(buf.getvalue(), media_type="image/jpeg")
        return Response(data, media_type="image/png")

    @app.get("/api/tasks/{task_id}/usage")
    def task_usage(task_id: str, eng: Engine = Depends(get_engine)):
        """用量与费用:按块汇总的接口调用次数、成功计费张数、output_tokens、费用估算与预算。"""
        eng.get_task(task_id)
        per: dict[str, dict] = {}
        for it in eng.list_items(task_id, prefix="stylize_"):
            d = per.setdefault(it["step_key"], {"frames": 0, "calls": 0, "billed": 0, "maybe_billed": 0,
                                                "output_tokens": 0, "cost_cny_est": 0.0, "cost_maybe_extra_cny_est": 0.0,
                                                "cached": 0, "degraded": 0, "failed": 0})
            d["frames"] += 1
            for k in ("calls", "billed", "maybe_billed", "output_tokens"):
                d[k] += int(it.get(k) or 0)
            d["cost_cny_est"] = round(d["cost_cny_est"] + float(it.get("cost_cny_est") or 0), 4)
            d["cost_maybe_extra_cny_est"] = round(d["cost_maybe_extra_cny_est"] + float(it.get("cost_maybe_extra_cny_est") or 0), 4)
            d["cached"] += 1 if it.get("cached") else 0
            d["degraded"] += 1 if it.get("degraded") else 0
            d["failed"] += 1 if it["status"] == "FAILED" else 0
        tot: dict[str, Any] = {}
        for d in per.values():
            for k, v in d.items():
                tot[k] = round(tot.get(k, 0) + v, 4) if isinstance(v, float) else tot.get(k, 0) + v
        s0 = eng.conn.execute("SELECT params FROM steps WHERE task_id=? AND key='stylize_0'", (task_id,)).fetchone()
        p = json.loads(s0["params"]) if s0 else {}
        return {"per_step": per, "total": tot, "budget_cny": p.get("budget_cny"), "price_cny": p.get("price_cny"),
                "provider": p.get("provider"), "model": p.get("model"),
                "note": "费用是按标价的估算,以平台账单为准;maybe_billed 是响应丢失/进程中途死亡导致的可能多付"}

    @app.get("/api/tasks/{task_id}/report")
    def task_report(task_id: str, eng: Engine = Depends(get_engine)):
        """各工序的耗时与尝试次数(从 attempts 汇总),与 usage 合在一起就是"参数/耗时/资源占用"报告。"""
        t = eng.get_task(task_id)
        steps = []
        for s in t["steps"]:
            att = eng.get_attempts(task_id, s["key"])
            steps.append({"key": s["key"], "type": s["type"], "status": s["status"], "cached": s["cached"],
                          "attempts": len(att), "failures": s["failures"], "lost": s["lost_count"],
                          "run_s": round(sum((a.get("duration") or 0) for a in att), 2),
                          "resource": s["resource"]})
        return {"task": {k: t[k] for k in ("id", "name", "status", "percent", "created_at", "finished_at")},
                "steps": steps, "usage": task_usage(task_id, eng),
                "wall_s": round((t["finished_at"] or t["updated_at"]) - t["created_at"], 2)}

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

    # ---- 步骤详情:输入 / 输出 / 参数。输入 = 依赖步骤的产物;输出 = 本步骤的产物;小文本直接内嵌预览
    PREVIEW_MAX = 20000
    KINDS = {".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image", ".gif": "image",
             ".mp4": "video", ".webm": "video", ".mov": "video",
             ".json": "text", ".txt": "text", ".md": "text", ".log": "text"}

    def _describe(eng: Engine, outputs: dict | None) -> list[dict]:
        items = []
        for name, o in sorted((outputs or {}).items()):
            kind = KINDS.get(Path(name).suffix.lower(), "file")
            it = {"name": name, "sha256": o.get("sha256"), "size": o.get("size"), "kind": kind}
            if kind == "text" and (o.get("size") or 0) <= PREVIEW_MAX:
                p = eng.store.path(o["sha256"])
                if p.is_file():
                    it["preview"] = p.read_text(encoding="utf-8", errors="replace")
            items.append(it)
        return items

    @app.get("/api/tasks/{task_id}/steps/{key}/io")
    def step_io(task_id: str, key: str, eng: Engine = Depends(get_engine)):
        t = eng.get_task(task_id)
        s = next((x for x in t["steps"] if x["key"] == key), None)
        if s is None:
            raise HTTPException(404, f"步骤不存在: {key}")
        by_key = {x["key"]: x for x in t["steps"]}
        return {"params": s["params"], "outputs": _describe(eng, s["outputs"]),
                "inputs": [{"from": d, "status": by_key[d]["status"], "artifacts": _describe(eng, by_key[d]["outputs"])}
                           for d in s["deps"] if d in by_key]}

    @app.get("/api/artifacts/{sha}")
    def artifact(sha: str, name: str = "", eng: Engine = Depends(get_engine)):
        """按内容哈希取产物(图片/视频/文本预览用)。只能取引擎登记过的产物,不接受任何路径。"""
        import mimetypes
        import re
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise HTTPException(422, "sha 必须是 64 位十六进制")
        if eng.conn.execute("SELECT 1 FROM artifacts WHERE sha256=?", (sha,)).fetchone() is None:
            raise HTTPException(404, "产物不存在")
        p = eng.store.path(sha)
        if not p.is_file():
            raise HTTPException(404, "产物文件缺失")
        return FileResponse(p, media_type=mimetypes.guess_type(Path(name).name)[0] or "application/octet-stream")

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

    # ------------------------------------------------------------ Agent:规划 / 追踪 / 诊断
    @app.post("/api/plan", status_code=201)
    def plan(req: PlanRequest, idempotency_key: str | None = Header(default=None),
             eng: Engine = Depends(get_engine)):
        """自然语言 → Planner(LLM + 护栏校验 + 修复回路)→ 提交。计划被护栏拒绝时返回 422 与每轮错误。"""
        planner = Planner(TracedLLM(llm_factory(), eng), eng)
        res = planner.plan(req.request, idempotency_key=idempotency_key)
        if not res.ok:
            return JSONResponse({"detail": "plan rejected by guardrails", **res.to_dict()}, status_code=422)
        return res.to_dict()

    @app.get("/api/plans")
    def plans(eng: Engine = Depends(get_engine)):
        return list_plans(eng)

    @app.get("/api/plans/{plan_id}")
    def plan_detail(plan_id: int, eng: Engine = Depends(get_engine)):
        p = get_plan(eng, plan_id)
        if p is None:
            raise NotFound(f"plan {plan_id}")
        return p

    @app.get("/api/llm_calls")
    def llm_calls(task_id: str | None = None, include_messages: bool = False,
                  limit: int = Query(100, ge=1, le=500), eng: Engine = Depends(get_engine)):
        return list_llm_calls(eng, task_id=task_id, include_messages=include_messages, limit=limit)

    @app.get("/api/llm_usage")
    def llm_usage(task_id: str | None = None, eng: Engine = Depends(get_engine)):
        return llm_usage_summary(eng, task_id)

    @app.post("/api/tasks/{task_id}/steps/{key}/diagnose", status_code=201)
    def diagnose(task_id: str, key: str, agent: DiagnosisAgent = Depends(get_diag)):
        return agent.start(task_id, key)

    @app.get("/api/diagnoses")
    def diagnoses(status: str | None = None, task_id: str | None = None, agent: DiagnosisAgent = Depends(get_diag)):
        return agent.list(status, task_id)

    @app.post("/api/diagnoses/{thread_id}/decision")
    def diagnosis_decision(thread_id: str, body: DecisionRequest, agent: DiagnosisAgent = Depends(get_diag)):
        return agent.resume(thread_id, body.approve, body.comment, body.who)

    # ------------------------------------------------------------ Dashboard
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    return app
