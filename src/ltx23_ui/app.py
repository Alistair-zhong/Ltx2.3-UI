from __future__ import annotations

import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .media import probe_duration
from .models import (
    DEFAULT_NEGATIVE_PROMPT,
    GenerationRequest,
    ValidationResult,
    frames_for_duration,
    validate_request,
)
from .runtime import PipelineRuntime

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
DATA_DIR = Path(os.environ.get("LTX_UI_DATA_DIR", "~/.ltx23-ui")).expanduser()
UPLOAD_DIR = DATA_DIR / "uploads"

runtime = PipelineRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    runtime.start()
    yield
    runtime.stop()


app = FastAPI(title="LTX-2.3 A2V UI", version="0.1.0", lifespan=lifespan)


class FrameRequest(BaseModel):
    duration: float = Field(gt=0)
    fps: float = Field(gt=0)


class ProbeRequest(BaseModel):
    path: str
    fps: float = Field(default=25.0, gt=0)
    start_time: float = Field(default=0.0, ge=0)
    max_duration: float | None = Field(default=None, gt=0)


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model_loaded": runtime.model_loaded,
        "queue_size": runtime.queue_size,
    }


@app.get("/api/defaults")
def defaults() -> dict:
    return {
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "constraints": {"resolution_multiple": 64, "frame_formula": "8k+1"},
    }


@app.post("/api/frames")
def calculate_frames(body: FrameRequest) -> dict:
    frames = frames_for_duration(body.duration, body.fps)
    return {"num_frames": frames, "video_duration": round(frames / body.fps, 3)}


@app.post("/api/probe")
def probe(body: ProbeRequest) -> dict:
    try:
        source_duration = probe_duration(body.path)
    except (FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    available = max(0.0, source_duration - body.start_time)
    selected = min(available, body.max_duration) if body.max_duration else available
    if selected <= 0:
        raise HTTPException(status_code=400, detail="音频起始时间超出了文件时长")
    frames = frames_for_duration(selected, body.fps)
    return {
        "source_duration": round(source_duration, 3),
        "selected_duration": round(selected, 3),
        "num_frames": frames,
        "video_duration": round(frames / body.fps, 3),
    }


@app.post("/api/validate", response_model=ValidationResult)
def validate(body: GenerationRequest) -> ValidationResult:
    return validate_request(body, runtime.active_key)


@app.post("/api/jobs", status_code=202)
def create_job(body: GenerationRequest) -> dict:
    try:
        job = runtime.submit(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return job.public()


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return runtime.list_jobs()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = runtime.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="任务不存在")
    return job.public()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if not runtime.cancel(job_id):
        raise HTTPException(status_code=409, detail="只能取消排队中的任务")
    return {"ok": True}


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    job = runtime.get_job(job_id)
    if not job or job.state != "completed":
        raise HTTPException(status_code=404, detail="视频尚未生成")
    path = Path(job.request.generation.output_path).expanduser()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="输出文件不存在")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.post("/api/model/unload")
def unload_model() -> dict:
    try:
        runtime.unload()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"ok": True}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> dict:
    safe_name = Path(file.filename or "upload.bin").name
    target = UPLOAD_DIR / safe_name
    counter = 1
    while target.exists():
        target = UPLOAD_DIR / f"{Path(safe_name).stem}-{counter}{Path(safe_name).suffix}"
        counter += 1
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return {"path": str(target), "name": target.name, "size": target.stat().st_size}


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def run() -> None:
    import uvicorn

    uvicorn.run("ltx23_ui.app:app", host="0.0.0.0", port=7860, reload=False)
