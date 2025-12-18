from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.alpr import process_video
from app.job_manager import JobManager, JobState


BASE_DIR = Path(__file__).resolve().parents[1]
JOBS_DIR = BASE_DIR / "jobs"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="ALPR YOLO11 + OpenOCR")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

jobs = JobManager()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


async def _run_job(
    *,
    job_id: str,
    input_video_path: Path,
    model_path: str,
    conf: float,
    iou: float,
    use_gpu: bool,
) -> None:
    output_dir = JOBS_DIR / job_id
    output_video_path = output_dir / "annotated.mp4"
    output_json_path = output_dir / "result.json"

    def progress_cb(p: float) -> None:
        jobs.update(job_id, progress=float(p))

    jobs.update(job_id, state=JobState.running, message="processing")

    try:
        await asyncio.to_thread(
            process_video,
            input_video_path=input_video_path,
            output_video_path=output_video_path,
            output_json_path=output_json_path,
            model_path=model_path,
            conf=conf,
            iou=iou,
            use_gpu=use_gpu,
            cache_dir=BASE_DIR / "models",
            progress_cb=progress_cb,
        )
        jobs.update(
            job_id,
            state=JobState.done,
            message="done",
            artifacts={
                "video": str(output_video_path),
                "json": str(output_json_path),
            },
        )
    except Exception as e:
        jobs.update(job_id, state=JobState.error, error=f"{type(e).__name__}: {e}")


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    model_path: str = Form(
        "morsetechlab/yolov11-license-plate-detection/license-plate-finetune-v1x.pt"
    ),
    use_gpu: bool = Form(True),
    conf: float = Form(0.25),
    iou: float = Form(0.45),
):
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_dir / "input.mp4"
    content = await video.read()
    input_path.write_bytes(content)

    jobs.create(job_id)
    jobs.update(job_id, message="queued")

    background_tasks.add_task(
        _run_job,
        job_id=job_id,
        input_video_path=input_path,
        model_path=model_path,
        conf=conf,
        iou=iou,
        use_gpu=use_gpu,
    )

    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return jobs.as_dict(job)


@app.get("/api/jobs/{job_id}/result/video")
def get_result_video(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state != JobState.done:
        raise HTTPException(status_code=409, detail="job not done")

    video_path = Path(job.artifacts.get("video", ""))
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="result video missing")

    return FileResponse(str(video_path), media_type="video/mp4", filename="annotated.mp4")


@app.get("/api/jobs/{job_id}/result/json")
def get_result_json(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state != JobState.done:
        raise HTTPException(status_code=409, detail="job not done")

    json_path = Path(job.artifacts.get("json", ""))
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="result json missing")

    return FileResponse(str(json_path), media_type="application/json", filename="result.json")
