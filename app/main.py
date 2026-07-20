import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.schemas import JobStatus, JobStatusResponse, JobSubmitResponse
from app.storage import job_output_path, job_upload_dir, save_upload
from pipeline.world_model.persona import PRESETS
from worker.celery_app import celery_app
from worker.tasks import harmonize_job

app = FastAPI(title="Harmonize")


@app.post("/jobs", response_model=JobSubmitResponse)
async def submit_job(
    character_image: UploadFile = File(..., description="Anime character reference image"),
    video: UploadFile = File(..., description="Real background video the character reacts to"),
    persona: str = Form("curious_vlogger", description=f"One of: {', '.join(PRESETS)}"),
    title: str = Form("Vlog"),
    theme: str = Form("exploring a real place"),
    tone: str = Form("casual vlog"),
    language: str = Form("ja"),
    anchor_x: float = Form(0.5),
    anchor_y: float = Form(0.85),
) -> JobSubmitResponse:
    if persona not in PRESETS:
        raise HTTPException(status_code=422, detail=f"Unknown persona '{persona}'. Available: {list(PRESETS)}")

    job_id = str(uuid.uuid4())
    upload_dir = job_upload_dir(job_id)

    image_path = save_upload(character_image, upload_dir / f"character{_suffix(character_image.filename)}")
    video_path = save_upload(video, upload_dir / f"video{_suffix(video.filename)}")

    harmonize_job.delay(
        job_id,
        str(image_path),
        str(video_path),
        persona,
        title,
        theme,
        tone,
        language,
        anchor_x,
        anchor_y,
    )
    return JobSubmitResponse(job_id=job_id)


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    result = celery_app.AsyncResult(job_id)

    if result.state == "PENDING":
        status = JobStatus.PENDING
    elif result.state == "FAILURE":
        status = JobStatus.FAILURE
    elif result.state == "SUCCESS":
        status = JobStatus.SUCCESS
    else:
        # custom states set via update_state in the task: PREPROCESSING / GENERATING
        status = JobStatus(result.state)

    detail = None
    if isinstance(result.info, dict):
        detail = result.info.get("detail")
    elif result.state == "FAILURE":
        detail = str(result.info)

    return JobStatusResponse(job_id=job_id, status=status, detail=detail)


@app.get("/jobs/{job_id}/result")
async def get_job_result(job_id: str) -> FileResponse:
    output_path = job_output_path(job_id)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Result not ready or job failed")
    return FileResponse(output_path, media_type="video/mp4", filename=f"{job_id}.mp4")


def _suffix(filename: str | None) -> str:
    if not filename or "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1]
