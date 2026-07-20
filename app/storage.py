import shutil
from pathlib import Path

from fastapi import UploadFile

from pipeline.config import settings


def job_upload_dir(job_id: str) -> Path:
    path = settings.uploads_dir / job_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_output_path(job_id: str) -> Path:
    """Matches pipeline.orchestrator.run_pipeline's own output path convention —
    it writes here directly, this is just how the API locates the result."""
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    return settings.outputs_dir / f"{job_id}.mp4"


def save_upload(upload: UploadFile, dest: Path) -> Path:
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest
