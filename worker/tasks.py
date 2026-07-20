from pipeline.orchestrator import run_pipeline
from pipeline.schemas import Goal
from pipeline.world_model.persona import get_preset
from worker.celery_app import celery_app
from worker.gpu_config import resolve_gpu_plan

# Resolved once per worker process from WORKER_GPUS — each worker process is
# pinned to a fixed GPU (or GPU group for FSDP) for its whole lifetime.
_GPU_PLAN = resolve_gpu_plan()


@celery_app.task(bind=True, name="worker.tasks.harmonize_job")
def harmonize_job(
    self,
    job_id: str,
    character_image_path: str,
    video_path: str,
    persona_name: str,
    title: str,
    theme: str,
    tone: str,
    language: str,
    anchor_x: float,
    anchor_y: float,
) -> dict:
    persona = get_preset(persona_name, character_image_path, speech_language=language)
    goal = Goal(title=title, theme=theme, tone=tone)

    def progress(stage: str) -> None:
        self.update_state(state=stage.upper(), meta={"detail": stage})

    try:
        output_path = run_pipeline(
            video_path=video_path,
            character_image_path=character_image_path,
            persona=persona,
            goal=goal,
            job_id=job_id,
            gpu_plan=_GPU_PLAN,
            anchor_seed=(anchor_x, anchor_y),
            progress=progress,
        )
    except Exception as exc:
        # Re-raised so Celery marks the task FAILURE; message surfaces via
        # AsyncResult.info in the API's status endpoint.
        raise RuntimeError(f"harmonize_job {job_id} failed: {exc}") from exc

    return {"job_id": job_id, "output_path": str(output_path)}
