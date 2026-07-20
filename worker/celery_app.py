from celery import Celery

from app.config import settings

celery_app = Celery(
    "harmonize",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_track_started=True,
    # One job occupies this worker's assigned GPU(s) for its full duration,
    # so each worker process must only ever run one task at a time.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

celery_app.autodiscover_tasks(["worker"])
