from celery import Celery

from app.config import settings

celery_app = Celery(
    "crowd_flow_analyzer",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # Keep concurrency=1 per worker process to avoid SQLite write conflicts.
    # For true parallelism across multiple videos, run multiple worker
    # containers (`docker compose up --scale worker=N`) and switch to
    # PostgreSQL via CFA_DATABASE_URL.
    worker_concurrency=1,
    # Prevent a single slow task from starving the queue
    worker_prefetch_multiplier=1,
)

celery_app.autodiscover_tasks(["app.worker"])
