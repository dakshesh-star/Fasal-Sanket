"""Celery configuration for the independently deployed processing worker."""
import os
from app import config as _config
from celery import Celery

redis_url = os.getenv("REDIS_URL")
if not redis_url:
    # Importing the FastAPI app locally should not require Redis. Job submission
    # still persists to the DB; the dispatcher records a clear queue error.
    redis_url = "redis://localhost:6379/0"

celery_app = Celery("fasal_sanket", broker=redis_url, backend=redis_url, include=["app.worker"])
celery_app.conf.update(
    task_serializer="json", result_serializer="json", accept_content=["json"],
    task_acks_late=True, task_reject_on_worker_lost=True, worker_prefetch_multiplier=1,
    broker_transport_options={"visibility_timeout": int(os.getenv("JOB_VISIBILITY_TIMEOUT", "21600"))},
    result_expires=3600, task_track_started=True,
    beat_schedule={"dispatch-durable-job-outbox": {
        "task": "app.worker.dispatch_outbox", "schedule": float(os.getenv("JOB_DISPATCH_INTERVAL", "10")),
    }},
    task_soft_time_limit=int(os.getenv("JOB_SOFT_TIMEOUT_SECONDS", "14400")),
    task_time_limit=int(os.getenv("JOB_HARD_TIMEOUT_SECONDS", "15000")),
)
