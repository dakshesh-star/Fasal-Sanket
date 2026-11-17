"""Durable satellite acquisition and experimental inference worker tasks."""
from __future__ import annotations

import os
import socket
import threading
from datetime import date

import requests
from celery import shared_task
from fastapi import HTTPException

from app import database
from app.satellite_pipeline import PipelineError, process_observations


@shared_task(name="app.worker.dispatch_outbox")
def dispatch_outbox():
    """Replay DB-committed jobs until the broker accepts them."""
    from app.worker import process_satellite_job
    sent = 0
    for row in database.pending_job_dispatches():
        job_id = row["job_id"]
        try:
            task = process_satellite_job.apply_async(args=[job_id])
            database.mark_job_dispatched(job_id, task.id)
            sent += 1
        except Exception as exc:
            database.mark_job_dispatch_error(job_id, f"{type(exc).__name__}: {exc}")
    return {"dispatched": sent}


@shared_task(bind=True, name="app.worker.process_satellite_job", max_retries=5,
             acks_late=True, reject_on_worker_lost=True)
def process_satellite_job(self, job_id: str):
    worker_id = f"{socket.gethostname()}:{os.getpid()}"
    job = database.claim_job(job_id, worker_id,
        stale_after_seconds=int(os.getenv("JOB_STALE_AFTER_SECONDS", "1800")))
    if not job:  # duplicate broker delivery, cancellation, or an active lease
        return {"job_id": job_id, "skipped": True}
    payload = job["request"]
    user = database.get_user_by_id(job["user_id"])
    field = database.get_field(job["user_id"], job["field_id"])
    if not user or not field:
        database.update_job(job_id, status="failed", current_stage="failed", error={"message": "Field or owner no longer exists."}, completed_at=database.now_iso())
        return {"job_id": job_id, "status": "failed"}

    def progress(stage, percent, **counts):
        if database.job_cancelled(job_id):
            raise InterruptedError("Job cancelled by user")
        database.update_job(job_id, status=stage, current_stage=stage, progress=percent,
                            heartbeat_at=database.now_iso(), **counts)

    try:
        stop_monitor = threading.Event()
        def heartbeat_monitor():
            while not stop_monitor.wait(30):
                current = database.get_job_by_id(job_id)
                if not current or current["status"] == "cancelled":
                    return
                acquisitions = database.list_satellite_acquisitions(job["user_id"], job["field_id"])
                statuses = {item["status"] for item in acquisitions}
                stage = ("generating_observations" if "processed" in statuses or "stored" in statuses
                         else "processing_imagery" if "assets_retrieved" in statuses
                         else "retrieving_assets" if "downloading_assets" in statuses
                         else current["current_stage"])
                percent = max(current["progress"], {"retrieving_assets": 25,
                    "processing_imagery": 60, "generating_observations": 82}.get(stage, current["progress"]))
                database.update_job(job_id, status=stage, current_stage=stage, progress=percent,
                    heartbeat_at=database.now_iso(),
                    assets_retrieved=sum(item["status"] in ("assets_retrieved", "processed", "stored") for item in acquisitions))
        monitor = threading.Thread(target=heartbeat_monitor, name=f"job-heartbeat-{job_id[:8]}", daemon=True)
        monitor.start()
        try:
            response = process_observations(job["user_id"], job["field_id"], payload["start_date"],
                payload["end_date"], float(payload.get("max_cloud_cover", 30)),
                on_progress=lambda stage, percent, **counts: progress(stage, percent, **counts),
                is_cancelled=lambda: database.job_cancelled(job_id))
        except PipelineError as exc:
            existing = database.list_observations(job["user_id"], job["field_id"])
            if job["kind"] != "identification" or not existing:
                raise
            # A metadata/asset outage must not discard observations already
            # persisted for this field. Inference remains possible on those rows.
            response = {"observation_count": len(existing), "scene_counts": exc.detail if isinstance(exc.detail, dict) else {},
                        "failed_acquisitions": [], "acquisition_warning": exc.detail}
        finally:
            stop_monitor.set()
            monitor.join(timeout=2)
        if response.get("cancelled"):
            return {"job_id": job_id, "status": "cancelled"}
        stored = int(response.get("observation_count", 0))
        counts = response.get("scene_counts", {})
        progress("generating_observations", 85, scenes_discovered=int(counts.get("discovered", 0)),
                 assets_retrieved=int(counts.get("assets_retrieved", 0)), valid_observations=stored)
        result = {"scene_counts": counts, "failed_acquisitions": response.get("failed_acquisitions", []),
                  "observation_count": stored, "available_acquisition_dates": response.get("available_acquisition_dates", []),
                  "acquisition_warning": response.get("acquisition_warning")}
        model_status = None
        if job["kind"] == "identification":
            progress("running_models", 90)
            try:
                from app import main as api_main
                identified = api_main.identify_field_crop(
                    field_id=job["field_id"], model_type=payload.get("model_type", "random_forest"),
                    target_error_rate=float(payload.get("target_error_rate", .1)), _job_id=job_id, user=user)
                result["identification"] = identified
                model_status = "experimental_candidate"
            except HTTPException as exc:
                if exc.status_code == 422:
                    model_status = "insufficient_evidence"
                    result["identification"] = {"status": model_status, "detail": exc.detail}
                else:
                    raise
        progress("completed", 100, scenes_discovered=int(result["scene_counts"].get("discovered", 0)),
                 assets_retrieved=int(result["scene_counts"].get("assets_retrieved", 0)),
                 valid_observations=stored, model_status=model_status, result=result,
                 completed_at=database.now_iso())
        return {"job_id": job_id, "status": "completed", **result}
    except InterruptedError:
        return {"job_id": job_id, "status": "cancelled"}
    except Exception as exc:
        transient = isinstance(exc, (requests.RequestException, TimeoutError, ConnectionError)) or (
            isinstance(exc, HTTPException) and exc.status_code in (502, 504))
        transient = transient or (isinstance(exc, PipelineError) and exc.retryable)
        current = database.get_job_by_id(job_id) or job
        if transient and int(current.get("attempts", 0)) < int(current.get("max_attempts", 3)):
            database.update_job(job_id, status="queued", current_stage="queued", heartbeat_at=None,
                error={"message": "Temporary satellite or network error; retry scheduled.", "type": type(exc).__name__})
            raise self.retry(exc=exc, countdown=min(30 * (2 ** self.request.retries), 900))
        if isinstance(exc, PipelineError):
            detail = exc.detail
        elif isinstance(exc, HTTPException):
            detail = exc.detail if isinstance(exc.detail, (dict, str, int, float, list)) else str(exc.detail)
        else:
            detail = f"{type(exc).__name__}: {exc}"
        database.update_job(job_id, status="failed", current_stage="failed", error=(detail if isinstance(detail, dict) else {"message": detail}), completed_at=database.now_iso())
        return {"job_id": job_id, "status": "failed", "error": detail}
