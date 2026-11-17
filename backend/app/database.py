"""SQLAlchemy Core persistence for local SQLite and production PostgreSQL."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import create_engine, inspect, text, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError

from app import config as _config  # load private local .env before reading DATABASE_URL
from app.geometry import area_hectares
from app.models import metadata

_db_setting = Path(os.getenv("FASAL_DATABASE_PATH", "data/fasal_sanket.sqlite3"))
DB_PATH = _db_setting if _db_setting.is_absolute() else Path(__file__).resolve().parents[1] / _db_setting
_engines: dict[str, Engine] = {}
_initialized: set[str] = set()


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):
            return "postgresql+psycopg://" + url[len("postgres://"):]
        if url.startswith("postgresql://"):
            return "postgresql+psycopg://" + url[len("postgresql://"):]
        return url
    if os.getenv("VERCEL"):
        raise RuntimeError("DATABASE_URL must point to persistent PostgreSQL on Vercel; function-local SQLite is not supported.")
    return "sqlite:///" + str(Path(DB_PATH).resolve()).replace("\\", "/")


def _engine() -> tuple[Engine, str, bool]:
    url = database_url()
    is_sqlite = url.startswith("sqlite:")
    engine = _engines.get(url)
    if engine is None:
        options = {"pool_pre_ping": True}
        if is_sqlite:
            options["connect_args"] = {"timeout": 10, "check_same_thread": False}
        else:
            serverless = bool(os.getenv("VERCEL"))
            options["pool_size"] = int(os.getenv("DB_POOL_SIZE", "1" if serverless else "3"))
            options["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "0" if serverless else "2"))
        engine = create_engine(url, **options)
        if is_sqlite:
            @event.listens_for(engine, "connect")
            def _enable_foreign_keys(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
        _engines[url] = engine
    return engine, url, is_sqlite


class _CompatRow:
    """Keep the old sqlite3.Row access pattern while executing through SQLAlchemy."""
    def __init__(self, mapping):
        self._values = dict(mapping)
        self._keys = list(self._values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._values[self._keys[key]]
        return self._values[key]

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def keys(self):
        return self._keys


class _CompatResult:
    def __init__(self, result):
        self._result = result
        self.rowcount = result.rowcount

    @staticmethod
    def _row(row):
        return _CompatRow(row._mapping) if row is not None else None

    def fetchone(self):
        return self._row(self._result.fetchone())

    def fetchall(self):
        return [self._row(row) for row in self._result.fetchall()]


class _CompatConnection:
    def __init__(self, connection):
        self._connection = connection

    def execute(self, statement: str, parameters=()):
        statement = statement.replace(" COLLATE NOCASE", "")
        if isinstance(parameters, dict):
            bound = parameters
        else:
            values = tuple(parameters or ())
            index = 0
            parts = statement.split("?")
            if len(parts) - 1 != len(values):
                raise ValueError(f"SQL bind count mismatch: {len(parts) - 1} placeholders for {len(values)} values")
            statement = parts[0]
            for value in values:
                statement += f":p{index}" + parts[index + 1]
                index += 1
            bound = {f"p{i}": value for i, value in enumerate(values)}
        return _CompatResult(self._connection.execute(text(statement), bound))


def initialize_database():
    """Create the local development schema once; production schema is Alembic-managed."""
    engine, url, is_sqlite = _engine()
    if url in _initialized:
        return
    if is_sqlite:
        Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        metadata.create_all(engine)
        # Keep local upgrades compatible with earlier SQLite files. PostgreSQL is
        # migrated explicitly by Alembic before the API/worker is started.
        inspector = inspect(engine)
        with engine.begin() as conn:
            if "users" in inspector.get_table_names() and "display_name" not in {c["name"] for c in inspector.get_columns("users")}:
                conn.execute(text("ALTER TABLE users ADD COLUMN display_name TEXT"))
            if "fields" in inspector.get_table_names():
                field_columns = {c["name"] for c in inspect(engine).get_columns("fields")}
                for name, declaration in (("area_hectares", "REAL"), ("scene_metadata_checked_at", "TEXT"),
                                          ("scene_metadata_count", "INTEGER NOT NULL DEFAULT 0")):
                    if name not in field_columns:
                        conn.execute(text(f"ALTER TABLE fields ADD COLUMN {name} {declaration}"))
                conn.execute(text("UPDATE fields SET crop='Unknown / Not identified' WHERE crop IS NULL OR trim(crop)=''"))
            if "observations" in inspector.get_table_names():
                obs_columns = {c["name"] for c in inspect(engine).get_columns("observations")}
                if "processing_status" not in obs_columns:
                    conn.execute(text("ALTER TABLE observations ADD COLUMN processing_status TEXT NOT NULL DEFAULT 'stored'"))
            if "predictions" in inspector.get_table_names():
                pred_columns = {c["name"] for c in inspect(engine).get_columns("predictions")}
                if "job_id" not in pred_columns:
                    conn.execute(text("ALTER TABLE predictions ADD COLUMN job_id VARCHAR(36)"))
                    conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_predictions_job_id ON predictions(job_id)"))
            if "fields" in inspector.get_table_names():
                rows = conn.execute(text("SELECT id,boundary_json FROM fields WHERE boundary_json IS NOT NULL")).fetchall()
                for field_id, boundary_json in rows:
                    try:
                        conn.execute(text("UPDATE fields SET area_hectares=:area WHERE id=:id"),
                                     {"area": area_hectares(json.loads(boundary_json)), "id": field_id})
                    except (ValueError, TypeError, json.JSONDecodeError):
                        continue
    else:
        # Fail fast on connectivity; do not create production schema at API startup.
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    _initialized.add(url)


@contextmanager
def connection():
    initialize_database()
    engine, _url, _is_sqlite = _engine()
    with engine.begin() as conn:
        yield _CompatConnection(conn)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _json_dump(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def create_user(email: str, password_hash: str, display_name: str | None = None):
    user_id = str(uuid.uuid4())
    with connection() as db:
        db.execute("INSERT INTO users(id,email,password_hash,created_at,display_name) VALUES(?,?,?,?,?)",
                   (user_id, email.lower(), password_hash, now_iso(), display_name))
    return {"id": user_id, "email": email.lower(), "display_name": display_name}


def get_user_by_email(email: str):
    with connection() as db:
        row = db.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
        return dict(row) if row else None


def get_user_by_id(user_id: str):
    with connection() as db:
        row = db.execute("SELECT id,email,created_at,display_name FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def save_refresh_session(token_hash: str, user_id: str, expires_at: int):
    with connection() as db:
        db.execute("INSERT INTO refresh_sessions(token_hash,user_id,expires_at,created_at) VALUES(?,?,?,?)",
                   (token_hash, user_id, expires_at, now_iso()))


def get_refresh_session(token_hash: str):
    with connection() as db:
        row = db.execute("SELECT * FROM refresh_sessions WHERE token_hash = ?", (token_hash,)).fetchone()
        return dict(row) if row else None


def revoke_refresh_session(token_hash: str):
    with connection() as db:
        db.execute("DELETE FROM refresh_sessions WHERE token_hash = ?", (token_hash,))


def list_fields(user_id: str):
    with connection() as db:
        rows = db.execute("SELECT * FROM fields WHERE user_id = ? ORDER BY created_at DESC", (user_id,)).fetchall()
        return [_field_dict(row) for row in rows]


def get_field(user_id: str, field_id: str):
    with connection() as db:
        row = db.execute("SELECT * FROM fields WHERE user_id = ? AND id = ?", (user_id, field_id)).fetchone()
        return _field_dict(row) if row else None


def save_field(user_id: str, values: dict, field_id: str | None = None):
    current = now_iso()
    measured_area = area_hectares(values.get("boundary"))
    crop = values.get("crop") or "Unknown / Not identified"
    with connection() as db:
        if field_id:
            result = db.execute("""UPDATE fields SET name=?,latitude=?,longitude=?,crop=?,planting_date=?,harvest_date=?,boundary_json=?,area_hectares=?,updated_at=? WHERE user_id=? AND id=?""",
                (values["name"], values["latitude"], values["longitude"], crop, values.get("planting_date"), values.get("harvest_date"), _json_dump(values["boundary"]) if values.get("boundary") else None, measured_area, current, user_id, field_id))
            if not result.rowcount:
                return None
        else:
            field_id = str(uuid.uuid4())
            db.execute("""INSERT INTO fields(id,user_id,name,latitude,longitude,crop,planting_date,harvest_date,boundary_json,area_hectares,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (field_id, user_id, values["name"], values["latitude"], values["longitude"], crop, values.get("planting_date"), values.get("harvest_date"), _json_dump(values["boundary"]) if values.get("boundary") else None, measured_area, current, current))
    return get_field(user_id, field_id)


def delete_field(user_id: str, field_id: str):
    with connection() as db:
        result = db.execute("DELETE FROM fields WHERE user_id=? AND id=?", (user_id, field_id))
        return result.rowcount > 0


def save_observation(user_id: str, field_id: str, observed_at: str, stats: dict, scene_id: str | None = None,
                     source: str = "Copernicus Sentinel-2 L2A STAC assets"):
    with connection() as db:
        if not db.execute("SELECT 1 FROM fields WHERE user_id=? AND id=?", (user_id, field_id)).fetchone():
            return None
        obs_id = str(uuid.uuid4())
        db.execute("""INSERT INTO observations(id,user_id,field_id,observed_at,source,scene_id,stats_json,processing_status,created_at)
          VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(field_id,observed_at,source) DO UPDATE SET
          scene_id=excluded.scene_id,stats_json=excluded.stats_json,processing_status='stored',created_at=excluded.created_at""",
          (obs_id, user_id, field_id, observed_at, source, scene_id, _json_dump(stats), "stored", now_iso()))
        row = db.execute("SELECT * FROM observations WHERE user_id=? AND field_id=? AND observed_at=? AND source=?",
                         (user_id, field_id, observed_at, source)).fetchone()
        return _observation_dict(row)


def upsert_satellite_acquisition(user_id: str, field_id: str, scene_id: str, acquisition_date: str,
                                 cloud_cover: float | None, status: str, required_assets: list[str],
                                 retrieved_assets: list[str] | None = None, error_message: str | None = None,
                                 observation_id: str | None = None):
    now = now_iso()
    with connection() as db:
        if not db.execute("SELECT 1 FROM fields WHERE user_id=? AND id=?", (user_id, field_id)).fetchone():
            return None
        acquisition_id = str(uuid.uuid4())
        db.execute("""INSERT INTO satellite_acquisitions(id,user_id,field_id,scene_id,acquisition_date,cloud_cover,status,
            required_assets_json,retrieved_assets_json,observation_id,error_message,created_at,updated_at)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(user_id,field_id,scene_id) DO UPDATE SET acquisition_date=excluded.acquisition_date,
            cloud_cover=excluded.cloud_cover,
            status=CASE WHEN satellite_acquisitions.status='stored' THEN 'stored' ELSE excluded.status END,
            required_assets_json=excluded.required_assets_json,
            retrieved_assets_json=CASE WHEN excluded.retrieved_assets_json='[]' THEN satellite_acquisitions.retrieved_assets_json ELSE excluded.retrieved_assets_json END,
            error_message=excluded.error_message,observation_id=COALESCE(excluded.observation_id,satellite_acquisitions.observation_id),updated_at=excluded.updated_at""",
            (acquisition_id,user_id,field_id,scene_id,acquisition_date,cloud_cover,status,_json_dump(required_assets),
             _json_dump(retrieved_assets or []),observation_id,error_message,now,now))
        row = db.execute("SELECT * FROM satellite_acquisitions WHERE user_id=? AND field_id=? AND scene_id=?",
                         (user_id,field_id,scene_id)).fetchone()
        return _acquisition_dict(row)


def update_satellite_acquisition(user_id: str, field_id: str, scene_id: str, status: str,
                                 retrieved_assets: list[str] | None = None, error_message: str | None = None,
                                 observation_id: str | None = None):
    now = now_iso()
    with connection() as db:
        db.execute("""UPDATE satellite_acquisitions SET status=?,retrieved_assets_json=?,error_message=?,
            observation_id=COALESCE(?,observation_id),updated_at=? WHERE user_id=? AND field_id=? AND scene_id=?""",
            (status,_json_dump(retrieved_assets or []),error_message,observation_id,now,user_id,field_id,scene_id))
        row = db.execute("SELECT * FROM satellite_acquisitions WHERE user_id=? AND field_id=? AND scene_id=?",
                         (user_id,field_id,scene_id)).fetchone()
        return _acquisition_dict(row) if row else None


def list_satellite_acquisitions(user_id: str, field_id: str):
    with connection() as db:
        rows = db.execute("SELECT * FROM satellite_acquisitions WHERE user_id=? AND field_id=? ORDER BY acquisition_date,scene_id",
                           (user_id,field_id)).fetchall()
        return [_acquisition_dict(row) for row in rows]


def save_scene_search(user_id: str, field_id: str, count: int):
    checked = now_iso()
    with connection() as db:
        cursor = db.execute("UPDATE fields SET scene_metadata_checked_at=?,scene_metadata_count=?,updated_at=? WHERE user_id=? AND id=?",
                            (checked, count, checked, user_id, field_id))
        return cursor.rowcount > 0


def list_observations(user_id: str, field_id: str):
    with connection() as db:
        rows = db.execute("SELECT * FROM observations WHERE user_id=? AND field_id=? ORDER BY observed_at",
                           (user_id,field_id)).fetchall()
        return [_observation_dict(row) for row in rows]


def workspace_summary(user_id: str):
    with connection() as db:
        fields_count = db.execute("SELECT COUNT(*) FROM fields WHERE user_id=?", (user_id,)).fetchone()[0]
        observations_count = db.execute("SELECT COUNT(*) FROM observations WHERE user_id=?", (user_id,)).fetchone()[0]
        latest = db.execute("SELECT * FROM observations WHERE user_id=? ORDER BY observed_at DESC LIMIT 12", (user_id,)).fetchall()
        predictions_count = db.execute("SELECT COUNT(*) FROM predictions WHERE user_id=?", (user_id,)).fetchone()[0]
        recent_predictions = db.execute("SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 5", (user_id,)).fetchall()
        return {"field_count": fields_count, "observation_count": observations_count, "prediction_count": predictions_count,
                "recent_observations": [_observation_dict(row) for row in reversed(latest)],
                "recent_predictions": [_prediction_dict(row) for row in recent_predictions],
                "predictions_available": predictions_count > 0}


def save_prediction(user_id: str, field_id: str, model_type: str, result: dict, job_id: str | None = None):
    prediction_id = str(uuid.uuid4())
    created = now_iso()
    with connection() as db:
        if not db.execute("SELECT 1 FROM fields WHERE id=? AND user_id=?", (field_id, user_id)).fetchone():
            return None
        if job_id:
            if not db.execute("SELECT 1 FROM processing_jobs WHERE id=? AND user_id=? AND field_id=?", (job_id,user_id,field_id)).fetchone():
                return None
            db.execute("""INSERT INTO predictions(id,user_id,field_id,created_at,model_type,provenance,result_json,job_id)
                VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(job_id) DO UPDATE SET
                created_at=excluded.created_at,model_type=excluded.model_type,provenance=excluded.provenance,result_json=excluded.result_json""",
                (prediction_id,user_id,field_id,created,model_type,"experimental model trained on synthetic data",_json_dump(result),job_id))
            row = db.execute("SELECT * FROM predictions WHERE job_id=?", (job_id,)).fetchone()
        else:
            db.execute("INSERT INTO predictions(id,user_id,field_id,created_at,model_type,provenance,result_json,job_id) VALUES(?,?,?,?,?,?,?,NULL)",
                       (prediction_id,user_id,field_id,created,model_type,"experimental model trained on synthetic data",_json_dump(result)))
            row = db.execute("SELECT * FROM predictions WHERE id=?", (prediction_id,)).fetchone()
        return _prediction_dict(row)


def list_predictions(user_id: str, field_id: str):
    with connection() as db:
        rows = db.execute("SELECT * FROM predictions WHERE user_id=? AND field_id=? ORDER BY created_at DESC", (user_id, field_id)).fetchall()
        return [_prediction_dict(row) for row in rows]


ACTIVE_JOB_STATES = ("queued", "discovering_scenes", "retrieving_assets", "processing_imagery",
                     "generating_observations", "running_models")
TERMINAL_JOB_STATES = ("completed", "failed", "cancelled")


def create_job(user_id: str, field_id: str, kind: str, request: dict, max_attempts: int = 3):
    payload = {"kind": kind, **request}
    request_json = _json_dump(payload)
    request_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    now = now_iso()
    try:
        with connection() as db:
            existing = db.execute("SELECT * FROM processing_jobs WHERE user_id=? AND field_id=? AND request_hash=? AND status IN (?,?,?,?,?,?) ORDER BY created_at DESC LIMIT 1",
                                  (user_id,field_id,request_hash,*ACTIVE_JOB_STATES)).fetchone()
            if existing:
                return _job_dict(existing), False
            if not db.execute("SELECT 1 FROM fields WHERE user_id=? AND id=?", (user_id,field_id)).fetchone():
                return None, False
            job_id, outbox_id = str(uuid.uuid4()), str(uuid.uuid4())
            db.execute("""INSERT INTO processing_jobs(id,user_id,field_id,kind,status,progress,current_stage,request_hash,request_json,
                attempts,max_attempts,scenes_discovered,assets_retrieved,valid_observations,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,0,?,0,0,0,?,?)""",
                (job_id,user_id,field_id,kind,"queued",0,"queued",request_hash,request_json,max_attempts,now,now))
            db.execute("INSERT INTO job_outbox(id,job_id,event_type,payload_json,status,attempts,created_at) VALUES(?,?,?,?,?,?,?)",
                (outbox_id,job_id,"process_satellite_job",_json_dump({"job_id":job_id}),"pending",0,now))
            row = db.execute("SELECT * FROM processing_jobs WHERE id=?", (job_id,)).fetchone()
            return _job_dict(row), True
    except IntegrityError:
        # A concurrent identical request may win the partial unique index race.
        existing = get_active_job(user_id, field_id, request_hash)
        if existing:
            return existing, False
        raise


def get_active_job(user_id: str, field_id: str, request_hash: str):
    with connection() as db:
        row = db.execute("SELECT * FROM processing_jobs WHERE user_id=? AND field_id=? AND request_hash=? AND status IN (?,?,?,?,?,?) ORDER BY created_at DESC LIMIT 1",
            (user_id,field_id,request_hash,*ACTIVE_JOB_STATES)).fetchone()
        return _job_dict(row) if row else None


def get_job(user_id: str, field_id: str, job_id: str):
    with connection() as db:
        row = db.execute("SELECT * FROM processing_jobs WHERE id=? AND user_id=? AND field_id=?", (job_id,user_id,field_id)).fetchone()
        job = _job_dict(row) if row else None
        if job:
            dispatch = db.execute("SELECT attempts,last_error FROM job_outbox WHERE job_id=?", (job_id,)).fetchone()
            job["dispatch_attempts"] = dispatch[0] if dispatch else 0
            job["dispatch_error"] = dispatch[1] if dispatch else None
        return job


def list_jobs(user_id: str, field_id: str, limit: int = 10):
    with connection() as db:
        rows = db.execute("SELECT * FROM processing_jobs WHERE user_id=? AND field_id=? ORDER BY created_at DESC LIMIT ?",
                           (user_id,field_id,max(1,min(int(limit),50)))).fetchall()
        jobs = [_job_dict(row) for row in rows]
        for job in jobs:
            dispatch = db.execute("SELECT attempts,last_error FROM job_outbox WHERE job_id=?", (job["id"],)).fetchone()
            job["dispatch_attempts"] = dispatch[0] if dispatch else 0
            job["dispatch_error"] = dispatch[1] if dispatch else None
        return jobs


def update_job(job_id: str, **values):
    allowed = {"status", "progress", "current_stage", "attempts", "scenes_discovered", "assets_retrieved",
               "valid_observations", "model_status", "result_json", "error_json", "celery_task_id", "worker_id",
               "heartbeat_at", "dispatched_at", "started_at", "completed_at"}
    if "result" in values:
        values["result_json"] = _json_dump(values.pop("result"))
    if "error" in values:
        values["error_json"] = _json_dump(values.pop("error"))
    values = {key: value for key, value in values.items() if key in allowed}
    if not values:
        return get_job_by_id(job_id)
    values["updated_at"] = now_iso()
    assignments = ",".join(f"{key}=?" for key in values)
    with connection() as db:
        db.execute(f"UPDATE processing_jobs SET {assignments} WHERE id=?", (*values.values(),job_id))
        row = db.execute("SELECT * FROM processing_jobs WHERE id=?", (job_id,)).fetchone()
        return _job_dict(row) if row else None


def get_job_by_id(job_id: str):
    with connection() as db:
        row = db.execute("SELECT * FROM processing_jobs WHERE id=?", (job_id,)).fetchone()
        return _job_dict(row) if row else None


def claim_job(job_id: str, worker_id: str, stale_after_seconds: int = 1800):
    now = datetime.now(timezone.utc)
    cutoff = (now - timedelta(seconds=stale_after_seconds)).isoformat()
    stamp = now.isoformat()
    placeholders = ",".join("?" for _ in ACTIVE_JOB_STATES[1:])
    with connection() as db:
        result = db.execute(f"""UPDATE processing_jobs SET status='discovering_scenes',current_stage='discovering_scenes',
            progress=CASE WHEN progress<1 THEN 1 ELSE progress END,worker_id=?,heartbeat_at=?,started_at=COALESCE(started_at,?),
            attempts=attempts+1,updated_at=? WHERE id=? AND
            (status='queued' OR (status IN ({placeholders}) AND (heartbeat_at IS NULL OR heartbeat_at<?)))""",
            (worker_id,stamp,stamp,stamp,job_id,*ACTIVE_JOB_STATES[1:],cutoff))
        if not result.rowcount:
            return None
        return _job_dict(db.execute("SELECT * FROM processing_jobs WHERE id=?", (job_id,)).fetchone())


def cancel_job(user_id: str, field_id: str, job_id: str):
    now = now_iso()
    with connection() as db:
        result = db.execute("""UPDATE processing_jobs SET status='cancelled',current_stage='cancelled',completed_at=?,
            updated_at=? WHERE id=? AND user_id=? AND field_id=? AND status IN (?,?,?,?,?,?)""",
            (now,now,job_id,user_id,field_id,*ACTIVE_JOB_STATES))
        row = db.execute("SELECT * FROM processing_jobs WHERE id=? AND user_id=? AND field_id=?", (job_id,user_id,field_id)).fetchone()
        return (_job_dict(row), bool(result.rowcount)) if row else (None, False)


def job_cancelled(job_id: str) -> bool:
    with connection() as db:
        row = db.execute("SELECT status FROM processing_jobs WHERE id=?", (job_id,)).fetchone()
        return bool(row and row[0] == "cancelled")


def pending_job_dispatches(limit: int = 100):
    with connection() as db:
        return [dict(row) for row in db.execute("SELECT * FROM job_outbox WHERE status='pending' ORDER BY created_at LIMIT ?",
                                                (max(1,min(limit,500)),)).fetchall()]


def mark_job_dispatched(job_id: str, task_id: str):
    now = now_iso()
    with connection() as db:
        db.execute("UPDATE job_outbox SET status='dispatched',dispatched_at=?,last_error=NULL WHERE job_id=? AND status='pending'", (now,job_id))
        db.execute("UPDATE processing_jobs SET dispatched_at=COALESCE(dispatched_at,?),celery_task_id=COALESCE(celery_task_id,?),updated_at=? WHERE id=?",
                   (now,task_id,now,job_id))


def mark_job_dispatch_error(job_id: str, message: str, max_attempts: int = 12):
    now = now_iso()
    with connection() as db:
        db.execute("UPDATE job_outbox SET attempts=attempts+1,last_error=? WHERE job_id=? AND status='pending'",
                   (message[:1000],job_id))
        row = db.execute("SELECT attempts FROM job_outbox WHERE job_id=?", (job_id,)).fetchone()
        if row and row[0] >= max_attempts:
            db.execute("UPDATE job_outbox SET status='failed' WHERE job_id=?", (job_id,))
            db.execute("UPDATE processing_jobs SET status='failed',current_stage='failed',error_json=?,completed_at=?,updated_at=? WHERE id=? AND status='queued'",
                       (_json_dump({"message":"The background queue is unavailable after repeated dispatch attempts."}),now,now,job_id))


def _field_dict(row):
    value = dict(row)
    boundary = value.pop("boundary_json", None)
    value["boundary"] = json.loads(boundary) if boundary else None
    if value.get("area_hectares") is not None:
        value["area_hectares"] = float(value["area_hectares"])
    for coord in ("latitude", "longitude"):
        if value.get(coord) is not None:
            value[coord] = float(value[coord])
    return value


def _observation_dict(row):
    value = dict(row)
    value["stats"] = json.loads(value.pop("stats_json"))
    return value


def _acquisition_dict(row):
    value = dict(row)
    if value.get("cloud_cover") is not None:
        value["cloud_cover"] = float(value["cloud_cover"])
    value["required_assets"] = json.loads(value.pop("required_assets_json"))
    value["retrieved_assets"] = json.loads(value.pop("retrieved_assets_json"))
    return value


def _prediction_dict(row):
    value = dict(row)
    value["result"] = json.loads(value.pop("result_json"))
    return value


def _job_dict(row):
    if not row:
        return None
    value = dict(row)
    value["request"] = json.loads(value.pop("request_json"))
    result_json, error_json = value.pop("result_json", None), value.pop("error_json", None)
    value["result"] = json.loads(result_json) if result_json else None
    value["error"] = json.loads(error_json) if error_json else None
    return value
