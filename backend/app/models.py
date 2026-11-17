"""SQLAlchemy Core schema shared by SQLite development and PostgreSQL production."""
from sqlalchemy import (
    CheckConstraint,
    Column,
    ForeignKey,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)

metadata = MetaData()

users = Table(
    "users", metadata,
    Column("id", String(36), primary_key=True),
    Column("email", String(320), nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("display_name", String(80)),
)

refresh_sessions = Table(
    "refresh_sessions", metadata,
    Column("token_hash", String(128), primary_key=True),
    Column("user_id", String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("expires_at", Integer, nullable=False),
    Column("created_at", String(40), nullable=False),
)

fields = Table(
    "fields", metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(120), nullable=False),
    Column("latitude", Float, nullable=False),
    Column("longitude", Float, nullable=False),
    Column("crop", String(120), nullable=False, server_default=text("'Unknown / Not identified'")),
    Column("planting_date", String(10)),
    Column("harvest_date", String(10)),
    Column("boundary_json", Text),
    Column("area_hectares", Float),
    Column("scene_metadata_checked_at", String(40)),
    Column("scene_metadata_count", Integer, nullable=False, server_default=text("0")),
    Column("created_at", String(40), nullable=False),
    Column("updated_at", String(40), nullable=False),
)
Index("ix_fields_user_id", fields.c.user_id)

observations = Table(
    "observations", metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("field_id", String(36), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
    Column("observed_at", String(40), nullable=False),
    Column("source", String(160), nullable=False),
    Column("scene_id", Text),
    Column("stats_json", Text, nullable=False),
    Column("processing_status", String(32), nullable=False, server_default=text("'stored'")),
    Column("created_at", String(40), nullable=False),
    UniqueConstraint("field_id", "observed_at", "source", name="uq_observations_field_date_source"),
)
Index("ix_observations_field", observations.c.user_id, observations.c.field_id, observations.c.observed_at)

satellite_acquisitions = Table(
    "satellite_acquisitions", metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("field_id", String(36), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
    Column("scene_id", String(512), nullable=False),
    Column("acquisition_date", String(10), nullable=False),
    Column("cloud_cover", Float),
    Column("status", String(40), nullable=False),
    Column("required_assets_json", Text, nullable=False, server_default=text("'[]'")),
    Column("retrieved_assets_json", Text, nullable=False, server_default=text("'[]'")),
    Column("observation_id", String(36), ForeignKey("observations.id", ondelete="SET NULL")),
    Column("error_message", Text),
    Column("created_at", String(40), nullable=False),
    Column("updated_at", String(40), nullable=False),
    UniqueConstraint("user_id", "field_id", "scene_id", name="uq_acquisitions_user_field_scene"),
)
Index("ix_satellite_acquisitions_field", satellite_acquisitions.c.user_id,
      satellite_acquisitions.c.field_id, satellite_acquisitions.c.acquisition_date)

predictions = Table(
    "predictions", metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("field_id", String(36), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
    Column("created_at", String(40), nullable=False),
    Column("model_type", String(40), nullable=False),
    Column("provenance", String(200), nullable=False),
    Column("result_json", Text, nullable=False),
    Column("job_id", String(36), ForeignKey("processing_jobs.id", ondelete="SET NULL"), unique=True),
)
Index("ix_predictions_field", predictions.c.user_id, predictions.c.field_id, predictions.c.created_at)

processing_jobs = Table(
    "processing_jobs", metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("field_id", String(36), ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
    Column("kind", String(40), nullable=False),
    Column("status", String(32), nullable=False),
    Column("progress", Integer, nullable=False, server_default=text("0")),
    Column("current_stage", String(40), nullable=False),
    Column("request_hash", String(64), nullable=False),
    Column("request_json", Text, nullable=False),
    Column("attempts", Integer, nullable=False, server_default=text("0")),
    Column("max_attempts", Integer, nullable=False, server_default=text("3")),
    Column("scenes_discovered", Integer, nullable=False, server_default=text("0")),
    Column("assets_retrieved", Integer, nullable=False, server_default=text("0")),
    Column("valid_observations", Integer, nullable=False, server_default=text("0")),
    Column("model_status", String(40)),
    Column("result_json", Text),
    Column("error_json", Text),
    Column("celery_task_id", String(128)),
    Column("worker_id", String(128)),
    Column("heartbeat_at", String(40)),
    Column("dispatched_at", String(40)),
    Column("created_at", String(40), nullable=False),
    Column("started_at", String(40)),
    Column("completed_at", String(40)),
    Column("updated_at", String(40), nullable=False),
    CheckConstraint("progress >= 0 AND progress <= 100", name="ck_processing_jobs_progress"),
    CheckConstraint("attempts >= 0 AND max_attempts > 0", name="ck_processing_jobs_attempts"),
)
Index("ix_processing_jobs_user_field", processing_jobs.c.user_id, processing_jobs.c.field_id,
      processing_jobs.c.created_at)
Index("ix_processing_jobs_status_created", processing_jobs.c.status, processing_jobs.c.created_at)
Index(
    "uq_processing_jobs_active_request",
    processing_jobs.c.user_id,
    processing_jobs.c.field_id,
    processing_jobs.c.request_hash,
    unique=True,
    sqlite_where=processing_jobs.c.status.in_(("queued", "discovering_scenes", "retrieving_assets",
        "processing_imagery", "generating_observations", "running_models")),
    postgresql_where=processing_jobs.c.status.in_(("queued", "discovering_scenes", "retrieving_assets",
        "processing_imagery", "generating_observations", "running_models")),
)

job_outbox = Table(
    "job_outbox", metadata,
    Column("id", String(36), primary_key=True),
    Column("job_id", String(36), ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False, unique=True),
    Column("event_type", String(80), nullable=False),
    Column("payload_json", Text, nullable=False),
    Column("status", String(24), nullable=False, server_default=text("'pending'")),
    Column("attempts", Integer, nullable=False, server_default=text("0")),
    Column("created_at", String(40), nullable=False),
    Column("dispatched_at", String(40)),
    Column("last_error", Text),
)
Index("ix_job_outbox_status_created", job_outbox.c.status, job_outbox.c.created_at)
