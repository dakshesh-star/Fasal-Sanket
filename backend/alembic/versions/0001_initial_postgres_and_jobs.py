"""Create the application schema and durable processing job/outbox tables."""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

ACTIVE = "'queued','discovering_scenes','retrieving_assets','processing_imagery','generating_observations','running_models'"


def upgrade():
    op.create_table("users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False, unique=True),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("display_name", sa.String(80)))
    op.create_table("refresh_sessions",
        sa.Column("token_hash", sa.String(128), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("expires_at", sa.Integer(), nullable=False), sa.Column("created_at", sa.String(40), nullable=False))
    op.create_table("fields",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("latitude", sa.Float(), nullable=False), sa.Column("longitude", sa.Float(), nullable=False),
        sa.Column("crop", sa.String(120), nullable=False, server_default=sa.text("'Unknown / Not identified'")),
        sa.Column("planting_date", sa.String(10)), sa.Column("harvest_date", sa.String(10)),
        sa.Column("boundary_json", sa.Text()), sa.Column("area_hectares", sa.Float()),
        sa.Column("scene_metadata_checked_at", sa.String(40)),
        sa.Column("scene_metadata_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.String(40), nullable=False), sa.Column("updated_at", sa.String(40), nullable=False))
    op.create_index("ix_fields_user_id", "fields", ["user_id"])
    op.create_table("processing_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("field_id", sa.String(36), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
        sa.Column("kind", sa.String(40), nullable=False), sa.Column("status", sa.String(32), nullable=False),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("current_stage", sa.String(40), nullable=False), sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("request_json", sa.Text(), nullable=False), sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("scenes_discovered", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("assets_retrieved", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("valid_observations", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("model_status", sa.String(40)), sa.Column("result_json", sa.Text()), sa.Column("error_json", sa.Text()),
        sa.Column("celery_task_id", sa.String(128)), sa.Column("worker_id", sa.String(128)),
        sa.Column("heartbeat_at", sa.String(40)), sa.Column("dispatched_at", sa.String(40)),
        sa.Column("created_at", sa.String(40), nullable=False), sa.Column("started_at", sa.String(40)),
        sa.Column("completed_at", sa.String(40)), sa.Column("updated_at", sa.String(40), nullable=False),
        sa.CheckConstraint("progress >= 0 AND progress <= 100", name="ck_processing_jobs_progress"),
        sa.CheckConstraint("attempts >= 0 AND max_attempts > 0", name="ck_processing_jobs_attempts"))
    op.create_index("ix_processing_jobs_user_field", "processing_jobs", ["user_id", "field_id", "created_at"])
    op.create_index("ix_processing_jobs_status_created", "processing_jobs", ["status", "created_at"])
    op.create_index("uq_processing_jobs_active_request", "processing_jobs", ["user_id", "field_id", "request_hash"],
        unique=True, postgresql_where=sa.text(f"status IN ({ACTIVE})"),
        sqlite_where=sa.text(f"status IN ({ACTIVE})"))
    op.create_table("observations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("field_id", sa.String(36), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
        sa.Column("observed_at", sa.String(40), nullable=False), sa.Column("source", sa.String(160), nullable=False),
        sa.Column("scene_id", sa.Text()), sa.Column("stats_json", sa.Text(), nullable=False),
        sa.Column("processing_status", sa.String(32), nullable=False, server_default="stored"),
        sa.Column("created_at", sa.String(40), nullable=False),
        sa.UniqueConstraint("field_id", "observed_at", "source", name="uq_observations_field_date_source"))
    op.create_index("ix_observations_field", "observations", ["user_id", "field_id", "observed_at"])
    op.create_table("satellite_acquisitions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("field_id", sa.String(36), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
        sa.Column("scene_id", sa.String(512), nullable=False), sa.Column("acquisition_date", sa.String(10), nullable=False),
        sa.Column("cloud_cover", sa.Float()), sa.Column("status", sa.String(40), nullable=False),
        sa.Column("required_assets_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("retrieved_assets_json", sa.Text(), nullable=False, server_default="[]"),
        sa.Column("observation_id", sa.String(36), sa.ForeignKey("observations.id", ondelete="SET NULL")),
        sa.Column("error_message", sa.Text()), sa.Column("created_at", sa.String(40), nullable=False),
        sa.Column("updated_at", sa.String(40), nullable=False),
        sa.UniqueConstraint("user_id", "field_id", "scene_id", name="uq_acquisitions_user_field_scene"))
    op.create_index("ix_satellite_acquisitions_field", "satellite_acquisitions", ["user_id", "field_id", "acquisition_date"])
    op.create_table("predictions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(36), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("field_id", sa.String(36), sa.ForeignKey("fields.id", ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.String(40), nullable=False), sa.Column("model_type", sa.String(40), nullable=False),
        sa.Column("provenance", sa.String(200), nullable=False), sa.Column("result_json", sa.Text(), nullable=False),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("processing_jobs.id", ondelete="SET NULL"), unique=True))
    op.create_index("ix_predictions_field", "predictions", ["user_id", "field_id", "created_at"])
    op.create_table("job_outbox",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("processing_jobs.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("event_type", sa.String(80), nullable=False), sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("status", sa.String(24), nullable=False, server_default="pending"),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.String(40), nullable=False), sa.Column("dispatched_at", sa.String(40)),
        sa.Column("last_error", sa.Text()))
    op.create_index("ix_job_outbox_status_created", "job_outbox", ["status", "created_at"])


def downgrade():
    op.drop_index("ix_job_outbox_status_created", table_name="job_outbox")
    op.drop_table("job_outbox")
    op.drop_index("ix_predictions_field", table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_satellite_acquisitions_field", table_name="satellite_acquisitions")
    op.drop_table("satellite_acquisitions")
    op.drop_index("ix_observations_field", table_name="observations")
    op.drop_table("observations")
    op.drop_index("uq_processing_jobs_active_request", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_status_created", table_name="processing_jobs")
    op.drop_index("ix_processing_jobs_user_field", table_name="processing_jobs")
    op.drop_table("processing_jobs")
    op.drop_index("ix_fields_user_id", table_name="fields")
    op.drop_table("fields")
    op.drop_table("refresh_sessions")
    op.drop_table("users")
