# Fasal Sanket: production setup

Deployment is intentionally not performed by this change. No production database, Redis service, CDSE S3 credentials, Vercel project link, or deployment authorization was available in this workspace.

## Service layout

```text
Browser ── same-origin requests ──> Vercel FastAPI + static frontend
                                       │
                                       ├── PostgreSQL (durable user, field, observation, job data)
                                       └── PostgreSQL outbox ──> Redis/Celery worker
                                                                    ├── CDSE STAC search
                                                                    ├── CDSE S3 asset retrieval
                                                                    └── Rasterio processing + persistence
```

Fast requests (authentication, saved fields, metadata discovery, and job submission/status) remain on the API. Bulk asset retrieval and raster work run in the worker. The worker writes acquisition stage and field observations to the same PostgreSQL database. A database outbox is committed with each job; its periodic dispatcher safely republishes pending messages after a queue outage. The broker may deliver more than once; workers claim jobs with a database lease, and observations are upserted by field/date/source.

## Local setup

1. Copy `.env.example` to a private `.env`; set `FASAL_JWT_SECRET` to a random value of at least 32 bytes. Set `APP_ENV=production` in production services. Keep `DATABASE_URL` blank for local SQLite. Never commit `.env`.
2. Install `backend/requirements.txt` into Python 3.12. Rasterio/GDAL must include a JP2-capable driver for actual CDSE assets.
3. Run the API from `backend/`: `uvicorn app.main:app --reload --port 8000`.
4. For background jobs, start Redis and run from `backend/`: `celery -A app.celery_app:celery_app worker --beat --loglevel=INFO --concurrency=1`.
5. The frontend uses the same origin in production and `localhost:8000` for local static previews.

Without Redis or a running worker, job submission still commits to SQLite/PostgreSQL and remains queued in the durable outbox. The UI exposes recorded dispatch errors when the dispatcher has attempted the broker; the outbox retries when the worker/Redis service is available again. A worker and Redis are required to process satellite jobs.

## Vercel API + frontend

The root `pyproject.toml` points Vercel's FastAPI runtime at the existing app; `vercel.json` includes the frontend files, backend source, and model artifacts. `requirements.txt` delegates to the existing backend dependency set. The existing routes remain unprefixed, so the existing same-origin frontend API URLs are preserved. Set `DATABASE_URL`, `FASAL_JWT_SECRET`, and `FORCE_HTTPS=true` in Vercel's encrypted environment settings. Set `CORS_ORIGINS` to exact trusted origins only if cross-origin access is required. `VERCEL` without `DATABASE_URL` fails fast rather than creating ephemeral SQLite data.

**Vercel function feasibility is not yet proven.** Its Python runtime supports FastAPI but limits uncompressed function bundles to 500 MB by default. This application imports PyTorch, SciPy, scikit-learn, NumPy and Rasterio, plus GDAL/JP2 native libraries. The complete API dependency set may exceed that bundle ceiling or fail its Linux native-library build. A Vercel build and deployed runtime check are required before saying the existing full API can run there. The current setup does not split a lightweight API runtime from ML-heavy routes. If the function build exceeds the ceiling, the smallest reliable change is to run the same FastAPI app as a container service alongside the worker and keep Vercel for the frontend; the routes and data model need not be migrated or rewritten.

The Python runtime currently supports FastAPI, and the Vercel documentation lists a 500 MB uncompressed function bundle limit. See [Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python) and [FastAPI on Vercel](https://vercel.com/docs/frameworks/backend/fastapi).

## PostgreSQL schema and importing the existing workspace

1. Create a managed PostgreSQL database with TLS and configure `DATABASE_URL` in both the Vercel project and worker service. Use a provider's pooled URL for serverless access when offered. The API defaults to one pooled connection per Vercel instance; tune the worker separately.
2. Before API/worker startup, from `backend/`, run `alembic upgrade head` with `DATABASE_URL` set. The API checks connectivity and deliberately does not auto-create production tables.
3. Back up the local SQLite file. With `DATABASE_URL` set, run from the repository root: `python backend/scripts/import_sqlite_to_postgres.py --source backend/data/fasal_sanket.sqlite3 --report sqlite_import_report.json`. The script opens the source read-only, refuses to merge into nonempty tables by default, preserves record IDs, and uses conflict-safe inserts. To merge into a populated target, inspect a backup first, then explicitly pass `--allow-nonempty`; existing destination rows are not overwritten. Review the report and reconcile any rows rejected by foreign keys before switching the frontend.
4. The importer report lists database row IDs and errors, not passwords or tokens. Store it securely; it may contain user identifiers.

## Worker service (Railway / Fly.io / equivalent)

Build the repository with `backend/Dockerfile` (Docker build context is repository root). Set `DATABASE_URL`, `REDIS_URL`, `FASAL_JWT_SECRET`, `CDSE_S3_ENDPOINT`, `CDSE_S3_ACCESS_KEY_ID`, and `CDSE_S3_SECRET_ACCESS_KEY` as encrypted service variables. Run the image's default Celery worker/beat command. Keep CDSE credentials worker-only; public STAC discovery does not require them. Configure at least one worker replica and a persistent managed Redis broker. Raster processing uses temporary directories only and persists observations/job status to PostgreSQL; local files are not treated as durable.

## Job and inference semantics

- `POST /fields/{field_id}/observations/process` now returns HTTP 202 with a durable `job` and `status_url`. The route is preserved; long downloads no longer run in that HTTP request.
- `POST /fields/{field_id}/identification-jobs` queues acquisition and then inference. Existing `POST /fields/{field_id}/identify` remains for immediate inference against already stored observations.
- Poll `GET /fields/{field_id}/jobs` and `GET /fields/{field_id}/jobs/{job_id}`; cancel with `POST /fields/{field_id}/jobs/{job_id}/cancel`. All job routes are scoped to authenticated owner and field.
- Scene discovery, asset availability, retrieval, raster processing, and stored observations are separate states. Missing S3 credentials or raster runtime are terminal, explanatory job errors rather than fabricated success.
- One valid real acquisition is sufficient for the current code path to produce an experimental model candidate; the model input has 20 temporal bins, with missing bins imputed. This is not evidence of a scientifically validated one-date crop model. Predictions remain labeled experimental because provided model artifacts were trained on synthetic observations. Real-world labeled field data and independent holdout evaluation are still needed before operational crop identification claims.

## Verification gates and current limits

Verified locally: SQLAlchemy SQLite persistence, migration against a clean SQLite database, job ownership/deduplication/claim/cancel/result handling, focused spectral/index tests, and API module import. The local workspace has Rasterio installed, but no CDSE S3 keys; no actual imagery was downloaded in this task. No managed PostgreSQL, Redis, production auth, worker delivery, Vercel build, production bundle size, or deployed URL was available to test. Do not treat the local SQLite migration as proof of PostgreSQL compatibility or the mock/local tests as CDSE end-to-end verification.

Required before deployment: provision PostgreSQL and Redis; supply CDSE S3 keys to the worker; run Alembic and migrate/back up existing SQLite records; run auth and the complete API test suite in an environment where ASGI tests can start; verify package bundle and Linux GDAL/JP2 support; execute one real saved-polygon acquisition end to end; inspect persisted observation rows and experimental report; only then deploy and check the public URL.
