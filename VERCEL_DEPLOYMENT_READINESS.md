# Vercel readiness status

**Not deployed.** The project now has a Vercel FastAPI entrypoint and configuration, but this full backend dependency set cannot currently be considered deployable as a Vercel Function.

## Implemented in the repository

- `pyproject.toml` points the Vercel FastAPI runtime at the existing app; frontend and all current API route paths remain in the same application.
- `vercel.json` bundles the frontend, backend source, and model artifacts for the function. The frontend production API base remains same-origin.
- `backend/app/database.py` supports SQLite locally and PostgreSQL through `DATABASE_URL`. Vercel without `DATABASE_URL` fails fast instead of writing to ephemeral function storage.
- Alembic manages the production schema. A read-only SQLite-to-PostgreSQL importer supports stable IDs and conflict-safe repeatable inserts.
- Vercel requests enqueue durable PostgreSQL job/outbox rows. An independent Celery worker and Redis broker perform image retrieval, raster processing, and optional model inference.
- Authentication and existing routes remain in FastAPI. The observation-processing route returns HTTP 202 with a durable job ID; job status, list, and cancel routes are owner/field scoped.

## Vercel function size limit

The official Python runtime supports FastAPI but sets a 500 MB uncompressed function bundle ceiling. In this checkout's installed environment, the `torch` package alone occupies 513.6 MiB; SciPy occupies 109.3 MiB, Rasterio 38.0 MiB, scikit-learn 40.6 MiB, NumPy 31.1 MiB, plus Pandas, model artifacts, GDAL/JP2 native libraries, and the rest of the API dependencies. Linux wheels will differ from this Windows measurement, but the full set is clearly above the function budget even before the other packages are included. There is no Vercel build here to verify a platform-specific exception, and the Python runtime does not automatically prune these packages from the monolithic FastAPI app.

**Smallest reliable external boundary:** keep the frontend on Vercel and run the same FastAPI app as a container service alongside the already required Celery worker, using the same PostgreSQL and Redis. This preserves all existing API routes and model artifacts and avoids splitting/rebuilding the application's inference and raster modules. `backend/Dockerfile` can run the worker by default; for the API service, override its command with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. If the API has a separate public origin, set that public origin in `frontend/api-config.js` (`window.FASAL_API_BASE_URL`) before serving the static frontend, and allow the Vercel origin in the API's exact `CORS_ORIGINS`. This value is public and is never a secret. A split lightweight Vercel API plus model proxy is possible but is a larger code change and has not been implemented.

## Remaining deployment gates

No PostgreSQL or Redis URL, CDSE S3 access/secret key, Vercel project/account link, or deployment approval path is available. Thus no real imagery was acquired, no production auth/database/worker connection was tested, and no public deployment was made. The full TestClient API suite also cannot start its AnyIO portal in this Windows sandbox: it stalls in `asyncio.windows_events` while creating a local socket pair. Direct auth/job route smoke checks and focused non-HTTP tests pass; rerun the full HTTP test suite in normal local CI or Linux.

See [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) for setup, migration/import, worker, scientific-integrity, and verification instructions. See [ARCHITECTURE_AUDIT.md](ARCHITECTURE_AUDIT.md) for the pre-change baseline audit.
