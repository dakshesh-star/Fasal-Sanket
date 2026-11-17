# Fasal Sanket

Fasal Sanket is a polygon-first agricultural intelligence research prototype. It connects saved field boundaries to Sentinel-2 Level-2A scene discovery, field-level vegetation measurements, temporal analysis, and experimental crop candidates.

> **Research status:** The included Random Forest and LSTM models were trained on synthetic observations. Outputs are experimental candidates and are not validated real-world crop identifications. No real satellite observations or crop labels are fabricated.

## Features

- Account registration and sign-in, private field polygons, and field-area calculation.
- Sentinel-2 L2A STAC discovery for saved polygons, with cloud and required-asset availability reporting.
- Asynchronous acquisition and processing jobs, with per-scene status and stored observation history.
- Field-level NDVI, NDWI, and EVI statistics from retrieved spectral assets.
- Experimental crop analysis and synthetic research tools, kept separate from real field observations.
- Reports, progressive resolution calendar, model evaluation charts, and confusion matrices.

## Run locally

Use Python 3.12. From this directory, in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r backend\requirements.txt
Copy-Item .env.example .env
```

Set a private `FASAL_JWT_SECRET` in `.env` before using authentication. To run the API and frontend together:

```powershell
Set-Location backend
..\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --env-file ..\.env
```

Open <http://127.0.0.1:8000>. The API documentation is at <http://127.0.0.1:8000/docs>. The landing page is `/`, the authenticated application is `/app`, and sign-in and account creation are at `/login` and `/signup`.

The local setup uses SQLite. For asynchronous acquisition and processing, configure PostgreSQL and Redis and run the API and Celery worker as described in [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md).

## Real Sentinel-2 imagery

Public STAC scene discovery does not require credentials. Downloading Sentinel-2 L2A spectral assets requires a Copernicus Data Space account and S3 access keys. Generate them in the [CDSE S3 credentials panel](https://eodata-s3keysmanager.dataspace.copernicus.eu/panel/s3-credentials) and set `CDSE_S3_ACCESS_KEY_ID` and `CDSE_S3_SECRET_ACCESS_KEY` in your private `.env`. See the [CDSE S3 guide](https://documentation.dataspace.copernicus.eu/APIs/S3.html). Never commit `.env` or share credentials in the repository.

For each usable acquisition, the processor reads B2, B3, B4, B8, B11, B12 and the scene classification layer, applies product reflectance offsets/scaling and nodata/cloud masking, clips to the field polygon, and stores field-level index statistics with acquisition and scene provenance. Processing jobs run in the background when the production queue is configured.

## Configuration

Start with `.env.example`. Important settings include:

- `FASAL_JWT_SECRET`: replace the development value with a long, random secret.
- `DATABASE_URL`: managed PostgreSQL URL for production; the local SQLite file is intentionally excluded from Git.
- `REDIS_URL`: broker for background jobs.
- `CDSE_S3_ACCESS_KEY_ID` and `CDSE_S3_SECRET_ACCESS_KEY`: private keys for downloading imagery assets.
- `CORS_ORIGINS`: explicit frontend origins when the frontend and API use different hosts.

## Deployment

The repository contains a Vercel configuration, but the full FastAPI dependency set includes PyTorch, scikit-learn, Rasterio/GDAL and raster-processing code. This combination exceeds or risks Vercel Function bundle/runtime constraints, and synchronous raster processing is unsuitable for function request limits. The recommended deployment keeps the static frontend on Vercel and runs the existing FastAPI app plus its Celery worker as container services using managed PostgreSQL and Redis.

Deployment has **not** been completed or production-verified. See [VERCEL_DEPLOYMENT_READINESS.md](VERCEL_DEPLOYMENT_READINESS.md) and [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) for architecture, setup, and remaining gates.

## Verification

Run the backend suite from the repository root:

```powershell
python -m pytest backend/tests -q
```

The full HTTP suite may require a normal local or Linux CI environment because the Windows sandbox used during development could not create the required local socket pair. Verify database, queue, CDSE access, model loading, polygon processing, index calculations, authentication, and crop-candidate readiness in the target deployment before using it with real data.

## Scientific limitations

- Model artifacts, training data, and supplied evaluation metrics are synthetic. Their scores do not establish accuracy on real Sentinel-2 fields.
- Sparse real observations can be accepted by the current model input path through temporal imputation; this does not make the resulting candidate reliable.
- Real labelled crop data, independent geographic/temporal validation, calibration, and out-of-distribution assessment are needed before operational crop identification claims.
- Password-reset email requires an email provider. Imagery download requires CDSE S3 credentials. Neither is included in this repository.

## Project structure

```text
backend/     FastAPI application, processing pipeline, models, migrations, tests
frontend/    Public site and authenticated single-page application
api/         Vercel Python entry point
screenshots/ Product screenshots
```

No license has been assigned to this project yet.
