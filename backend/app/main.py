"""
Fasal Sanket - FastAPI Backend Application

Progressive crop-identification API powered by temporal Sentinel-2 observations
and the signature Taxonomic Resolution Calendar.
"""

import os
import json
import logging
import requests
from datetime import date, datetime, timedelta
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import hashlib
import time

import numpy as np
from sqlalchemy.exc import IntegrityError
from fastapi import FastAPI, HTTPException, Query, status, Depends, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import sys
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.schemas import (
    HealthResponse, CropsResponse, CropInfo, ModelsResponse, ModelMetadata,
    ModelsMetricsResponse, PredictRequest, PredictResponse,
    ProgressivePredictRequest, ProgressivePredictResponse, CalendarResponse,
    Credentials, AuthTokens, UserResponse, FieldCreate
)
from app.taxonomy import TAXONOMY_TREE, LEVEL_DESCRIPTIONS
from data.generator import CROPS, CROP_TO_ID, FEATURE_NAMES
from app import database
from app.security import (
    current_user, hash_password, verify_password, issue_access_token,
    refresh_token, hash_refresh_token, ACCESS_TOKEN_SECONDS, REFRESH_TOKEN_SECONDS
)
from app.satellite_processing import retrieve_polygon_statistics, CDSECredentialsMissing
from app.geometry import geometry_from_geojson
from app.stac_client import search_sentinel2_l2a
from app.sentinel_assets import (ASSET_KEYS, assets_available, s3_credentials_configured,
    process_stac_item, CDSES3CredentialsMissing, RasterDependenciesMissing)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fasal_sanket")

# Global Inference Engine loaded once at startup
engine = None


def get_engine():
    global engine
    if engine is None:
        from app.inference import InferenceEngine
        models_dir = os.path.join(backend_dir, "models")
        engine = InferenceEngine(models_dir=models_dir)
    return engine


# Botanical Crop Knowledge Metadata
CROP_CATALOG = [
    {
        "id": 0,
        "name": "Wheat",
        "scientific_name": "Triticum aestivum",
        "season": "Rabi (Winter/Spring)",
        "agronomic_type": "Cereal Grain / Monocot",
        "description": "Temperate rabi cereal with high chlorophyll absorption during winter vegetative peak and rapid dry-down before harvest.",
        "taxonomy": TAXONOMY_TREE["Wheat"],
        "phenology_peak": "Mid-season (t ~0.55)",
        "spectral_profile": {"peak_ndvi": 0.82, "flowering_signature": 0.12, "water_index_swir": -0.15}
    },
    {
        "id": 1,
        "name": "Rice",
        "scientific_name": "Oryza sativa",
        "season": "Kharif (Monsoon)",
        "agronomic_type": "Semi-aquatic Cereal / Monocot",
        "description": "Characterized by early-season standing water absorption in SWIR/NIR during transplanting, followed by rapid tillering to high canopy density.",
        "taxonomy": TAXONOMY_TREE["Rice"],
        "phenology_peak": "Late-monsoon (t ~0.58)",
        "spectral_profile": {"peak_ndvi": 0.86, "flowering_signature": 0.08, "water_index_swir": 0.35}
    },
    {
        "id": 2,
        "name": "Maize",
        "scientific_name": "Zea mays",
        "season": "Kharif / Spring",
        "agronomic_type": "Coarse Grain C4 / Monocot",
        "description": "C4 photosynthetic cereal exhibiting rapid early leaf area index growth, steep peak NDVI (~0.88), and early cob maturation.",
        "taxonomy": TAXONOMY_TREE["Maize"],
        "phenology_peak": "Early-mid season (t ~0.48)",
        "spectral_profile": {"peak_ndvi": 0.88, "flowering_signature": 0.10, "water_index_swir": -0.05}
    },
    {
        "id": 3,
        "name": "Mustard",
        "scientific_name": "Brassica juncea",
        "season": "Rabi (Winter)",
        "agronomic_type": "Oilseed / Dicot",
        "description": "Rabi broadleaf oilseed with distinctive bright yellow flowering flush at ~40% of season that alters Green/Red spectral ratios, followed by silique filling.",
        "taxonomy": TAXONOMY_TREE["Mustard"],
        "phenology_peak": "Mid-winter (t ~0.45)",
        "spectral_profile": {"peak_ndvi": 0.79, "flowering_signature": 0.42, "water_index_swir": -0.10}
    },
    {
        "id": 4,
        "name": "Cotton",
        "scientific_name": "Gossypium hirsutum",
        "season": "Kharif (Long Duration)",
        "agronomic_type": "Fiber Crop / Dicot",
        "description": "Extended vegetative establishment period with broad leafy bush canopy, prolonged flowering/boll maturation plateau, and late boll opening.",
        "taxonomy": TAXONOMY_TREE["Cotton"],
        "phenology_peak": "Post-monsoon extended plateau (t ~0.62)",
        "spectral_profile": {"peak_ndvi": 0.81, "flowering_signature": 0.15, "water_index_swir": -0.02}
    }
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.initialize_database()
    logger.info("Starting Fasal Sanket Agricultural Intelligence Backend...")
    logger.info("ML artifacts load on the first model request; auth and workspace startup stay lightweight.")
    yield
    logger.info("Shutting down Fasal Sanket Backend.")


app = FastAPI(
    title="Fasal Sanket - Agricultural Intelligence API",
    description=(
        "Progressive crop identification system using temporal Sentinel-2 observations. "
        "Answers: 'How specifically can we identify a crop, and when during the growing season can we identify it reliably?'"
    ),
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for local development and frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5500,http://localhost:5500").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _issue_session(response: Response, user: dict, remember_session: bool = True):
    refresh = refresh_token()
    expires = int(time.time()) + REFRESH_TOKEN_SECONDS
    database.save_refresh_session(hash_refresh_token(refresh), user["id"], expires)
    response.set_cookie(
        "fasal_refresh", refresh, httponly=True, secure=os.getenv("FORCE_HTTPS", "false").lower() == "true",
        samesite="lax", max_age=REFRESH_TOKEN_SECONDS if remember_session else None, path="/auth"
    )
    return AuthTokens(
        access_token=issue_access_token(user["id"]), expires_in=ACCESS_TOKEN_SECONDS,
        user={"id": user["id"], "email": user["email"]}
    )


@app.post("/auth/register", response_model=AuthTokens, status_code=status.HTTP_201_CREATED, tags=["Authentication"])
async def register(credentials: Credentials, response: Response):
    if not any(char.isalpha() for char in credentials.password) or not any(char.isdigit() for char in credentials.password):
        raise HTTPException(status_code=422, detail="Password must include at least one letter and one number")
    try:
        user = database.create_user(credentials.email, hash_password(credentials.password), credentials.name.strip() if credentials.name else None)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="An account with this email already exists") from exc
    return _issue_session(response, user)


@app.post("/auth/login", response_model=AuthTokens, tags=["Authentication"])
async def login(credentials: Credentials, response: Response):
    user = database.get_user_by_email(credentials.email)
    if not user or not verify_password(credentials.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect")
    return _issue_session(response, user, credentials.remember_session)


@app.post("/auth/refresh", response_model=AuthTokens, tags=["Authentication"])
async def refresh_session(request: Request, response: Response):
    presented = request.cookies.get("fasal_refresh")
    if not presented:
        raise HTTPException(status_code=401, detail="Refresh session is missing")
    token_hash = hash_refresh_token(presented)
    session = database.get_refresh_session(token_hash)
    if not session or session["expires_at"] <= int(time.time()):
        database.revoke_refresh_session(token_hash)
        response.delete_cookie("fasal_refresh", path="/auth", httponly=True, samesite="lax")
        raise HTTPException(status_code=401, detail="Refresh session is invalid or expired")
    database.revoke_refresh_session(token_hash)
    user = database.get_user_by_id(session["user_id"])
    if not user:
        response.delete_cookie("fasal_refresh", path="/auth", httponly=True, samesite="lax")
        raise HTTPException(status_code=401, detail="User account no longer exists")
    return _issue_session(response, user)


@app.post("/auth/logout", status_code=status.HTTP_204_NO_CONTENT, tags=["Authentication"])
async def logout(request: Request, response: Response, user: dict = Depends(current_user)):
    presented = request.cookies.get("fasal_refresh")
    if presented:
        session = database.get_refresh_session(hash_refresh_token(presented))
        if session and session["user_id"] == user["id"]:
            database.revoke_refresh_session(hash_refresh_token(presented))
    response.delete_cookie("fasal_refresh", path="/auth", httponly=True, samesite="lax")


@app.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_profile(user: dict = Depends(current_user)):
    return user


@app.get("/workspace/summary", tags=["Workspace"])
async def workspace_summary(user: dict = Depends(current_user)):
    return database.workspace_summary(user["id"])


@app.get("/fields", tags=["Fields"])
async def list_user_fields(user: dict = Depends(current_user)):
    fields = database.list_fields(user["id"])
    return {"fields": fields, "count": len(fields)}


@app.post("/fields", status_code=status.HTTP_201_CREATED, tags=["Fields"])
async def create_user_field(payload: FieldCreate, user: dict = Depends(current_user)):
    return database.save_field(user["id"], payload.model_dump())


@app.get("/fields/{field_id}", tags=["Fields"])
async def get_user_field(field_id: str, user: dict = Depends(current_user)):
    field = database.get_field(user["id"], field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return field


@app.put("/fields/{field_id}", tags=["Fields"])
async def update_user_field(field_id: str, payload: FieldCreate, user: dict = Depends(current_user)):
    field = database.save_field(user["id"], payload.model_dump(), field_id=field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    return field


@app.delete("/fields/{field_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["Fields"])
async def delete_user_field(field_id: str, user: dict = Depends(current_user)):
    if not database.delete_field(user["id"], field_id):
        raise HTTPException(status_code=404, detail="Field not found")


@app.get("/fields/{field_id}/observations", tags=["Fields"])
async def get_field_observations(field_id: str, user: dict = Depends(current_user)):
    if not database.get_field(user["id"], field_id):
        raise HTTPException(status_code=404, detail="Field not found")
    observations = database.list_observations(user["id"], field_id)
    return {"observations": observations, "count": len(observations), "source": "CDSE Sentinel-2 L2A", "synthetic": False}


@app.get("/fields/{field_id}/satellite/scenes", tags=["Satellite"])
async def search_field_scenes(
    field_id: str, start_date: date = Query(...), end_date: date = Query(...),
    max_cloud_cover: float = Query(30, ge=0, le=100), limit: int = Query(50, ge=1, le=100),
    user: dict = Depends(current_user),
):
    """Discover and classify STAC records without claiming pixels were retrieved."""
    field = database.get_field(user["id"], field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    geometry = geometry_from_geojson(field.get("boundary"))
    if not geometry:
        raise HTTPException(status_code=422, detail="Polygon boundary required for spatial intersection")
    if start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date")
    try:
        features = search_sentinel2_l2a(geometry, start_date.isoformat(), end_date.isoformat(), max_items=1000)
    except requests.RequestException as exc:
        logger.warning("CDSE polygon STAC search failed: %s", exc)
        raise HTTPException(status_code=502, detail="Copernicus polygon scene search is unavailable") from exc
    prior = {row["scene_id"]: row for row in database.list_satellite_acquisitions(user["id"], field_id)}
    scenes = []
    cloud_filtered = assets_missing = downloadable = 0
    for item in features:
        props = item.get("properties", {})
        cloud = props.get("eo:cloud_cover")
        scene_id = item.get("id")
        if not scene_id:
            continue
        assets_ok, missing_assets = assets_available(item)
        is_cloud_filtered = cloud is not None and cloud > max_cloud_cover
        previous = prior.get(scene_id, {})
        status_value = previous.get("status")
        if not status_value or status_value in ("discovered", "credentials_required", "assets_missing", "cloud_filtered"):
            status_value = "cloud_filtered" if is_cloud_filtered else "assets_missing" if not assets_ok else "available" if s3_credentials_configured() else "credentials_required"
        if is_cloud_filtered:
            cloud_filtered += 1
        if not assets_ok:
            assets_missing += 1
        can_download = assets_ok and not is_cloud_filtered and s3_credentials_configured()
        downloadable += int(can_download)
        acquisition_date = (props.get("datetime") or "")[:10]
        database.upsert_satellite_acquisition(user["id"], field_id, scene_id, acquisition_date,
            cloud, status_value, [ASSET_KEYS[k] for k in ASSET_KEYS], previous.get("retrieved_assets", []),
            previous.get("error_message"), previous.get("observation_id"))
        scenes.append({"id": scene_id, "datetime": props.get("datetime"), "acquisition_date": acquisition_date, "cloud_cover": cloud,
            "collection": item.get("collection"), "geometry": item.get("geometry"),
            "asset_status": status_value, "assets_available": assets_ok, "missing_assets": missing_assets,
            "downloadable": can_download, "retrieved_assets": previous.get("retrieved_assets", []),
            "observation_id": previous.get("observation_id"), "error_message": previous.get("error_message"),
            "assets": {k: {"href": v.get("href"), "type": v.get("type"), "title": v.get("title")}
                       for k, v in item.get("assets", {}).items()}})
    scenes.sort(key=lambda x: x.get("datetime") or "")
    observations = database.list_observations(user["id"], field_id)
    acquisitions = database.list_satellite_acquisitions(user["id"], field_id)
    failed = sum(a["status"] == "failed" for a in acquisitions)
    stored_scenes = sum(a["status"] == "stored" for a in acquisitions)
    dates = sorted({s["acquisition_date"] for s in scenes if not (s["cloud_cover"] is not None and s["cloud_cover"] > max_cloud_cover)
        and s["assets_available"]})
    database.save_scene_search(user["id"], field_id, len(features))
    counts = {"discovered": len(scenes), "downloadable": downloadable, "assets_available": sum(s["assets_available"] for s in scenes),
        "processed_observations": len(observations), "stored_scenes": stored_scenes, "failed": failed,
        "cloud_filtered": cloud_filtered, "assets_missing": assets_missing, "available_acquisition_dates": dates}
    return {"field_id": field_id, "scenes": scenes, "count": len(scenes), "counts": counts,
        "credential_configured": s3_credentials_configured(), "source": "Copernicus Data Space STAC",
        "workflow_status": {"metadata_retrieved": True, "assets_downloaded": stored_scenes > 0,
            "reflectance_processed": stored_scenes > 0, "field_statistics_calculated": len(observations) > 0},
        "truncated": len(features) >= 1000}


@app.get("/fields/{field_id}/satellite/acquisitions", tags=["Satellite"])
async def get_field_acquisition_status(field_id: str, user: dict = Depends(current_user)):
    if not database.get_field(user["id"], field_id):
        raise HTTPException(status_code=404, detail="Field not found")
    acquisitions = database.list_satellite_acquisitions(user["id"], field_id)
    observations = database.list_observations(user["id"], field_id)
    return {"field_id": field_id, "acquisitions": acquisitions,
        "counts": {"discovered": len(acquisitions),
            "downloadable": sum(a["status"] in ("available", "downloading_assets", "assets_retrieved", "processed") for a in acquisitions) if s3_credentials_configured() else 0,
            "assets_retrieved": sum(a["status"] in ("assets_retrieved", "processed", "stored") for a in acquisitions),
            "processed": sum(a["status"] in ("processed", "stored") for a in acquisitions),
            "stored_observations": len(observations), "failed": sum(a["status"] == "failed" for a in acquisitions),
            "cloud_filtered": sum(a["status"] == "cloud_filtered" for a in acquisitions)},
        "credential_configured": s3_credentials_configured()}


@app.post("/fields/{field_id}/observations/process", status_code=status.HTTP_202_ACCEPTED, tags=["Satellite"])
async def process_field_observations(
    field_id: str, start_date: date = Query(...), end_date: date = Query(...),
    max_cloud_cover: float = Query(30, ge=0, le=100), user: dict = Depends(current_user),
):
    """Enqueue durable satellite retrieval and raster processing; poll the returned job."""
    field = database.get_field(user["id"], field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    if not field.get("boundary"):
        raise HTTPException(status_code=422, detail="A saved Polygon or MultiPolygon boundary is required.")
    if start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date")
    if (end_date - start_date).days > 400:
        raise HTTPException(status_code=422, detail="A job may span at most 400 days")
    job, created = database.create_job(user["id"], field_id, "observations", {
        "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
        "max_cloud_cover": max_cloud_cover,
    })
    if job is None:
        raise HTTPException(status_code=404, detail="Field not found")
    return {"job": job, "created": created, "status_url": f"/fields/{field_id}/jobs/{job['id']}"}


@app.post("/fields/{field_id}/identification-jobs", status_code=status.HTTP_202_ACCEPTED, tags=["Inference"])
async def enqueue_identification_job(
    field_id: str, model_type: str = Query("random_forest", pattern="^(random_forest|lstm)$"),
    target_error_rate: float = Query(0.10, ge=0.01, le=0.50),
    start_date: date | None = Query(None), end_date: date | None = Query(None),
    max_cloud_cover: float = Query(30, ge=0, le=100), user: dict = Depends(current_user),
):
    field = database.get_field(user["id"], field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    if not field.get("boundary"):
        raise HTTPException(status_code=422, detail="A saved field polygon is required.")
    today = date.today()
    start = start_date or (date.fromisoformat(field["planting_date"]) if field.get("planting_date") else today - timedelta(days=365))
    end = end_date or (date.fromisoformat(field["harvest_date"]) if field.get("harvest_date") and date.fromisoformat(field["harvest_date"]) <= today else today)
    if start > end or (end-start).days > 400:
        raise HTTPException(status_code=422, detail="The identification acquisition window must be ordered and at most 400 days")
    job, created = database.create_job(user["id"], field_id, "identification", {
        "model_type": model_type, "target_error_rate": target_error_rate, "start_date": start.isoformat(),
        "end_date": end.isoformat(), "max_cloud_cover": max_cloud_cover,
    })
    return {"job": job, "created": created, "status_url": f"/fields/{field_id}/jobs/{job['id']}"}


@app.get("/fields/{field_id}/jobs", tags=["Jobs"])
async def get_field_jobs(field_id: str, user: dict = Depends(current_user)):
    if not database.get_field(user["id"], field_id):
        raise HTTPException(status_code=404, detail="Field not found")
    return {"jobs": database.list_jobs(user["id"], field_id)}


@app.get("/fields/{field_id}/jobs/{job_id}", tags=["Jobs"])
async def get_field_job(field_id: str, job_id: str, user: dict = Depends(current_user)):
    job = database.get_job(user["id"], field_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job}


@app.post("/fields/{field_id}/jobs/{job_id}/cancel", tags=["Jobs"])
async def cancel_field_job(field_id: str, job_id: str, user: dict = Depends(current_user)):
    job, changed = database.cancel_job(user["id"], field_id, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job": job, "cancelled": changed}


@app.post("/fields/{field_id}/identify", tags=["Inference"])
def identify_field_crop(
    field_id: str, model_type: str = Query("random_forest", pattern="^(random_forest|lstm)$"),
    target_error_rate: float = Query(0.10, ge=0.01, le=0.50),
    _job_id: Optional[str] = Query(None, include_in_schema=False), user: dict = Depends(current_user),
):
    """Produce explicitly experimental candidates from real observations; never uses synthetic evidence as real reliability."""
    field = database.get_field(user["id"], field_id)
    if not field:
        raise HTTPException(status_code=404, detail="Field not found")
    if not field.get("boundary"):
        raise HTTPException(status_code=422, detail="A saved field polygon is required for crop identification")
    rows = database.list_observations(user["id"], field_id)
    usable = []
    for row in rows:
        bands = row.get("stats", {}).get("bands", {})
        names = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "EVI"]
        values = [bands.get(name, {}).get("mean") for name in names]
        if row.get("stats", {}).get("valid_pixel_count", 0) > 0 and all(v is not None and np.isfinite(v) for v in values):
            usable.append({**row, "features": np.asarray(values, dtype=np.float32)})
    if len(usable) < 1:
        raise HTTPException(status_code=422, detail={"message": "Insufficient real satellite evidence.",
            "usable_observations": len(usable), "required_observations": 1,
            "next_step": "Acquire at least one clear, polygon-clipped Sentinel-2 observation. No crop candidate was generated."})
    usable.sort(key=lambda x: x["observed_at"])
    try:
        start_date = date.fromisoformat(field.get("planting_date") or usable[0]["observed_at"])
        end_date = date.fromisoformat(field.get("harvest_date") or usable[-1]["observed_at"])
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Planting/harvest dates are invalid") from exc
    if end_date <= start_date:
        start_date, end_date = date.fromisoformat(usable[0]["observed_at"]), date.fromisoformat(usable[-1]["observed_at"])
    duration = max((end_date - start_date).days, 1)
    checkpoints = []
    previous_probabilities = None
    previous_observation_count = 0
    eng = get_engine()
    for fraction in (0.2, 0.4, 0.6, 0.8, 1.0):
        cutoff = start_date + timedelta(days=round(duration * fraction))
        available = [r for r in usable if date.fromisoformat(r["observed_at"]) <= cutoff]
        base = {"checkpoint": int(fraction * 100), "checkpoint_date": cutoff.isoformat(),
            "observation_count": len(available), "date_range": [available[0]["observed_at"], available[-1]["observed_at"]] if available else None}
        if not available:
            checkpoints.append({**base, "status": "insufficient_observations", "candidate_crop": None,
                "model_confidence": None, "empirical_reliability": None, "supported_taxonomic_level": 0,
                "explanation": "No processed observation was available by this season checkpoint; future observations were excluded."})
            continue
        sequence = np.full((1, 20, 9), np.nan, dtype=np.float32)
        buckets = {}
        for row in available:
            offset = max(0, (date.fromisoformat(row["observed_at"]) - start_date).days)
            slot = min(19, max(0, int(np.ceil(offset / duration * 20)) - 1))
            buckets.setdefault(slot, []).append(row["features"])
        for slot, samples in buckets.items():
            sequence[0, slot] = np.mean(np.stack(samples), axis=0)
        base["model_input_temporal_bins"] = 20
        base["observed_temporal_bins"] = len(buckets)
        base["imputed_temporal_bins"] = 20 - len(buckets)
        try:
            result = eng.predict_single_stage(sequence, model_type=model_type,
                stage_fraction=fraction, target_error_rate=target_error_rate, use_empirical_evidence=False)
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=f"Experimental model inference unavailable: {exc}") from exc
        probabilities = result["probabilities"]
        spectral_delta = None
        if previous_observation_count and len(available) > previous_observation_count:
            old = usable[previous_observation_count - 1]["features"]
            new = available[-1]["features"]
            spectral_delta = {"NDVI_delta": round(float(new[6] - old[6]), 4),
                "NDWI_delta": round(float(new[7] - old[7]), 4), "EVI_delta": round(float(new[8] - old[8]), 4)}
        changes = None
        if previous_probabilities:
            changes = {crop: round(probabilities[crop] - previous_probabilities[crop], 4) for crop in probabilities}
        checkpoints.append({**base, "status": "experimental_candidate", "candidate_crop": result["predicted_crop"],
            "probabilities": probabilities, "model_confidence": result["taxonomic_resolution"]["model_confidence"],
            "empirical_reliability": None, "required_reliability": round(1.0 - target_error_rate, 4),
            "supported_taxonomic_level": 0, "supported_taxonomic_label": "Insufficient real-world validation evidence",
            "probability_change_from_previous": changes, "new_observation_spectral_change": spectral_delta,
            "explanation": f"Experimental candidate from a model trained and evaluated on synthetic data. {len(buckets)} of 20 seasonal input bins contain real observations; the rest are imputed. Confidence is not empirical accuracy; no taxonomic level is unlocked."})
        previous_probabilities = probabilities
        previous_observation_count = len(available)
    final = next((stage for stage in reversed(checkpoints) if stage.get("candidate_crop")), None)
    observations = [r["stats"] | {"observed_at": r["observed_at"], "source": r["source"]} for r in usable]
    report = {"field": {"id": field["id"], "name": field["name"], "area_hectares": field.get("area_hectares"),
            "latitude": field["latitude"], "longitude": field["longitude"], "boundary": field["boundary"],
            "analysis_period": [start_date.isoformat(), end_date.isoformat()]},
        "satellite_evidence": {"source": "Copernicus Sentinel-2 L2A", "observation_count": len(usable),
            "acquisition_dates": [r["observed_at"] for r in usable],
            "cloud_percentages": [r["stats"].get("cloud_contaminated_pixel_percentage") for r in usable],
            "vegetation_history": [{"date": r["observed_at"], "NDVI": r["stats"].get("mean_ndvi"),
                "NDWI": r["stats"].get("mean_ndwi"), "EVI": r["stats"].get("mean_evi")} for r in usable]},
        "identification": {"status": "experimental_candidate" if final else "insufficient_observations",
            "candidate_crop": final.get("candidate_crop") if final else None,
            "identified_crop": None, "user_display_label": "Unknown / Not identified",
            "scientific_name": None,
            "taxonomic_level": 0, "model_type": model_type,
            "model_training_provenance": "synthetic training dataset", "model_confidence": final.get("model_confidence") if final else None,
            "empirical_reliability": None, "validation_warning": "Insufficient real-world validation evidence.",
            "out_of_distribution_status": "Not assessed; no labelled real crop reference dataset is connected."},
        "checkpoints": checkpoints, "prediction_evolution": "Probability and spectral changes are shown per checkpoint; no prior candidate is asserted as a validated identification.",
        "decision_explanation": "The model input feature order matches the stored band means [B2, B3, B4, B8, B11, B12, NDVI, NDWI, EVI]. The model/scaler were trained on synthetic data; results are experimental only.",
        "synthetic_evidence_used_for_taxonomy": False}
    stored = database.save_prediction(user["id"], field_id, model_type, report, job_id=_job_id)
    return {"prediction_id": stored["id"], "report": report, "created_at": stored["created_at"]}


@app.get("/fields/{field_id}/predictions", tags=["Inference"])
async def get_field_predictions(field_id: str, user: dict = Depends(current_user)):
    if not database.get_field(user["id"], field_id):
        raise HTTPException(status_code=404, detail="Field not found")
    predictions = database.list_predictions(user["id"], field_id)
    return {"predictions": predictions, "count": len(predictions)}


@app.get("/satellite/scenes", tags=["Satellite"])
def search_satellite_scenes(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    start_date: date = Query(...),
    end_date: date = Query(...),
    max_cloud_cover: float = Query(30, ge=0, le=100),
    limit: int = Query(30, ge=1, le=100),
):
    """Search public Sentinel-2 L2A scene metadata through the CDSE STAC API.

    Discovery results are metadata only; no pixel assets are downloaded or represented
    as field reflectance. Protected assets may require CDSE credentials.
    """
    if start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date")
    # Approximate a 1 km square around the requested point for scene discovery.
    dlat = 0.0045
    dlon = 0.0045 / max(abs(np.cos(np.radians(latitude))), 0.1)
    endpoint = os.getenv("CDSE_STAC_URL", "https://stac.dataspace.copernicus.eu/v1/search")
    payload = {
        "collections": [os.getenv("CDSE_STAC_COLLECTION", "sentinel-2-l2a")],
        "bbox": [longitude-dlon, latitude-dlat, longitude+dlon, latitude+dlat],
        "datetime": f"{start_date.isoformat()}T00:00:00Z/{end_date.isoformat()}T23:59:59Z",
        "limit": limit,
        "query": {"eo:cloud_cover": {"lte": max_cloud_cover}},
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=(5, 25))
        response.raise_for_status()
        items = response.json().get("features", [])
    except requests.RequestException as exc:
        logger.warning("CDSE STAC search failed: %s", exc)
        raise HTTPException(status_code=502, detail="Copernicus scene search is unavailable. Check network access or CDSE_STAC_URL.") from exc
    scenes = []
    for item in items:
        props = item.get("properties", {})
        cloud = props.get("eo:cloud_cover")
        if cloud is not None and cloud > max_cloud_cover:
            continue
        scenes.append({
            "id": item.get("id"), "title": props.get("title") or item.get("id"),
            "datetime": props.get("datetime"), "cloud_cover": cloud,
            "collection": item.get("collection"), "geometry": item.get("geometry"),
            "assets": {k: {"href": v.get("href"), "type": v.get("type"), "title": v.get("title")}
                       for k, v in item.get("assets", {}).items()},
        })
    scenes.sort(key=lambda item: item.get("datetime") or "")
    return {"scenes": scenes, "count": len(scenes), "source": "Copernicus Data Space Ecosystem STAC", "collection": payload["collections"][0], "is_real_scene_metadata": True, "reflectance_processed": False}


@app.get("/satellite/weather", tags=["Satellite"])
def get_weather_support(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
    start_date: date = Query(...),
    end_date: date = Query(...),
):
    """Retrieve supporting daily weather from Open-Meteo; this is not satellite data."""
    if start_date > end_date:
        raise HTTPException(status_code=422, detail="start_date must be on or before end_date")
    endpoint = "https://archive-api.open-meteo.com/v1/archive" if end_date < date.today() else "https://api.open-meteo.com/v1/forecast"
    try:
        response = requests.get(endpoint, params={
            "latitude": latitude, "longitude": longitude,
            "start_date": start_date.isoformat(), "end_date": end_date.isoformat(),
            "daily": "temperature_2m_mean,precipitation_sum",
            "timezone": "UTC",
        }, timeout=(5, 20))
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Open-Meteo request failed: %s", exc)
        raise HTTPException(status_code=502, detail="Weather service is unavailable") from exc
    return {"source": "Open-Meteo", "data": response.json(), "is_satellite_data": False}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Returns application health, model load status, and device info."""
    loaded = engine.loaded_status if engine is not None else {"pipeline": False, "random_forest": False, "lstm": False}
    return HealthResponse(
        status="healthy",
        models_loaded=loaded,
        device="cpu",
        version="1.0.0",
        is_synthetic=True
    )


@app.get("/crops", response_model=CropsResponse, tags=["Agronomy"])
async def get_crops():
    """Lists the 5 initial target crops and their botanical taxonomic hierarchies."""
    return CropsResponse(
        crops=[CropInfo(**crop) for crop in CROP_CATALOG],
        count=len(CROP_CATALOG),
        is_synthetic_demonstration=True
    )


@app.get("/models", response_model=ModelsResponse, tags=["Machine Learning"])
async def get_models():
    """Lists available crop identification models and their artifact statuses."""
    eng = get_engine()
    models_list = []

    # Random Forest
    rf_ready = eng.loaded_status["random_forest"]
    rf_acc = eng.rf_metadata.get("full_season_test_accuracy")
    rf_f1 = eng.rf_metadata.get("full_season_test_macro_f1")
    models_list.append(ModelMetadata(
        name="random_forest",
        model_type="Random Forest Classifier (Tabular)",
        status="ready" if rf_ready else "artifact_missing",
        full_season_test_accuracy=rf_acc,
        full_season_test_macro_f1=rf_f1,
        input_features=FEATURE_NAMES
    ))

    # LSTM
    lstm_ready = eng.loaded_status["lstm"]
    lstm_acc = eng.lstm_metadata.get("full_season_test_accuracy")
    lstm_f1 = eng.lstm_metadata.get("full_season_test_macro_f1")
    models_list.append(ModelMetadata(
        name="lstm",
        model_type="PyTorch BiLSTM Sequence Classifier",
        status="ready" if lstm_ready else "artifact_missing",
        full_season_test_accuracy=lstm_acc,
        full_season_test_macro_f1=lstm_f1,
        input_features=FEATURE_NAMES
    ))

    available = [m.name for m in models_list if m.status == "ready"]
    return ModelsResponse(models=models_list, available_models=available)


@app.get("/models/metrics", response_model=ModelsMetricsResponse, tags=["Machine Learning"])
async def get_model_metrics():
    """Retrieves progressive evaluation metrics and confusion matrices across checkpoints."""
    metrics_path = os.path.join(backend_dir, "evaluation", "metrics.json")
    if not os.path.exists(metrics_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics artifact not found. Please run 'py backend/evaluation/evaluate.py' first."
        )

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    return ModelsMetricsResponse(
        dataset_info=metrics_data.get("dataset", {}),
        random_forest=metrics_data.get("random_forest"),
        lstm=metrics_data.get("lstm"),
        target_reliability_threshold=0.90,
        is_synthetic=True
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict_crop(request: PredictRequest):
    """
    Predicts crop identity and evaluates taxonomic resolution at a specific observation checkpoint.
    """
    eng = get_engine()

    try:
        obs_array = np.array(request.observations, dtype=np.float32)
        result = eng.predict_single_stage(
            observations=obs_array,
            model_type=request.model_type,
            stage_fraction=request.stage_fraction,
            target_error_rate=request.target_error_rate
        )
        return PredictResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {str(e)}")


@app.post("/predict/progressive", response_model=ProgressivePredictResponse, tags=["Inference"])
async def predict_progressive_crop(request: ProgressivePredictRequest):
    """
    Progressively evaluates the crop observation sequence across all 5 checkpoints
    (20%, 40%, 60%, 80%, 100%) to produce the full Taxonomic Resolution Calendar.
    """
    eng = get_engine()

    try:
        obs_array = np.array(request.observations, dtype=np.float32)
        result = eng.predict_progressive(
            observations=obs_array,
            model_type=request.model_type,
            target_error_rate=request.target_error_rate
        )
        return ProgressivePredictResponse(**result)

    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Progressive prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal progressive prediction error: {str(e)}")


@app.get("/calendar", response_model=CalendarResponse, tags=["Taxonomic Resolution"])
async def get_taxonomic_calendar(
    target_error_rate: float = Query(0.10, ge=0.01, le=0.50, description="Error budget threshold (e.g. 0.10 -> 90% reliability)"),
    model_type: str = Query("random_forest", description="Model whose validation evidence to use: 'random_forest' or 'lstm'")
):
    """
    Returns the system-wide Taxonomic Resolution Calendar mapping observation checkpoints
    to the highest supported biological specificity levels according to empirical evidence.
    Supports model-specific evidence (RF or LSTM).
    """
    eng = get_engine()
    # Select validation evidence from the requested model
    if model_type == "lstm" and eng.lstm_metadata:
        evidence = eng.lstm_metadata.get("progressive_val_evidence", {})
    else:
        evidence = eng.rf_metadata.get("progressive_val_evidence", eng.taxonomy_engine.validation_evidence)
    req_reliability = 1.0 - target_error_rate

    stages_info = []
    checkpoints = ["20%", "40%", "60%", "80%", "100%"]

    for chk in checkpoints:
        stage_ev = evidence.get(chk, {})
        supported_level = 0
        supported_name = "None"
        for lvl in [1, 2, 3, 4]:
            if stage_ev.get(f"level_{lvl}", 0.0) >= req_reliability:
                supported_level = lvl
                supported_name = f"Level {lvl} ({LEVEL_DESCRIPTIONS[lvl]})"

        stages_info.append({
            "stage": chk,
            "supported_level": supported_level,
            "supported_level_description": supported_name,
            "validation_evidence": stage_ev,
            "target_error_rate": target_error_rate,
            "required_reliability": req_reliability,
            "explanation": (
                f"At {chk} season progress, empirical reliability warrants {supported_name}. "
                if supported_level > 0 else
                f"At {chk} season progress, insufficient evidence meets the required {req_reliability:.0%} threshold."
            )
        })

    return CalendarResponse(
        description="Progressive taxonomic resolution levels supported under the configured error budget.",
        target_error_rate=target_error_rate,
        required_reliability=req_reliability,
        stages=stages_info,
        crops_represented=CROPS,
        is_synthetic_demonstration=True
    )


@app.get("/sample-field/{crop_name}", tags=["Agronomy Demo"])
async def get_sample_field(crop_name: str):
    """
    Convenience endpoint returning realistic synthetic Sentinel-2 temporal observation
    sequence for an exemplary field of the selected crop. Useful for live UI testing.
    """
    crop_cap = crop_name.capitalize()
    if crop_cap not in CROPS:
        raise HTTPException(status_code=404, detail=f"Unknown crop '{crop_name}'. Available: {CROPS}")

    from data.generator import simulate_phenology, compute_spectral_indices
    timesteps = np.linspace(0, 1, 20)
    rng = np.random.RandomState(42 + CROP_TO_ID[crop_cap])

    b2, b3, b4, b8, b11, b12 = simulate_phenology(
        crop=crop_cap,
        timesteps=timesteps,
        sowing_offset=0.0,
        vigor=1.0,
        rng=rng
    )
    ndvi, ndwi, evi = compute_spectral_indices(b2, b3, b4, b8, b11)

    observations = np.stack([b2, b3, b4, b8, b11, b12, ndvi, ndwi, evi], axis=-1).tolist()

    return {
        "crop": crop_cap,
        "features": FEATURE_NAMES,
        "num_timesteps": len(observations),
        "observations": observations,
        "taxonomy": TAXONOMY_TREE[crop_cap],
        "is_synthetic": True
    }


# Mount frontend static web files
frontend_dir = os.path.join(os.path.dirname(backend_dir), "frontend")
if os.path.exists(frontend_dir):
    for public_path in ("/app", "/login", "/signup", "/forgot-password", "/privacy", "/about"):
        app.add_api_route(public_path, lambda: FileResponse(os.path.join(frontend_dir, "index.html")), methods=["GET"], include_in_schema=False)
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

