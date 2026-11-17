"""Worker-side Sentinel-2 L2A discovery, asset retrieval and field summaries."""
from __future__ import annotations

import logging
from typing import Callable

import requests

from app import database
from app.geometry import geometry_from_geojson
from app.sentinel_assets import (ASSET_KEYS, assets_available, s3_credentials_configured,
    process_stac_item, CDSES3CredentialsMissing, RasterDependenciesMissing)
from app.stac_client import search_sentinel2_l2a

logger = logging.getLogger(__name__)


def _asset_error_code(exc):
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return (response.get("Error", {}).get("Code"),
                int(response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0))
    return (None, 0)


class PipelineError(RuntimeError):
    def __init__(self, detail, *, retryable=False):
        super().__init__(detail.get("message", str(detail)) if isinstance(detail, dict) else str(detail))
        self.detail = detail
        self.retryable = retryable


def process_observations(user_id: str, field_id: str, start_date: str, end_date: str,
                         max_cloud_cover: float = 30, *, on_progress: Callable | None = None,
                         is_cancelled: Callable[[], bool] | None = None) -> dict:
    field = database.get_field(user_id, field_id)
    if not field or not field.get("boundary"):
        raise PipelineError({"message": "A saved field Polygon or MultiPolygon is required."})
    geometry = geometry_from_geojson(field["boundary"])
    if not geometry:
        raise PipelineError({"message": "The saved field geometry is invalid or empty."})
    if on_progress:
        on_progress("discovering_scenes", 5)
    try:
        scenes = search_sentinel2_l2a(geometry, start_date, end_date, max_items=1000)
    except requests.RequestException as exc:
        raise PipelineError({"message": "Copernicus Data Space scene discovery failed.", "stage": "scene_discovery"}, retryable=True) from exc

    accepted, cloud_filtered, missing_assets = [], 0, 0
    required = [ASSET_KEYS[k] for k in ASSET_KEYS]
    for item in scenes:
        props = item.get("properties", {})
        scene_id = item.get("id")
        if not scene_id:
            continue
        cloud = props.get("eo:cloud_cover")
        available, _missing = assets_available(item)
        status = "cloud_filtered" if cloud is not None and cloud > max_cloud_cover else "assets_missing" if not available else "available"
        cloud_filtered += status == "cloud_filtered"
        missing_assets += status == "assets_missing"
        database.upsert_satellite_acquisition(user_id, field_id, scene_id,
            (props.get("datetime") or "")[:10], cloud, status, required)
        if status == "available":
            accepted.append(item)
    database.save_scene_search(user_id, field_id, len(scenes))
    if on_progress:
        on_progress("retrieving_assets", 15, scenes_discovered=len(scenes))

    if accepted and not s3_credentials_configured():
        for item in accepted:
            database.update_satellite_acquisition(user_id, field_id, item["id"], "credentials_required",
                error_message="CDSE S3 credentials are not configured.")
        raise PipelineError({"message": "Catalog scenes were found, but asset retrieval requires CDSE S3 credentials.",
            "next_step": "Set CDSE_S3_ACCESS_KEY_ID and CDSE_S3_SECRET_ACCESS_KEY in the worker secret store.",
            "scenes_discovered": len(scenes), "downloadable": 0, "cloud_filtered": cloud_filtered,
            "assets_missing": missing_assets, "failed": 0})

    eligible = accepted[:100]
    daily: dict[str, list] = {}
    failures = []
    for index, item in enumerate(eligible):
        if is_cancelled and is_cancelled():
            break
        props = item.get("properties", {})
        scene_id = item["id"]
        acquired = (props.get("datetime") or "")[:10]
        database.update_satellite_acquisition(user_id, field_id, scene_id, "downloading_assets", error_message=None)
        if on_progress:
            on_progress("retrieving_assets", min(65, 15 + int(50 * index / max(len(eligible), 1))),
                        scenes_discovered=len(scenes))
        def mark_retrieved(names, sid=scene_id):
            database.update_satellite_acquisition(user_id, field_id, sid, "assets_retrieved", names)
            if on_progress:
                on_progress("processing_imagery", min(70, 40 + int(20 * index / max(len(eligible), 1))),
                            scenes_discovered=len(scenes))
        try:
            stats = process_stac_item(item, field["boundary"], on_assets_retrieved=mark_retrieved)
            stats["scene_id"] = scene_id
            daily.setdefault(acquired, []).append(stats)
            database.update_satellite_acquisition(user_id, field_id, scene_id, "processed", required)
        except CDSES3CredentialsMissing as exc:
            database.update_satellite_acquisition(user_id, field_id, scene_id, "credentials_required", error_message=str(exc)[:500])
            raise PipelineError({"message": str(exc), "next_step": "Verify the CDSE S3 key pair and endpoint in the worker environment.",
                                 "scenes_discovered": len(scenes), "failed": len(failures)}) from exc
        except RasterDependenciesMissing as exc:
            raise PipelineError({"message": str(exc), "next_step": "Install Rasterio and its JP2-capable GDAL runtime in the worker image."}) from exc
        except requests.RequestException as exc:
            logger.warning("Temporary Sentinel-2 retrieval failure scene=%s type=%s", scene_id, type(exc).__name__)
            failures.append({"scene_id": scene_id, "message": "Temporary network error"})
            database.update_satellite_acquisition(user_id, field_id, scene_id, "failed", error_message="Temporary network error")
        except Exception as exc:
            error_code, http_status = _asset_error_code(exc)
            if error_code in {"AccessDenied", "InvalidAccessKeyId", "SignatureDoesNotMatch", "InvalidToken"}:
                database.update_satellite_acquisition(user_id, field_id, scene_id, "credentials_required",
                    error_message="CDSE rejected the configured S3 credentials.")
                raise PipelineError({"message": "Copernicus Data Space rejected the configured S3 access keys.",
                    "next_step": "Verify the CDSE S3 key pair and endpoint in the worker's secret store.",
                    "scene_id": scene_id}) from exc
            if error_code in {"SlowDown", "ServiceUnavailable", "InternalError", "RequestTimeout"} or http_status >= 500:
                raise PipelineError({"message": "Copernicus asset storage returned a temporary service error.",
                    "scene_id": scene_id}, retryable=True) from exc
            if type(exc).__name__ in {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError",
                                      "ConnectionClosedError", "HTTPClientError"}:
                raise PipelineError({"message": "Temporary connection failure while retrieving Copernicus assets.",
                    "scene_id": scene_id}, retryable=True) from exc
            logger.exception("Sentinel-2 asset processing failed for scene %s", scene_id)
            failures.append({"scene_id": scene_id, "message": str(exc)[:300]})
            database.update_satellite_acquisition(user_id, field_id, scene_id, "failed", error_message=str(exc)[:500])

    if is_cancelled and is_cancelled():
        return {"cancelled": True, "scene_counts": {"discovered": len(scenes), "downloadable": len(eligible),
            "cloud_filtered": cloud_filtered, "assets_missing": missing_assets}}

    if on_progress:
        on_progress("generating_observations", 78, scenes_discovered=len(scenes))
    observations = []
    source = "Copernicus Sentinel-2 L2A STAC assets"
    for acquired, tiles in sorted(daily.items()):
        valid = [tile for tile in tiles if int(tile.get("valid_pixel_count") or 0) > 0]
        total = sum(int(tile.get("valid_pixel_count") or 0) for tile in valid)
        if not total:
            continue
        names = ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "EVI"]
        bands = {}
        for name in names:
            entries = [tile["bands"][name] for tile in valid if tile.get("bands", {}).get(name, {}).get("mean") is not None]
            weights = [max(0, int(entry.get("valid_pixel_count") or 0)) for entry in entries]
            denominator = sum(weights) or 1
            bands[name] = {"mean": sum(entry["mean"] * weight for entry, weight in zip(entries, weights)) / denominator,
                "median": sum(entry["median"] * weight for entry, weight in zip(entries, weights)) / denominator,
                "min": min(entry["min"] for entry in entries), "max": max(entry["max"] for entry in entries),
                "valid_pixel_count": sum(weights), "nodata_pixel_count": sum(int(entry.get("nodata_pixel_count", 0)) for entry in entries)}
        inside = sum(int(tile.get("inside_pixel_count", 0)) for tile in valid)
        cloud_weight = sum(tile.get("cloud_contaminated_pixel_percentage", 0) * tile.get("inside_pixel_count", 0) for tile in valid)
        stats = {"bands": bands, "mean_ndvi": bands["NDVI"]["mean"], "median_ndvi": bands["NDVI"]["median"],
            "min_ndvi": bands["NDVI"]["min"], "max_ndvi": bands["NDVI"]["max"], "mean_ndwi": bands["NDWI"]["mean"],
            "mean_evi": bands["EVI"]["mean"], "valid_pixel_count": total,
            "cloud_contaminated_pixel_percentage": round(cloud_weight / max(inside, 1), 3),
            "inside_pixel_count": inside, "reflectance_scale": valid[0]["reflectance_scale"],
            "cloud_mask": "SCL 3,8,9,10 cloud/shadow; 0,1,11 no-data/saturated/snow excluded",
            "processing_status": "stored", "processed_scene_count": len(valid)}
        scene_ids = "|".join(sorted(tile["scene_id"] for tile in valid))
        observation = database.save_observation(user_id, field_id, acquired, stats, scene_id=scene_ids, source=source)
        if observation:
            observations.append(observation)
            for tile in valid:
                database.update_satellite_acquisition(user_id, field_id, tile["scene_id"], "stored", required,
                                                       observation_id=observation["id"])
    acquisition_rows = database.list_satellite_acquisitions(user_id, field_id)
    counts = {"discovered": len(scenes), "downloadable": len(eligible),
        "assets_retrieved": sum(a["status"] in ("assets_retrieved", "processed", "stored") for a in acquisition_rows),
        "processed": sum(a["status"] in ("processed", "stored") for a in acquisition_rows),
        "failed": len(failures), "cloud_filtered": cloud_filtered, "assets_missing": missing_assets}
    return {"observations": observations, "observation_count": len(observations), "scene_counts": counts,
            "failed_acquisitions": failures, "truncated": len(accepted) > 100,
            "available_acquisition_dates": sorted({(x.get("properties", {}).get("datetime") or "")[:10] for x in eligible})}
