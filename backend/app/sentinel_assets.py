"""CDSE STAC asset download and Sentinel-2 L2A field extraction."""
from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


ASSET_KEYS = {
    "B2": "B02_10m", "B3": "B03_10m", "B4": "B04_10m", "B8": "B08_10m",
    "B11": "B11_20m", "B12": "B12_20m", "SCL": "SCL_20m",
}
REFLECTANCE_BANDS = ("B2", "B3", "B4", "B8", "B11", "B12")
SCL_CLOUD_CLASSES = (3, 8, 9, 10)
SCL_INVALID_CLASSES = (0, 1, 3, 8, 9, 10, 11)
L2A_BAND_IDS = {"B2": 1, "B3": 2, "B4": 3, "B8": 7, "B11": 11, "B12": 12}


class CDSES3CredentialsMissing(RuntimeError):
    pass


class RasterDependenciesMissing(RuntimeError):
    pass


def assets_available(item: dict) -> tuple[bool, list[str]]:
    assets = item.get("assets", {})
    missing = [key for key in ASSET_KEYS.values() if not assets.get(key, {}).get("href")]
    return not missing, missing


def s3_credentials_configured() -> bool:
    return bool(os.getenv("CDSE_S3_ACCESS_KEY_ID") and os.getenv("CDSE_S3_SECRET_ACCESS_KEY"))


def _s3_client():
    access = os.getenv("CDSE_S3_ACCESS_KEY_ID")
    secret = os.getenv("CDSE_S3_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise CDSES3CredentialsMissing(
            "CDSE S3 keys are not configured. Add CDSE_S3_ACCESS_KEY_ID and "
            "CDSE_S3_SECRET_ACCESS_KEY to the private .env, then restart the app."
        )
    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise RasterDependenciesMissing("Install backend dependencies, including boto3 and rasterio, to process STAC assets.") from exc
    return boto3.client(
        "s3", endpoint_url=os.getenv("CDSE_S3_ENDPOINT", "https://eodata.dataspace.copernicus.eu"),
        aws_access_key_id=access, aws_secret_access_key=secret,
        region_name="default", config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def reflectance_from_dn(dn, offset: float = -1000.0, quantification_value: float = 10000.0):
    """Apply Sentinel-2 L2A BOA reflectance conversion; DN 0 remains nodata."""
    values = np.asarray(dn, dtype=np.float32)
    result = (values + float(offset)) / float(quantification_value)
    return np.where(values == 0, np.nan, result)


def calculate_window(bands: dict[str, np.ndarray], scl: np.ndarray, inside: np.ndarray,
                     valid_masks: dict[str, np.ndarray], offsets: dict[str, float], quantification_value: float):
    """Return reflectance and index pixels after geometry, nodata, and SCL masking."""
    reflectance = {
        name: reflectance_from_dn(bands[name], offsets.get(name, -1000.0), quantification_value)
        for name in REFLECTANCE_BANDS
    }
    nodata_valid = np.asarray(inside, dtype=bool).copy()
    for name in REFLECTANCE_BANDS:
        nodata_valid &= np.asarray(valid_masks[name], dtype=bool) & np.isfinite(reflectance[name])
    scl = np.asarray(scl)
    cloud = nodata_valid & np.isin(scl, SCL_CLOUD_CLASSES)
    valid = nodata_valid & ~np.isin(scl, SCL_INVALID_CLASSES)
    blue, green, red, nir = (reflectance[k] for k in ("B2", "B3", "B4", "B8"))
    ndvi_den = nir + red
    ndwi_den = green + nir
    evi_den = nir + 6 * red - 7.5 * blue + 1
    valid &= (np.abs(ndvi_den) > 1e-8) & (np.abs(ndwi_den) > 1e-8) & (np.abs(evi_den) > 1e-8)
    with np.errstate(divide="ignore", invalid="ignore"):
        indices = {
            "NDVI": (nir - red) / ndvi_den,
            "NDWI": (green - nir) / ndwi_den,
            "EVI": 2.5 * (nir - red) / evi_den,
        }
    for arr in indices.values():
        valid &= np.isfinite(arr)
    return reflectance, indices, valid, cloud, nodata_valid


def _metadata_radiometry(xml_path: Path) -> tuple[float, dict[str, float]]:
    root = ET.parse(xml_path).getroot()
    def local(tag):
        return tag.rsplit("}", 1)[-1]
    quant = 10000.0
    offsets = {band: -1000.0 for band in REFLECTANCE_BANDS}
    for node in root.iter():
        name = local(node.tag)
        if name == "BOA_QUANTIFICATION_VALUE" and node.text:
            quant = float(node.text)
        if name == "BOA_ADD_OFFSET":
            try:
                band_id = int(node.attrib.get("band_id", "-1"))
                value = float(node.text or "")
                band = next((b for b, idx in L2A_BAND_IDS.items() if idx == band_id), None)
                if band:
                    offsets[band] = value
            except (ValueError, TypeError):
                continue
    return quant, offsets


def _download_item(item: dict, output: Path) -> dict[str, Path]:
    client = _s3_client()
    ok, missing = assets_available(item)
    if not ok:
        raise ValueError("Required STAC assets are missing: " + ", ".join(missing))
    names = {**ASSET_KEYS, "metadata": "product_metadata"}
    paths = {}
    for name, asset_key in names.items():
        href = item.get("assets", {}).get(asset_key, {}).get("href")
        if not href:
            if name == "metadata":
                continue
            raise ValueError(f"STAC item is missing required asset {asset_key}.")
        parsed = urlparse(href)
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
            raise ValueError(f"Unsupported STAC asset URL for {asset_key}; expected s3://bucket/key.")
        path = output / f"{name}{Path(parsed.path).suffix or '.jp2'}"
        client.download_file(parsed.netloc, parsed.path.lstrip("/"), str(path))
        paths[name] = path
    if "metadata" not in paths:
        raise ValueError("STAC item has no product_metadata asset needed for exact L2A reflectance scaling.")
    return paths


def _stats(values: np.ndarray, bins: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {"mean": None, "median": None, "min": None, "max": None, "valid_pixel_count": 0,
                "nodata_pixel_count": 0, "histogram": np.zeros(len(bins) - 1, dtype=np.int64)}
    hist, _ = np.histogram(np.clip(values, bins[0], bins[-1]), bins=bins)
    cumulative = np.cumsum(hist)
    median_bin = int(np.searchsorted(cumulative, (values.size + 1) // 2))
    median = float((bins[median_bin] + bins[median_bin + 1]) / 2)
    return {"mean": float(np.mean(values)), "median": median, "min": float(np.min(values)), "max": float(np.max(values)),
            "valid_pixel_count": int(values.size), "nodata_pixel_count": 0, "histogram": hist}


def process_stac_item(item: dict, geometry: dict, chunk_size: int = 512, on_assets_retrieved=None) -> dict:
    """Download the item bands, clip by polygon, and calculate masked per-pixel statistics."""
    try:
        import rasterio
        from rasterio.features import geometry_mask, geometry_window
        from rasterio.transform import window_transform
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import Resampling, transform_geom
    except ImportError as exc:
        raise RasterDependenciesMissing("Install backend dependencies including rasterio before processing Sentinel-2 JP2 assets.") from exc

    with tempfile.TemporaryDirectory(prefix="fasal-s2-") as temp:
        paths = _download_item(item, Path(temp))
        if on_assets_retrieved:
            on_assets_retrieved(list(paths))
        quant, offsets = _metadata_radiometry(paths["metadata"])
        with rasterio.open(paths["B2"]) as reference:
            shape_geometry = transform_geom("EPSG:4326", reference.crs, geometry, precision=-1)
            try:
                field_window = geometry_window(reference, [shape_geometry], boundless=False)
            except Exception:
                raise ValueError("This scene tile does not overlap the saved field polygon.")
            sources, views = {}, {}
            try:
                for name in REFLECTANCE_BANDS + ("SCL",):
                    src = rasterio.open(paths[name])
                    sources[name] = src
                    if name == "B2" or (src.crs == reference.crs and src.transform == reference.transform and
                                         src.width == reference.width and src.height == reference.height):
                        views[name] = src
                    else:
                        views[name] = WarpedVRT(src, crs=reference.crs, transform=reference.transform,
                            width=reference.width, height=reference.height,
                            resampling=Resampling.nearest if name == "SCL" else Resampling.bilinear,
                            src_nodata=src.nodata, nodata=0)
                names = REFLECTANCE_BANDS + ("NDVI", "NDWI", "EVI")
                bin_edges = {name: (np.linspace(-10, 10, 4097) if name == "EVI" else
                    np.linspace(-1.5, 1.5, 3073) if name in ("NDVI", "NDWI") else np.linspace(-1.0, 2.0, 3073))
                    for name in names}
                accum = {name: {"count": 0, "sum": 0.0, "min": float("inf"), "max": float("-inf"),
                    "histogram": np.zeros(len(bin_edges[name]) - 1, dtype=np.int64)} for name in names}
                counters = {"inside_pixel_count": 0, "cloud_pixel_count": 0, "valid_pixel_count": 0}
                col0, row0 = int(field_window.col_off), int(field_window.row_off)
                width, height = int(field_window.width), int(field_window.height)
                for row in range(row0, row0 + height, chunk_size):
                    for col in range(col0, col0 + width, chunk_size):
                        win = rasterio.windows.Window(col, row, min(chunk_size, col0 + width - col), min(chunk_size, row0 + height - row))
                        inside = geometry_mask([shape_geometry], out_shape=(int(win.height), int(win.width)),
                            transform=window_transform(win, reference.transform), invert=True, all_touched=False)
                        if not inside.any():
                            continue
                        values, masks = {}, {}
                        for name in REFLECTANCE_BANDS:
                            src = views[name]
                            values[name] = src.read(1, window=win)
                            masks[name] = src.read_masks(1, window=win) > 0
                            if src.nodata is not None:
                                masks[name] &= values[name] != src.nodata
                        scl = views["SCL"].read(1, window=win)
                        reflectance, indices, valid, cloud, region_valid = calculate_window(
                            values, scl, inside, masks, offsets, quant)
                        counters["inside_pixel_count"] += int(inside.sum())
                        counters["cloud_pixel_count"] += int(cloud.sum())
                        counters["valid_pixel_count"] += int(valid.sum())
                        for name in names:
                            values_for_band = (reflectance[name] if name in REFLECTANCE_BANDS else indices[name])[valid]
                            if not values_for_band.size:
                                continue
                            stat = accum[name]
                            stat["count"] += int(values_for_band.size)
                            stat["sum"] += float(np.sum(values_for_band, dtype=np.float64))
                            stat["min"] = min(stat["min"], float(np.min(values_for_band)))
                            stat["max"] = max(stat["max"], float(np.max(values_for_band)))
                            clipped = np.clip(values_for_band, bin_edges[name][0], bin_edges[name][-1])
                            stat["histogram"] += np.histogram(clipped, bins=bin_edges[name])[0]
                if not counters["valid_pixel_count"]:
                    raise ValueError("No clear, non-nodata pixels remain inside this field for the scene.")
                bands = {}
                for name, stat in accum.items():
                    bins, count = bin_edges[name], stat["count"]
                    median_bin = int(np.searchsorted(np.cumsum(stat["histogram"]), (count + 1) // 2))
                    bands[name] = {"mean": stat["sum"] / count, "median": float((bins[median_bin] + bins[median_bin + 1]) / 2),
                        "min": stat["min"], "max": stat["max"], "valid_pixel_count": count, "nodata_pixel_count": 0,
                        "median_approximation": "histogram midpoint"}
                n = counters["valid_pixel_count"]
                return {"bands": bands, "mean_ndvi": bands["NDVI"]["mean"], "median_ndvi": bands["NDVI"]["median"],
                    "min_ndvi": bands["NDVI"]["min"], "max_ndvi": bands["NDVI"]["max"],
                    "mean_ndwi": bands["NDWI"]["mean"], "mean_evi": bands["EVI"]["mean"],
                    "valid_pixel_count": n, "inside_pixel_count": counters["inside_pixel_count"],
                    "cloud_contaminated_pixel_percentage": round(100*counters["cloud_pixel_count"] /
                        max(counters["inside_pixel_count"], 1), 3),
                    "processing_status": "processed", "reflectance_scale": "(DN + BOA_ADD_OFFSET) / BOA_QUANTIFICATION_VALUE",
                    "pixel_resolution_m": 10, "scene_id": item.get("id"),
                    "acquisition_date": (item.get("properties", {}).get("datetime") or "")[:10]}
            finally:
                for name, view in views.items():
                    if view is not sources.get(name):
                        view.close()
                for src in sources.values():
                    src.close()

