"""Authenticated Sentinel-2 L2A polygon statistics via CDSE Statistical API."""
import os
import requests

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
STATS_URL = "https://sh.dataspace.copernicus.eu/statistics/v1"

EVALSCRIPT = '''//VERSION=3
function setup() {
  return { input: [{ bands: ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "dataMask"] }],
    output: [
      { id: "reflectance", bands: ["B2", "B3", "B4", "B8", "B11", "B12"], sampleType: "FLOAT32" },
      { id: "indices", bands: ["NDVI", "NDWI", "EVI"], sampleType: "FLOAT32" },
      { id: "scl", bands: 1, sampleType: "UINT8" },
      { id: "dataMask", bands: ["reflectance", "indices"] }
    ] };
}
function evaluatePixel(s) {
  const clear = s.SCL !== 3 && s.SCL !== 8 && s.SCL !== 9 && s.SCL !== 10 && s.SCL !== 11;
  const den = s.B08 + s.B04;
  const wden = s.B03 + s.B08;
  const eden = s.B08 + 6*s.B04 - 7.5*s.B02 + 1;
  const valid = s.dataMask && clear && den !== 0 && wden !== 0 && eden !== 0;
  return { reflectance: [s.B02, s.B03, s.B04, s.B08, s.B11, s.B12],
    indices: [valid ? (s.B08-s.B04)/den : 0,
      valid ? (s.B03-s.B08)/wden : 0,
      valid ? 2.5*(s.B08-s.B04)/eden : 0],
    scl: [s.SCL], dataMask: [s.dataMask && clear, s.dataMask && clear && den !== 0 && wden !== 0 && eden !== 0] };
}
'''


class CDSECredentialsMissing(Exception):
    pass


def _geometry(boundary):
    if not boundary:
        return None
    kind = boundary.get("type")
    if kind == "Feature":
        return boundary.get("geometry")
    if kind == "FeatureCollection":
        features = boundary.get("features", [])
        return features[0].get("geometry") if len(features) == 1 else None
    if kind in {"Polygon", "MultiPolygon"}:
        return boundary
    return None


def retrieve_polygon_statistics(boundary, start_date: str, end_date: str):
    """Return daily polygon stats; CDSE outputs are scaled surface reflectance (0..1)."""
    client_id = os.getenv("CDSE_OAUTH_CLIENT_ID")
    client_secret = os.getenv("CDSE_OAUTH_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise CDSECredentialsMissing("Configure CDSE_OAUTH_CLIENT_ID and CDSE_OAUTH_CLIENT_SECRET to retrieve pixels.")
    geometry = _geometry(boundary)
    if not geometry:
        raise ValueError("A single Polygon or MultiPolygon boundary is required for field-level processing.")
    token_response = requests.post(TOKEN_URL, data={
        "grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret,
    }, timeout=(5, 20))
    token_response.raise_for_status()
    token = token_response.json().get("access_token")
    if not token:
        raise RuntimeError("CDSE did not return an OAuth access token.")
    payload = {
        "input": {"bounds": {"geometry": geometry,
            "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"mosaickingOrder": "leastCC"}}]},
        "aggregation": {"timeRange": {"from": f"{start_date}T00:00:00Z", "to": f"{end_date}T23:59:59Z"},
            "aggregationInterval": {"of": "P1D"}, "evalscript": EVALSCRIPT, "resx": 10, "resy": 10},
        "calculations": {"indices": {"statistics": {"default": {"percentiles": {"k": [50]}}}},
            "scl": {"histograms": {"default": {"bins": list(range(13))}}}},
    }
    response = requests.post(STATS_URL, headers={"Authorization": f"Bearer {token}",
        "Content-Type": "application/json", "Accept": "application/json"}, json=payload, timeout=(10, 120))
    response.raise_for_status()
    results = []
    for item in response.json().get("data", []):
        outputs = item.get("outputs", {})
        reflectance = outputs.get("reflectance", {}).get("bands", {})
        indices = outputs.get("indices", {}).get("bands", {})
        bands = {}
        for name in ["B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "EVI"]:
            stats = (reflectance if name.startswith("B") else indices).get(name, {}).get("stats", {})
            bands[name] = {"mean": stats.get("mean"), "median": stats.get("percentiles", {}).get("50.0"),
                "min": stats.get("min"), "max": stats.get("max"), "valid_pixel_count": stats.get("sampleCount", 0),
                "nodata_pixel_count": stats.get("noDataCount", 0)}
        ndvi = bands["NDVI"]
        if ndvi["valid_pixel_count"]:
            interval = item.get("interval", {})
            scl_bands = outputs.get("scl", {}).get("bands", {})
            scl_histogram = next(iter(scl_bands.values()), {}).get("histogram", {}).get("bins", [])
            scl_counts = {int(bin_["lowEdge"]): int(bin_["count"]) for bin_ in scl_histogram if "lowEdge" in bin_}
            total_scl = sum(scl_counts.values())
            cloud_pixels = sum(scl_counts.get(code, 0) for code in (3, 8, 9, 10))
            results.append({"observed_at": interval.get("from", "")[:10], "bands": bands,
                "mean_ndvi": ndvi["mean"], "median_ndvi": ndvi["median"], "min_ndvi": ndvi["min"],
                "max_ndvi": ndvi["max"], "mean_ndwi": bands["NDWI"]["mean"], "mean_evi": bands["EVI"]["mean"],
                "valid_pixel_count": ndvi["valid_pixel_count"],
                "cloud_contaminated_pixel_percentage": round(100 * cloud_pixels / total_scl, 3) if total_scl else None,
                "cloud_percentage_note": None if total_scl else "SCL histogram did not contain valid pixels."})
    return results
