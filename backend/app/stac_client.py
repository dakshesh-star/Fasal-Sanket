"""Small, paginated client for the public Copernicus Data Space STAC catalogue."""
import os
import requests

STAC_URL = "https://stac.dataspace.copernicus.eu/v1/search"


def search_sentinel2_l2a(geometry: dict, start_date: str, end_date: str, max_items: int = 1000) -> list[dict]:
    endpoint = os.getenv("CDSE_STAC_URL", STAC_URL)
    page_size = min(100, max(1, max_items))
    payload = {"collections": [os.getenv("CDSE_STAC_COLLECTION", "sentinel-2-l2a")],
        "intersects": geometry,
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": page_size, "query": {"eo:cloud_cover": {"lte": 100}}}
    response = requests.post(endpoint, json=payload, timeout=(5, 45))
    response.raise_for_status()
    page = response.json()
    items, seen = [], set()
    while True:
        for item in page.get("features", []):
            if item.get("id") not in seen:
                seen.add(item.get("id"))
                items.append(item)
                if len(items) >= max_items:
                    return items
        next_link = next((link for link in page.get("links", []) if link.get("rel") == "next"), None)
        if not next_link:
            break
        href = next_link.get("href", endpoint)
        method = next_link.get("method", "GET").upper()
        if method == "POST":
            response = requests.post(href, json=next_link.get("body", payload),
                headers=next_link.get("headers", {}), timeout=(5, 45))
        else:
            response = requests.get(href, timeout=(5, 45))
        response.raise_for_status()
        page = response.json()
    return items
