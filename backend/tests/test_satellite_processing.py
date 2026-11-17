from app import satellite_processing as processing


def test_cdse_statistics_payload_and_response_parsing(monkeypatch):
    monkeypatch.setenv("CDSE_OAUTH_CLIENT_ID", "test-client")
    monkeypatch.setenv("CDSE_OAUTH_CLIENT_SECRET", "test-secret")

    class Response:
        def __init__(self, value):
            self.value = value

        def raise_for_status(self):
            return None

        def json(self):
            return self.value

    reflectance = {name: {"stats": {"mean": .2, "min": .1, "max": .3, "sampleCount": 50, "noDataCount": 0}}
                   for name in ["B2", "B3", "B4", "B8", "B11", "B12"]}
    indices = {name: {"stats": {"mean": value, "min": value-.1, "max": value+.1,
                                 "sampleCount": 40, "noDataCount": 10, "percentiles": {"50.0": value}}}
               for name, value in [("NDVI", .6), ("NDWI", -.2), ("EVI", .4)]}
    response_json = {"data": [{"interval": {"from": "2026-02-01T00:00:00Z", "to": "2026-02-02T00:00:00Z"},
        "outputs": {"reflectance": {"bands": reflectance}, "indices": {"bands": indices},
                    "scl": {"bands": {"B0": {"histogram": {"bins": [
                        {"lowEdge": 3, "count": 2}, {"lowEdge": 4, "count": 6}, {"lowEdge": 8, "count": 2}]}}}}}}]}
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return Response({"access_token": "mock-token"}) if url == processing.TOKEN_URL else Response(response_json)

    monkeypatch.setattr(processing.requests, "post", fake_post)
    boundary = {"type": "Polygon", "coordinates": [[[77, 28], [77.1, 28], [77.1, 28.1], [77, 28.1], [77, 28]]]}
    rows = processing.retrieve_polygon_statistics(boundary, "2026-02-01", "2026-02-28")
    assert len(calls) == 2
    payload = calls[1][1]["json"]
    assert payload["input"]["bounds"]["geometry"] == boundary
    assert payload["aggregation"]["resx"] == 10
    assert "2.5*(s.B08-s.B04)/eden" in payload["aggregation"]["evalscript"]
    assert rows[0]["mean_ndvi"] == .6 and rows[0]["median_ndvi"] == .6
    assert rows[0]["bands"]["B12"]["mean"] == .2
    assert rows[0]["cloud_contaminated_pixel_percentage"] == 40.0
