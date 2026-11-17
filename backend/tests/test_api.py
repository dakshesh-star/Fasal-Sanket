"""
Integration tests for FastAPI backend endpoints
"""

import pytest
import sys
import os
import uuid
from pathlib import Path
from fastapi.testclient import TestClient

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_loaded" in data


def test_crops_endpoint(client):
    response = client.get("/crops")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 5
    crop_names = [c["name"] for c in data["crops"]]
    assert "Wheat" in crop_names
    assert "Rice" in crop_names
    assert "Maize" in crop_names
    assert "Mustard" in crop_names
    assert "Cotton" in crop_names


def test_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) >= 2


def test_calendar_endpoint(client):
    response = client.get("/calendar?target_error_rate=0.10")
    assert response.status_code == 200
    data = response.json()
    assert len(data["stages"]) == 5
    assert data["target_error_rate"] == 0.10


def test_sample_field_endpoint(client):
    response = client.get("/sample-field/wheat")
    assert response.status_code == 200
    data = response.json()
    assert data["crop"] == "Wheat"
    assert len(data["observations"]) == 20
    assert len(data["observations"][0]) == 9


def test_predict_endpoint_flow(client):
    sample_resp = client.get("/sample-field/mustard")
    assert sample_resp.status_code == 200
    obs = sample_resp.json()["observations"]

    predict_payload = {
        "model_type": "random_forest",
        "stage_fraction": 0.8,
        "target_error_rate": 0.10,
        "observations": obs
    }
    pred_resp = client.post("/predict", json=predict_payload)
    assert pred_resp.status_code == 200
    res = pred_resp.json()
    assert "predicted_crop" in res
    assert "taxonomic_resolution" in res
    assert res["predicted_crop"] in ["Mustard", "Wheat", "Rice", "Maize", "Cotton"]


def test_predict_progressive_endpoint_flow(client):
    sample_resp = client.get("/sample-field/cotton")
    assert sample_resp.status_code == 200
    obs = sample_resp.json()["observations"]

    prog_payload = {
        "model_type": "random_forest",
        "target_error_rate": 0.10,
        "observations": obs
    }
    prog_resp = client.post("/predict/progressive", json=prog_payload)
    assert prog_resp.status_code == 200
    res = prog_resp.json()
    assert len(res["stages"]) == 5
    assert res["final_predicted_crop"] in ["Cotton", "Wheat", "Rice", "Maize", "Mustard"]


def test_satellite_scene_search_returns_catalog_metadata(client, monkeypatch):
    class CatalogResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"features": [{
                "id": "scene-1", "collection": "sentinel-2-l2a",
                "properties": {"datetime": "2026-10-01T10:30:00Z", "eo:cloud_cover": 8.5},
                "geometry": {"type": "Point", "coordinates": [77.2, 28.6]},
                "assets": {"B04": {"href": "https://example.test/B04.tif", "type": "image/tiff"}},
            }]}

    monkeypatch.setattr("app.main.requests.post", lambda *args, **kwargs: CatalogResponse())
    response = client.get(
        "/satellite/scenes?latitude=28.6&longitude=77.2&start_date=2026-09-01&end_date=2026-10-31&max_cloud_cover=20"
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert data["is_real_scene_metadata"] is True
    assert data["reflectance_processed"] is False
    assert data["scenes"][0]["cloud_cover"] == 8.5
    assert data["scenes"][0]["assets"]["B04"]["href"].endswith("B04.tif")


def test_satellite_scene_search_rejects_reversed_dates(client):
    response = client.get(
        "/satellite/scenes?latitude=28.6&longitude=77.2&start_date=2026-10-31&end_date=2026-09-01"
    )
    assert response.status_code == 422


def test_registration_refresh_and_private_field_access(client, monkeypatch, tmp_path):
    from app import database
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "auth-test.sqlite3")
    email = f"researcher-{uuid.uuid4().hex}@example.org"
    registered = client.post("/auth/register", json={"name": "Field Researcher", "email": email, "password": "Research123"})
    assert registered.status_code == 201
    tokens = registered.json()
    assert "access_token" in tokens and "refresh_token" not in tokens
    assert "fasal_refresh" in client.cookies
    assert client.get("/fields").status_code == 401

    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    me = client.get("/auth/me", headers=headers).json()
    assert me["email"] == email and me["display_name"] == "Field Researcher"
    created = client.post("/fields", headers=headers, json={
        "name": "North plot", "latitude": 28.6, "longitude": 77.2,
        "crop": "Wheat", "boundary": {"type": "Polygon", "coordinates": [[[77.1,28.5],[77.2,28.5],[77.2,28.6],[77.1,28.5]]]},
    })
    assert created.status_code == 201
    field_id = created.json()["id"]
    assert client.get("/fields", headers=headers).json()["count"] == 1

    logged_in = client.post("/auth/login", json={"email": email, "password": "Research123", "remember_session": False})
    assert logged_in.status_code == 200
    assert logged_in.json()["user"]["email"] == email
    assert "max-age=" not in logged_in.headers["set-cookie"].lower()

    refreshed = client.post("/auth/refresh")
    assert refreshed.status_code == 200
    assert refreshed.json()["access_token"] != tokens["access_token"]
    assert client.post("/auth/logout", headers=headers).status_code == 204
    assert client.post("/auth/refresh").status_code == 401
    other = client.post("/auth/register", json={"email": f"other-{uuid.uuid4().hex}@example.org", "password": "Research456"})
    other_headers = {"Authorization": f"Bearer {other.json()['access_token']}"}
    assert client.get(f"/fields/{field_id}", headers=other_headers).status_code == 404


def test_login_rejects_incorrect_password(client, monkeypatch, tmp_path):
    from app import database
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "login-test.sqlite3")
    email = f"login-{uuid.uuid4().hex}@example.org"
    assert client.post("/auth/register", json={"email": email, "password": "Research123"}).status_code == 201
    response = client.post("/auth/login", json={"email": email, "password": "WrongPass456"})
    assert response.status_code == 401


def test_field_validation_and_persistence_across_database_reinitialization(client, monkeypatch, tmp_path):
    from app import database
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "persistent-test.sqlite3")
    email = f"persist-{uuid.uuid4().hex}@example.org"
    token = client.post("/auth/register", json={"email": email, "password": "Research123"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    bad = client.post("/fields", headers=headers, json={"name": "  ", "latitude": 300, "longitude": 0})
    assert bad.status_code == 422
    bad_ring = client.post("/fields", headers=headers, json={"name": "Bad polygon", "latitude": 1, "longitude": 1,
        "boundary": {"type": "Polygon", "coordinates": [[[0,0],[1,1],[2,2]]]}})
    assert bad_ring.status_code == 422
    created = client.post("/fields", headers=headers, json={"name": "Persist me", "latitude": 28.6, "longitude": 77.2,
        "planting_date": "2026-01-01", "harvest_date": "2026-05-01",
        "boundary": {"type": "Polygon", "coordinates": [[[77.1,28.5],[77.2,28.5],[77.2,28.6],[77.1,28.5],[77.1,28.5]]]}})
    assert created.status_code == 201
    field_id = created.json()["id"]
    database.initialize_database()  # mirrors startup after a backend restart
    assert client.get(f"/fields/{field_id}", headers=headers).json()["name"] == "Persist me"
    updated = client.put(f"/fields/{field_id}", headers=headers, json={"name": "Edited", "latitude": 28.6, "longitude": 77.2})
    assert updated.status_code == 200 and updated.json()["name"] == "Edited"
    assert client.delete(f"/fields/{field_id}", headers=headers).status_code == 204
    assert client.get(f"/fields/{field_id}", headers=headers).status_code == 404


def test_field_processing_requires_polygon_and_credentials(client, monkeypatch, tmp_path):
    from app import database
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "process-test.sqlite3")
    monkeypatch.delenv("CDSE_S3_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("CDSE_S3_SECRET_ACCESS_KEY", raising=False)
    email = f"process-{uuid.uuid4().hex}@example.org"
    token = client.post("/auth/register", json={"email": email, "password": "Research123"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    point = client.post("/fields", headers=headers, json={"name": "Point only", "latitude": 28.6, "longitude": 77.2}).json()
    no_polygon = client.post(f"/fields/{point['id']}/observations/process?start_date=2026-01-01&end_date=2026-01-10", headers=headers)
    assert no_polygon.status_code == 422
    polygon = client.post("/fields", headers=headers, json={"name": "Boundary", "latitude": 28.6, "longitude": 77.2,
        "boundary": {"type": "Polygon", "coordinates": [[[77.1,28.5],[77.2,28.5],[77.2,28.6],[77.1,28.5],[77.1,28.5]]]}}).json()
    monkeypatch.setattr("app.main.search_sentinel2_l2a", lambda *args, **kwargs: [])
    queued = client.post(f"/fields/{polygon['id']}/observations/process?start_date=2026-01-01&end_date=2026-01-10", headers=headers)
    assert queued.status_code == 202
    assert queued.json()["job"]["status"] == "queued"
    assert queued.json()["job"]["kind"] == "observations"
    assert client.get(queued.json()["status_url"], headers=headers).json()["job"]["id"] == queued.json()["job"]["id"]


def test_authenticated_polygon_processing_is_durable_and_returns_job(client, monkeypatch, tmp_path):
    from app import database
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "observations-test.sqlite3")
    email = f"observations-{uuid.uuid4().hex}@example.org"
    token = client.post("/auth/register", json={"email": email, "password": "Research123"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    field = client.post("/fields", headers=headers, json={"name": "Measured polygon", "latitude": 28.6, "longitude": 77.2,
        "boundary": {"type": "Polygon", "coordinates": [[[77.1,28.5],[77.2,28.5],[77.2,28.6],[77.1,28.6],[77.1,28.5]]]}}).json()
    response = client.post(f"/fields/{field['id']}/observations/process?start_date=2026-01-01&end_date=2026-02-28", headers=headers)
    assert response.status_code == 202
    job = response.json()["job"]
    assert job["status"] == "queued" and job["field_id"] == field["id"] and job["user_id"] == field["user_id"]
    assert client.get(response.json()["status_url"], headers=headers).json()["job"]["id"] == job["id"]
    assert client.get(f"/fields/{field['id']}/observations", headers=headers).json()["count"] == 0
    summary = client.get("/workspace/summary", headers=headers).json()
    assert summary["field_count"] == 1 and summary["observation_count"] == 0


def test_polygon_first_field_defaults_crop_unknown_and_calculates_area(client, monkeypatch, tmp_path):
    from app import database
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "polygon-first.sqlite3")
    email = f"polygon-{uuid.uuid4().hex}@example.org"
    token = client.post("/auth/register", json={"email": email, "password": "Research123"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    created = client.post("/fields", headers=headers, json={"name": "Unknown field", "boundary": {
        "type": "Polygon", "coordinates": [[[77,28],[77.01,28],[77.01,28.01],[77,28.01],[77,28]]]}})
    assert created.status_code == 201
    field = created.json()
    assert field["crop"] == "Unknown / Not identified"
    assert field["latitude"] is not None and field["longitude"] is not None
    assert field["area_hectares"] > 0
    bowtie = client.post("/fields", headers=headers, json={"name": "Crossing ring", "boundary": {
        "type": "Polygon", "coordinates": [[[77,28],[77.02,28.02],[77,28.02],[77.02,28],[77,28]]]}})
    assert bowtie.status_code == 422


def test_polygon_scene_discovery_uses_intersects_and_reports_metadata_only(client, monkeypatch, tmp_path):
    from app import database
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "intersects.sqlite3")
    email = f"stac-field-{uuid.uuid4().hex}@example.org"
    token = client.post("/auth/register", json={"email": email, "password": "Research123"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    field = client.post("/fields", headers=headers, json={"name": "Polygon", "boundary": {
        "type": "Polygon", "coordinates": [[[77,28],[77.01,28],[77.01,28.01],[77,28.01],[77,28]]]}}).json()
    class CatalogResponse:
        def raise_for_status(self): return None
        def json(self):
            return {"features": [{"id": "S2-test", "collection": "sentinel-2-l2a",
                "properties": {"datetime": "2026-02-01T05:00:00Z", "eo:cloud_cover": 5}, "assets": {}}]}
    calls=[]
    def fake_post(url, **kwargs): calls.append(kwargs["json"]); return CatalogResponse()
    monkeypatch.setattr("app.main.requests.post", fake_post)
    result = client.get(f"/fields/{field['id']}/satellite/scenes?start_date=2026-01-01&end_date=2026-02-28", headers=headers)
    assert result.status_code == 200
    assert calls[0]["intersects"] == field["boundary"]
    assert result.json()["count"] == 1
    assert result.json()["workflow_status"] == {"metadata_retrieved": True, "assets_downloaded": False,
        "reflectance_processed": False, "field_statistics_calculated": False}


def test_identification_requires_real_observations_without_guessing(client, monkeypatch, tmp_path):
    from app import database
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "identify-empty.sqlite3")
    email = f"identify-{uuid.uuid4().hex}@example.org"
    token = client.post("/auth/register", json={"email": email, "password": "Research123"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    field = client.post("/fields", headers=headers, json={"name": "Unobserved", "boundary": {
        "type": "Polygon", "coordinates": [[[77,28],[77.01,28],[77.01,28.01],[77,28.01],[77,28]]]}}).json()
    result = client.post(f"/fields/{field['id']}/identify", headers=headers)
    assert result.status_code == 422
    assert result.json()["detail"]["usable_observations"] == 0
    assert "No crop candidate was generated" in result.json()["detail"]["next_step"]


def test_one_observation_is_model_compatible_but_reports_imputed_season(client, monkeypatch, tmp_path):
    from app import database
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "identify-one-date.sqlite3")
    email = f"identify-one-{uuid.uuid4().hex}@example.org"
    token = client.post("/auth/register", json={"email": email, "password": "Research123"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    field = client.post("/fields", headers=headers, json={"name": "One date", "boundary": {
        "type": "Polygon", "coordinates": [[[77,28],[77.01,28],[77.01,28.01],[77,28.01],[77,28]]]}}).json()
    bands = {name: {"mean": val, "median": val, "min": val, "max": val, "valid_pixel_count": 25, "nodata_pixel_count": 0}
        for name,val in [("B2",.1),("B3",.2),("B4",.15),("B8",.4),("B11",.22),("B12",.2),
                         ("NDVI",.4545),("NDWI",-.3333),("EVI",.48)]}
    database.save_observation(field["user_id"], field["id"], "2026-01-01", {"bands": bands, "valid_pixel_count": 25})
    result = client.post(f"/fields/{field['id']}/identify?model_type=random_forest", headers=headers)
    assert result.status_code == 200
    stage = result.json()["report"]["checkpoints"][-1]
    assert stage["status"] == "experimental_candidate"
    assert stage["observed_temporal_bins"] == 1
    assert stage["imputed_temporal_bins"] == 19
    assert "imputed" in stage["explanation"]
    assert stage["supported_taxonomic_level"] == 0


def test_real_observation_candidates_are_experimental_and_do_not_unlock_taxonomy(client, monkeypatch, tmp_path):
    from app import database
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "identify-real-sequence.sqlite3")
    email = f"identify-sequence-{uuid.uuid4().hex}@example.org"
    token = client.post("/auth/register", json={"email": email, "password": "Research123"}).json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    field = client.post("/fields", headers=headers, json={"name": "Observed field", "planting_date": "2026-01-01",
        "harvest_date": "2026-02-01", "boundary": {"type": "Polygon", "coordinates": [[[77,28],[77.01,28],[77.01,28.01],[77,28.01],[77,28]]]}}).json()
    bands={name:{"mean":val,"median":val,"min":val,"max":val,"valid_pixel_count":50,"nodata_pixel_count":0}
        for name,val in [("B2",.1),("B3",.2),("B4",.15),("B8",.4),("B11",.22),("B12",.2),("NDVI",.4545),("NDWI",-.3333),("EVI",.48)]}
    for i, day in enumerate(["2026-01-01","2026-01-15","2026-02-01"]):
        database.save_observation(field["user_id"], field["id"], day, {"bands":bands,"mean_ndvi":.45+i*.05,
            "median_ndvi":.45+i*.05,"min_ndvi":.3,"max_ndvi":.7,"mean_ndwi":-.33,"mean_evi":.48,
            "valid_pixel_count":50,"cloud_contaminated_pixel_percentage":2.0})
    result=client.post(f"/fields/{field['id']}/identify?model_type=random_forest&target_error_rate=0.1",headers=headers)
    assert result.status_code == 200
    report=result.json()["report"]
    assert report["identification"]["status"] == "experimental_candidate"
    assert report["identification"]["identified_crop"] is None
    assert report["identification"]["user_display_label"] == "Unknown / Not identified"
    assert report["identification"]["model_training_provenance"] == "synthetic training dataset"
    assert report["identification"]["empirical_reliability"] is None
    assert all(s["supported_taxonomic_level"] == 0 for s in report["checkpoints"] if s["candidate_crop"])
    assert len(report["checkpoints"]) == 5
    assert [s["observation_count"] for s in report["checkpoints"]] == [1, 1, 2, 2, 3]
    history=client.get(f"/fields/{field['id']}/predictions",headers=headers)
    assert history.status_code == 200 and history.json()["count"] == 1
