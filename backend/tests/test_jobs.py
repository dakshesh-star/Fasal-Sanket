from pathlib import Path

from app import database
from app import satellite_pipeline
import pytest


def _workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", Path(tmp_path) / "jobs.sqlite3")
    user = database.create_user("jobs@example.test", "hash", "Job Test")
    field = database.save_field(user["id"], {"name": "Test field", "latitude": 28.6,
        "longitude": 77.2, "boundary": {"type": "Polygon", "coordinates": [[[77.2, 28.6],
        [77.201, 28.6], [77.201, 28.601], [77.2, 28.6]]]}})
    return user, field


def test_job_is_durable_and_duplicate_active_request_is_idempotent(tmp_path, monkeypatch):
    user, field = _workspace(tmp_path, monkeypatch)
    payload = {"start_date": "2026-01-01", "end_date": "2026-04-01", "max_cloud_cover": 30}
    first, created = database.create_job(user["id"], field["id"], "observations", payload)
    duplicate, duplicate_created = database.create_job(user["id"], field["id"], "observations", payload)
    assert created is True
    assert duplicate_created is False
    assert duplicate["id"] == first["id"]
    assert len(database.pending_job_dispatches()) == 1
    assert database.get_job(user["id"], field["id"], first["id"])["request"] == {
        "kind": "observations", **payload}


def test_job_claim_cancel_and_persisted_error_result(tmp_path, monkeypatch):
    user, field = _workspace(tmp_path, monkeypatch)
    job, _ = database.create_job(user["id"], field["id"], "identification", {"model_type": "random_forest"})
    claimed = database.claim_job(job["id"], "test-worker")
    assert claimed["status"] == "discovering_scenes"
    assert claimed["attempts"] == 1
    assert database.claim_job(job["id"], "second-worker") is None
    updated = database.update_job(job["id"], status="failed", current_stage="failed",
        error={"message": "missing CDSE credentials"}, result={"scene_counts": {"discovered": 3}})
    assert updated["error"] == {"message": "missing CDSE credentials"}
    assert updated["result"] == {"scene_counts": {"discovered": 3}}
    cancelled, changed = database.cancel_job(user["id"], field["id"], job["id"])
    assert changed is False  # terminal jobs cannot be changed by a late cancel
    assert cancelled["status"] == "failed"


def test_job_status_and_cancel_are_scoped_to_owner_and_field(tmp_path, monkeypatch):
    owner, field = _workspace(tmp_path, monkeypatch)
    other = database.create_user("other@example.test", "hash", "Other")
    job, _ = database.create_job(owner["id"], field["id"], "observations", {"start_date": "2026-01-01"})
    assert database.get_job(other["id"], field["id"], job["id"]) is None
    assert database.cancel_job(other["id"], field["id"], job["id"])[0] is None
    assert database.cancel_job(owner["id"], field["id"], job["id"])[1] is True
    assert database.job_cancelled(job["id"]) is True


def test_job_prediction_is_idempotent_and_owner_field_bound(tmp_path, monkeypatch):
    owner, field = _workspace(tmp_path, monkeypatch)
    job, _ = database.create_job(owner["id"], field["id"], "identification", {"model_type": "random_forest"})
    first = database.save_prediction(owner["id"], field["id"], "random_forest", {"candidate": "Wheat"}, job_id=job["id"])
    replay = database.save_prediction(owner["id"], field["id"], "random_forest", {"candidate": "Rice"}, job_id=job["id"])
    assert first["id"] == replay["id"]
    assert replay["result"] == {"candidate": "Rice"}
    assert len(database.list_predictions(owner["id"], field["id"])) == 1
    other = database.create_user("third@example.test", "hash", "Third")
    assert database.save_prediction(other["id"], field["id"], "random_forest", {}, job_id=job["id"]) is None


def _scene(scene_id, acquired="2026-02-01", cloud=5):
    assets = {key: {"href": f"s3://eodata/{key}.jp2"} for key in satellite_pipeline.ASSET_KEYS.values()}
    assets["product_metadata"] = {"href": "s3://eodata/product.xml"}
    return {"id": scene_id, "properties": {"datetime": acquired + "T05:00:00Z", "eo:cloud_cover": cloud}, "assets": assets}


def _tile(scene_id, ndvi):
    names = ("B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDWI", "EVI")
    bands = {name: {"mean": ndvi if name == "NDVI" else 0.2, "median": ndvi if name == "NDVI" else 0.2,
        "min": 0.1, "max": 0.8, "valid_pixel_count": 25, "nodata_pixel_count": 0} for name in names}
    return {"scene_id": scene_id, "bands": bands, "valid_pixel_count": 25, "inside_pixel_count": 30,
        "cloud_contaminated_pixel_percentage": 4.0, "reflectance_scale": "metadata offset / quantification value"}


def test_pipeline_reports_credentials_block_without_fabricating_observations(tmp_path, monkeypatch):
    user, field = _workspace(tmp_path, monkeypatch)
    monkeypatch.setattr(satellite_pipeline, "search_sentinel2_l2a", lambda *a, **k: [_scene("scene-a")])
    monkeypatch.setattr(satellite_pipeline, "s3_credentials_configured", lambda: False)
    with pytest.raises(satellite_pipeline.PipelineError, match="credentials"):
        satellite_pipeline.process_observations(user["id"], field["id"], "2026-01-01", "2026-03-01")
    acquisition = database.list_satellite_acquisitions(user["id"], field["id"])[0]
    assert acquisition["status"] == "credentials_required"
    assert database.list_observations(user["id"], field["id"]) == []
    assert database.get_field(user["id"], field["id"])["scene_metadata_count"] == 1


def test_pipeline_persists_actual_processor_statistics_idempotently(tmp_path, monkeypatch):
    user, field = _workspace(tmp_path, monkeypatch)
    scenes = [_scene("scene-b"), _scene("scene-a")]
    monkeypatch.setattr(satellite_pipeline, "search_sentinel2_l2a", lambda *a, **k: scenes)
    monkeypatch.setattr(satellite_pipeline, "s3_credentials_configured", lambda: True)
    def fake_process(item, geometry, on_assets_retrieved=None):
        if on_assets_retrieved:
            on_assets_retrieved(["B02_10m", "B03_10m"])
        return _tile(item["id"], .4 if item["id"] == "scene-a" else .6)
    monkeypatch.setattr(satellite_pipeline, "process_stac_item", fake_process)
    first = satellite_pipeline.process_observations(user["id"], field["id"], "2026-01-01", "2026-03-01")
    again = satellite_pipeline.process_observations(user["id"], field["id"], "2026-01-01", "2026-03-01")
    observations = database.list_observations(user["id"], field["id"])
    assert first["observation_count"] == 1 and again["observation_count"] == 1
    assert len(observations) == 1
    assert observations[0]["stats"]["mean_ndvi"] == pytest.approx(.5)
    assert observations[0]["scene_id"] == "scene-a|scene-b"
    assert {row["status"] for row in database.list_satellite_acquisitions(user["id"], field["id"])} == {"stored"}
