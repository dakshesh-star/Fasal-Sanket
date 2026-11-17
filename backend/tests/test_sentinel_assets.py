import numpy as np

from app.sentinel_assets import assets_available, calculate_window, reflectance_from_dn


def test_l2a_reflectance_uses_boa_offset_and_keeps_zero_as_nodata():
    result = reflectance_from_dn(np.array([0, 1000, 6000]), offset=-1000, quantification_value=10000)
    assert np.isnan(result[0])
    assert result[1] == 0
    assert result[2] == .5


def test_indices_use_reflectance_with_geometry_nodata_and_scl_masks():
    shape = (2, 2)
    dn = {"B2": np.full(shape, 2000), "B3": np.full(shape, 3000), "B4": np.full(shape, 2500),
          "B8": np.full(shape, 5000), "B11": np.full(shape, 4000), "B12": np.full(shape, 3500)}
    scl = np.array([[4, 4], [8, 11]])
    inside = np.ones(shape, dtype=bool)
    masks = {name: np.ones(shape, dtype=bool) for name in dn}
    reflectance, indices, valid, clouds, _ = calculate_window(dn, scl, inside, masks,
        {name: -1000 for name in dn}, 10000)
    assert valid.sum() == 2
    assert clouds.sum() == 1
    assert np.isclose(indices["NDVI"][0, 0], (0.4 - 0.15) / (0.4 + 0.15))
    assert np.isclose(indices["NDWI"][0, 0], (0.2 - 0.4) / (0.2 + 0.4))
    assert np.isclose(indices["EVI"][0, 0], 2.5*(0.4-.15)/(0.4+6*.15-7.5*.1+1))
    assert np.isclose(reflectance["B2"][0, 0], .1)


def test_scene_downloadability_requires_all_reflectance_and_scl_assets():
    item = {"assets": {key: {"href": f"s3://eodata/{key}.jp2"} for key in
        ["B02_10m", "B03_10m", "B04_10m", "B08_10m", "B11_20m", "B12_20m", "SCL_20m"]}}
    ok, missing = assets_available(item)
    assert ok and not missing
    del item["assets"]["SCL_20m"]
    ok, missing = assets_available(item)
    assert not ok and missing == ["SCL_20m"]
