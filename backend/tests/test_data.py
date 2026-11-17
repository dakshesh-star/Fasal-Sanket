"""
Tests for Synthetic Sentinel-2 Temporal Data Generation and Field Splitting
"""

import pytest
import numpy as np
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from data.generator import generate_synthetic_dataset, split_by_fields, CROPS, FEATURE_NAMES


def test_dataset_generation_shapes_and_crops():
    num_per_crop = 20
    X, y, field_ids, meta = generate_synthetic_dataset(num_fields_per_crop=num_per_crop, seed=123)

    expected_total = len(CROPS) * num_per_crop
    assert X.shape == (expected_total, 20, len(FEATURE_NAMES))
    assert len(y) == expected_total
    assert len(field_ids) == expected_total
    assert len(np.unique(y)) == 5
    assert meta["num_crops"] == 5


def test_field_splitting_prevents_leakage():
    X, y, field_ids, meta = generate_synthetic_dataset(num_fields_per_crop=30, seed=42)
    train_split, val_split, test_split = split_by_fields(X, y, field_ids, seed=42)

    X_train, y_train, f_train = train_split
    X_val, y_val, f_val = val_split
    X_test, y_test, f_test = test_split

    set_train = set(f_train)
    set_val = set(f_val)
    set_test = set(f_test)

    # Assert ZERO intersection between any splits
    assert len(set_train.intersection(set_val)) == 0, "Train and Val share fields!"
    assert len(set_train.intersection(set_test)) == 0, "Train and Test share fields!"
    assert len(set_val.intersection(set_test)) == 0, "Val and Test share fields!"


def test_spectral_ranges_and_indices():
    X, y, _, _ = generate_synthetic_dataset(num_fields_per_crop=5, seed=99)
    # NDVI (index 6) should be within [-1, 1]
    ndvi = X[:, :, 6]
    valid_ndvi = ndvi[~np.isnan(ndvi)]
    assert np.all(valid_ndvi >= -1.0) and np.all(valid_ndvi <= 1.0)
