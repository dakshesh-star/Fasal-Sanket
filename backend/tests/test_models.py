"""
Tests for Feature Preprocessing Pipeline and Progressive Truncation
"""

import pytest
import numpy as np
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.pipeline import PreprocessingPipeline


def test_pipeline_validation_and_imputation():
    pipeline = PreprocessingPipeline()

    # Create dummy observation with NaNs
    dummy = np.ones((2, 20, 9), dtype=np.float32)
    dummy[0, 5, :] = np.nan  # Cloud masked timestep

    cleaned = pipeline.impute_missing(dummy)
    assert not np.isnan(cleaned).any()
    assert cleaned.shape == (2, 20, 9)


def test_progressive_truncation_no_future_leakage():
    pipeline = PreprocessingPipeline()
    T = 20
    dummy = np.random.rand(1, T, 9).astype(np.float32)

    # Truncate at 40% (8 timesteps observed)
    truncated, steps = pipeline.truncate_sequence(dummy, fraction=0.40)
    assert steps == 8
    # Assert past observed steps match exactly
    np.testing.assert_array_equal(truncated[:, :steps, :], dummy[:, :steps, :])
    # Assert future steps (8..19) are strictly zeroed (no future leakage!)
    assert np.all(truncated[:, steps:, :] == 0.0)


def test_tabular_feature_extraction():
    pipeline = PreprocessingPipeline()
    dummy = np.random.rand(3, 20, 9).astype(np.float32)
    tab_feat = pipeline.extract_tabular_features(dummy, fraction=0.6)
    assert tab_feat.ndim == 2
    assert tab_feat.shape[0] == 3
    assert not np.isnan(tab_feat).any()
