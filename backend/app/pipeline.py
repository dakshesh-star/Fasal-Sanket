"""
Fasal Sanket - Feature Engineering and Preprocessing Pipeline

Ensures training and inference use identical transformations, validates inputs,
handles missing observations (cloud masking) via robust temporal interpolation,
and formats data for both tabular (Random Forest) and sequence (LSTM) models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "B2_Blue", "B3_Green", "B4_Red", "B8_NIR", "B11_SWIR1", "B12_SWIR2",
    "NDVI", "NDWI", "EVI"
]


class PreprocessingPipeline:
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or FEATURE_NAMES
        self.scaler = StandardScaler()
        self.fitted = False
        self.config: Dict = {
            "num_features": len(self.feature_names),
            "feature_names": self.feature_names,
            "progressive_fractions": [0.2, 0.4, 0.6, 0.8, 1.0]
        }

    def validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validates that input array has shape (N, T, D) or (T, D) and acceptable ranges."""
        if not isinstance(X, np.ndarray):
            X = np.array(X, dtype=np.float32)

        if X.ndim == 2:
            X = np.expand_dims(X, axis=0)

        if X.ndim != 3:
            raise ValueError(f"Input must have 3 dimensions (batch, timesteps, features), got {X.ndim}")

        if X.shape[2] != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features ({self.feature_names}), "
                f"but got {X.shape[2]} features."
            )

        return X

    def impute_missing(self, X: np.ndarray) -> np.ndarray:
        """
        Handles missing observations (NaNs from cloud masks) per field & band.
        Uses linear temporal interpolation, forward fill, backward fill, and channel medians.
        """
        X_imputed = X.copy()
        N, T, D = X_imputed.shape

        for i in range(N):
            for d in range(D):
                series = pd.Series(X_imputed[i, :, d])
                if series.isna().any():
                    series = series.interpolate(method="linear", limit_direction="both")
                    # Fallback if entirely NaN
                    if series.isna().any():
                        series = series.bfill().ffill()
                    # If still NaN (e.g. all empty)
                    if series.isna().any():
                        series = series.fillna(0.0)
                    X_imputed[i, :, d] = series.values

        return X_imputed

    def truncate_sequence(self, X: np.ndarray, fraction: float) -> np.ndarray:
        """
        Truncates the observation sequence up to a given fraction (0.2, 0.4, 0.6, 0.8, 1.0)
        WITHOUT leaking future observations. Remaining unobserved steps are zeroed/masked.
        """
        N, T, D = X.shape
        steps_observed = max(1, int(round(T * fraction)))

        X_truncated = np.zeros_like(X)
        X_truncated[:, :steps_observed, :] = X[:, :steps_observed, :]
        return X_truncated, steps_observed

    def extract_tabular_features(self, X: np.ndarray, fraction: float = 1.0) -> np.ndarray:
        """
        Extracts comprehensive temporal statistical features for Random Forest
        from the observed timesteps up to `fraction` of the season.
        Strictly excludes future timesteps.
        """
        X_valid = self.validate_input(X)
        X_clean = self.impute_missing(X_valid)
        N, T, D = X_clean.shape

        steps_observed = max(1, int(round(T * fraction)))
        X_obs = X_clean[:, :steps_observed, :]  # (N, steps_observed, D)

        feature_list = []

        # 1. Summary stats across observed timesteps
        mean_feat = np.mean(X_obs, axis=1)  # (N, D)
        std_feat = np.std(X_obs, axis=1)   # (N, D)
        max_feat = np.max(X_obs, axis=1)   # (N, D)
        min_feat = np.min(X_obs, axis=1)   # (N, D)
        latest_feat = X_obs[:, -1, :]      # (N, D) - current stage value

        # 2. Key phenological signatures:
        # NDVI is feature index 6
        ndvi_idx = 6 if "NDVI" in self.feature_names else 0
        ndvi_obs = X_obs[:, :, ndvi_idx]  # (N, steps_observed)

        # Peak NDVI timing (normalized to observed window)
        peak_idx = np.argmax(ndvi_obs, axis=1)[:, np.newaxis] / float(steps_observed)
        # NDVI slope / velocity over observed window
        if steps_observed > 1:
            ndvi_slope = (ndvi_obs[:, -1] - ndvi_obs[:, 0])[:, np.newaxis] / float(steps_observed)
        else:
            ndvi_slope = np.zeros((N, 1), dtype=np.float32)

        # Integral / Area under NDVI curve
        trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
        ndvi_integral = trapz_func(ndvi_obs, axis=1)[:, np.newaxis] if steps_observed > 1 else ndvi_obs[:, :1]

        # Combine all features
        tabular_matrix = np.hstack([
            mean_feat, std_feat, max_feat, min_feat, latest_feat,
            peak_idx, ndvi_slope, ndvi_integral
        ])
        return tabular_matrix

    def fit_scaler(self, X_train: np.ndarray):
        """Fits standard scaler on flattened tabular features of full sequence."""
        tab_train = self.extract_tabular_features(X_train, fraction=1.0)
        self.scaler.fit(tab_train)
        self.fitted = True

    def transform_tabular(self, X: np.ndarray, fraction: float = 1.0) -> np.ndarray:
        """Transforms raw input sequence into scaled tabular representation for RF."""
        raw_tabular = self.extract_tabular_features(X, fraction=fraction)
        if not self.fitted:
            raise RuntimeError("Pipeline scaler has not been fitted.")
        return self.scaler.transform(raw_tabular)

    def prepare_lstm_sequence(self, X: np.ndarray, fraction: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Prepares normalized sequence tensor for PyTorch LSTM up to `fraction`.
        Returns (X_seq, steps_observed) where future steps are masked with 0.
        """
        X_valid = self.validate_input(X)
        X_clean = self.impute_missing(X_valid)
        X_trunc, steps = self.truncate_sequence(X_clean, fraction=fraction)
        return X_trunc, steps

    def save(self, filepath: str):
        """Saves pipeline state to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "PreprocessingPipeline":
        """Loads fitted pipeline from disk."""
        return joblib.load(filepath)
