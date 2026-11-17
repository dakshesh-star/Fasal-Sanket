"""
Fasal Sanket - Synthetic Sentinel-2 Temporal Crop Observation Generator

Generates realistic temporal multi-spectral observations for 5 target crops:
1. Wheat (Triticum aestivum) - Rabi cereal
2. Rice (Oryza sativa) - Kharif semi-aquatic cereal
3. Maize (Zea mays) - Coarse grain
4. Mustard (Brassica juncea) - Rabi oilseed
5. Cotton (Gossypium hirsutum) - Long duration Kharif fiber crop

IMPORTANT: This dataset is strictly SYNTHETIC, engineered for algorithmic
and architectural evaluation of the progressive Taxonomic Resolution Calendar.
Data splitting is grouped by independent simulated fields to eliminate leakage.
"""

import os
import json
import numpy as np
from typing import Dict, Tuple, Optional

# Constants
RANDOM_SEED = 42
CROPS = ["Wheat", "Rice", "Maize", "Mustard", "Cotton"]
CROP_TO_ID = {crop: i for i, crop in enumerate(CROPS)}
ID_TO_CROP = {i: crop for crop, i in CROP_TO_ID.items()}

# Taxonomy hierarchy mapping for the 5 crops
TAXONOMY_MAP = {
    "Wheat": {
        "level_1": "Monocotyledoneae (Monocots)",
        "level_2": "Poaceae (True Grasses)",
        "level_3": "Triticeae",
        "level_4": "Triticum aestivum (Wheat)"
    },
    "Rice": {
        "level_1": "Monocotyledoneae (Monocots)",
        "level_2": "Poaceae (True Grasses)",
        "level_3": "Oryzeae",
        "level_4": "Oryza sativa (Rice)"
    },
    "Maize": {
        "level_1": "Monocotyledoneae (Monocots)",
        "level_2": "Poaceae (True Grasses)",
        "level_3": "Maydeae",
        "level_4": "Zea mays (Maize)"
    },
    "Mustard": {
        "level_1": "Dicotyledoneae (Dicots)",
        "level_2": "Brassicaceae (Mustard Family)",
        "level_3": "Brassica",
        "level_4": "Brassica juncea (Indian Mustard)"
    },
    "Cotton": {
        "level_1": "Dicotyledoneae (Dicots)",
        "level_2": "Malvaceae (Mallow Family)",
        "level_3": "Gossypium",
        "level_4": "Gossypium hirsutum (Cotton)"
    }
}

# Spectral Channels simulated (Sentinel-2 like)
# B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR1), B12 (SWIR2)
# Indices: NDVI, NDWI, EVI
FEATURE_NAMES = [
    "B2_Blue", "B3_Green", "B4_Red", "B8_NIR", "B11_SWIR1", "B12_SWIR2",
    "NDVI", "NDWI", "EVI"
]

NUM_TIMESTEPS = 20  # 20 observations spaced across the normalized growing sequence (e.g. 10-day intervals)


def simulate_phenology(
    crop: str,
    timesteps: np.ndarray,
    sowing_offset: float,
    vigor: float,
    rng: np.random.RandomState
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates Sentinel-2 bottom-of-atmosphere surface reflectance (0.0 to 1.0)
    for 6 bands based on asymmetric double-logistic phenological growth models.
    """
    t = timesteps - sowing_offset
    t_norm = (t - t.min()) / (t.max() - t.min() + 1e-6)

    # Base soil reflectance
    soil_b2 = 0.08 + rng.normal(0, 0.01)
    soil_b3 = 0.12 + rng.normal(0, 0.01)
    soil_b4 = 0.18 + rng.normal(0, 0.015)
    soil_b8 = 0.22 + rng.normal(0, 0.02)
    soil_b11 = 0.28 + rng.normal(0, 0.02)
    soil_b12 = 0.24 + rng.normal(0, 0.02)

    if crop == "Wheat":
        # Rabi cereal: Sow early, peak ~0.55 of season, rapid yellowing/senescence
        peak_t = 0.55 + rng.normal(0, 0.03)
        width = 0.22 * vigor
        canopy = np.exp(-((t_norm - peak_t) ** 2) / (2 * width ** 2))
        canopy = np.clip(canopy, 0.0, 1.0) * vigor

        # Chlorophyll absorption drops Red, scatters NIR
        b4 = np.clip(soil_b4 * (1 - 0.72 * canopy) + rng.normal(0, 0.01, size=len(t)), 0.02, 0.6)
        b8 = np.clip(soil_b8 + (0.55 * canopy) + rng.normal(0, 0.015, size=len(t)), 0.05, 0.85)
        b2 = np.clip(soil_b2 * (1 - 0.50 * canopy) + rng.normal(0, 0.008, size=len(t)), 0.01, 0.4)
        b3 = np.clip(soil_b3 * (1 - 0.20 * canopy) + rng.normal(0, 0.01, size=len(t)), 0.02, 0.45)
        # SWIR drops with canopy water, rises sharply at senescence
        senescence = np.clip((t_norm - (peak_t + 0.15)) * 3.0, 0.0, 1.0)
        b11 = np.clip(soil_b11 * (1 - 0.45 * canopy) + 0.12 * senescence + rng.normal(0, 0.012, size=len(t)), 0.05, 0.65)
        b12 = np.clip(soil_b12 * (1 - 0.40 * canopy) + 0.10 * senescence + rng.normal(0, 0.012, size=len(t)), 0.04, 0.6)

    elif crop == "Rice":
        # Kharif semi-aquatic cereal: Distinctive water background (low NIR, low SWIR) early on
        flood_stage = np.clip(1.0 - t_norm * 4.0, 0.0, 1.0)
        peak_t = 0.58 + rng.normal(0, 0.03)
        width = 0.20 * vigor
        canopy = np.exp(-((t_norm - peak_t) ** 2) / (2 * width ** 2))
        canopy = np.clip(canopy, 0.0, 1.0) * vigor

        b4 = np.clip(
            (soil_b4 * (1 - 0.5 * flood_stage)) * (1 - 0.75 * canopy) + rng.normal(0, 0.01, size=len(t)),
            0.02, 0.6
        )
        b8 = np.clip(
            (soil_b8 * (1 - 0.65 * flood_stage)) + (0.60 * canopy) + rng.normal(0, 0.015, size=len(t)),
            0.04, 0.88
        )
        b2 = np.clip(soil_b2 * (1 - 0.45 * canopy) + rng.normal(0, 0.008, size=len(t)), 0.01, 0.4)
        b3 = np.clip(soil_b3 * (1 - 0.15 * canopy) + rng.normal(0, 0.01, size=len(t)), 0.02, 0.45)
        b11 = np.clip(
            (soil_b11 * (1 - 0.75 * flood_stage)) * (1 - 0.45 * canopy) + 0.08 * (t_norm > 0.8) + rng.normal(0, 0.012, size=len(t)),
            0.02, 0.6
        )
        b12 = np.clip(
            (soil_b12 * (1 - 0.75 * flood_stage)) * (1 - 0.40 * canopy) + 0.08 * (t_norm > 0.8) + rng.normal(0, 0.012, size=len(t)),
            0.02, 0.55
        )

    elif crop == "Maize":
        # Kharif/Spring coarse grain: Fast vegetative growth, high peak, rapid dry-down
        peak_t = 0.48 + rng.normal(0, 0.025)
        width = 0.18 * vigor
        canopy = np.exp(-((t_norm - peak_t) ** 2) / (2 * width ** 2))
        canopy = np.clip(canopy, 0.0, 1.0) * vigor

        b4 = np.clip(soil_b4 * (1 - 0.78 * canopy) + rng.normal(0, 0.01, size=len(t)), 0.02, 0.6)
        b8 = np.clip(soil_b8 + (0.65 * canopy) + rng.normal(0, 0.015, size=len(t)), 0.05, 0.90)
        b2 = np.clip(soil_b2 * (1 - 0.55 * canopy) + rng.normal(0, 0.008, size=len(t)), 0.01, 0.4)
        b3 = np.clip(soil_b3 * (1 + 0.10 * canopy) + rng.normal(0, 0.01, size=len(t)), 0.02, 0.45)
        b11 = np.clip(soil_b11 * (1 - 0.50 * canopy) + 0.14 * (t_norm > 0.7) + rng.normal(0, 0.012, size=len(t)), 0.04, 0.65)
        b12 = np.clip(soil_b12 * (1 - 0.45 * canopy) + 0.12 * (t_norm > 0.7) + rng.normal(0, 0.012, size=len(t)), 0.03, 0.60)

    elif crop == "Mustard":
        # Rabi oilseed: Early rosette growth, distinctive yellow flower bloom at t_norm ~0.40
        peak_t = 0.45 + rng.normal(0, 0.03)
        width = 0.20 * vigor
        canopy = np.exp(-((t_norm - peak_t) ** 2) / (2 * width ** 2))
        canopy = np.clip(canopy, 0.0, 1.0) * vigor

        flower_t = 0.40 + rng.normal(0, 0.02)
        flower_effect = np.exp(-((t_norm - flower_t) ** 2) / (2 * (0.05) ** 2))

        b4 = np.clip(soil_b4 * (1 - 0.65 * canopy) + 0.07 * flower_effect + rng.normal(0, 0.01, size=len(t)), 0.03, 0.6)
        b8 = np.clip(soil_b8 + (0.50 * canopy) - 0.04 * flower_effect + rng.normal(0, 0.015, size=len(t)), 0.05, 0.82)
        b2 = np.clip(soil_b2 * (1 - 0.40 * canopy) + rng.normal(0, 0.008, size=len(t)), 0.01, 0.4)
        b3 = np.clip(soil_b3 * (1 - 0.10 * canopy) + 0.10 * flower_effect + rng.normal(0, 0.01, size=len(t)), 0.03, 0.50)
        b11 = np.clip(soil_b11 * (1 - 0.40 * canopy) + 0.09 * (t_norm > 0.7) + rng.normal(0, 0.012, size=len(t)), 0.05, 0.65)
        b12 = np.clip(soil_b12 * (1 - 0.35 * canopy) + 0.08 * (t_norm > 0.7) + rng.normal(0, 0.012, size=len(t)), 0.04, 0.60)

    elif crop == "Cotton":
        # Long-duration Kharif: Slow vegetative start, broad extended plateau, gradual senescence
        peak_t = 0.62 + rng.normal(0, 0.03)
        width = 0.28 * vigor
        canopy = np.exp(-((t_norm - peak_t) ** 2) / (2 * width ** 2))
        canopy = np.clip(canopy, 0.0, 1.0) * vigor

        b4 = np.clip(soil_b4 * (1 - 0.68 * canopy) + rng.normal(0, 0.01, size=len(t)), 0.03, 0.6)
        b8 = np.clip(soil_b8 + (0.52 * canopy) + rng.normal(0, 0.015, size=len(t)), 0.05, 0.82)
        b2 = np.clip(soil_b2 * (1 - 0.45 * canopy) + rng.normal(0, 0.008, size=len(t)), 0.01, 0.4)
        b3 = np.clip(soil_b3 * (1 - 0.15 * canopy) + rng.normal(0, 0.01, size=len(t)), 0.02, 0.45)
        b11 = np.clip(soil_b11 * (1 - 0.38 * canopy) + 0.06 * (t_norm > 0.85) + rng.normal(0, 0.012, size=len(t)), 0.05, 0.65)
        b12 = np.clip(soil_b12 * (1 - 0.35 * canopy) + 0.05 * (t_norm > 0.85) + rng.normal(0, 0.012, size=len(t)), 0.04, 0.60)

    else:
        raise ValueError(f"Unknown crop: {crop}")

    return b2, b3, b4, b8, b11, b12


def compute_spectral_indices(
    b2: np.ndarray, b3: np.ndarray, b4: np.ndarray, b8: np.ndarray, b11: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes standard vegetation & water indices from surface reflectance."""
    eps = 1e-6
    # NDVI = (NIR - Red) / (NIR + Red)
    ndvi = np.clip((b8 - b4) / (b8 + b4 + eps), -1.0, 1.0)

    # NDWI (Gao) = (NIR - SWIR) / (NIR + SWIR) (Canopy water content)
    ndwi = np.clip((b8 - b11) / (b8 + b11 + eps), -1.0, 1.0)

    # EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    evi_denom = b8 + 6.0 * b4 - 7.5 * b2 + 1.0
    evi = np.clip(2.5 * (b8 - b4) / (evi_denom + eps), -1.5, 1.5)

    return ndvi, ndwi, evi


def generate_synthetic_dataset(
    num_fields_per_crop: int = 200,
    num_timesteps: int = NUM_TIMESTEPS,
    cloud_missing_prob: float = 0.08,
    seed: int = RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Generates a full synthetic dataset with independent field identifiers.

    Returns:
    - X: Array of shape (N_fields, num_timesteps, num_features)
    - y: Array of shape (N_fields,) containing integer crop labels 0..4
    - field_ids: Array of shape (N_fields,) indicating simulated field IDs
    - metadata: Dictionary containing generator configs and synthetic disclaimer
    """
    rng = np.random.RandomState(seed)
    timesteps = np.linspace(0, 1, num_timesteps)

    total_fields = len(CROPS) * num_fields_per_crop
    num_features = len(FEATURE_NAMES)

    X = np.zeros((total_fields, num_timesteps, num_features), dtype=np.float32)
    y = np.zeros((total_fields,), dtype=np.int64)
    field_ids = []

    idx = 0
    for crop_idx, crop_name in enumerate(CROPS):
        for field_num in range(num_fields_per_crop):
            field_id = f"{crop_name[:2].upper()}_FIELD_{field_num:04d}"
            field_ids.append(field_id)

            # Field specific natural parameters
            sowing_offset = rng.normal(0.0, 0.04)
            vigor = float(np.clip(rng.normal(1.0, 0.12), 0.70, 1.30))

            # Simulate base reflectance bands
            b2, b3, b4, b8, b11, b12 = simulate_phenology(
                crop=crop_name,
                timesteps=timesteps,
                sowing_offset=sowing_offset,
                vigor=vigor,
                rng=rng
            )

            # Compute indices
            ndvi, ndwi, evi = compute_spectral_indices(b2, b3, b4, b8, b11)

            # Assemble features for all timesteps
            field_features = np.stack([
                b2, b3, b4, b8, b11, b12, ndvi, ndwi, evi
            ], axis=-1)

            # Simulate occasional cloud contamination / missing observations
            cloud_mask = rng.rand(num_timesteps) < cloud_missing_prob
            if np.any(cloud_mask):
                field_features[cloud_mask, :] = np.nan

            X[idx] = field_features
            y[idx] = crop_idx
            idx += 1

    field_ids = np.array(field_ids)

    metadata = {
        "dataset_type": "SYNTHETIC_SENTINEL2_SIMULATION",
        "notice": "This dataset consists of synthetic Sentinel-2-like temporal observations. Not for operational crop estimation.",
        "random_seed": seed,
        "num_crops": len(CROPS),
        "crops": CROPS,
        "crop_to_id": CROP_TO_ID,
        "id_to_crop": ID_TO_CROP,
        "features": FEATURE_NAMES,
        "num_timesteps": num_timesteps,
        "num_fields_per_crop": num_fields_per_crop,
        "total_fields": total_fields,
        "cloud_missing_prob": cloud_missing_prob,
        "taxonomy": TAXONOMY_MAP
    }

    return X, y, field_ids, metadata


def split_by_fields(
    X: np.ndarray,
    y: np.ndarray,
    field_ids: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = RANDOM_SEED
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Performs field-level stratified splitting.
    Crucial: Entire fields are assigned to either Train, Val, or Test.
    Zero field leakage across splits!
    """
    rng = np.random.RandomState(seed)
    unique_crops = np.unique(y)

    train_indices = []
    val_indices = []
    test_indices = []

    for c in unique_crops:
        c_indices = np.where(y == c)[0]
        rng.shuffle(c_indices)

        n_total = len(c_indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_indices.extend(c_indices[:n_train])
        val_indices.extend(c_indices[n_train:n_train + n_val])
        test_indices.extend(c_indices[n_train + n_val:])

    train_idx = np.array(train_indices)
    val_idx = np.array(val_indices)
    test_idx = np.array(test_indices)

    # Verify zero overlap between field sets
    train_fields = set(field_ids[train_idx])
    val_fields = set(field_ids[val_idx])
    test_fields = set(field_ids[test_idx])

    assert len(train_fields.intersection(val_fields)) == 0, "Leakage detected between Train and Val!"
    assert len(train_fields.intersection(test_fields)) == 0, "Leakage detected between Train and Test!"
    assert len(val_fields.intersection(test_fields)) == 0, "Leakage detected between Val and Test!"

    return (
        (X[train_idx], y[train_idx], field_ids[train_idx]),
        (X[val_idx], y[val_idx], field_ids[val_idx]),
        (X[test_idx], y[test_idx], field_ids[test_idx])
    )


def generate_and_save(out_dir: Optional[str] = None) -> str:
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    print("Generating synthetic Sentinel-2 temporal dataset...")
    X, y, field_ids, meta = generate_synthetic_dataset(num_fields_per_crop=200, seed=RANDOM_SEED)

    train_split, val_split, test_split = split_by_fields(X, y, field_ids, seed=RANDOM_SEED)
    X_train, y_train, f_train = train_split
    X_val, y_val, f_val = val_split
    X_test, y_test, f_test = test_split

    npz_path = os.path.join(out_dir, "synthetic_crops.npz")
    np.savez_compressed(
        npz_path,
        X_train=X_train, y_train=y_train, f_train=f_train,
        X_val=X_val, y_val=y_val, f_val=f_val,
        X_test=X_test, y_test=y_test, f_test=f_test
    )

    meta_path = os.path.join(out_dir, "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Dataset generated and saved to {npz_path}")
    print(f"Train fields: {len(f_train)}, Val fields: {len(f_val)}, Test fields: {len(f_test)}")
    return npz_path


if __name__ == "__main__":
    generate_and_save()
