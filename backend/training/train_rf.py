"""
Fasal Sanket - Random Forest Training with Progressive Multi-Stage Representation

Trains Random Forest classifier on independent field-grouped synthetic data.
Uses progressive stage representations so the model accurately recognizes crop
phenology at 20%, 40%, 60%, 80%, and 100% of the season without future leakage.
Evaluates progressive stage metrics on held-out validation fields to compute
empirical reliability for each taxonomic level.
Saves fitted model artifact to backend/models/rf_classifier.joblib
"""

import os
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

import sys
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from app.pipeline import PreprocessingPipeline
from app.taxonomy import TAXONOMY_TREE
from data.generator import generate_and_save, CROPS, CROP_TO_ID


def evaluate_taxonomy_levels(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes accuracy at each of the 4 biological taxonomy levels:
    Level 1: Clade (Monocot vs Dicot)
    Level 2: Family (Poaceae, Brassicaceae, Malvaceae)
    Level 3: Tribe/Genus (Triticeae, Oryzeae, Maydeae, Brassica, Gossypium)
    Level 4: Species (Wheat, Rice, Maize, Mustard, Cotton)
    """
    lvl4_acc = float(accuracy_score(y_true, y_pred))
    true_crops = [CROPS[idx] for idx in y_true]
    pred_crops = [CROPS[idx] for idx in y_pred]

    lvl1_true = [TAXONOMY_TREE[c]["level_1"] for c in true_crops]
    lvl1_pred = [TAXONOMY_TREE[c]["level_1"] for c in pred_crops]
    lvl1_acc = float(accuracy_score(lvl1_true, lvl1_pred))

    lvl2_true = [TAXONOMY_TREE[c]["level_2"] for c in true_crops]
    lvl2_pred = [TAXONOMY_TREE[c]["level_2"] for c in pred_crops]
    lvl2_acc = float(accuracy_score(lvl2_true, lvl2_pred))

    lvl3_true = [TAXONOMY_TREE[c]["level_3"] for c in true_crops]
    lvl3_pred = [TAXONOMY_TREE[c]["level_3"] for c in pred_crops]
    lvl3_acc = float(accuracy_score(lvl3_true, lvl3_pred))

    return {
        "level_1": round(lvl1_acc, 4),
        "level_2": round(lvl2_acc, 4),
        "level_3": round(lvl3_acc, 4),
        "level_4": round(lvl4_acc, 4)
    }


def train_random_forest():
    data_dir = os.path.join(backend_dir, "data")
    npz_path = os.path.join(data_dir, "synthetic_crops.npz")
    models_dir = os.path.join(backend_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    if not os.path.exists(npz_path):
        print("Data not found, generating synthetic dataset first...")
        generate_and_save(data_dir)

    print("Loading synthetic field dataset...")
    data = np.load(npz_path)
    X_train = data["X_train"]
    y_train = data["y_train"]
    f_train = data["f_train"]

    X_val = data["X_val"]
    y_val = data["y_val"]
    f_val = data["f_val"]

    X_test = data["X_test"]
    y_test = data["y_test"]
    f_test = data["f_test"]

    print(f"Dataset loaded: Train={len(f_train)} fields, Val={len(f_val)} fields, Test={len(f_test)} fields")

    # Fit pipeline
    pipeline = PreprocessingPipeline()
    print("Fitting feature pipeline scaler on training fields...")
    pipeline.fit_scaler(X_train)
    pipeline_path = os.path.join(models_dir, "scaler.joblib")
    pipeline.save(pipeline_path)
    print(f"Pipeline saved to {pipeline_path}")

    # Build progressive multi-stage training set so model learns each checkpoint's phenological signature
    stages = [0.2, 0.4, 0.6, 0.8, 1.0]
    X_train_multi = []
    y_train_multi = []

    for stg in stages:
        X_stg = pipeline.transform_tabular(X_train, fraction=stg)
        X_train_multi.append(X_stg)
        y_train_multi.append(y_train)

    X_train_augmented = np.vstack(X_train_multi)
    y_train_augmented = np.concatenate(y_train_multi)

    print(f"Training Random Forest on multi-stage representations ({len(y_train_augmented)} samples, 150 trees)...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_augmented, y_train_augmented)

    # Full season evaluation
    X_test_full = pipeline.transform_tabular(X_test, fraction=1.0)
    y_test_pred = rf.predict(X_test_full)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    print(f"\nRandom Forest Full-Season Test Accuracy: {test_acc:.4f}, Macro F1: {test_f1:.4f}")

    # Progressive Stage Evaluation (20%, 40%, 60%, 80%, 100%)
    progressive_val_evidence = {}
    progressive_test_metrics = {}

    print("\n--- Progressive Evaluation on Held-Out Validation Fields ---")
    for stg in stages:
        stg_key = f"{int(stg * 100)}%"
        # Extract features strictly up to stage fraction (NO future observations!)
        X_val_stg = pipeline.transform_tabular(X_val, fraction=stg)
        y_val_stg_pred = rf.predict(X_val_stg)

        tax_metrics = evaluate_taxonomy_levels(y_val, y_val_stg_pred)
        progressive_val_evidence[stg_key] = tax_metrics

        X_test_stg = pipeline.transform_tabular(X_test, fraction=stg)
        y_test_stg_pred = rf.predict(X_test_stg)
        test_stage_acc = float(accuracy_score(y_test, y_test_stg_pred))
        test_stage_f1 = float(f1_score(y_test, y_test_stg_pred, average="macro"))

        progressive_test_metrics[stg_key] = {
            "accuracy": round(test_stage_acc, 4),
            "macro_f1": round(test_stage_f1, 4),
            "taxonomy_accuracy": tax_metrics
        }
        print(f"Stage {stg_key:>4}: Level 1 (Clade)={tax_metrics['level_1']:.1%}, "
              f"Level 2 (Family)={tax_metrics['level_2']:.1%}, "
              f"Level 3 (Tribe)={tax_metrics['level_3']:.1%}, "
              f"Level 4 (Species Acc)={tax_metrics['level_4']:.1%}")

    # Save fitted model artifact
    rf_model_path = os.path.join(models_dir, "rf_classifier.joblib")
    joblib.dump(rf, rf_model_path)
    print(f"\nRandom Forest artifact saved to {rf_model_path}")

    # Save metadata and evidence
    metadata = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 150,
        "max_depth": 14,
        "num_classes": len(CROPS),
        "crops": CROPS,
        "features": pipeline.feature_names,
        "num_train_fields": len(f_train),
        "num_val_fields": len(f_val),
        "num_test_fields": len(f_test),
        "full_season_test_accuracy": round(float(test_acc), 4),
        "full_season_test_macro_f1": round(float(test_f1), 4),
        "progressive_val_evidence": progressive_val_evidence,
        "progressive_test_metrics": progressive_test_metrics,
        "is_synthetic": True
    }

    meta_path = os.path.join(models_dir, "rf_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Random Forest metadata and validation evidence saved to {meta_path}")
    return rf, metadata


if __name__ == "__main__":
    train_random_forest()
