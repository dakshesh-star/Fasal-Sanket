"""
Fasal Sanket - Comprehensive Progressive Model Evaluation

Evaluates fitted Random Forest and LSTM sequence models across held-out test fields
at 5 observation checkpoints: 20%, 40%, 60%, 80%, 100% of the growing season.

Calculates:
- Actual accuracy, macro F1, per-class precision/recall/F1
- Confusion matrices
- Calibration measures: Multiclass Brier Score and Expected Calibration Error (ECE)
- Generates publication-quality botanical evaluation figures in backend/evaluation/plots/
- Outputs machine-readable benchmark data to backend/evaluation/metrics.json
"""

import os
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

import sys
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from app.pipeline import PreprocessingPipeline
from app.taxonomy import TAXONOMY_TREE
from data.generator import CROPS, CROP_TO_ID

try:
    import torch
    from training.train_lstm import CropLSTM
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_multiclass_brier_score(y_true: np.ndarray, probas: np.ndarray) -> float:
    """Computes mean squared error of predicted probability distributions against one-hot targets."""
    n_samples, n_classes = probas.shape
    one_hot = np.zeros((n_samples, n_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), y_true] = 1.0
    brier = np.mean(np.sum((probas - one_hot) ** 2, axis=1))
    return float(brier)


def compute_expected_calibration_error(y_true: np.ndarray, probas: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE)."""
    confidences = np.max(probas, axis=1)
    predictions = np.argmax(probas, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def evaluate_taxonomy(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    true_crops = [CROPS[idx] for idx in y_true]
    pred_crops = [CROPS[idx] for idx in y_pred]

    lvl1_true = [TAXONOMY_TREE[c]["level_1"] for c in true_crops]
    lvl1_pred = [TAXONOMY_TREE[c]["level_1"] for c in pred_crops]

    lvl2_true = [TAXONOMY_TREE[c]["level_2"] for c in true_crops]
    lvl2_pred = [TAXONOMY_TREE[c]["level_2"] for c in pred_crops]

    lvl3_true = [TAXONOMY_TREE[c]["level_3"] for c in true_crops]
    lvl3_pred = [TAXONOMY_TREE[c]["level_3"] for c in pred_crops]

    return {
        "level_1_clade": float(accuracy_score(lvl1_true, lvl1_pred)),
        "level_2_family": float(accuracy_score(lvl2_true, lvl2_pred)),
        "level_3_tribe": float(accuracy_score(lvl3_true, lvl3_pred)),
        "level_4_species": float(accuracy_score(y_true, y_pred))
    }


def run_full_evaluation():
    data_dir = os.path.join(backend_dir, "data")
    models_dir = os.path.join(backend_dir, "models")
    eval_dir = os.path.join(backend_dir, "evaluation")
    plots_dir = os.path.join(eval_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    npz_path = os.path.join(data_dir, "synthetic_crops.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Dataset not found at {npz_path}. Run training or data generation first.")

    data = np.load(npz_path)
    X_test, y_test, f_test = data["X_test"], data["y_test"], data["f_test"]
    print(f"Loaded {len(f_test)} test fields across {len(CROPS)} crops.")

    pipeline_path = os.path.join(models_dir, "scaler.joblib")
    pipeline = PreprocessingPipeline.load(pipeline_path)

    rf_path = os.path.join(models_dir, "rf_classifier.joblib")
    rf_model = joblib.load(rf_path)

    lstm_model = None
    lstm_path = os.path.join(models_dir, "lstm_classifier.pt")
    if TORCH_AVAILABLE and os.path.exists(lstm_path):
        lstm_model = CropLSTM(input_size=9, hidden_size=64, num_layers=2, num_classes=len(CROPS))
        lstm_model.load_state_dict(torch.load(lstm_path, map_location="cpu", weights_only=True))
        lstm_model.eval()
        print("PyTorch LSTM model successfully loaded.")

    stages = [0.2, 0.4, 0.6, 0.8, 1.0]
    stage_labels = ["20%", "40%", "60%", "80%", "100%"]

    results = {
        "dataset": {
            "num_test_fields": len(f_test),
            "num_classes": len(CROPS),
            "crops": CROPS,
            "disclaimer": "Metrics calculated on held-out synthetic Sentinel-2 observations."
        },
        "random_forest": {},
        "lstm": {}
    }

    rf_accuracies = []
    rf_species_accs = []
    rf_family_accs = []
    rf_clade_accs = []

    lstm_accuracies = []
    lstm_species_accs = []
    lstm_family_accs = []
    lstm_clade_accs = []

    # Store confusion matrices for plotting
    rf_cm_early = None
    rf_cm_late = None

    for idx, stg in enumerate(stages):
        stg_lbl = stage_labels[idx]

        # 1. Random Forest progressive
        X_tab = pipeline.transform_tabular(X_test, fraction=stg)
        rf_preds = rf_model.predict(X_tab)
        rf_probs = rf_model.predict_proba(X_tab)

        rf_acc = float(accuracy_score(y_test, rf_preds))
        rf_f1 = float(f1_score(y_test, rf_preds, average="macro"))
        prec, rec, f1, _ = precision_recall_fscore_support(y_test, rf_preds, labels=range(len(CROPS)), zero_division=0)
        rf_cm = confusion_matrix(y_test, rf_preds).tolist()
        rf_brier = compute_multiclass_brier_score(y_test, rf_probs)
        rf_ece = compute_expected_calibration_error(y_test, rf_probs)
        rf_tax = evaluate_taxonomy(y_test, rf_preds)

        rf_accuracies.append(rf_acc)
        rf_clade_accs.append(rf_tax["level_1_clade"])
        rf_family_accs.append(rf_tax["level_2_family"])
        rf_species_accs.append(rf_tax["level_4_species"])

        if stg == 0.4:
            rf_cm_early = np.array(rf_cm)
        elif stg == 1.0:
            rf_cm_late = np.array(rf_cm)

        results["random_forest"][stg_lbl] = {
            "accuracy": round(rf_acc, 4),
            "macro_f1": round(rf_f1, 4),
            "brier_score": round(rf_brier, 4),
            "expected_calibration_error": round(rf_ece, 4),
            "taxonomic_accuracy": rf_tax,
            "per_class": {
                crop: {
                    "precision": round(float(prec[i]), 4),
                    "recall": round(float(rec[i]), 4),
                    "f1": round(float(f1[i]), 4)
                }
                for i, crop in enumerate(CROPS)
            },
            "confusion_matrix": rf_cm
        }

        # 2. LSTM progressive
        if lstm_model is not None:
            X_lstm, steps = pipeline.prepare_lstm_sequence(X_test, fraction=stg)
            lens = torch.full((len(X_test),), steps, dtype=torch.long)
            with torch.no_grad():
                logits = lstm_model(torch.tensor(X_lstm, dtype=torch.float32), lengths=lens)
                lstm_probs = torch.softmax(logits, dim=1).cpu().numpy()
                lstm_preds = np.argmax(lstm_probs, axis=1)

            l_acc = float(accuracy_score(y_test, lstm_preds))
            l_f1 = float(f1_score(y_test, lstm_preds, average="macro"))
            l_prec, l_rec, l_f1_per, _ = precision_recall_fscore_support(y_test, lstm_preds, labels=range(len(CROPS)), zero_division=0)
            l_cm = confusion_matrix(y_test, lstm_preds).tolist()
            l_brier = compute_multiclass_brier_score(y_test, lstm_probs)
            l_ece = compute_expected_calibration_error(y_test, lstm_probs)
            l_tax = evaluate_taxonomy(y_test, lstm_preds)

            lstm_accuracies.append(l_acc)
            lstm_clade_accs.append(l_tax["level_1_clade"])
            lstm_family_accs.append(l_tax["level_2_family"])
            lstm_species_accs.append(l_tax["level_4_species"])

            results["lstm"][stg_lbl] = {
                "accuracy": round(l_acc, 4),
                "macro_f1": round(l_f1, 4),
                "brier_score": round(l_brier, 4),
                "expected_calibration_error": round(l_ece, 4),
                "taxonomic_accuracy": l_tax,
                "per_class": {
                    crop: {
                        "precision": round(float(l_prec[i]), 4),
                        "recall": round(float(l_rec[i]), 4),
                        "f1": round(float(l_f1_per[i]), 4)
                    }
                    for i, crop in enumerate(CROPS)
                },
                "confusion_matrix": l_cm
            }

    # Save metrics JSON
    metrics_path = os.path.join(eval_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    # Generate Visualization Figures
    # 1. Progressive Accuracy Curve
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.plot(stage_labels, rf_accuracies, marker="o", color="#2d6a4f", linewidth=2.5, label="Random Forest (Tabular)")
    if lstm_model is not None:
        plt.plot(stage_labels, lstm_accuracies, marker="s", color="#d4a373", linewidth=2.5, label="LSTM (Sequence)")

    plt.axhline(0.90, color="#d90429", linestyle="--", linewidth=1.5, label="Target Reliability Threshold (90% / Error <= 10%)")
    plt.title("Fasal Sanket - Progressive Crop Identification Accuracy", fontsize=14, pad=12, fontweight="bold", color="#1b4332")
    plt.xlabel("Observation Progress Across Growing Season", fontsize=11, labelpad=8)
    plt.ylabel("Held-Out Test Accuracy", fontsize=11, labelpad=8)
    plt.ylim(0.40, 1.02)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="#f8f9fa", edgecolor="#ced4da")
    plt.tight_layout()
    plot1_path = os.path.join(plots_dir, "progressive_accuracy_curve.png")
    plt.savefig(plot1_path)
    plt.close()
    print(f"Generated plot: {plot1_path}")

    # 2. Taxonomic Resolution Level Comparison Heatmap / Progression
    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.plot(stage_labels, rf_clade_accs, marker="^", color="#1b4332", linewidth=2, label="Level 1: Clade (Monocot vs Dicot)")
    plt.plot(stage_labels, rf_family_accs, marker="d", color="#40916c", linewidth=2, label="Level 2: Family (Poaceae, etc.)")
    plt.plot(stage_labels, rf_species_accs, marker="o", color="#b7094c", linewidth=2.5, label="Level 4: Species (Exact Crop)")
    plt.axhline(0.90, color="#e63946", linestyle="--", linewidth=1.5, label="Target Reliability (90%)")

    plt.title("Taxonomic Resolution by Observation Stage (RF Model)", fontsize=13, pad=12, fontweight="bold", color="#1b4332")
    plt.xlabel("Temporal Observation Checkpoint", fontsize=11)
    plt.ylabel("Accuracy across Taxonomic Levels", fontsize=11)
    plt.ylim(0.50, 1.02)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="#f8f9fa")
    plt.tight_layout()
    plot2_path = os.path.join(plots_dir, "taxonomic_resolution_levels.png")
    plt.savefig(plot2_path)
    plt.close()
    print(f"Generated plot: {plot2_path}")

    # 3. Confusion Matrix at 40% vs 100%
    if rf_cm_early is not None and rf_cm_late is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

        for ax, cm, title in zip(axes, [rf_cm_early, rf_cm_late], ["At 40% Observation (Mid-Emergence)", "At 100% Observation (Harvest)"]):
            im = ax.imshow(cm, cmap="YlGn", interpolation="nearest")
            ax.set_title(title, fontsize=12, fontweight="bold", color="#1b4332")
            ax.set_xticks(range(len(CROPS)))
            ax.set_yticks(range(len(CROPS)))
            ax.set_xticklabels(CROPS, rotation=35, ha="right")
            ax.set_yticklabels(CROPS)
            ax.set_xlabel("Predicted Crop")
            ax.set_ylabel("True Crop")

            # Annotate numbers
            for i in range(len(CROPS)):
                for j in range(len(CROPS)):
                    color = "white" if cm[i, j] > cm.max() / 2 else "black"
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontweight="bold")

        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
        fig.suptitle("Fasal Sanket - Confusion Matrix Evolution Across Season", fontsize=14, y=1.02, fontweight="bold", color="#1b4332")
        plot3_path = os.path.join(plots_dir, "confusion_matrices_progressive.png")
        plt.savefig(plot3_path, bbox_inches="tight")
        plt.close()
        print(f"Generated plot: {plot3_path}")

    return results


if __name__ == "__main__":
    run_full_evaluation()
