"""
Fasal Sanket - Model Inference and Progressive Prediction Engine

Loads saved Random Forest and LSTM model artifacts at startup.
Provides unified prediction and progressive calendar resolution interfaces.
Returns clear diagnostics if model artifacts are missing.
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, List, Optional, Any, Union

import sys
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.pipeline import PreprocessingPipeline
from app.taxonomy import TaxonomicResolutionEngine, TAXONOMY_TREE
from data.generator import CROPS, CROP_TO_ID, ID_TO_CROP

try:
    import torch
    from training.train_lstm import CropLSTM
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class InferenceEngine:
    """Manages loaded model artifacts and progressive predictions."""

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or os.path.join(backend_dir, "models")
        self.pipeline: Optional[PreprocessingPipeline] = None
        self.rf_model = None
        self.rf_metadata: Dict = {}
        self.lstm_model = None
        self.lstm_metadata: Dict = {}
        self.taxonomy_engine = TaxonomicResolutionEngine(target_error_rate=0.10)
        self.loaded_status = {
            "pipeline": False,
            "random_forest": False,
            "lstm": False
        }
        self.load_artifacts()

    def load_artifacts(self):
        """Loads all fitted model artifacts and metadata from disk."""
        # 1. Pipeline / Scaler
        scaler_path = os.path.join(self.models_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            try:
                self.pipeline = PreprocessingPipeline.load(scaler_path)
                self.loaded_status["pipeline"] = True
            except Exception as e:
                print(f"Warning: Failed to load pipeline from {scaler_path}: {e}")

        # 2. Random Forest
        rf_path = os.path.join(self.models_dir, "rf_classifier.joblib")
        rf_meta_path = os.path.join(self.models_dir, "rf_metadata.json")
        if os.path.exists(rf_path):
            try:
                self.rf_model = joblib.load(rf_path)
                self.loaded_status["random_forest"] = True
                if os.path.exists(rf_meta_path):
                    with open(rf_meta_path, "r") as f:
                        self.rf_metadata = json.load(f)
                        # Configure taxonomy engine with empirical validation evidence
                        val_ev = self.rf_metadata.get("progressive_val_evidence", {})
                        if val_ev:
                            self.taxonomy_engine.set_validation_evidence(val_ev)
            except Exception as e:
                print(f"Warning: Failed to load Random Forest from {rf_path}: {e}")

        # 3. PyTorch LSTM
        lstm_path = os.path.join(self.models_dir, "lstm_classifier.pt")
        lstm_meta_path = os.path.join(self.models_dir, "lstm_metadata.json")
        if TORCH_AVAILABLE and os.path.exists(lstm_path):
            try:
                model = CropLSTM(input_size=9, hidden_size=64, num_layers=2, num_classes=len(CROPS))
                model.load_state_dict(torch.load(lstm_path, map_location="cpu", weights_only=True))
                model.eval()
                self.lstm_model = model
                self.loaded_status["lstm"] = True
                if os.path.exists(lstm_meta_path):
                    with open(lstm_meta_path, "r") as f:
                        self.lstm_metadata = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load LSTM from {lstm_path}: {e}")

    def check_readiness(self, model_type: str = "random_forest"):
        """Validates that requested model artifact and pipeline are loaded."""
        if not self.loaded_status["pipeline"] or self.pipeline is None:
            raise FileNotFoundError(
                "Preprocessing pipeline artifact is missing. "
                "Please run model training first: 'py backend/training/train_rf.py'"
            )

        if model_type == "random_forest":
            if not self.loaded_status["random_forest"] or self.rf_model is None:
                raise FileNotFoundError(
                    "Random Forest model artifact is missing. "
                    "Please run: 'py backend/training/train_rf.py'"
                )
        elif model_type == "lstm":
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is not available in the current Python environment.")
            if not self.loaded_status["lstm"] or self.lstm_model is None:
                raise FileNotFoundError(
                    "LSTM model artifact is missing. "
                    "Please run: 'py backend/training/train_lstm.py'"
                )
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Choose 'random_forest' or 'lstm'.")

    def predict_single_stage(
        self,
        observations: np.ndarray,
        model_type: str = "random_forest",
        stage_fraction: float = 1.0,
        target_error_rate: float = 0.10,
        use_empirical_evidence: bool = True
    ) -> Dict[str, Any]:
        """
        Runs inference on an observation sequence up to stage_fraction (0.2, 0.4, 0.6, 0.8, 1.0).
        """
        self.check_readiness(model_type)
        obs_array = self.pipeline.validate_input(observations)

        if model_type == "random_forest":
            X_tab = self.pipeline.transform_tabular(obs_array, fraction=stage_fraction)
            probs = self.rf_model.predict_proba(X_tab)[0]
            pred_idx = int(np.argmax(probs))
            pred_crop = CROPS[pred_idx]
            crop_probs = {crop: round(float(probs[i]), 4) for i, crop in enumerate(CROPS)}

        elif model_type == "lstm":
            X_seq, steps = self.pipeline.prepare_lstm_sequence(obs_array, fraction=stage_fraction)
            lens = torch.tensor([steps], dtype=torch.long)
            with torch.no_grad():
                logits = self.lstm_model(torch.tensor(X_seq, dtype=torch.float32), lengths=lens)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_crop = CROPS[pred_idx]
            crop_probs = {crop: round(float(probs[i]), 4) for i, crop in enumerate(CROPS)}

        # Apply Taxonomic Resolution Engine policy
        resolution_engine = self.taxonomy_engine if use_empirical_evidence else TaxonomicResolutionEngine(target_error_rate=target_error_rate)
        tax_result = resolution_engine.resolve(
            predicted_crop=pred_crop,
            stage_fraction=stage_fraction,
            model_probabilities=crop_probs,
            override_error_rate=target_error_rate
        )

        return {
            "model_type": model_type,
            "stage_fraction": stage_fraction,
            "observation_progress": f"{int(round(stage_fraction * 100))}%",
            "predicted_crop": pred_crop,
            "probabilities": crop_probs,
            "taxonomic_resolution": tax_result,
            "is_synthetic_demonstration": use_empirical_evidence
        }

    def predict_progressive(
        self,
        observations: np.ndarray,
        model_type: str = "random_forest",
        target_error_rate: float = 0.10
    ) -> Dict[str, Any]:
        """
        Evaluates the sequence at all 5 progressive stages:
        20%, 40%, 60%, 80%, 100% of the growing season.
        """
        stages = [0.2, 0.4, 0.6, 0.8, 1.0]
        calendar_progression = []

        for stg in stages:
            stage_res = self.predict_single_stage(
                observations=observations,
                model_type=model_type,
                stage_fraction=stg,
                target_error_rate=target_error_rate
            )
            calendar_progression.append(stage_res)

        final_stage = calendar_progression[-1]

        return {
            "model_type": model_type,
            "target_error_rate": target_error_rate,
            "final_predicted_crop": final_stage["predicted_crop"],
            "stages": calendar_progression,
            "is_synthetic_demonstration": True
        }
