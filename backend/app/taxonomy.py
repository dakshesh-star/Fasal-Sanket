"""
Fasal Sanket - Taxonomic Resolution Engine & Reliability Policy

Governs progressive crop identification across 4 levels of botanical specificity:
- Level 1: Clade / Agronomic Class (Monocots vs. Dicots)
- Level 2: Botanical Family (Poaceae, Brassicaceae, Malvaceae)
- Level 3: Tribe / Genus (Triticeae, Oryzeae, Maydeae, Brassica, Gossypium)
- Level 4: Species (Triticum aestivum, Oryza sativa, Zea mays, Brassica juncea, Gossypium hirsutum)

Signature Principle:
A model must NEVER be forced to produce a fine-grained classification when empirical
validation evidence does not meet the configured error budget (e.g. max 10% error rate).
Raw model softmax confidence is not treated as proof of reliability.
"""

from typing import Dict, List, Optional, Any
import numpy as np

# Ground-truth biological taxonomy hierarchy for the target crops
TAXONOMY_TREE = {
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

LEVEL_DESCRIPTIONS = {
    1: "Clade / Agronomic Type (Monocot vs Dicot)",
    2: "Botanical Family (e.g., Poaceae, Brassicaceae)",
    3: "Tribe / Genus (e.g., Triticeae, Brassica)",
    4: "Species (Fine-grained agricultural crop identity)"
}


class TaxonomicResolutionEngine:
    """
    Evaluates model predictions against empirical validation evidence
    and a strict error budget to select the highest warranted taxonomic level.
    """

    def __init__(self, target_error_rate: float = 0.10):
        """
        :param target_error_rate: Maximum permissible error rate (default 0.10 -> 90% reliability required)
        """
        self.target_error_rate = target_error_rate
        self.min_reliability_threshold = 1.0 - target_error_rate
        self.validation_evidence: Dict[str, Dict] = {}

    def set_validation_evidence(self, evidence: Dict[str, Dict]):
        """
        Sets empirical validation metrics computed on held-out simulated fields.
        Structure:
        {
            "20%": {"level_1": 0.95, "level_2": 0.88, "level_3": 0.72, "level_4": 0.64},
            "40%": {"level_1": 0.98, "level_2": 0.93, "level_3": 0.85, "level_4": 0.79},
            ...
        }
        """
        self.validation_evidence = evidence

    def get_taxonomy_for_crop(self, crop: str) -> Dict[str, str]:
        """Returns the full 4-level taxonomy path for a crop."""
        return TAXONOMY_TREE.get(crop, {
            "level_1": f"Generic Class (Unknown)",
            "level_2": f"Generic Family (Unknown)",
            "level_3": f"Generic Genus (Unknown)",
            "level_4": f"Generic Species ({crop})"
        })

    def resolve(
        self,
        predicted_crop: str,
        stage_fraction: float,
        model_probabilities: Dict[str, float],
        override_error_rate: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Determines the supported taxonomic resolution level for a given prediction at an observation stage.
        """
        target_error = override_error_rate if override_error_rate is not None else self.target_error_rate
        req_reliability = 1.0 - target_error

        stage_pct = int(round(stage_fraction * 100))
        stage_key = f"{stage_pct}%"

        # Lookup empirical validation evidence for this stage
        stage_evidence = self.validation_evidence.get(stage_key, {})

        taxonomy_path = self.get_taxonomy_for_crop(predicted_crop)

        # Check resolution levels from specific (Level 4) down to broad (Level 1)
        selected_level = 0
        selected_label = "Insufficient Evidence for Any Classification"
        level_reliability = 0.0
        insufficient_evidence_reason = None
        supported_levels = []

        # Iterate 1..4 to check which meet the reliability requirement
        for lvl in [1, 2, 3, 4]:
            lvl_key = f"level_{lvl}"
            emp_rel = stage_evidence.get(lvl_key, 0.0)

            # Check if empirical reliability meets requirement
            meets_req = emp_rel >= req_reliability

            supported_levels.append({
                "level": lvl,
                "name": f"Level {lvl}",
                "description": LEVEL_DESCRIPTIONS[lvl],
                "label": taxonomy_path[lvl_key],
                "empirical_reliability": round(float(emp_rel), 4),
                "required_reliability": round(float(req_reliability), 4),
                "meets_threshold": meets_req
            })

            if meets_req:
                selected_level = lvl
                selected_label = taxonomy_path[lvl_key]
                level_reliability = emp_rel

        # If species level wasn't reached, formulate an explanation
        if selected_level < 4:
            higher_lvl = selected_level + 1
            higher_rel = stage_evidence.get(f"level_{higher_lvl}", 0.0)
            insufficient_evidence_reason = (
                f"Level {higher_lvl} ({LEVEL_DESCRIPTIONS.get(higher_lvl)}) validation reliability "
                f"({higher_rel:.1%}) fails error budget threshold of <= {target_error:.0%} error rate "
                f"(requires >= {req_reliability:.0%} reliability). Withholding fine-grained classification."
            )

        # Compute entropy / uncertainty of the model output
        probs = np.array(list(model_probabilities.values()), dtype=np.float32)
        probs = probs / (np.sum(probs) + 1e-9)
        entropy = -float(np.sum(probs * np.log(probs + 1e-9)))
        max_prob = float(np.max(probs))

        return {
            "observation_progress": stage_pct,
            "observation_fraction": stage_fraction,
            "predicted_crop": predicted_crop,
            "model_confidence": round(max_prob, 4),
            "prediction_entropy": round(entropy, 4),
            "selected_resolution_level": selected_level,
            "selected_resolution_name": f"Level {selected_level}" if selected_level > 0 else "None",
            "selected_taxonomic_label": selected_label,
            "empirical_reliability": round(float(level_reliability), 4),
            "target_error_rate": target_error,
            "required_reliability": round(float(req_reliability), 4),
            "insufficient_evidence_warning": insufficient_evidence_reason,
            "full_taxonomy_path": taxonomy_path,
            "level_breakdown": supported_levels,
            "is_synthetic_demonstration": True
        }

    def generate_calendar(
        self,
        predicted_crops_per_stage: Dict[float, str],
        probabilities_per_stage: Dict[float, Dict[str, float]],
        override_error_rate: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Generates full Taxonomic Resolution Calendar across all observation checkpoints (0.2, 0.4, 0.6, 0.8, 1.0).
        """
        stages = [0.2, 0.4, 0.6, 0.8, 1.0]
        calendar_entries = []

        for stg in stages:
            crop = predicted_crops_per_stage.get(stg, "Wheat")
            probs = probabilities_per_stage.get(stg, {c: 0.2 for c in TAXONOMY_TREE.keys()})
            entry = self.resolve(
                predicted_crop=crop,
                stage_fraction=stg,
                model_probabilities=probs,
                override_error_rate=override_error_rate
            )
            calendar_entries.append(entry)

        return calendar_entries
