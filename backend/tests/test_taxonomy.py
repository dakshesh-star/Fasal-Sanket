"""
Tests for Taxonomic Resolution Engine and Reliability Policy
"""

import pytest
import sys
import os

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.taxonomy import TaxonomicResolutionEngine, TAXONOMY_TREE


def test_taxonomy_error_budget_enforcement():
    # Target error rate 10% requires >= 90% empirical reliability
    engine = TaxonomicResolutionEngine(target_error_rate=0.10)

    # Simulated validation evidence: Level 1 and 2 pass threshold, Level 3 and 4 fail
    engine.set_validation_evidence({
        "40%": {
            "level_1": 0.96,  # Passes (>= 0.90)
            "level_2": 0.91,  # Passes (>= 0.90)
            "level_3": 0.82,  # Fails (< 0.90)
            "level_4": 0.74   # Fails (< 0.90)
        }
    })

    result = engine.resolve(
        predicted_crop="Wheat",
        stage_fraction=0.40,
        model_probabilities={"Wheat": 0.85, "Rice": 0.05, "Maize": 0.04, "Mustard": 0.03, "Cotton": 0.03}
    )

    # Must select Level 2 (Family), NOT Level 4 (Species) even though model had 85% raw confidence
    assert result["selected_resolution_level"] == 2
    assert "Poaceae" in result["selected_taxonomic_label"]
    assert result["insufficient_evidence_warning"] is not None
    assert "Level 3" in result["insufficient_evidence_warning"]


def test_taxonomy_full_season_resolution():
    engine = TaxonomicResolutionEngine(target_error_rate=0.10)
    engine.set_validation_evidence({
        "100%": {
            "level_1": 0.99,
            "level_2": 0.98,
            "level_3": 0.96,
            "level_4": 0.94   # Passes >= 0.90
        }
    })

    result = engine.resolve(
        predicted_crop="Mustard",
        stage_fraction=1.0,
        model_probabilities={"Mustard": 0.92, "Wheat": 0.02, "Rice": 0.02, "Maize": 0.02, "Cotton": 0.02}
    )

    assert result["selected_resolution_level"] == 4
    assert "Brassica juncea" in result["selected_taxonomic_label"]
    assert result["insufficient_evidence_warning"] is None


def test_taxonomy_calendar_generation():
    engine = TaxonomicResolutionEngine(target_error_rate=0.10)
    engine.set_validation_evidence({
        "20%": {"level_1": 0.92, "level_2": 0.75, "level_3": 0.60, "level_4": 0.52},
        "40%": {"level_1": 0.95, "level_2": 0.91, "level_3": 0.78, "level_4": 0.70},
        "60%": {"level_1": 0.98, "level_2": 0.95, "level_3": 0.91, "level_4": 0.85},
        "80%": {"level_1": 0.99, "level_2": 0.98, "level_3": 0.94, "level_4": 0.92},
        "100%": {"level_1": 0.99, "level_2": 0.99, "level_3": 0.97, "level_4": 0.95}
    })

    calendar = engine.generate_calendar(
        predicted_crops_per_stage={0.2: "Wheat", 0.4: "Wheat", 0.6: "Wheat", 0.8: "Wheat", 1.0: "Wheat"},
        probabilities_per_stage={stg: {"Wheat": 0.8} for stg in [0.2, 0.4, 0.6, 0.8, 1.0]}
    )

    assert len(calendar) == 5
    # Check progressive resolution levels increasing: 1 -> 2 -> 3 -> 4 -> 4
    levels = [entry["selected_resolution_level"] for entry in calendar]
    assert levels == [1, 2, 3, 4, 4]
