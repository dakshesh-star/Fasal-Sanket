"""
Fasal Sanket - Typed Pydantic Schemas for FastAPI
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, field_validator, model_validator
import re
from datetime import date
import math
from app.geometry import polygon_centroid, area_hectares, geometry_from_geojson, is_simple_ring


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall health status")
    models_loaded: Dict[str, bool] = Field(..., description="Load status for models and pipelines")
    device: str = Field(..., description="Compute device in use")
    version: str = Field(..., description="API version")
    is_synthetic: bool = True


class CropInfo(BaseModel):
    id: int
    name: str
    scientific_name: str
    season: str
    agronomic_type: str
    description: str
    taxonomy: Dict[str, str]
    phenology_peak: str
    spectral_profile: Dict[str, float]


class CropsResponse(BaseModel):
    crops: List[CropInfo]
    count: int
    is_synthetic_demonstration: bool = True


class ModelMetadata(BaseModel):
    name: str
    model_type: str
    status: str
    full_season_test_accuracy: Optional[float] = None
    full_season_test_macro_f1: Optional[float] = None
    input_features: List[str]


class ModelsResponse(BaseModel):
    models: List[ModelMetadata]
    available_models: List[str]


class ModelsMetricsResponse(BaseModel):
    dataset_info: Dict[str, Any]
    random_forest: Optional[Dict[str, Any]] = None
    lstm: Optional[Dict[str, Any]] = None
    target_reliability_threshold: float = 0.90
    is_synthetic: bool = True


class PredictRequest(BaseModel):
    model_type: str = Field("random_forest", description="Model to use: 'random_forest' or 'lstm'")
    stage_fraction: float = Field(1.0, ge=0.1, le=1.0, description="Stage of season observed: 0.2, 0.4, 0.6, 0.8, 1.0")
    target_error_rate: float = Field(0.10, ge=0.01, le=0.50, description="Error budget threshold (e.g. 0.10 -> 90% reliability)")
    # observations can be a 2D list of shape (num_timesteps, 9 features)
    observations: List[List[float]] = Field(
        ...,
        description="Temporal sequence of 9 Sentinel-2 features: [B2, B3, B4, B8, B11, B12, NDVI, NDWI, EVI]"
    )


class ProgressivePredictRequest(BaseModel):
    model_type: str = Field("random_forest", description="Model to use: 'random_forest' or 'lstm'")
    target_error_rate: float = Field(0.10, ge=0.01, le=0.50, description="Error budget threshold")
    observations: List[List[float]] = Field(
        ...,
        description="Temporal sequence of 9 Sentinel-2 features: [B2, B3, B4, B8, B11, B12, NDVI, NDWI, EVI]"
    )


class LevelBreakdownItem(BaseModel):
    level: int
    name: str
    description: str
    label: str
    empirical_reliability: float
    required_reliability: float
    meets_threshold: bool


class TaxonomicResolutionResult(BaseModel):
    observation_progress: int
    observation_fraction: float
    predicted_crop: str
    model_confidence: float
    prediction_entropy: float
    selected_resolution_level: int
    selected_resolution_name: str
    selected_taxonomic_label: str
    empirical_reliability: float
    target_error_rate: float
    required_reliability: float
    insufficient_evidence_warning: Optional[str] = None
    full_taxonomy_path: Dict[str, str]
    level_breakdown: List[LevelBreakdownItem]
    is_synthetic_demonstration: bool = True


class PredictResponse(BaseModel):
    model_type: str
    stage_fraction: float
    observation_progress: str
    predicted_crop: str
    probabilities: Dict[str, float]
    taxonomic_resolution: TaxonomicResolutionResult
    is_synthetic_demonstration: bool = True


class ProgressivePredictResponse(BaseModel):
    model_type: str
    target_error_rate: float
    final_predicted_crop: str
    stages: List[PredictResponse]
    is_synthetic_demonstration: bool = True


class CalendarResponse(BaseModel):
    title: str = "Fasal Sanket Taxonomic Resolution Calendar"
    description: str
    target_error_rate: float
    required_reliability: float
    stages: List[Dict[str, Any]]
    crops_represented: List[str]
    is_synthetic_demonstration: bool = True


class ErrorResponse(BaseModel):
    error: str
    detail: str
    remedy: Optional[str] = None


class Credentials(BaseModel):
    email: str = Field(..., min_length=3, max_length=254)
    password: str = Field(..., min_length=8, max_length=128)
    name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    remember_session: bool = True

    @field_validator("email")
    @classmethod
    def validate_email(cls, value):
        value = value.strip().lower()
        if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
            raise ValueError("A valid email address is required")
        return value


class AuthTokens(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]


class UserResponse(BaseModel):
    id: str
    email: str
    created_at: Optional[str] = None
    display_name: Optional[str] = None


class FieldCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    crop: Optional[str] = Field("Unknown / Not identified", max_length=60)
    planting_date: Optional[str] = None
    harvest_date: Optional[str] = None
    boundary: Optional[Dict[str, Any]] = None

    @field_validator("boundary")
    @classmethod
    def validate_boundary(cls, value):
        if value is None:
            return value
        geometry = geometry_from_geojson(value)
        if not geometry or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError("boundary must contain a GeoJSON Polygon or MultiPolygon")
        def ring_ok(ring):
            return (isinstance(ring, list) and len(ring) >= 4 and ring[0] == ring[-1]
                    and all(isinstance(pt, (list, tuple)) and len(pt) >= 2
                            and all(isinstance(v, (int, float)) and math.isfinite(v) for v in pt[:2])
                            and -180 <= pt[0] <= 180 and -90 <= pt[1] <= 90 for pt in ring))
        coords = geometry.get("coordinates")
        if geometry["type"] == "Polygon":
            valid = isinstance(coords, list) and len(coords) > 0 and all(ring_ok(r) for r in coords)
        else:
            valid = isinstance(coords, list) and len(coords) > 0 and all(
                isinstance(poly, list) and len(poly) > 0 and all(ring_ok(r) for r in poly) for poly in coords)
        if not valid:
            raise ValueError("GeoJSON polygon rings need at least four valid WGS84 coordinates and must close")
        rings = [ring for poly in coords for ring in poly] if geometry["type"] == "MultiPolygon" and isinstance(coords, list) else coords if geometry["type"] == "Polygon" and isinstance(coords, list) else []
        if any(not is_simple_ring(ring) for ring in rings):
            raise ValueError("GeoJSON polygon rings cannot cross themselves")
        if not area_hectares(value):
            raise ValueError("GeoJSON polygon must enclose a nonzero area")
        return value

    @model_validator(mode="after")
    def derive_location_from_boundary(self):
        if self.boundary:
            center = polygon_centroid(self.boundary)
            if center:
                if self.latitude is None:
                    self.latitude = center[0]
                if self.longitude is None:
                    self.longitude = center[1]
        if self.latitude is None or self.longitude is None:
            raise ValueError("Latitude and longitude are required when no polygon boundary is provided")
        return self

    @field_validator("name")
    @classmethod
    def validate_name(cls, value):
        if not value.strip():
            raise ValueError("Field name cannot be empty")
        return value.strip()

    @field_validator("planting_date", "harvest_date")
    @classmethod
    def validate_date(cls, value):
        if value is not None:
            try:
                date.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("Dates must use YYYY-MM-DD format") from exc
        return value

    @field_validator("harvest_date")
    @classmethod
    def harvest_after_planting(cls, value, info):
        planting = info.data.get("planting_date")
        if value and planting and value < planting:
            raise ValueError("Harvest date must be on or after planting date")
        return value
