"""HarvestIQ Pydantic Schemas for Request & Response Validation."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class CropRecommendationRequest(BaseModel):
    """
    Input schema for crop recommendation.
    Validates the 7 agro-climatic parameters within strict physiological boundaries.
    Supports both descriptive names ('nitrogen') and agronomic abbreviations ('N').
    """

    nitrogen: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        validation_alias=AliasChoices("nitrogen", "N"),
        description="Ratio of Nitrogen content in soil (0 – 200 kg/ha)",
    )
    phosphorus: float = Field(
        ...,
        ge=0.0,
        le=200.0,
        validation_alias=AliasChoices("phosphorus", "P"),
        description="Ratio of Phosphorus content in soil (0 – 200 kg/ha)",
    )
    potassium: float = Field(
        ...,
        ge=0.0,
        le=300.0,
        validation_alias=AliasChoices("potassium", "K"),
        description="Ratio of Potassium content in soil (0 – 300 kg/ha)",
    )
    temperature: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        validation_alias=AliasChoices("temperature", "Temperature"),
        description="Ambient temperature in degrees Celsius (0 – 50 °C)",
    )
    humidity: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        validation_alias=AliasChoices("humidity", "Humidity"),
        description="Relative atmospheric humidity percentage (0 – 100 %)",
    )
    ph: float = Field(
        ...,
        ge=3.0,
        le=10.0,
        validation_alias=AliasChoices("ph", "pH", "PH"),
        description="Soil pH value (3.0 – 10.0)",
    )
    rainfall: float = Field(
        ...,
        ge=0.0,
        le=400.0,
        validation_alias=AliasChoices("rainfall", "Rainfall"),
        description="Precipitation / rainfall in millimeters (0 – 400 mm)",
    )

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "nitrogen": 90.0,
                "phosphorus": 42.0,
                "potassium": 43.0,
                "temperature": 20.87,
                "humidity": 82.0,
                "ph": 6.5,
                "rainfall": 202.93,
            }
        },
    )

    def to_feature_list(self) -> List[float]:
        """Convert input parameters to the exact feature order expected by ML pipelines."""
        return [
            self.nitrogen,
            self.phosphorus,
            self.potassium,
            self.temperature,
            self.humidity,
            self.ph,
            self.rainfall,
        ]

    def to_dict(self) -> Dict[str, float]:
        """Export parameters as a standard dictionary."""
        return {
            "nitrogen": self.nitrogen,
            "phosphorus": self.phosphorus,
            "potassium": self.potassium,
            "temperature": self.temperature,
            "humidity": self.humidity,
            "ph": self.ph,
            "rainfall": self.rainfall,
        }


class RecommendationItem(BaseModel):
    """Ranked crop recommendation candidate with raw score and confidence."""

    crop: str = Field(..., description="Recommended crop name")
    score: float = Field(..., description="Uncalibrated model score / probability (0.0 to 1.0)")
    confidence_percentage: float = Field(..., description="Confidence percentage (0.0 to 100.0%)")
    rank: int = Field(..., description="Rank position (1 = primary recommendation)")


class CropAlternative(BaseModel):
    """Secondary crop recommendation candidate with probability score."""

    crop: str = Field(..., description="Crop class label")
    confidence_percentage: float = Field(..., description="Calculated model confidence percentage")


class CropRecommendationResponse(BaseModel):
    """
    Standard response payload for crop recommendation.
    Returns the selected crop, top-three ranked alternatives, model metadata, and uncalibrated scores.
    """

    success: bool = Field(..., description="Whether the recommendation was generated successfully")
    recommended_crop: Optional[str] = Field(None, description="Primary predicted crop label")
    recommendations: List[RecommendationItem] = Field(
        default_factory=list, description="Top-3 ranked crop recommendations with actual model scores"
    )
    model_name: Optional[str] = Field("RandomForestClassifier", description="Selected ML model name")
    model_version: Optional[str] = Field("1.0.0", description="Model version")
    score_type: Optional[str] = Field(
        "uncalibrated model score", description="Type of probability score (uncalibrated ensemble voting)"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of response generation",
    )
    # Backward-compatible fields
    status: str = Field("success", description="Status indicator: 'success' | 'model_unavailable' | 'error'")
    message: str = Field("", description="Informational message or recommendation summary")
    confidence: Optional[float] = Field(None, description="Prediction confidence percentage")
    alternatives: Optional[List[CropAlternative]] = Field(
        default=None, description="Top alternative crop recommendations"
    )
    validated_input: Optional[Dict[str, float]] = Field(None, description="Echo of validated input features")
    model_status: str = Field("ready", description="'ready' | 'not_trained'")


class ModelInfoResponse(BaseModel):
    """Comprehensive metadata and evaluation metrics for the active ML model."""

    model_name: str = Field(..., description="Name of the selected machine learning algorithm")
    model_version: str = Field("1.0.0", description="Production model version")
    is_loaded: bool = Field(..., description="Whether the model is loaded in memory and ready for inference")
    score_type: str = Field("uncalibrated model score", description="Probability calibration status")
    pipeline_steps: List[str] = Field(default_factory=list, description="Ordered stages in the sklearn Pipeline")
    supported_crops: List[str] = Field(default_factory=list, description="List of all 22 supported crop classes")
    classes_count: int = Field(0, description="Total count of recognized crop classes")
    features: List[str] = Field(default_factory=list, description="Input feature names in required order")
    features_count: int = Field(7, description="Number of expected input parameters")
    evaluation_metrics: Optional[Dict[str, Any]] = Field(
        None, description="Empirical evaluation metrics from benchmark validation"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC timestamp of model information query",
    )


class HealthResponse(BaseModel):
    """System health check response schema."""

    status: str = Field("healthy", description="Application operational status")
    service: str = Field("HarvestIQ API", description="Service name")
    version: str = Field("1.0.0", description="API version")
    model_loaded: bool = Field(..., description="Whether a trained ML model is currently loaded in memory")
    model_name: Optional[str] = Field(None, description="Active model algorithm name")
    dataset_present: bool = Field(..., description="Whether a training dataset file exists in backend/data/")
    python_version: str = Field(..., description="Active Python runtime version")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC health check timestamp",
    )


class ParameterBounds(BaseModel):
    """Boundary limits and metadata for a single agro-climatic parameter."""

    min: float
    max: float
    unit: str
    label: str
    description: str


class ParameterBoundsResponse(BaseModel):
    """Metadata response for all supported parameter bounds."""

    parameters: Dict[str, ParameterBounds]
