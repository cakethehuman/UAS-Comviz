"""Pydantic v2 request/response schemas for the FastAPI inference service."""

from typing import Literal

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    """Structured error detail payload."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """Standard error wrapper returned on validation or runtime errors."""

    error: ErrorDetail


class ImageMetadata(BaseModel):
    """Validated metadata extracted from the uploaded image."""

    filename: str
    content_type: str
    format: str
    width: int
    height: int
    aspect_ratio: float
    size_bytes: int
    original_mode: str


class ClassProbability(BaseModel):
    """Probability summary for a single target class."""

    index: int
    label: str
    probability: float
    std: float | None = None


class FoldPrediction(BaseModel):
    """Single-fold prediction record for ensemble diagnostics."""

    fold: int
    weight_file: str
    predicted_index: int
    predicted_label: str
    confidence: float


class ModelPredictionResult(BaseModel):
    """Prediction, uncertainty metrics, and fold breakdown for one model slot."""

    model_key: Literal["efficientnet_b0", "vit"]
    display_name: str
    architecture: str
    status: Literal["ready", "unavailable", "error"]
    reason: str | None = None
    expected_weights: list[str] = Field(default_factory=list)
    predicted_index: int | None = None
    predicted_label: str | None = None
    confidence: float | None = None
    top2_margin: float | None = None
    normalized_entropy: float | None = None
    low_confidence_warning: bool = False
    warning_reasons: list[str] = Field(default_factory=list)
    inference_time_ms: float | None = None
    fold_count: int = 0
    fold_agreement: str | None = None
    fold_agreement_ratio: float | None = None
    probabilities: list[ClassProbability] = Field(default_factory=list)
    fold_predictions: list[FoldPrediction] = Field(default_factory=list)


class ComparisonSummary(BaseModel):
    """Head-to-head comparison metrics across EfficientNetB0 and ViT."""

    both_available: bool
    models_agree: bool | None = None
    confidence_delta: float | None = None
    latency_delta_ms: float | None = None
    higher_confidence_model: str | None = None
    faster_model: str | None = None


class CompareResponse(BaseModel):
    """Unified side-by-side prediction response."""

    status: Literal["ok"] = "ok"
    classes: list[str]
    image: ImageMetadata
    models: dict[str, ModelPredictionResult]
    comparison: ComparisonSummary


class SinglePredictionResponse(BaseModel):
    """Prediction response for a single model endpoint."""

    status: Literal["ok"] = "ok"
    classes: list[str]
    image: ImageMetadata
    result: ModelPredictionResult


class ModelRegistryInfo(BaseModel):
    """Registry status and metadata for a configured model slot."""

    model_key: str
    display_name: str
    architecture: str
    inductive_bias: str
    status: Literal["ready", "unavailable"]
    fold_count: int
    weight_files: list[str]
    expected_patterns: list[str]
    input_size: tuple[int, int] | None = None
    parameter_count: int | None = None
    reason: str | None = None


class HealthResponse(BaseModel):
    """API health and readiness summary."""

    status: Literal["ok", "degraded"]
    api_version: str
    ready_models: list[str]
    unavailable_models: list[str]
    classes: list[str]
    max_upload_mb: float
    allowed_formats: list[str]


class ModelsResponse(BaseModel):
    """Detailed registry report for all model slots."""

    models: dict[str, ModelRegistryInfo]
    classes: list[str]
