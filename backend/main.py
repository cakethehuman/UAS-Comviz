"""FastAPI application entrypoint for EfficientNetB0 vs ViT classification."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import ALLOWED_PIL_FORMATS, API_VERSION, CLASS_LABELS, MAX_FILE_SIZE_BYTES
from backend.models_service import registry
from backend.schemas import (
    CompareResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    SinglePredictionResponse,
)
from backend.validation import ImageValidationError, validate_and_decode_image


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load and warm up available Keras models on application startup."""
    registry.load_all()
    yield


app = FastAPI(
    title="EfficientNetB0 vs ViT Classifier API",
    version=API_VERSION,
    description="Validated 6-class vegetable classification and head-to-head model comparison API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ImageValidationError)
async def image_validation_exception_handler(
    _request: Request,
    exc: ImageValidationError,
) -> JSONResponse:
    """Return structured JSON errors for upload and image decoding failures."""
    payload = ErrorResponse(error=ErrorDetail(code=exc.code, message=exc.message))
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Return service health, ready model slots, and upload constraints."""
    ready_models, unavailable_models = registry.readiness_lists()
    return HealthResponse(
        status="ok" if ready_models else "degraded",
        api_version=API_VERSION,
        ready_models=ready_models,
        unavailable_models=unavailable_models,
        classes=CLASS_LABELS,
        max_upload_mb=round(MAX_FILE_SIZE_BYTES / (1024 * 1024), 2),
        allowed_formats=sorted(ALLOWED_PIL_FORMATS),
    )


@app.get("/models", response_model=ModelsResponse, tags=["models"])
async def list_models() -> ModelsResponse:
    """Return registry metadata for EfficientNetB0 and ViT model slots."""
    return ModelsResponse(models=registry.get_registry_info(), classes=CLASS_LABELS)


@app.post("/models/reload", response_model=ModelsResponse, tags=["models"])
async def reload_models() -> ModelsResponse:
    """Re-scan backend/model/ and reload weights without restarting the server."""
    registry.load_all()
    return ModelsResponse(models=registry.get_registry_info(), classes=CLASS_LABELS)


@app.post("/predict/compare", response_model=CompareResponse, tags=["predictions"])
async def predict_compare(file: UploadFile = File(...)) -> CompareResponse:
    """Validate image once and run head-to-head inference across EfficientNetB0 and ViT."""
    file_bytes = await file.read()
    rgb_image, metadata = validate_and_decode_image(
        file_bytes,
        file.filename,
        file.content_type,
    )
    models, comparison = registry.compare(rgb_image)
    return CompareResponse(
        classes=CLASS_LABELS,
        image=metadata,
        models=models,
        comparison=comparison,
    )


async def _predict_single(model_key: str, file: UploadFile) -> SinglePredictionResponse:
    """Validate uploaded image and predict with a single named model slot."""
    file_bytes = await file.read()
    rgb_image, metadata = validate_and_decode_image(
        file_bytes,
        file.filename,
        file.content_type,
    )
    result = registry.predict_model(model_key, rgb_image)
    if result.status == "unavailable":
        raise HTTPException(
            status_code=503,
            detail={
                "code": "MODEL_UNAVAILABLE",
                "message": result.reason or f"Model '{model_key}' weights are not loaded.",
                "expected_weights": result.expected_weights,
            },
        )
    if result.status == "error":
        raise HTTPException(
            status_code=500,
            detail={
                "code": "INFERENCE_ERROR",
                "message": result.reason or f"Inference failed for model '{model_key}'.",
            },
        )
    return SinglePredictionResponse(classes=CLASS_LABELS, image=metadata, result=result)


@app.post("/predict/efficientnet", response_model=SinglePredictionResponse, tags=["predictions"])
async def predict_efficientnet(file: UploadFile = File(...)) -> SinglePredictionResponse:
    """Run inference using only the EfficientNetB0 ensemble."""
    return await _predict_single("efficientnet_b0", file)


@app.post("/predict/vit", response_model=SinglePredictionResponse, tags=["predictions"])
async def predict_vit(file: UploadFile = File(...)) -> SinglePredictionResponse:
    """Run inference using only the Vision Transformer (ViT) slot."""
    return await _predict_single("vit", file)


@app.post("/predict_ef", response_model=SinglePredictionResponse, tags=["predictions"])
async def predict_ef_legacy(file: UploadFile = File(...)) -> SinglePredictionResponse:
    """Backwards-compatible route delegating to EfficientNetB0."""
    return await _predict_single("efficientnet_b0", file)
