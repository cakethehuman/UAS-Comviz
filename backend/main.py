from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from io import BytesIO
import os
from pathlib import Path
import threading
import time
import warnings

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
import keras_hub  # noqa: F401
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
import tensorflow as tf

warnings.filterwarnings("ignore", message=".*structure of `inputs`.*", category=UserWarning)

API_VERSION = "2.0.0"
CLASSES: tuple[str, ...] = (
    "Bean",
    "Bitter melon",
    "Cucumber",
    "Green Chili",
    "Ladies finger",
    "Pointed gourd",
)
IMAGE_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.60
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent / "model")))


class PredictionResponse(BaseModel):
    """Single-model ensemble prediction DTO."""

    model: str
    label: str
    confidence: float
    low_confidence: bool
    inference_ms: float
    probabilities: dict[str, float]


class CompareResponse(BaseModel):
    """Side-by-side comparison DTO across all loaded models."""

    agree: bool
    results: dict[str, PredictionResponse]


class HealthResponse(BaseModel):
    """Service readiness and loaded fold counts DTO."""

    api_version: str = API_VERSION
    status: str
    model_count: int
    models: dict[str, int]
    classes: list[str]


def preprocess_bytes(image_bytes: bytes) -> tf.Tensor:
    """Decode uploaded image bytes and match the exact [0, 255] float32 training pipeline."""
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        with Image.open(BytesIO(image_bytes)) as img:
            img.load()
            rgb_img = ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=400, detail="Invalid or corrupt image file.") from exc

    img_tensor = tf.convert_to_tensor(np.asarray(rgb_img, dtype=np.uint8))
    resized = tf.image.resize(img_tensor, [IMAGE_SIZE, IMAGE_SIZE])
    img_u8 = tf.cast(tf.round(resized), tf.uint8)
    img_f32 = tf.cast(img_u8, tf.float32)
    return tf.expand_dims(img_f32, axis=0)


class InferenceStrategy(ABC):
    """Abstract inference strategy for a model family."""

    name: str

    @property
    @abstractmethod
    def fold_count(self) -> int:
        """Number of loaded fold models."""

    @abstractmethod
    def load(self) -> None:
        """Load and warm up checkpoints from disk."""

    @abstractmethod
    def predict(self, x: tf.Tensor) -> PredictionResponse:
        """Run inference on a preprocessed (1, 224, 224, 3) float32 tensor."""


class KerasEnsembleStrategy(InferenceStrategy):
    """5-fold Keras model averaging strategy."""

    def __init__(self, name: str, prefix: str, model_dir: Path) -> None:
        self.name = name
        self._prefix = prefix
        self._model_dir = model_dir
        self._models: list[tf.keras.Model] = []

    @property
    def fold_count(self) -> int:
        return len(self._models)

    def _discover_paths(self) -> list[Path]:
        if not self._model_dir.exists():
            return []
        ordered = [self._model_dir / f"{self._prefix}_fold{i}.keras" for i in range(5)]
        existing = [p for p in ordered if p.is_file()]
        return existing if existing else sorted(self._model_dir.glob(f"{self._prefix}_fold*.keras"))

    def load(self) -> None:
        loaded: list[tf.keras.Model] = []
        dummy = tf.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32)
        for path in self._discover_paths():
            model = tf.keras.models.load_model(path, compile=False)
            _ = model(dummy, training=False)
            loaded.append(model)
        self._models = loaded

    def predict(self, x: tf.Tensor) -> PredictionResponse:
        if not self._models:
            raise HTTPException(status_code=503, detail=f"{self.name} models are not loaded.")

        t0 = time.perf_counter()
        fold_probs = [model(x, training=False).numpy()[0] for model in self._models]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        mean_probs = np.mean(fold_probs, axis=0)
        idx = int(np.argmax(mean_probs))
        conf = float(mean_probs[idx])

        return PredictionResponse(
            model=self.name,
            label=CLASSES[idx],
            confidence=round(conf, 4),
            low_confidence=conf < LOW_CONFIDENCE_THRESHOLD,
            inference_ms=round(elapsed_ms, 2),
            probabilities={cls: round(float(mean_probs[i]), 4) for i, cls in enumerate(CLASSES)},
        )


class ModelRegistry:
    """Thread-safe singleton registry managing DINOv2 and EfficientNetB0 strategies."""

    _ALIASES: dict[str, str] = {
        "dinov2": "dinov2",
        "deno": "dinov2",
        "dino": "dinov2",
        "efb0": "efb0",
        "efficientnet": "efb0",
        "efficientnet_b0": "efb0",
    }

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self._load_lock = threading.RLock()
        self._infer_lock = threading.Lock()
        self._strategies: dict[str, InferenceStrategy] = {
            "dinov2": KerasEnsembleStrategy("DINOv2", "dinov2", model_dir),
            "efb0": KerasEnsembleStrategy("EfficientNetB0", "efb0", model_dir),
        }

    def load_all(self) -> None:
        with self._load_lock:
            for strategy in self._strategies.values():
                strategy.load()

    def resolve(self, key: str) -> InferenceStrategy:
        canonical = self._ALIASES.get(key.lower().strip())
        if canonical is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model '{key}'. Use 'dinov2' (or 'deno') or 'efb0'.",
            )
        return self._strategies[canonical]

    def strategies(self) -> dict[str, InferenceStrategy]:
        return dict(self._strategies)

    def fold_counts(self) -> dict[str, int]:
        return {key: strategy.fold_count for key, strategy in self._strategies.items()}

    def predict_one(self, model_key: str, image_bytes: bytes) -> PredictionResponse:
        tensor = preprocess_bytes(image_bytes)
        strategy = self.resolve(model_key)
        with self._infer_lock:
            return strategy.predict(tensor)

    def compare_all(self, image_bytes: bytes) -> CompareResponse:
        tensor = preprocess_bytes(image_bytes)
        with self._infer_lock:
            results = {
                key: strategy.predict(tensor)
                for key, strategy in self._strategies.items()
                if strategy.fold_count > 0
            }
        if not results:
            raise HTTPException(status_code=503, detail="No models are loaded.")
        agree = len({res.label for res in results.values()}) == 1
        return CompareResponse(agree=agree, results=results)


_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """FastAPI dependency returning the singleton ModelRegistry."""
    return _registry


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _registry.load_all()
    yield


app = FastAPI(title="Green Vegetable Classifier API", version=API_VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health(registry: ModelRegistry = Depends(get_registry)) -> HealthResponse:
    counts = registry.fold_counts()
    total = sum(counts.values())
    return HealthResponse(
        api_version=API_VERSION,
        status="ok" if total > 0 else "degraded",
        model_count=total,
        models=counts,
        classes=list(CLASSES),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = Query("dinov2", description="Model selector: dinov2 (deno) or efb0"),
    registry: ModelRegistry = Depends(get_registry),
) -> PredictionResponse:
    image_bytes = await file.read()
    return await run_in_threadpool(registry.predict_one, model, image_bytes)


@app.post("/predict/compare", response_model=CompareResponse)
async def predict_compare(
    file: UploadFile = File(...),
    registry: ModelRegistry = Depends(get_registry),
) -> CompareResponse:
    image_bytes = await file.read()
    return await run_in_threadpool(registry.compare_all, image_bytes)
