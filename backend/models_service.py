"""Thread-safe model registry, warm-up, ensemble inference, and uncertainty diagnostics."""

from dataclasses import dataclass, field
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
from PIL import Image
import tensorflow as tf

from backend.config import (
    CLASS_LABELS,
    DEFAULT_IMAGE_SIZE,
    HIGH_ENTROPY_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD,
    MIN_FOLD_AGREEMENT_RATIO,
    MODEL_DIR,
    MODEL_SPECS,
)
from backend.schemas import (
    ClassProbability,
    ComparisonSummary,
    FoldPrediction,
    ModelPredictionResult,
    ModelRegistryInfo,
)
from backend.validation import preprocess_for_model


@dataclass
class _LoadedModelSlot:
    """Internal representation of a loaded model slot (single or multi-fold)."""

    model_key: str
    models: list[Any] = field(default_factory=list)
    weight_files: list[str] = field(default_factory=list)
    input_size: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
    parameter_count: int | None = None
    load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return len(self.models) > 0 and self.load_error is None


def _ensure_optional_keras_hub() -> None:
    """Import keras_hub if available so KerasHub ViT layers deserialize automatically."""
    try:
        import keras_hub  # noqa: F401
    except ImportError:
        pass


def _extract_input_size(model: Any) -> tuple[int, int]:
    """Extract static (height, width) from a Keras model's input_shape."""
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list) and shape:
        shape = shape[0]
    if isinstance(shape, tuple) and len(shape) >= 4:
        h, w = shape[1], shape[2]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return (h, w)
    return (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)


def _to_probabilities(vector: np.ndarray) -> np.ndarray:
    """Validate finite 6-class output and normalize logits/probabilities safely."""
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    if arr.shape != (len(CLASS_LABELS),) or not np.all(np.isfinite(arr)):
        raise ValueError(
            f"Unexpected model output shape {arr.shape}; expected ({len(CLASS_LABELS)},) finite values."
        )

    if np.any(arr < 0.0) or abs(float(np.sum(arr)) - 1.0) > 1e-2:
        shifted = arr - np.max(arr)
        exp_vals = np.exp(shifted)
        return exp_vals / np.sum(exp_vals)

    total = float(np.sum(arr))
    if total <= 0.0:
        return np.full(len(CLASS_LABELS), 1.0 / len(CLASS_LABELS), dtype=np.float64)
    return arr / total


class ModelRegistry:
    """Manages EfficientNetB0 and ViT weights, warm-up, and ensemble predictions."""

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self._model_dir = model_dir
        self._lock = threading.RLock()
        self._slots: dict[str, _LoadedModelSlot] = {}

    def _discover_weights(self, patterns: list[str]) -> list[Path]:
        """Find unique matching weight files sorted deterministically."""
        if not self._model_dir.exists():
            return []
        found: dict[Path, None] = {}
        for pattern in patterns:
            for candidate in sorted(self._model_dir.glob(pattern)):
                if candidate.is_file():
                    found[candidate.resolve()] = None
        return list(found.keys())

    def load_all(self) -> None:
        """Discover, load, validate, and warm up all configured model slots."""
        new_slots: dict[str, _LoadedModelSlot] = {}
        keras_hub_loaded = False

        for model_key, spec in MODEL_SPECS.items():
            weight_paths = self._discover_weights(spec["patterns"])
            slot = _LoadedModelSlot(model_key=model_key)

            if not weight_paths:
                slot.load_error = spec["expected_hint"]
                new_slots[model_key] = slot
                continue

            if model_key == "vit" and not keras_hub_loaded:
                _ensure_optional_keras_hub()
                keras_hub_loaded = True

            try:
                loaded_models: list[Any] = []
                weight_names: list[str] = []
                slot_input_size: tuple[int, int] | None = None

                for path in weight_paths:
                    try:
                        model = tf.keras.models.load_model(path, compile=False)
                    except Exception:
                        if keras_hub_loaded:
                            raise
                        _ensure_optional_keras_hub()
                        keras_hub_loaded = True
                        model = tf.keras.models.load_model(path, compile=False)

                    out_shape = getattr(model, "output_shape", None)
                    if not isinstance(out_shape, tuple) or out_shape[-1] != len(CLASS_LABELS):
                        raise ValueError(
                            f"Model '{path.name}' output shape {out_shape} does not match {len(CLASS_LABELS)} classes."
                        )

                    input_size = _extract_input_size(model)
                    if slot_input_size is None:
                        slot_input_size = input_size

                    dummy = np.zeros((1, input_size[0], input_size[1], 3), dtype=np.float32)
                    _ = model(dummy, training=False)

                    loaded_models.append(model)
                    weight_names.append(path.name)

                slot.models = loaded_models
                slot.weight_files = weight_names
                slot.input_size = slot_input_size or (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE)
                slot.parameter_count = int(loaded_models[0].count_params())
            except Exception as exc:  # noqa: BLE001
                slot.models = []
                slot.weight_files = []
                slot.load_error = f"Failed to load {spec['display_name']} weights: {exc}"

            new_slots[model_key] = slot

        with self._lock:
            self._slots = new_slots

    def get_registry_info(self) -> dict[str, ModelRegistryInfo]:
        """Return public metadata for every configured model slot."""
        with self._lock:
            slots_snapshot = dict(self._slots)

        report: dict[str, ModelRegistryInfo] = {}
        for model_key, spec in MODEL_SPECS.items():
            slot = slots_snapshot.get(model_key)
            is_ready = bool(slot and slot.is_ready)
            report[model_key] = ModelRegistryInfo(
                model_key=model_key,
                display_name=spec["display_name"],
                architecture=spec["architecture"],
                inductive_bias=spec["inductive_bias"],
                status="ready" if is_ready else "unavailable",
                fold_count=len(slot.models) if is_ready and slot else 0,
                weight_files=list(slot.weight_files) if is_ready and slot else [],
                expected_patterns=list(spec["patterns"]),
                input_size=slot.input_size if is_ready and slot else None,
                parameter_count=slot.parameter_count if is_ready and slot else None,
                reason=None if is_ready else (slot.load_error if slot else spec["expected_hint"]),
            )
        return report

    def readiness_lists(self) -> tuple[list[str], list[str]]:
        """Return (ready_model_keys, unavailable_model_keys)."""
        info = self.get_registry_info()
        ready = [key for key, item in info.items() if item.status == "ready"]
        unavailable = [key for key, item in info.items() if item.status != "ready"]
        return ready, unavailable

    def predict_model(self, model_key: str, rgb_image: Image.Image) -> ModelPredictionResult:
        """Run inference and uncertainty diagnostics for a single model slot."""
        spec = MODEL_SPECS[model_key]
        with self._lock:
            slot = self._slots.get(model_key)
            if slot is None or not slot.is_ready:
                return ModelPredictionResult(
                    model_key=model_key,  # type: ignore[arg-type]
                    display_name=spec["display_name"],
                    architecture=spec["architecture"],
                    status="unavailable",
                    reason=slot.load_error if slot else spec["expected_hint"],
                    expected_weights=list(spec["patterns"]),
                )
            models = list(slot.models)
            weight_files = list(slot.weight_files)
            input_size = slot.input_size

        batch = preprocess_for_model(rgb_image, target_size=input_size)

        try:
            t0 = time.perf_counter()
            raw_outputs = [model(batch, training=False).numpy()[0] for model in models]
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            fold_probs = np.stack([_to_probabilities(vec) for vec in raw_outputs], axis=0)
        except Exception as exc:  # noqa: BLE001
            return ModelPredictionResult(
                model_key=model_key,  # type: ignore[arg-type]
                display_name=spec["display_name"],
                architecture=spec["architecture"],
                status="error",
                reason=f"Inference failed: {exc}",
                expected_weights=list(spec["patterns"]),
            )

        fold_count = fold_probs.shape[0]
        mean_probs = np.mean(fold_probs, axis=0)
        mean_probs = mean_probs / np.sum(mean_probs)
        std_probs = np.std(fold_probs, axis=0) if fold_count > 1 else None

        predicted_idx = int(np.argmax(mean_probs))
        predicted_label = CLASS_LABELS[predicted_idx]
        confidence = float(mean_probs[predicted_idx])

        sorted_probs = np.sort(mean_probs)[::-1]
        top2_margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

        max_entropy = math.log(len(CLASS_LABELS))
        entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-12)))
        normalized_entropy = max(0.0, min(1.0, entropy / max_entropy))

        fold_predictions: list[FoldPrediction] = []
        agree_count = 0
        for idx, (weight_name, probs) in enumerate(zip(weight_files, fold_probs)):
            fold_pred_idx = int(np.argmax(probs))
            if fold_pred_idx == predicted_idx:
                agree_count += 1
            fold_predictions.append(
                FoldPrediction(
                    fold=idx,
                    weight_file=weight_name,
                    predicted_index=fold_pred_idx,
                    predicted_label=CLASS_LABELS[fold_pred_idx],
                    confidence=round(float(probs[fold_pred_idx]), 4),
                )
            )

        fold_agreement_ratio = agree_count / fold_count
        fold_agreement = f"{agree_count}/{fold_count}"

        warning_reasons: list[str] = []
        if confidence < LOW_CONFIDENCE_THRESHOLD:
            warning_reasons.append(
                f"Low top-1 confidence ({confidence * 100:.1f}% < {LOW_CONFIDENCE_THRESHOLD * 100:.0f}%)"
            )
        if normalized_entropy > HIGH_ENTROPY_THRESHOLD:
            warning_reasons.append(
                f"High predictive uncertainty (entropy {normalized_entropy:.2f} > {HIGH_ENTROPY_THRESHOLD:.2f})"
            )
        if fold_count > 1 and fold_agreement_ratio < MIN_FOLD_AGREEMENT_RATIO:
            warning_reasons.append(
                f"Weak fold consensus ({fold_agreement} folds agree on '{predicted_label}')"
            )

        probabilities = [
            ClassProbability(
                index=idx,
                label=label,
                probability=round(float(mean_probs[idx]), 6),
                std=round(float(std_probs[idx]), 6) if std_probs is not None else None,
            )
            for idx, label in enumerate(CLASS_LABELS)
        ]
        probabilities.sort(key=lambda item: item.probability, reverse=True)

        return ModelPredictionResult(
            model_key=model_key,  # type: ignore[arg-type]
            display_name=spec["display_name"],
            architecture=spec["architecture"],
            status="ready",
            expected_weights=list(spec["patterns"]),
            predicted_index=predicted_idx,
            predicted_label=predicted_label,
            confidence=round(confidence, 6),
            top2_margin=round(top2_margin, 6),
            normalized_entropy=round(normalized_entropy, 6),
            low_confidence_warning=len(warning_reasons) > 0,
            warning_reasons=warning_reasons,
            inference_time_ms=round(elapsed_ms, 2),
            fold_count=fold_count,
            fold_agreement=fold_agreement,
            fold_agreement_ratio=round(fold_agreement_ratio, 4),
            probabilities=probabilities,
            fold_predictions=fold_predictions,
        )

    def compare(
        self,
        rgb_image: Image.Image,
    ) -> tuple[dict[str, ModelPredictionResult], ComparisonSummary]:
        """Run both EfficientNetB0 and ViT slots and synthesize head-to-head comparison metrics."""
        ef_res = self.predict_model("efficientnet_b0", rgb_image)
        vit_res = self.predict_model("vit", rgb_image)
        models = {"efficientnet_b0": ef_res, "vit": vit_res}

        if ef_res.status == "ready" and vit_res.status == "ready":
            ef_conf = ef_res.confidence or 0.0
            vit_conf = vit_res.confidence or 0.0
            ef_ms = ef_res.inference_time_ms or 0.0
            vit_ms = vit_res.inference_time_ms or 0.0

            higher_conf = (
                "tie"
                if abs(ef_conf - vit_conf) < 1e-4
                else ("efficientnet_b0" if ef_conf > vit_conf else "vit")
            )
            faster = (
                "tie"
                if abs(ef_ms - vit_ms) < 0.5
                else ("efficientnet_b0" if ef_ms < vit_ms else "vit")
            )
            summary = ComparisonSummary(
                both_available=True,
                models_agree=ef_res.predicted_index == vit_res.predicted_index,
                confidence_delta=round(ef_conf - vit_conf, 6),
                latency_delta_ms=round(ef_ms - vit_ms, 2),
                higher_confidence_model=higher_conf,
                faster_model=faster,
            )
        else:
            summary = ComparisonSummary(both_available=False)

        return models, summary


registry = ModelRegistry()
