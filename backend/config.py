"""Configuration constants and model specifications for the classifier backend."""

from pathlib import Path

API_VERSION = "1.0.0"

CLASS_LABELS: list[str] = [
    "Bean",
    "Bitter melon",
    "Cucumber",
    "Green Chili",
    "Ladies finger",
    "Pointed gourd",
]

DEFAULT_IMAGE_SIZE = 224
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
MIN_IMAGE_DIM = 32
MAX_IMAGE_DIM = 8192
MAX_IMAGE_PIXELS = 40_000_000

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}

LOW_CONFIDENCE_THRESHOLD = 0.60
HIGH_ENTROPY_THRESHOLD = 0.65
MIN_FOLD_AGREEMENT_RATIO = 0.60

MODEL_DIR = Path(__file__).resolve().parent / "model"

MODEL_SPECS: dict[str, dict] = {
    "efficientnet_b0": {
        "display_name": "EfficientNetB0",
        "architecture": "Convolutional Neural Network (CNN)",
        "inductive_bias": "Local translation invariance, depthwise separable convolutions & compound scaling",
        "patterns": ["efb0_*.keras", "efb0.keras"],
        "expected_hint": "Place efb0_0.keras..efb0_4.keras or efb0.keras in backend/model/",
    },
    "vit": {
        "display_name": "Vision Transformer (ViT)",
        "architecture": "Transformer Self-Attention over Image Patches",
        "inductive_bias": "Global patch-to-patch self-attention & long-range spatial context",
        "patterns": ["vit_*.keras", "vit.keras"],
        "expected_hint": "Place vit_0.keras..vit_4.keras or vit.keras in backend/model/ and reload",
    },
}
