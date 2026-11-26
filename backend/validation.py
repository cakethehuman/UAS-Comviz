"""Image upload validation, safe decoding, and preprocessing utilities."""

from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    ALLOWED_PIL_FORMATS,
    DEFAULT_IMAGE_SIZE,
    MAX_FILE_SIZE_BYTES,
    MAX_IMAGE_DIM,
    MAX_IMAGE_PIXELS,
    MIN_IMAGE_DIM,
)
from backend.schemas import ImageMetadata


class ImageValidationError(Exception):
    """Domain exception raised when an uploaded image fails validation."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _normalize_content_type(content_type: str | None) -> str:
    """Strip parameters and lowercase MIME content types."""
    if not content_type:
        return ""
    return content_type.split(";", 1)[0].strip().lower()


def validate_and_decode_image(
    file_bytes: bytes,
    filename: str | None,
    content_type: str | None,
) -> tuple[Image.Image, ImageMetadata]:
    """Validate raw upload bytes, decode safely, fix EXIF orientation, and convert to RGB."""
    if not file_bytes:
        raise ImageValidationError(
            status_code=400,
            code="EMPTY_FILE",
            message="Uploaded file is empty.",
        )

    size_bytes = len(file_bytes)
    if size_bytes > MAX_FILE_SIZE_BYTES:
        max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
        raise ImageValidationError(
            status_code=413,
            code="FILE_TOO_LARGE",
            message=f"File size ({size_bytes / (1024 * 1024):.2f} MB) exceeds the {max_mb:.0f} MB limit.",
        )

    clean_filename = Path(filename).name if filename else "upload"
    suffix = Path(clean_filename).suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ImageValidationError(
            status_code=415,
            code="UNSUPPORTED_FILE_EXTENSION",
            message=f"Unsupported file extension '{suffix}'. Allowed extensions: {allowed}.",
        )

    norm_ct = _normalize_content_type(content_type)
    if norm_ct and norm_ct != "application/octet-stream" and norm_ct not in ALLOWED_CONTENT_TYPES:
        raise ImageValidationError(
            status_code=415,
            code="UNSUPPORTED_MEDIA_TYPE",
            message=f"Unsupported content type '{norm_ct}'. Allowed types: JPEG, PNG, WEBP.",
        )

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

    try:
        img = Image.open(BytesIO(file_bytes))
    except Image.DecompressionBombError as exc:
        raise ImageValidationError(
            status_code=413,
            code="IMAGE_TOO_LARGE_PIXELS",
            message="Image pixel count exceeds the safety limit.",
        ) from exc
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(
            status_code=422,
            code="CORRUPT_IMAGE",
            message="Unable to identify or decode image file. Ensure it is a valid JPEG, PNG, or WEBP.",
        ) from exc

    img_format = (img.format or "").upper()
    original_mode = img.mode
    if img_format not in ALLOWED_PIL_FORMATS:
        raise ImageValidationError(
            status_code=415,
            code="UNSUPPORTED_IMAGE_FORMAT",
            message=f"Unsupported image format '{img_format or 'unknown'}'. Allowed formats: JPEG, PNG, WEBP.",
        )

    try:
        img.load()
    except (Image.DecompressionBombError,) as exc:
        raise ImageValidationError(
            status_code=413,
            code="IMAGE_TOO_LARGE_PIXELS",
            message="Image pixel count exceeds the safety limit.",
        ) from exc
    except OSError as exc:
        raise ImageValidationError(
            status_code=422,
            code="CORRUPT_IMAGE",
            message="Image data is truncated or corrupted.",
        ) from exc

    img = ImageOps.exif_transpose(img)
    width, height = img.size

    if width < MIN_IMAGE_DIM or height < MIN_IMAGE_DIM or width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
        raise ImageValidationError(
            status_code=422,
            code="INVALID_DIMENSIONS",
            message=(
                f"Image dimensions ({width}x{height}) must be between "
                f"{MIN_IMAGE_DIM}x{MIN_IMAGE_DIM} and {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM} pixels."
            ),
        )

    if "A" in img.mode or "transparency" in img.info:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        rgb_image = background
    else:
        rgb_image = img.convert("RGB")

    metadata = ImageMetadata(
        filename=clean_filename,
        content_type=norm_ct or f"image/{img_format.lower()}",
        format=img_format,
        width=width,
        height=height,
        aspect_ratio=round(width / height, 3),
        size_bytes=size_bytes,
        original_mode=original_mode,
    )
    return rgb_image, metadata


def preprocess_for_model(
    rgb_image: Image.Image,
    target_size: tuple[int, int] = (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE),
) -> np.ndarray:
    """Resize RGB image with area interpolation while preserving [0, 255] float32 range."""
    image_array = np.asarray(rgb_image, dtype=np.float32)
    resized = tf.image.resize(
        image_array,
        [target_size[0], target_size[1]],
        method="area",
    ).numpy()
    return np.expand_dims(resized, axis=0).astype(np.float32, copy=False)
