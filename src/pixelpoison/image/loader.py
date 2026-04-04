"""Image loading, validation, and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image, ExifTags

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
MIN_SIZE = 224
MAX_SIZE = 4096
INTERNAL_MAX_SIZE = 1024  # Resize to this for optimization; map perturbation back later


@dataclass
class ImageMeta:
    """Metadata about the loaded image for later reconstruction."""

    original_path: str
    original_size: tuple[int, int]  # (H, W)
    original_format: str  # "JPEG", "PNG", etc.
    scale_factor: float  # 1.0 if no resize, <1.0 if downscaled
    working_size: tuple[int, int]  # (H, W) after any resize
    exif_data: Optional[dict] = field(default_factory=dict)


def load_image(path: str) -> tuple[torch.Tensor, ImageMeta]:
    """Load an image from disk and prepare it for optimization.

    Returns:
        Tuple of (image_tensor, metadata).
        - image_tensor: float32 in [0, 1], shape (1, 3, H, W), NOT normalized.
        - metadata: Original image info for reconstruction.

    Raises:
        ValueError: If image is too small, too large, or unsupported format.
        FileNotFoundError: If path doesn't exist.
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path_obj)

    # Detect format
    fmt = img.format
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format: {fmt}. "
            f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    # Convert to RGB (handle RGBA, grayscale, etc.)
    img = img.convert("RGB")
    w, h = img.size  # PIL uses (width, height)

    # Validate size
    if h < MIN_SIZE or w < MIN_SIZE:
        raise ValueError(
            f"Image too small: {w}x{h}. Minimum size is {MIN_SIZE}x{MIN_SIZE}."
        )
    if h > MAX_SIZE or w > MAX_SIZE:
        raise ValueError(
            f"Image too large: {w}x{h}. Maximum size is {MAX_SIZE}x{MAX_SIZE}."
        )

    # Extract EXIF data (best effort)
    exif_data = {}
    try:
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif_data[tag_name] = str(value)
    except Exception:
        pass

    original_size = (h, w)

    # Resize if larger than internal max
    scale_factor = 1.0
    if h > INTERNAL_MAX_SIZE or w > INTERNAL_MAX_SIZE:
        scale = INTERNAL_MAX_SIZE / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        scale_factor = scale
        h, w = new_h, new_w

    # Convert to tensor
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3) in [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

    meta = ImageMeta(
        original_path=str(path_obj.resolve()),
        original_size=original_size,
        original_format=fmt,
        scale_factor=scale_factor,
        working_size=(h, w),
        exif_data=exif_data,
    )

    return tensor, meta
