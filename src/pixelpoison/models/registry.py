"""Model registry — specs, download, caching for CLIP and SigLIP models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a single surrogate model."""

    id: str  # Internal identifier (e.g., "clip-vit-b-32")
    display_name: str  # Human-readable (e.g., "CLIP ViT-B/32")
    open_clip_model: str  # open_clip model name
    pretrained: str  # open_clip pretrained tag
    family: str  # "clip" or "siglip"
    size_mb: int  # Approximate download size
    min_tier: int  # Minimum tier required
    norm_mean: tuple[float, float, float]
    norm_std: tuple[float, float, float]


# CLIP normalization constants (ImageNet-derived)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# SigLIP normalization constants
_SIGLIP_MEAN = (0.5, 0.5, 0.5)
_SIGLIP_STD = (0.5, 0.5, 0.5)

MODELS: dict[str, ModelSpec] = {
    "clip-vit-b-32": ModelSpec(
        id="clip-vit-b-32",
        display_name="CLIP ViT-B/32",
        open_clip_model="ViT-B-32",
        pretrained="openai",
        family="clip",
        size_mb=400,
        min_tier=1,
        norm_mean=_CLIP_MEAN,
        norm_std=_CLIP_STD,
    ),
    "clip-vit-b-16": ModelSpec(
        id="clip-vit-b-16",
        display_name="CLIP ViT-B/16",
        open_clip_model="ViT-B-16",
        pretrained="openai",
        family="clip",
        size_mb=600,
        min_tier=2,
        norm_mean=_CLIP_MEAN,
        norm_std=_CLIP_STD,
    ),
    "clip-vit-l-14": ModelSpec(
        id="clip-vit-l-14",
        display_name="CLIP ViT-L/14",
        open_clip_model="ViT-L-14",
        pretrained="openai",
        family="clip",
        size_mb=1700,
        min_tier=2,
        norm_mean=_CLIP_MEAN,
        norm_std=_CLIP_STD,
    ),
    "siglip-vit-so400m": ModelSpec(
        id="siglip-vit-so400m",
        display_name="SigLIP ViT-SO400M/14",
        open_clip_model="ViT-SO400M-14-SigLIP2",
        pretrained="webli",
        family="siglip",
        size_mb=1500,
        min_tier=2,
        norm_mean=_SIGLIP_MEAN,
        norm_std=_SIGLIP_STD,
    ),
    "clip-vit-l-14-336": ModelSpec(
        id="clip-vit-l-14-336",
        display_name="CLIP ViT-L/14@336px",
        open_clip_model="ViT-L-14-336",
        pretrained="openai",
        family="clip",
        size_mb=1700,
        min_tier=3,
        norm_mean=_CLIP_MEAN,
        norm_std=_CLIP_STD,
    ),
}


def get_models_for_tier(tier: int) -> list[ModelSpec]:
    """Return all models available at the given tier (cumulative)."""
    return [m for m in MODELS.values() if m.min_tier <= tier]


def get_cache_dir(custom_dir: Optional[str] = None) -> Path:
    """Return the model cache directory."""
    if custom_dir:
        return Path(custom_dir)
    return Path.home() / ".pixelpoison" / "models"


def get_total_download_size(tier: int) -> int:
    """Return total download size in MB for all models at a tier."""
    return sum(m.size_mb for m in get_models_for_tier(tier))
