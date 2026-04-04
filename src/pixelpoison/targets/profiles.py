"""Target VLM preprocessing profiles for target-aware optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class TargetProfile:
    """Preprocessing profile for a target VLM."""

    name: str
    display_name: str
    max_resolution: int  # Max dimension before downscale (longest or shortest edge)
    resize_method: str  # "bicubic" or "bilinear"
    preserve_aspect_ratio: bool
    uses_tiling: bool
    tile_size: Optional[int]  # Tile size if uses_tiling is True
    encoder_family: str  # "clip", "siglip", "proprietary", "unknown"
    recommended_surrogates: tuple[str, ...] = ()
    recommended_strategies: tuple[str, ...] = ()
    confidence: str = "medium"  # "high", "medium", "low"


PROFILES: dict[str, TargetProfile] = {
    "gpt4o": TargetProfile(
        name="gpt4o",
        display_name="GPT-4o (OpenAI)",
        max_resolution=768,
        resize_method="bicubic",
        preserve_aspect_ratio=True,
        uses_tiling=True,
        tile_size=512,
        encoder_family="proprietary",
        recommended_surrogates=("clip-vit-b-32", "clip-vit-l-14", "siglip-vit-so400m"),
        recommended_strategies=("cotta", "m_attack"),
        confidence="medium",
    ),
    "gpt5": TargetProfile(
        name="gpt5",
        display_name="GPT-5 (OpenAI)",
        max_resolution=768,
        resize_method="bicubic",
        preserve_aspect_ratio=True,
        uses_tiling=True,
        tile_size=512,
        encoder_family="proprietary",
        recommended_surrogates=("clip-vit-b-32", "clip-vit-l-14", "siglip-vit-so400m"),
        recommended_strategies=("cotta", "m_attack", "ipga"),
        confidence="low",
    ),
    "claude": TargetProfile(
        name="claude",
        display_name="Claude (Anthropic)",
        max_resolution=1568,
        resize_method="bicubic",
        preserve_aspect_ratio=True,
        uses_tiling=False,
        tile_size=None,
        encoder_family="unknown",
        recommended_surrogates=("siglip-vit-so400m", "clip-vit-l-14", "clip-vit-b-16"),
        recommended_strategies=("cotta", "ipga", "m_attack"),
        confidence="medium",
    ),
    "gemini": TargetProfile(
        name="gemini",
        display_name="Gemini 2.5 (Google)",
        max_resolution=1024,
        resize_method="bicubic",
        preserve_aspect_ratio=True,
        uses_tiling=False,
        tile_size=None,
        encoder_family="proprietary",
        recommended_surrogates=("clip-vit-l-14", "siglip-vit-so400m", "clip-vit-b-32"),
        recommended_strategies=("m_attack", "cotta", "ipga"),
        confidence="low",
    ),
    "opensource": TargetProfile(
        name="opensource",
        display_name="Open-Source VLMs (LLaVA, InternVL)",
        max_resolution=336,
        resize_method="bicubic",
        preserve_aspect_ratio=False,
        uses_tiling=False,
        tile_size=None,
        encoder_family="clip",
        recommended_surrogates=("clip-vit-l-14-336", "clip-vit-b-32", "clip-vit-b-16"),
        recommended_strategies=("pgd_baseline", "m_attack", "ipga"),
        confidence="high",
    ),
}

# Sampling weights for auto mode — stochastic profile selection during optimization
AUTO_WEIGHTS = {
    "gpt4o": 0.15,
    "gpt5": 0.15,
    "claude": 0.30,  # Hardest target — proportional attention
    "gemini": 0.20,
    None: 0.20,  # No preprocessing (direct CLIP input)
}


def get_profile(name: str) -> Optional[TargetProfile]:
    """Get a target profile by name. Returns None for 'auto'."""
    if name == "auto":
        return None
    return PROFILES.get(name)


def list_profiles() -> list[TargetProfile]:
    """Return all available target profiles."""
    return list(PROFILES.values())
