"""Composite scoring — weighted combination of all metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CandidateScores:
    """All scores for a single candidate adversarial image."""

    clip_score: float  # Mean CLIP/SigLIP cosine similarity [0, 1]
    jpeg_survival: float  # 1.0 if survives JPEG, 0.0 if not
    image_quality: float  # SSIM [0, 1]
    ensemble_agreement: float  # Cross-model agreement [0, 1]
    vlm_proxy: float  # VLM proxy score [0, 1] (Tier 3 only, else 0)
    composite: float = 0.0  # Weighted composite (computed)


# Tier-dependent scoring weights
SCORING_WEIGHTS = {
    1: {"clip": 0.70, "jpeg": 0.20, "quality": 0.10, "vlm": 0.00},
    2: {"clip": 0.60, "jpeg": 0.20, "quality": 0.10, "vlm": 0.10},  # vlm = ensemble_agreement
    3: {"clip": 0.40, "jpeg": 0.20, "quality": 0.10, "vlm": 0.30},
}


def compute_composite(scores: CandidateScores, tier: int) -> float:
    """Compute tier-dependent weighted composite score.

    At Tier 2, the 'vlm' weight uses ensemble_agreement as a proxy.
    At Tier 3, it uses the actual VLM proxy score.

    Returns:
        Composite score in [0, 1].
    """
    w = SCORING_WEIGHTS[tier]

    vlm_value = scores.vlm_proxy if tier == 3 else scores.ensemble_agreement

    composite = (
        w["clip"] * scores.clip_score
        + w["jpeg"] * scores.jpeg_survival
        + w["quality"] * scores.image_quality
        + w["vlm"] * vlm_value
    )

    scores.composite = composite
    return composite
