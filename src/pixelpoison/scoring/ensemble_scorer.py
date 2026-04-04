"""Ensemble agreement scoring — measures cross-model consistency."""

from __future__ import annotations

import numpy as np


def compute_ensemble_agreement(per_model_scores: dict[str, float]) -> float:
    """Compute how much the ensemble models agree on alignment quality.

    High agreement suggests the perturbation generalizes across architectures,
    making it more likely to transfer to the target VLM.

    Formula: agreement = 1.0 - std(scores) / (mean(scores) + epsilon)

    Args:
        per_model_scores: Dict of model_id → cosine similarity score.

    Returns:
        Agreement score in [0, 1]. 1.0 = perfect agreement.
    """
    if len(per_model_scores) < 2:
        return 1.0  # Single model trivially agrees with itself

    scores = list(per_model_scores.values())
    mean_val = np.mean(scores)
    std_val = np.std(scores)

    if mean_val < 1e-8:
        return 0.0

    agreement = 1.0 - std_val / (mean_val + 1e-8)
    return float(np.clip(agreement, 0.0, 1.0))
