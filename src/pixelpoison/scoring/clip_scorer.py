"""CLIP embedding alignment scoring."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_clip_score(
    adversarial: torch.Tensor,
    target_embeddings: dict[str, torch.Tensor],
    ensemble,
) -> tuple[float, dict[str, float]]:
    """Compute average CLIP cosine similarity across all ensemble models.

    Args:
        adversarial: Adversarial image tensor (1, 3, H, W) in [0, 1].
        target_embeddings: Dict of model_id → target text embedding.
        ensemble: CLIPEnsemble instance (loaded).

    Returns:
        Tuple of (mean_score, per_model_scores).
        Scores are cosine similarity in [-1, 1], practical range [0, 0.9].
    """
    per_model = {}

    with torch.no_grad():
        image_embeddings = ensemble.encode_image(adversarial)

    for model_id, img_emb in image_embeddings.items():
        target_emb = target_embeddings[model_id]
        sim = F.cosine_similarity(img_emb, target_emb, dim=-1).item()
        per_model[model_id] = sim

    mean_score = sum(per_model.values()) / len(per_model) if per_model else 0.0
    return mean_score, per_model
