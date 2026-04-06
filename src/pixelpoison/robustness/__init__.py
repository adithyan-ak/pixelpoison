"""JPEG robustness — DiffJPEG simulation and DCT mid-frequency targeting."""

from __future__ import annotations

import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from pixelpoison.attacks.base import AttackConfig, CandidateResult
from pixelpoison.robustness.diff_jpeg import DiffJPEG
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


def _actual_jpeg_roundtrip(tensor: torch.Tensor, quality: int = 85) -> torch.Tensor:
    """Save tensor as JPEG and reload — tests real (non-differentiable) JPEG survival."""
    arr = (tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
        np.uint8
    )
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    img_reloaded = Image.open(buf).convert("RGB")
    arr_reloaded = np.array(img_reloaded, dtype=np.float32) / 255.0
    return torch.from_numpy(arr_reloaded).permute(2, 0, 1).unsqueeze(0).to(tensor.device)


def check_jpeg_survival(
    adversarial: torch.Tensor,
    target_embeddings: dict[str, torch.Tensor],
    ensemble,
    quality: int = 85,
    threshold: float = 0.10,
) -> tuple[bool, float]:
    """Check if adversarial image survives JPEG compression.

    Args:
        adversarial: Adversarial image tensor (1, 3, H, W) in [0, 1].
        target_embeddings: Target text embeddings per model.
        ensemble: CLIPEnsemble instance.
        quality: JPEG quality for the roundtrip test.
        threshold: Maximum allowed CLIP score degradation (fraction).

    Returns:
        Tuple of (survives: bool, degradation: float).
    """
    # Score before JPEG
    pre_scores = []
    with torch.no_grad():
        pre_embs = ensemble.encode_image(adversarial)
        for mid, emb in pre_embs.items():
            sim = F.cosine_similarity(emb, target_embeddings[mid], dim=-1).item()
            pre_scores.append(sim)
    pre_mean = sum(pre_scores) / len(pre_scores)

    # JPEG roundtrip
    jpeg_tensor = _actual_jpeg_roundtrip(adversarial, quality)

    # Score after JPEG
    post_scores = []
    with torch.no_grad():
        post_embs = ensemble.encode_image(jpeg_tensor)
        for mid, emb in post_embs.items():
            sim = F.cosine_similarity(emb, target_embeddings[mid], dim=-1).item()
            post_scores.append(sim)
    post_mean = sum(post_scores) / len(post_scores)

    # Check degradation
    if pre_mean > 0:
        degradation = (pre_mean - post_mean) / pre_mean
    else:
        degradation = 1.0

    survives = degradation < threshold
    return survives, degradation


def jpeg_harden(
    candidate: CandidateResult,
    clean_image: torch.Tensor,
    target_embeddings: dict[str, torch.Tensor],
    ensemble,
    config: AttackConfig,
    refinement_iterations: int = 100,
) -> CandidateResult:
    """Post-optimization JPEG refinement pass.

    Re-optimizes the candidate's perturbation specifically for JPEG survival
    by running additional iterations through the differentiable JPEG layer.

    Args:
        candidate: The candidate to refine.
        clean_image: Original clean image tensor.
        target_embeddings: Target text embeddings per model.
        ensemble: CLIPEnsemble instance.
        config: Attack configuration.
        refinement_iterations: Number of refinement iterations.

    Returns:
        Refined CandidateResult with updated scores and jpeg_survives flag.
    """
    device = clean_image.device
    diff_jpeg = DiffJPEG(quality=config.jpeg_quality).to(device)

    # Start from the candidate's perturbation
    delta = candidate.perturbation.clone().detach().requires_grad_(True)
    alpha = max(2.0 * config.epsilon / max(refinement_iterations, 10), 2.0 / 255.0)

    # MI-FGSM momentum for JPEG refinement
    momentum = torch.zeros_like(delta.data, device=device)
    mu = 1.0

    for _ in range(refinement_iterations):
        if delta.grad is not None:
            delta.grad.zero_()

        x_adv = (clean_image + delta).clamp(0, 1)

        # Apply differentiable JPEG
        x_jpeg = diff_jpeg(x_adv)

        # Compute loss through JPEG-compressed version
        total_loss = torch.tensor(0.0, device=device)
        for model_id in ensemble.loaded_models:
            emb = ensemble.encode_image_single(x_jpeg, model_id)
            target_emb = target_embeddings[model_id]
            loss = -F.cosine_similarity(emb, target_emb, dim=-1).mean()
            total_loss = total_loss + loss

        total_loss = total_loss / ensemble.model_count
        total_loss.backward()

        # MI-FGSM momentum update
        grad = delta.grad.data
        grad_norm = grad / (torch.mean(torch.abs(grad)) + 1e-12)
        momentum = mu * momentum + grad_norm

        with torch.no_grad():
            delta.data = delta.data - alpha * momentum.sign()
            delta.data = delta.data.clamp(-config.epsilon, config.epsilon)
            delta.data = (clean_image + delta.data).clamp(0, 1) - clean_image

    # Project final perturbation to epsilon-ball
    with torch.no_grad():
        delta.data = delta.data.clamp(-config.epsilon, config.epsilon)

    # Final adversarial image
    adversarial = (clean_image + delta.data).clamp(0, 1)

    # Check JPEG survival
    survives, degradation = check_jpeg_survival(
        adversarial, target_embeddings, ensemble, config.jpeg_quality
    )

    # Recompute scores
    final_per_model = {}
    with torch.no_grad():
        embeddings = ensemble.encode_image(adversarial)
        for model_id, emb in embeddings.items():
            sim = F.cosine_similarity(emb, target_embeddings[model_id], dim=-1).item()
            final_per_model[model_id] = sim

    mean_clip = sum(final_per_model.values()) / len(final_per_model)

    return CandidateResult(
        strategy_name=candidate.strategy_name,
        adversarial_image=adversarial.detach(),
        perturbation=delta.data.detach(),
        per_model_scores=final_per_model,
        clip_score=mean_clip,
        psnr=compute_psnr(clean_image, adversarial.detach()),
        ssim=compute_ssim(clean_image, adversarial.detach()),
        jpeg_survives=survives,
        iterations_used=candidate.iterations_used + refinement_iterations,
        time_seconds=candidate.time_seconds,
        metadata={**candidate.metadata, "jpeg_degradation": degradation},
    )
