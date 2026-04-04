"""M-Attack: Random-crop local-to-global feature matching.

Research basis: M-Attack (NeurIPS 2025) + SGMA semantic guidance + TATM typography augmentation.

Key insight: standard PGD produces uniform noise that VLM vision encoders ignore.
Random cropping forces semantic energy to distribute non-uniformly, concentrating
in regions that all models attend to.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


def _compute_affine_from_crop(
    crop_box: tuple[float, float, float, float],
    src_h: int,
    src_w: int,
) -> torch.Tensor:
    """Compute a 2x3 affine matrix that maps the crop region to the full output.

    Args:
        crop_box: (y1, x1, y2, x2) in normalized [0, 1] coords.
        src_h: Source image height.
        src_w: Source image width.

    Returns:
        (1, 2, 3) affine matrix for F.affine_grid.
    """
    y1, x1, y2, x2 = crop_box

    # Convert [0, 1] coords to [-1, 1] for grid_sample
    cx = (x1 + x2) - 1.0  # Center in [-1, 1]
    cy = (y1 + y2) - 1.0
    sx = x2 - x1  # Scale
    sy = y2 - y1

    theta = torch.tensor(
        [[[sx, 0, cx], [0, sy, cy]]],
        dtype=torch.float32,
    )
    return theta


def _differentiable_crop_resize(
    x: torch.Tensor,
    crop_box: tuple[float, float, float, float],
    target_size: int = 224,
) -> torch.Tensor:
    """Differentiable random crop and resize using grid_sample.

    Gradients flow back from the cropped output to the full input image.

    Args:
        x: Image tensor (B, C, H, W).
        crop_box: (y1, x1, y2, x2) in normalized [0, 1] coords.
        target_size: Output size (square).

    Returns:
        Cropped and resized tensor (B, C, target_size, target_size).
    """
    theta = _compute_affine_from_crop(crop_box, x.shape[2], x.shape[3]).to(x.device)
    grid = F.affine_grid(theta, [x.shape[0], x.shape[1], target_size, target_size], align_corners=True)
    return F.grid_sample(x, grid, mode="bilinear", align_corners=True)


def _sample_random_crop(
    h: int,
    w: int,
    relevance_map: Optional[torch.Tensor] = None,
    scale_range: tuple[float, float] = (0.5, 1.0),
    aspect_range: tuple[float, float] = (0.8, 1.2),
) -> tuple[float, float, float, float]:
    """Sample a random crop box, optionally biased by semantic relevance.

    Args:
        h: Image height.
        w: Image width.
        relevance_map: Optional (H, W) tensor with semantic relevance scores.
            If provided, 70% of crops are biased toward high-relevance regions.
        scale_range: (min_scale, max_scale) relative to image size.
        aspect_range: (min_aspect, max_aspect) ratio.

    Returns:
        (y1, x1, y2, x2) in normalized [0, 1] coords.
    """
    scale = random.uniform(*scale_range)
    aspect = random.uniform(*aspect_range)

    crop_h = min(1.0, scale * math.sqrt(aspect))
    crop_w = min(1.0, scale / math.sqrt(aspect))

    # Determine center
    if relevance_map is not None and random.random() < 0.7:
        # Bias toward high-relevance region
        flat = relevance_map.flatten()
        # Weighted sampling
        probs = F.softmax(flat * 3.0, dim=0)  # Temperature to sharpen
        idx = torch.multinomial(probs, 1).item()
        cy = (idx // relevance_map.shape[1]) / relevance_map.shape[0]
        cx = (idx % relevance_map.shape[1]) / relevance_map.shape[1]
    else:
        # Uniform random center
        cy = random.uniform(crop_h / 2, 1.0 - crop_h / 2)
        cx = random.uniform(crop_w / 2, 1.0 - crop_w / 2)

    y1 = max(0.0, cy - crop_h / 2)
    y2 = min(1.0, cy + crop_h / 2)
    x1 = max(0.0, cx - crop_w / 2)
    x2 = min(1.0, cx + crop_w / 2)

    return (y1, x1, y2, x2)


def _compute_semantic_map(
    image: torch.Tensor,
    ensemble,
    target_embeddings: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Compute SGMA semantic relevance map via simplified GradCAM.

    Identifies which image regions all models attend to.

    Args:
        image: Clean image tensor (1, 3, H, W).
        ensemble: CLIPEnsemble instance.
        target_embeddings: Target text embeddings.

    Returns:
        (H, W) relevance map normalized to [0, 1].
    """
    device = image.device
    _, _, h, w = image.shape
    image_grad = image.clone().requires_grad_(True)

    relevance_maps = []

    for model_id in ensemble.loaded_models:
        if image_grad.grad is not None:
            image_grad.grad.zero_()

        emb = ensemble.encode_image_single(image_grad, model_id)
        target = target_embeddings[model_id]
        sim = F.cosine_similarity(emb, target, dim=-1).mean()
        sim.backward()

        if image_grad.grad is not None:
            # Gradient magnitude as relevance
            grad_map = image_grad.grad.abs().mean(dim=(0, 1))  # (H, W)
            relevance_maps.append(grad_map.detach())

    if not relevance_maps:
        return torch.ones(h, w, device=device)

    # Average across models
    avg_map = torch.stack(relevance_maps).mean(dim=0)

    # Normalize to [0, 1]
    min_val = avg_map.min()
    max_val = avg_map.max()
    if max_val - min_val > 1e-8:
        avg_map = (avg_map - min_val) / (max_val - min_val)
    else:
        avg_map = torch.ones_like(avg_map)

    return avg_map


class MAttackStrategy(AttackStrategy):
    """M-Attack: Transferable targeted attack via random cropping.

    At each iteration, a random crop of the adversarial image is extracted
    and aligned with the target embedding. The gradient flows through the
    differentiable crop+resize operation back to the full image pixels.

    Over many iterations with random crops, gradient accumulates non-uniformly,
    concentrating on semantically important regions.
    """

    @property
    def name(self) -> str:
        return "m_attack"

    @property
    def required_tier(self) -> int:
        return 2

    @property
    def required_models(self) -> list[str]:
        return []

    def optimize(
        self,
        clean_image: torch.Tensor,
        target_embeddings: dict[str, torch.Tensor],
        ensemble,
        config: AttackConfig,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> CandidateResult:
        start_time = time.time()
        device = clean_image.device

        if config.seed is not None:
            torch.manual_seed(config.seed)

        _, _, h, w = clean_image.shape

        # Precompute semantic relevance map (skip in quick mode)
        relevance_map = None
        if not config.quick:
            relevance_map = _compute_semantic_map(clean_image, ensemble, target_embeddings)

        # Initialize perturbation
        delta = torch.zeros_like(clean_image, requires_grad=True, device=device)
        step_size = config.epsilon / max(config.iterations * 0.5, 1.0)

        best_score = -float("inf")
        best_delta = delta.data.clone()
        no_improve_count = 0

        for iteration in range(config.iterations):
            if delta.grad is not None:
                delta.grad.zero_()

            x_adv = (clean_image + delta).clamp(0, 1)

            # Sample random crop
            crop_box = _sample_random_crop(h, w, relevance_map)

            # Differentiable crop + resize to 224x224
            x_crop = _differentiable_crop_resize(x_adv, crop_box, target_size=224)

            # Compute loss across ensemble
            total_loss = torch.tensor(0.0, device=device)
            for model_id in ensemble.loaded_models:
                emb = ensemble.encode_image_single(x_crop, model_id)
                target_emb = target_embeddings[model_id]
                loss = -F.cosine_similarity(emb, target_emb, dim=-1).mean()
                total_loss = total_loss + loss

            total_loss = total_loss / ensemble.model_count
            total_loss.backward()

            # PGD update — gradient flows through grid_sample back to full delta
            with torch.no_grad():
                delta.data = delta.data - step_size * delta.grad.sign()
                delta.data = delta.data.clamp(-config.epsilon, config.epsilon)
                delta.data = (clean_image + delta.data).clamp(0, 1) - clean_image

            # Periodically check full-image score (not just crop score)
            if (iteration + 1) % 25 == 0 or iteration == config.iterations - 1:
                with torch.no_grad():
                    full_adv = (clean_image + delta.data).clamp(0, 1)
                    embs = ensemble.encode_image(full_adv)
                    sims = [
                        F.cosine_similarity(embs[mid], target_embeddings[mid], dim=-1).item()
                        for mid in embs
                    ]
                    current_score = sum(sims) / len(sims)

                if current_score > best_score:
                    best_score = current_score
                    best_delta = delta.data.clone()
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if progress_callback:
                    progress_callback(iteration, current_score)

                early_stop_threshold = 0.75 if config.quick else 0.85
                if current_score > early_stop_threshold:
                    break
                if no_improve_count >= 10:  # Checked every 25 iters, so 250 stale iters
                    break

        # Final result
        adversarial = (clean_image + best_delta).clamp(0, 1)

        final_per_model = {}
        with torch.no_grad():
            embeddings = ensemble.encode_image(adversarial)
            for model_id, emb in embeddings.items():
                sim = F.cosine_similarity(emb, target_embeddings[model_id], dim=-1).item()
                final_per_model[model_id] = sim

        mean_clip = sum(final_per_model.values()) / len(final_per_model)

        return CandidateResult(
            strategy_name=self.name,
            adversarial_image=adversarial.detach(),
            perturbation=best_delta.detach(),
            per_model_scores=final_per_model,
            clip_score=mean_clip,
            psnr=compute_psnr(clean_image, adversarial.detach()),
            ssim=compute_ssim(clean_image, adversarial.detach()),
            iterations_used=iteration + 1,
            time_seconds=time.time() - start_time,
            metadata={"used_semantic_guidance": relevance_map is not None},
        )
