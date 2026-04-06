"""M-Attack: Random-crop local-to-global feature matching with MI-FGSM + TIM.

Research basis:
- M-Attack (NeurIPS 2025): random-crop local-to-global matching
- SGMA semantic guidance + TATM typography augmentation
- MI-FGSM momentum: Dong et al. (CVPR 2018)
- TIM (Translation-Invariant Method): Dong et al. (CVPR 2019)

Key insight: standard PGD produces uniform noise that VLM vision encoders ignore.
Random cropping forces semantic energy to distribute non-uniformly, concentrating
in regions that all models attend to. MI-FGSM momentum stabilizes gradient
direction across random crops for better convergence.
"""

from __future__ import annotations

import math
import random
import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


def _get_gaussian_kernel_2d(kernel_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2D Gaussian kernel for TIM gradient smoothing."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = g / g.sum()
    kernel_2d = kernel_1d.outer(kernel_1d)
    return kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)


def _differentiable_crop_resize(
    x: torch.Tensor,
    crop_box: tuple[float, float, float, float],
    target_size: int = 224,
) -> torch.Tensor:
    """Differentiable random crop and resize via slicing + interpolate.

    Uses tensor slicing (gradient flows to cropped pixels) and F.interpolate
    (differentiable on all backends) instead of grid_sample, whose backward
    is not implemented on MPS.
    """
    _, _, h, w = x.shape
    y1, x1, y2, x2 = crop_box

    py1 = max(0, min(int(y1 * h), h - 1))
    py2 = max(py1 + 1, min(int(y2 * h), h))
    px1 = max(0, min(int(x1 * w), w - 1))
    px2 = max(px1 + 1, min(int(x2 * w), w))

    cropped = x[:, :, py1:py2, px1:px2]
    return F.interpolate(
        cropped, size=(target_size, target_size),
        mode="bilinear", align_corners=False,
    )


def _sample_random_crop(
    h: int,
    w: int,
    relevance_map: Optional[torch.Tensor] = None,
    scale_range: tuple[float, float] = (0.5, 1.0),
    aspect_range: tuple[float, float] = (0.8, 1.2),
) -> tuple[float, float, float, float]:
    """Sample a random crop box, optionally biased by semantic relevance."""
    scale = random.uniform(*scale_range)
    aspect = random.uniform(*aspect_range)

    crop_h = min(1.0, scale * math.sqrt(aspect))
    crop_w = min(1.0, scale / math.sqrt(aspect))

    if relevance_map is not None and random.random() < 0.7:
        flat = relevance_map.flatten()
        probs = F.softmax(flat * 3.0, dim=0)
        idx = torch.multinomial(probs, 1).item()
        cy = (idx // relevance_map.shape[1]) / relevance_map.shape[0]
        cx = (idx % relevance_map.shape[1]) / relevance_map.shape[1]
    else:
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
    """Compute SGMA semantic relevance map via simplified GradCAM."""
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
            grad_map = image_grad.grad.abs().mean(dim=(0, 1))
            relevance_maps.append(grad_map.detach())

    if not relevance_maps:
        return torch.ones(h, w, device=device)

    avg_map = torch.stack(relevance_maps).mean(dim=0)

    min_val = avg_map.min()
    max_val = avg_map.max()
    if max_val - min_val > 1e-8:
        avg_map = (avg_map - min_val) / (max_val - min_val)
    else:
        avg_map = torch.ones_like(avg_map)

    return avg_map


class MAttackStrategy(AttackStrategy):
    """M-Attack with MI-FGSM momentum + TIM for transferable targeted attacks.

    At each iteration, a random crop of the adversarial image is extracted
    and aligned with the target embedding. MI-FGSM momentum accumulates
    gradients across different random crops, building a stable estimate of
    the optimal perturbation direction. TIM smooths gradients to reduce
    position-dependent overfitting.
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

        # Setup TIM (Gaussian kernel for gradient smoothing)
        tim_kernel = _get_gaussian_kernel_2d(kernel_size=7, sigma=1.5).to(device)
        tim_padding = 7 // 2

        # Initialize perturbation and MI-FGSM momentum
        delta = torch.zeros_like(clean_image, requires_grad=True, device=device)
        momentum = torch.zeros_like(clean_image, device=device)
        mu = 1.0  # MI-FGSM momentum decay
        alpha = config.step_size

        best_score = -float("inf")
        best_delta = delta.data.clone()
        no_improve_count = 0

        for iteration in range(config.iterations):
            if delta.grad is not None:
                delta.grad.zero_()

            x_adv = (clean_image + delta).clamp(0, 1)

            # Sample multiple random crops per iteration for gradient averaging
            n_crops = 3
            total_loss = torch.tensor(0.0, device=device)

            for _ in range(n_crops):
                crop_box = _sample_random_crop(h, w, relevance_map)
                x_crop = _differentiable_crop_resize(x_adv, crop_box, target_size=224)

                for model_id in ensemble.loaded_models:
                    emb = ensemble.encode_image_single(x_crop, model_id)
                    target_emb = target_embeddings[model_id]
                    loss = -F.cosine_similarity(emb, target_emb, dim=-1).mean()
                    total_loss = total_loss + loss

            total_loss = total_loss / (n_crops * ensemble.model_count)

            # Also add full-image loss to maintain global coherence
            for model_id in ensemble.loaded_models:
                emb_full = ensemble.encode_image_single(x_adv, model_id)
                target_emb = target_embeddings[model_id]
                full_loss = -F.cosine_similarity(emb_full, target_emb, dim=-1).mean()
                total_loss = total_loss + 0.5 * full_loss

            total_loss.backward()

            # --- TIM: smooth gradient ---
            grad = delta.grad.data
            grad = F.conv2d(grad, tim_kernel, padding=tim_padding, groups=3)

            # --- MI-FGSM: momentum update ---
            grad_norm = grad / (torch.mean(torch.abs(grad)) + 1e-12)
            momentum = mu * momentum + grad_norm

            with torch.no_grad():
                delta.data = delta.data - alpha * momentum.sign()
                delta.data = delta.data.clamp(-config.epsilon, config.epsilon)
                delta.data = (clean_image + delta.data).clamp(0, 1) - clean_image

            # Check full-image score periodically
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

                # Early stopping (relaxed)
                if current_score > 0.95:
                    break
                if no_improve_count >= 20:  # Checked every 25 iters = 500 stale iters
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
            metadata={
                "used_semantic_guidance": relevance_map is not None,
                "momentum_decay": mu,
                "crops_per_iteration": n_crops,
            },
        )
