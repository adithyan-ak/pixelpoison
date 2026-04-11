"""Adversarial Example Soups: Meta-strategy that averages perturbations (Tier 1).

Research basis: Wortsman et al., "Model Soups" (ICML 2022) adapted to
adversarial perturbations. Also inspired by "Boosting Adversarial
Transferability across Model Genus" (AAAI 2024).

Key insight: Averaging perturbations from multiple optimization runs with
different hyperparameters (step size, momentum, augmentation settings)
produces a more robust perturbation that transfers better than any individual
run. This is because averaging smooths out model-specific artifacts while
preserving the common adversarial signal.

This is a zero-cost improvement: run the same total iterations split across
M configs, then average the resulting perturbations.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


def _run_mi_fgsm_variant(
    clean_image: torch.Tensor,
    target_embeddings: dict[str, torch.Tensor],
    ensemble,
    epsilon: float,
    step_size: float,
    iterations: int,
    mu: float,
    dim_prob: float,
    seed: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """Run a single MI-FGSM variant and return the perturbation.

    Args:
        clean_image: Original image tensor.
        target_embeddings: Target text embeddings per model.
        ensemble: CLIPEnsemble instance.
        epsilon: Max perturbation L-inf.
        step_size: PGD step size.
        iterations: Number of optimization iterations.
        mu: Momentum decay factor.
        dim_prob: Probability of applying DIM (random resize).
        seed: Random seed.
        device: Torch device.

    Returns:
        Best perturbation tensor found during optimization.
    """
    if seed is not None:
        torch.manual_seed(seed)

    delta = torch.zeros_like(clean_image, requires_grad=True, device=device)
    momentum = torch.zeros_like(clean_image)

    best_score = -float("inf")
    best_delta = delta.data.clone()

    for iteration in range(iterations):
        if delta.grad is not None:
            delta.grad.zero_()

        x_adv = (clean_image + delta).clamp(0, 1)

        # Optional DIM: random resize and pad
        if torch.rand(1).item() < dim_prob:
            _, _, h, w = x_adv.shape
            scale = 0.85 + 0.3 * torch.rand(1).item()  # U(0.85, 1.15)
            new_h = max(16, int(h * scale))
            new_w = max(16, int(w * scale))
            x_input = F.interpolate(
                x_adv, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
            x_input = F.interpolate(
                x_input, size=(h, w), mode="bilinear", align_corners=False
            )
        else:
            x_input = x_adv

        # Ensemble loss
        total_loss = torch.tensor(0.0, device=device)
        for model_id in ensemble.loaded_models:
            emb = ensemble.encode_image_single(x_input, model_id)
            target_emb = target_embeddings[model_id]
            loss = -F.cosine_similarity(emb, target_emb, dim=-1).mean()
            total_loss = total_loss + loss
        total_loss = total_loss / ensemble.model_count
        total_loss.backward()

        grad = delta.grad.data.clone()

        # MI-FGSM momentum
        grad_norm = grad / (grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-12)
        momentum = mu * momentum + grad_norm

        # PGD update
        with torch.no_grad():
            delta.data = delta.data - step_size * momentum.sign()
            delta.data = delta.data.clamp(-epsilon, epsilon)
            delta.data = (clean_image + delta.data).clamp(0, 1) - clean_image

        # Track best
        if (iteration + 1) % 25 == 0 or iteration == iterations - 1:
            with torch.no_grad():
                full_adv = (clean_image + delta.data).clamp(0, 1)
                embs = ensemble.encode_image(full_adv)
                sims = [
                    F.cosine_similarity(
                        embs[mid], target_embeddings[mid], dim=-1
                    ).item()
                    for mid in embs
                ]
                score = sum(sims) / len(sims)

            if score > best_score:
                best_score = score
                best_delta = delta.data.clone()

    return best_delta


# Soup configurations: diverse hyperparameters for ensemble averaging
SOUP_CONFIGS = [
    # (step_size_mult, momentum, dim_prob, description)
    (1.0, 1.0, 0.7, "standard MI-FGSM + DIM"),
    (2.0, 1.0, 0.0, "aggressive step, no DIM"),
    (0.5, 0.9, 0.5, "conservative step, lighter momentum"),
    (1.5, 1.0, 0.9, "medium step, heavy DIM"),
    (1.0, 0.7, 0.3, "standard step, light momentum + DIM"),
    (3.0, 1.0, 0.7, "very aggressive step + DIM"),
    (0.75, 1.0, 0.0, "small step, no augmentation"),
    (1.5, 0.5, 0.5, "medium step, half momentum"),
]


class SoupsStrategy(AttackStrategy):
    """Adversarial Example Soups: average perturbations from diverse configs.

    Runs M independent MI-FGSM variants with different hyperparameters
    (step size, momentum, DIM probability), each for iterations/M steps.
    The final perturbation is the average of all M results, projected
    back to the epsilon-ball.

    This produces smoother, more transferable perturbations without
    increasing total computation.
    """

    @property
    def name(self) -> str:
        return "soups"

    @property
    def required_tier(self) -> int:
        return 1

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

        n_configs = min(len(SOUP_CONFIGS), 8)
        iters_per_config = max(config.iterations // n_configs, 50)

        perturbations = []

        for i in range(n_configs):
            step_mult, mu, dim_prob, _desc = SOUP_CONFIGS[i]
            step_size = config.step_size * step_mult

            seed = config.seed + i + 1 if config.seed is not None else None

            delta = _run_mi_fgsm_variant(
                clean_image=clean_image,
                target_embeddings=target_embeddings,
                ensemble=ensemble,
                epsilon=config.epsilon,
                step_size=step_size,
                iterations=iters_per_config,
                mu=mu,
                dim_prob=dim_prob,
                seed=seed,
                device=device,
            )
            perturbations.append(delta)

            if progress_callback:
                # Report progress as fraction of configs completed
                equiv_iter = (i + 1) * iters_per_config
                with torch.no_grad():
                    adv = (clean_image + delta).clamp(0, 1)
                    embs = ensemble.encode_image(adv)
                    sims = [
                        F.cosine_similarity(
                            embs[mid], target_embeddings[mid], dim=-1
                        ).item()
                        for mid in embs
                    ]
                    score = sum(sims) / len(sims)
                progress_callback(equiv_iter, score)

            # Free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Average all perturbations (the "soup")
        with torch.no_grad():
            avg_delta = torch.stack(perturbations).mean(dim=0)
            # Project back to epsilon-ball
            avg_delta = avg_delta.clamp(-config.epsilon, config.epsilon)
            avg_delta = (clean_image + avg_delta).clamp(0, 1) - clean_image

        # Final adversarial image
        adversarial = (clean_image + avg_delta).clamp(0, 1)

        final_per_model = {}
        with torch.no_grad():
            embeddings = ensemble.encode_image(adversarial)
            for model_id, emb in embeddings.items():
                sim = F.cosine_similarity(
                    emb, target_embeddings[model_id], dim=-1
                ).item()
                final_per_model[model_id] = sim

        mean_clip = sum(final_per_model.values()) / len(final_per_model)

        return CandidateResult(
            strategy_name=self.name,
            adversarial_image=adversarial.detach(),
            perturbation=avg_delta.detach(),
            per_model_scores=final_per_model,
            clip_score=mean_clip,
            psnr=compute_psnr(clean_image, adversarial.detach()),
            ssim=compute_ssim(clean_image, adversarial.detach()),
            iterations_used=iters_per_config * n_configs,
            time_seconds=time.time() - start_time,
            metadata={
                "n_configs": n_configs,
                "iters_per_config": iters_per_config,
            },
        )
