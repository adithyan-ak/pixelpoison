"""CWA: Common Weakness Attack (Tier 2).

Research basis: "Rethinking Model Ensemble in Transfer-based Adversarial
Attacks" (ICLR 2024, arXiv:2303.09105).

Key insight: Standard ensemble attacks average gradients, which can cancel out
diverse gradient signals. CWA uses Sharpness-Aware Minimization (SAM) to find
flat loss regions that generalize across models, combined with Cosine Similarity
Encourager (CSE) that sequentially aligns with each model's gradient direction.

This finds adversarial examples in flat regions of the loss landscape that
transfer better because they don't rely on sharp features of any single model.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


class CWAStrategy(AttackStrategy):
    """CWA: Common Weakness Attack with SAM + CSE.

    Two-phase per-iteration update:
    1. SAM inner step: probe the sharpest direction around current point
    2. CSE: from the SAM-perturbed point, sequentially align with each
       model's gradient using L2-normalized steps

    The net direction is accumulated with MI-FGSM momentum for the final
    sign-gradient PGD update.
    """

    @property
    def name(self) -> str:
        return "cwa"

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

        # Hyperparameters (from CWA paper)
        mu = 1.0           # Outer momentum decay
        sam_radius = config.epsilon / 15.0  # SAM perturbation radius
        cse_step = 0.02    # CSE inner step size (beta in paper, scaled down)
        cse_momentum = 0.8  # CSE inner momentum decay

        # Initialize
        delta = torch.zeros_like(clean_image, requires_grad=True, device=device)
        outer_momentum = torch.zeros_like(clean_image)

        best_score = -float("inf")
        best_delta = delta.data.clone()
        no_improve_count = 0
        iteration = 0

        model_ids = list(ensemble.loaded_models)

        for iteration in range(config.iterations):
            # === Phase 1: SAM inner ascent step ===
            # Compute ensemble gradient at current point
            if delta.grad is not None:
                delta.grad.zero_()

            x_adv = (clean_image + delta).clamp(0, 1)
            total_loss = torch.tensor(0.0, device=device)
            for model_id in model_ids:
                emb = ensemble.encode_image_single(x_adv, model_id)
                target_emb = target_embeddings[model_id]
                loss = -F.cosine_similarity(emb, target_emb, dim=-1).mean()
                total_loss = total_loss + loss
            total_loss = total_loss / ensemble.model_count
            total_loss.backward()

            ensemble_grad = delta.grad.data.clone()

            # SAM: move to sharpest point within radius
            with torch.no_grad():
                sam_perturbation = sam_radius * ensemble_grad.sign()
                delta_sam = (delta.data + sam_perturbation).clamp(
                    -config.epsilon, config.epsilon
                )
                delta_sam = (clean_image + delta_sam).clamp(0, 1) - clean_image

            # === Phase 2: CSE sequential model alignment ===
            # Starting from SAM-perturbed point, sequentially align with each model
            delta_cse = delta_sam.clone()
            inner_momentum = torch.zeros_like(clean_image)

            for model_id in model_ids:
                # Compute single-model gradient at current CSE point
                delta_temp = delta_cse.clone().detach().requires_grad_(True)
                x_cse = (clean_image + delta_temp).clamp(0, 1)

                emb = ensemble.encode_image_single(x_cse, model_id)
                target_emb = target_embeddings[model_id]
                loss = -F.cosine_similarity(emb, target_emb, dim=-1).mean()
                loss.backward()

                model_grad = delta_temp.grad.data

                # L2-normalize the gradient (key CSE step: equal weighting)
                grad_l2 = model_grad / (
                    torch.norm(model_grad, p=2) + 1e-12
                )

                # CSE inner momentum
                inner_momentum = cse_momentum * inner_momentum + grad_l2

                # CSE step
                with torch.no_grad():
                    delta_cse = delta_cse - cse_step * inner_momentum
                    delta_cse = delta_cse.clamp(-config.epsilon, config.epsilon)
                    delta_cse = (clean_image + delta_cse).clamp(0, 1) - clean_image

            # === Outer update: net direction with MI-FGSM momentum ===
            with torch.no_grad():
                # Net direction from original to CSE-refined point
                direction = delta_cse - delta.data

                # Normalize
                dir_norm = direction / (
                    direction.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-12
                )

                # Outer momentum accumulation
                outer_momentum = mu * outer_momentum + dir_norm

                # PGD sign-gradient update
                delta.data = delta.data - config.step_size * outer_momentum.sign()
                delta.data = delta.data.clamp(-config.epsilon, config.epsilon)
                delta.data = (clean_image + delta.data).clamp(0, 1) - clean_image

            # Periodically evaluate
            if (iteration + 1) % 10 == 0 or iteration == config.iterations - 1:
                with torch.no_grad():
                    full_adv = (clean_image + delta.data).clamp(0, 1)
                    embs = ensemble.encode_image(full_adv)
                    sims = [
                        F.cosine_similarity(
                            embs[mid], target_embeddings[mid], dim=-1
                        ).item()
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

                if current_score > 0.95:
                    break
                if no_improve_count >= 20:
                    break

        # Final adversarial image
        adversarial = (clean_image + best_delta).clamp(0, 1)

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
            perturbation=best_delta.detach(),
            per_model_scores=final_per_model,
            clip_score=mean_clip,
            psnr=compute_psnr(clean_image, adversarial.detach()),
            ssim=compute_ssim(clean_image, adversarial.detach()),
            iterations_used=iteration + 1,
            time_seconds=time.time() - start_time,
            metadata={"sam_radius": sam_radius, "cse_step": cse_step},
        )
