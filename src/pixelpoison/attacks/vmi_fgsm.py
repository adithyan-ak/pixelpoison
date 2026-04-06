"""VMI-FGSM: Variance-Tuned Momentum Iterative FGSM (Tier 1).

Research basis: Wang & He, "Enhancing the Transferability of Adversarial
Attacks through Variance Tuning" (CVPR 2021, arXiv:2103.15571).

Key insight: Standard MI-FGSM accumulates momentum from point gradients,
which can oscillate. VMI-FGSM estimates gradient variance by sampling
neighbors around the current point and uses this variance to stabilize
the gradient direction, significantly improving transferability.

At each iteration:
1. Compute current gradient
2. Tune it with variance from previous iteration's neighborhood sampling
3. Accumulate with momentum
4. Sample N neighbors to compute variance for next iteration
5. PGD sign-gradient update
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


class VMIFGSMStrategy(AttackStrategy):
    """VMI-FGSM: Variance-Tuned Momentum Iterative FGSM.

    Enhances MI-FGSM with gradient variance tuning. At each step, the
    gradient is augmented with the variance signal from neighborhood
    sampling, which stabilizes the update direction across different
    models and makes the perturbation more transferable.
    """

    @property
    def name(self) -> str:
        return "vmi_fgsm"

    @property
    def required_tier(self) -> int:
        return 1

    @property
    def required_models(self) -> list[str]:
        return []

    def _compute_loss_grad(
        self,
        x_adv: torch.Tensor,
        delta: torch.Tensor,
        target_embeddings: dict[str, torch.Tensor],
        ensemble,
    ) -> torch.Tensor:
        """Compute gradient of ensemble loss w.r.t. delta."""
        if delta.grad is not None:
            delta.grad.zero_()

        device = x_adv.device
        total_loss = torch.tensor(0.0, device=device)
        for model_id in ensemble.loaded_models:
            emb = ensemble.encode_image_single(x_adv, model_id)
            target_emb = target_embeddings[model_id]
            loss = -F.cosine_similarity(emb, target_emb, dim=-1).mean()
            total_loss = total_loss + loss

        total_loss = total_loss / ensemble.model_count
        total_loss.backward()

        return delta.grad.data.clone()

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

        # Hyperparameters
        mu = 1.0          # Momentum decay
        beta = 1.5         # Neighborhood radius multiplier
        n_samples = 20     # Neighborhood samples for variance estimate

        # Initialize
        delta = torch.zeros_like(clean_image, requires_grad=True, device=device)
        momentum = torch.zeros_like(clean_image)
        variance = torch.zeros_like(clean_image)

        best_score = -float("inf")
        best_delta = delta.data.clone()
        no_improve_count = 0
        iteration = 0

        for iteration in range(config.iterations):
            x_adv = (clean_image + delta).clamp(0, 1)

            # Compute current gradient
            current_grad = self._compute_loss_grad(
                x_adv, delta, target_embeddings, ensemble
            )

            # Variance-tuned gradient: add variance from previous iteration
            tuned_grad = current_grad + variance

            # Normalize and accumulate momentum
            grad_norm = tuned_grad / (
                tuned_grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-12
            )
            momentum = mu * momentum + grad_norm

            # PGD update
            with torch.no_grad():
                delta.data = delta.data - config.step_size * momentum.sign()
                delta.data = delta.data.clamp(-config.epsilon, config.epsilon)
                delta.data = (clean_image + delta.data).clamp(0, 1) - clean_image

            # Compute variance for next iteration by sampling neighbors
            neighbor_grad_sum = torch.zeros_like(clean_image)
            for _ in range(n_samples):
                # Sample uniform random perturbation in neighborhood
                r = (2 * torch.rand_like(delta.data) - 1) * beta * config.epsilon

                # Need gradient w.r.t. delta for the neighbor
                delta_temp = delta.data.clone().detach().requires_grad_(True)
                x_temp = (clean_image + delta_temp + r).clamp(0, 1)

                total_loss = torch.tensor(0.0, device=device)
                for model_id in ensemble.loaded_models:
                    emb = ensemble.encode_image_single(x_temp, model_id)
                    target_emb = target_embeddings[model_id]
                    loss = -F.cosine_similarity(emb, target_emb, dim=-1).mean()
                    total_loss = total_loss + loss
                total_loss = total_loss / ensemble.model_count
                total_loss.backward()

                neighbor_grad_sum += delta_temp.grad.data

            # Variance = mean neighbor gradient - current gradient
            variance = neighbor_grad_sum / n_samples - current_grad

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
            metadata={"beta": beta, "n_samples": n_samples},
        )
