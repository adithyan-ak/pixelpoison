"""PGD Baseline attack with X-Transfer dynamic ensemble weighting."""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


class PGDBaseline(AttackStrategy):
    """Standard Projected Gradient Descent with X-Transfer dynamic ensemble weighting.

    Research basis: Madry et al. (ICLR 2018) + X-Transfer (ICML 2025).

    At each iteration:
    1. Compute per-model gradients
    2. Weight gradients by diversity (X-Transfer): models providing diverse
       gradient signals get higher weight, preventing ensemble collapse
    3. Update perturbation via sign gradient + project to epsilon-ball
    """

    @property
    def name(self) -> str:
        return "pgd_baseline"

    @property
    def required_tier(self) -> int:
        return 1

    @property
    def required_models(self) -> list[str]:
        return []  # Uses whatever models are loaded in the ensemble

    def _compute_xtransfer_weights(
        self, gradients: dict[str, torch.Tensor], temperature: float = 0.5
    ) -> dict[str, float]:
        """Compute X-Transfer dynamic weights based on gradient diversity.

        Models whose gradients are dissimilar to others get higher weight,
        promoting gradient diversity and reducing overfitting to any single surrogate.
        """
        model_ids = list(gradients.keys())
        if len(model_ids) <= 1:
            return {mid: 1.0 for mid in model_ids}

        # Flatten gradients for cosine similarity
        flat_grads = {
            mid: g.flatten() for mid, g in gradients.items()
        }

        diversities = {}
        for i, mid_i in enumerate(model_ids):
            sims = []
            for j, mid_j in enumerate(model_ids):
                if i == j:
                    continue
                sim = F.cosine_similarity(
                    flat_grads[mid_i].unsqueeze(0),
                    flat_grads[mid_j].unsqueeze(0),
                ).item()
                sims.append(sim)
            # Higher diversity = lower average similarity to others
            diversities[mid_i] = 1.0 - (sum(sims) / len(sims))

        # Softmax with temperature
        div_tensor = torch.tensor([diversities[mid] for mid in model_ids])
        weights = F.softmax(div_tensor / temperature, dim=0)

        return {mid: weights[i].item() for i, mid in enumerate(model_ids)}

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

        # Seed for reproducibility
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # Initialize perturbation
        delta = torch.zeros_like(clean_image, requires_grad=True, device=device)

        best_score = -float("inf")
        best_delta = delta.data.clone()
        no_improve_count = 0

        for iteration in range(config.iterations):
            # Construct adversarial image
            x_adv = (clean_image + delta).clamp(0, 1)

            # Compute per-model gradients sequentially
            per_model_grads = {}
            per_model_losses = {}

            for model_id in ensemble.loaded_models:
                if delta.grad is not None:
                    delta.grad.zero_()

                # Forward pass through this model
                x_adv_curr = (clean_image + delta).clamp(0, 1)
                emb = ensemble.encode_image_single(x_adv_curr, model_id)
                target_emb = target_embeddings[model_id]

                loss = -F.cosine_similarity(emb, target_emb, dim=-1).mean()
                loss.backward()

                per_model_grads[model_id] = delta.grad.data.clone()
                per_model_losses[model_id] = -loss.item()  # Store as positive similarity

            # X-Transfer dynamic weighting
            weights = self._compute_xtransfer_weights(per_model_grads)

            # Weighted gradient
            weighted_grad = torch.zeros_like(delta.data)
            for model_id, grad in per_model_grads.items():
                weighted_grad += weights[model_id] * grad

            # PGD update: sign gradient step
            with torch.no_grad():
                delta.data = delta.data - config.step_size * weighted_grad.sign()

                # Project to epsilon-ball
                delta.data = delta.data.clamp(-config.epsilon, config.epsilon)

                # Ensure valid image range
                delta.data = (clean_image + delta.data).clamp(0, 1) - clean_image

            # Track best score
            current_score = sum(per_model_losses.values()) / len(per_model_losses)

            if current_score > best_score:
                best_score = current_score
                best_delta = delta.data.clone()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Progress callback
            if progress_callback:
                progress_callback(iteration, current_score)

            # Early stopping
            early_stop_threshold = 0.75 if config.quick else 0.85
            if current_score > early_stop_threshold:
                break
            if no_improve_count >= 50:
                break
            if torch.isnan(delta.data).any() or torch.isinf(delta.data).any():
                delta.data = best_delta
                break

        # Final adversarial image
        adversarial = (clean_image + best_delta).clamp(0, 1)

        # Compute final scores
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
            metadata={"final_weights": weights},
        )
