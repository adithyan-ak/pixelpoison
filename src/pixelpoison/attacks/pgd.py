"""PGD attack with MI-FGSM momentum, DIM, TIM, and X-Transfer ensemble weighting.

Research basis:
- PGD: Madry et al. (ICLR 2018)
- MI-FGSM momentum: Dong et al. (CVPR 2018)
- DIM (Diverse Input Method): Xie et al. (CVPR 2019)
- TIM (Translation-Invariant Method): Dong et al. (CVPR 2019)
- SIM (Scale-Invariant Method): Lin et al. (ICLR 2020)
- X-Transfer dynamic ensemble weighting (ICML 2025)
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.augmentation.transforms import AugmentationPipeline
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


def _get_gaussian_kernel_2d(kernel_size: int = 15, sigma: float = 3.0) -> torch.Tensor:
    """Create a 2D Gaussian kernel for TIM (Translation-Invariant Method).

    Convolving the gradient with this kernel smooths out position-specific
    features, making the perturbation more transferable across models with
    different spatial sensitivities.
    """
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = g / g.sum()
    kernel_2d = kernel_1d.outer(kernel_1d)
    # Shape for grouped conv: (3, 1, K, K) to apply per-channel
    return kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)


class PGDBaseline(AttackStrategy):
    """PGD with MI-FGSM momentum + DIM + TIM + SIM + X-Transfer.

    Key improvements over vanilla PGD for transferability:
    1. MI-FGSM momentum (mu=1.0): stabilizes gradient direction across iterations,
       preventing oscillation and improving convergence to transferable features.
    2. DIM (p=0.7): random resize+pad at each iteration prevents overfitting
       to specific spatial features of the surrogate models.
    3. TIM: Gaussian kernel convolution on gradients smooths position-dependent
       signals, improving transfer to models with different receptive fields.
    4. SIM: compute loss over multiple scaled copies of the image, forcing the
       perturbation to be effective at multiple resolutions.
    5. X-Transfer: diversity-weighted ensemble gradients.
    """

    @property
    def name(self) -> str:
        return "pgd_baseline"

    @property
    def required_tier(self) -> int:
        return 1

    @property
    def required_models(self) -> list[str]:
        return []

    def _compute_xtransfer_weights(
        self, gradients: dict[str, torch.Tensor], temperature: float = 0.5
    ) -> dict[str, float]:
        """Compute X-Transfer dynamic weights based on gradient diversity."""
        model_ids = list(gradients.keys())
        if len(model_ids) <= 1:
            return {mid: 1.0 for mid in model_ids}

        flat_grads = {mid: g.flatten() for mid, g in gradients.items()}

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
            diversities[mid_i] = 1.0 - (sum(sims) / len(sims))

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

        if config.seed is not None:
            torch.manual_seed(config.seed)

        # --- Setup DIM (input diversity) ---
        augmenter = AugmentationPipeline(
            enable_resize=True, enable_blur=True, enable_jitter=True,
        ).to(device)

        # --- Setup TIM (Gaussian kernel for gradient smoothing) ---
        tim_kernel = _get_gaussian_kernel_2d(kernel_size=15, sigma=3.0).to(device)
        tim_padding = 15 // 2

        # --- SIM: number of scale copies ---
        n_scale_copies = 5

        # Initialize perturbation and momentum
        delta = torch.zeros_like(clean_image, requires_grad=True, device=device)
        momentum = torch.zeros_like(clean_image, device=device)
        mu = 1.0  # MI-FGSM decay factor (standard: 1.0)
        alpha = config.step_size

        best_score = -float("inf")
        best_delta = delta.data.clone()
        no_improve_count = 0

        for iteration in range(config.iterations):
            if delta.grad is not None:
                delta.grad.zero_()

            # Compute per-model gradients sequentially (memory-efficient)
            per_model_grads = {}
            per_model_losses = {}

            for model_id in ensemble.loaded_models:
                if delta.grad is not None:
                    delta.grad.zero_()

                # --- SIM: accumulate loss across scale copies ---
                scale_loss = torch.tensor(0.0, device=device)
                for si in range(n_scale_copies):
                    scale_factor = 1.0 / (2 ** si)
                    x_adv = (clean_image + delta).clamp(0, 1)

                    if si > 0:
                        x_scaled = x_adv * scale_factor
                    else:
                        x_scaled = x_adv

                    # --- DIM: stochastic input diversity ---
                    x_div = augmenter(x_scaled)

                    emb = ensemble.encode_image_single(x_div, model_id)
                    target_emb = target_embeddings[model_id]
                    loss = -F.cosine_similarity(emb, target_emb, dim=-1).mean()
                    scale_loss = scale_loss + loss

                scale_loss = scale_loss / n_scale_copies
                scale_loss.backward()

                per_model_grads[model_id] = delta.grad.data.clone()
                per_model_losses[model_id] = -scale_loss.item()

            # X-Transfer dynamic weighting
            weights = self._compute_xtransfer_weights(per_model_grads)

            # Weighted gradient aggregation
            weighted_grad = torch.zeros_like(delta.data)
            for model_id, grad in per_model_grads.items():
                weighted_grad += weights[model_id] * grad

            # --- TIM: smooth gradient with Gaussian kernel ---
            weighted_grad = F.conv2d(
                weighted_grad, tim_kernel, padding=tim_padding, groups=3,
            )

            # --- MI-FGSM: momentum update ---
            # Normalize gradient by L1 norm (standard MI-FGSM normalization)
            grad_norm = weighted_grad / (torch.mean(torch.abs(weighted_grad)) + 1e-12)
            momentum = mu * momentum + grad_norm

            # PGD update: sign of momentum
            with torch.no_grad():
                delta.data = delta.data - alpha * momentum.sign()
                delta.data = delta.data.clamp(-config.epsilon, config.epsilon)
                delta.data = (clean_image + delta.data).clamp(0, 1) - clean_image

            # Track best score
            current_score = sum(per_model_losses.values()) / len(per_model_losses)

            if current_score > best_score:
                best_score = current_score
                best_delta = delta.data.clone()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if progress_callback:
                progress_callback(iteration, current_score)

            # Early stopping (relaxed thresholds)
            if current_score > 0.95:
                break
            if no_improve_count >= 200:
                break
            if torch.isnan(delta.data).any() or torch.isinf(delta.data).any():
                delta.data = best_delta
                break

        # Final adversarial image
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
            metadata={"final_weights": weights, "momentum_decay": mu},
        )
