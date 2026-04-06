"""SSA: Spectrum Simulation Attack (Tier 1).

Research basis: Long et al., "Frequency Domain Model Augmentation for
Adversarial Attack" (ECCV 2022, arXiv:2207.05382).

Key insight: Instead of needing many surrogate models, SSA simulates model
diversity by randomly perturbing frequency spectrum coefficients before each
gradient computation. This makes the perturbation transferable across models
with different frequency sensitivities.

Uses torch.fft.fft2/ifft2 for correct differentiable spectrum transformation.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


def _spectrum_transform(
    x: torch.Tensor,
    rho: float = 0.5,
    sigma: float = 16.0 / 255.0,
) -> torch.Tensor:
    """Apply SSA spectrum transformation via FFT (differentiable).

    T(x) = IFFT( FFT(x + xi) * M )
    where xi ~ N(0, sigma^2) and M ~ U(1-rho, 1+rho) applied to magnitude.

    Args:
        x: Image tensor (B, C, H, W) in [0, 1].
        rho: Width of uniform scaling range for spectrum coefficients.
        sigma: Standard deviation of input noise.

    Returns:
        Spectrum-transformed image (B, C, H, W).
    """
    # Add Gaussian noise for input diversity
    xi = torch.randn_like(x) * sigma
    x_noisy = x + xi

    # Forward FFT (full complex, differentiable)
    freq = torch.fft.fft2(x_noisy)

    # Random multiplicative mask on spectrum magnitude
    # M ~ U(1-rho, 1+rho) — perturbs each frequency component independently
    m = 1.0 - rho + 2.0 * rho * torch.rand(
        freq.shape, device=x.device, dtype=x.dtype
    )
    freq_masked = freq * m

    # Inverse FFT back to spatial domain
    x_transformed = torch.fft.ifft2(freq_masked).real

    return x_transformed.clamp(0, 1)


class SSAStrategy(AttackStrategy):
    """SSA: Spectrum Simulation Attack.

    Simulates model diversity via frequency-domain augmentation. At each
    iteration, N spectrum-transformed copies are generated, gradients are
    averaged, and accumulated with MI-FGSM momentum for the PGD update.

    N is kept small (5) to balance diversity vs compute cost with an ensemble.
    """

    @property
    def name(self) -> str:
        return "ssa"

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

        # Hyperparameters
        rho = 0.5                # Spectrum coefficient scaling range
        sigma = config.epsilon   # Input noise std
        n_samples = 5            # Spectrum transforms per iteration (reduced for ensemble)
        mu = 1.0                 # MI-FGSM momentum decay

        # Initialize perturbation and momentum
        delta = torch.zeros_like(clean_image, requires_grad=True, device=device)
        momentum = torch.zeros_like(clean_image)

        best_score = -float("inf")
        best_delta = delta.data.clone()
        no_improve_count = 0
        iteration = 0

        for iteration in range(config.iterations):
            # Accumulate gradients over N spectrum-transformed copies
            avg_grad = torch.zeros_like(clean_image)

            for _ in range(n_samples):
                if delta.grad is not None:
                    delta.grad.zero_()

                x_adv = (clean_image + delta).clamp(0, 1)

                # Apply spectrum transformation
                x_transformed = _spectrum_transform(x_adv, rho=rho, sigma=sigma)

                # Compute loss across ensemble
                total_loss = torch.tensor(0.0, device=device)
                for model_id in ensemble.loaded_models:
                    emb = ensemble.encode_image_single(x_transformed, model_id)
                    target_emb = target_embeddings[model_id]
                    loss = -F.cosine_similarity(emb, target_emb, dim=-1).mean()
                    total_loss = total_loss + loss

                total_loss = total_loss / ensemble.model_count
                total_loss.backward()

                avg_grad += delta.grad.data.clone()

            avg_grad = avg_grad / n_samples

            # MI-FGSM momentum accumulation
            avg_grad_norm = avg_grad / (
                avg_grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-12
            )
            momentum = mu * momentum + avg_grad_norm

            # PGD update
            with torch.no_grad():
                delta.data = delta.data - config.step_size * momentum.sign()
                delta.data = delta.data.clamp(-config.epsilon, config.epsilon)
                delta.data = (clean_image + delta.data).clamp(0, 1) - clean_image

            # Periodically evaluate full score
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
            metadata={"rho": rho, "n_samples": n_samples},
        )
