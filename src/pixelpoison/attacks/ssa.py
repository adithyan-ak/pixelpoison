"""SSA: Spectrum Simulation Attack (Tier 1).

Research basis: Long et al., "Frequency Domain Model Augmentation for
Adversarial Attack" (ECCV 2022, arXiv:2207.05382).

Key insight: Instead of needing many surrogate models, SSA simulates model
diversity by randomly perturbing DCT spectrum coefficients before each gradient
computation. This makes the perturbation transferable across models with
different frequency sensitivities.

At each iteration:
1. Sample N spectrum-transformed copies of the current adversarial image
2. Average gradients across all N copies
3. Accumulate with MI-FGSM momentum
4. PGD sign-gradient update
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


def _dct_2d(x: torch.Tensor) -> torch.Tensor:
    """Full-image 2D DCT via FFT (differentiable).

    Args:
        x: (B, C, H, W) tensor.

    Returns:
        DCT coefficients (B, C, H, W).
    """
    # Type-II DCT via real FFT
    # DCT along height
    n_h = x.shape[2]
    v_h = torch.cat([x[:, :, ::2, :], x[:, :, 1::2, :].flip(dims=[2])], dim=2)
    fft_h = torch.fft.rfft(v_h, dim=2, n=n_h)
    k_h = torch.arange(n_h, device=x.device, dtype=x.dtype)
    shift_h = torch.exp(-1j * torch.pi * k_h / (2 * n_h)).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    dct_h = (fft_h * shift_h[:, :, : fft_h.shape[2], :]).real

    # DCT along width
    n_w = x.shape[3]
    v_w = torch.cat([dct_h[:, :, :, ::2], dct_h[:, :, :, 1::2].flip(dims=[3])], dim=3)
    fft_w = torch.fft.rfft(v_w, dim=3, n=n_w)
    k_w = torch.arange(n_w, device=x.device, dtype=x.dtype)
    shift_w = torch.exp(-1j * torch.pi * k_w / (2 * n_w)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    dct_out = (fft_w * shift_w[:, :, :, : fft_w.shape[3]]).real

    return dct_out


def _idct_2d(x: torch.Tensor) -> torch.Tensor:
    """Full-image 2D inverse DCT via FFT (differentiable).

    Args:
        x: DCT coefficients (B, C, H, W).

    Returns:
        Spatial domain tensor (B, C, H, W).
    """
    # Inverse DCT along width
    n_w = x.shape[3]
    k_w = torch.arange(n_w, device=x.device, dtype=x.dtype)
    shift_w = torch.exp(1j * torch.pi * k_w / (2 * n_w)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    # Extend to complex
    x_complex = x * shift_w[:, :, :, : x.shape[3]]
    ifft_w = torch.fft.irfft(x_complex, dim=3, n=n_w)
    # Unshuffle
    out_w = torch.zeros_like(ifft_w)
    half_w = (n_w + 1) // 2
    out_w[:, :, :, ::2] = ifft_w[:, :, :, :half_w]
    out_w[:, :, :, 1::2] = ifft_w[:, :, :, half_w:].flip(dims=[3])

    # Inverse DCT along height
    n_h = out_w.shape[2]
    k_h = torch.arange(n_h, device=x.device, dtype=x.dtype)
    shift_h = torch.exp(1j * torch.pi * k_h / (2 * n_h)).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    w_complex = out_w * shift_h[:, :, : out_w.shape[2], :]
    ifft_h = torch.fft.irfft(w_complex, dim=2, n=n_h)
    out_h = torch.zeros_like(ifft_h)
    half_h = (n_h + 1) // 2
    out_h[:, :, ::2, :] = ifft_h[:, :, :half_h, :]
    out_h[:, :, 1::2, :] = ifft_h[:, :, half_h:, :].flip(dims=[2])

    return out_h


def _spectrum_transform(
    x: torch.Tensor,
    rho: float = 0.5,
    sigma: float = 16.0 / 255.0,
) -> torch.Tensor:
    """Apply SSA spectrum transformation.

    T(x) = IDCT( DCT(x + xi) * M )
    where xi ~ N(0, sigma^2) and M ~ U(1-rho, 1+rho).

    Args:
        x: Image tensor (B, C, H, W) in [0, 1].
        rho: Width of uniform scaling range for DCT coefficients.
        sigma: Standard deviation of input noise.

    Returns:
        Spectrum-transformed image (B, C, H, W).
    """
    # Add Gaussian noise
    xi = torch.randn_like(x) * sigma
    x_noisy = x + xi

    # Forward DCT
    dct_coeffs = _dct_2d(x_noisy)

    # Random multiplicative mask on DCT coefficients
    m = 1.0 - rho + 2.0 * rho * torch.rand_like(dct_coeffs)
    dct_masked = dct_coeffs * m

    # Inverse DCT
    x_transformed = _idct_2d(dct_masked)

    return x_transformed.clamp(0, 1)


class SSAStrategy(AttackStrategy):
    """SSA: Spectrum Simulation Attack.

    Simulates model diversity via DCT-domain augmentation. At each iteration,
    N spectrum-transformed copies are generated, gradients are averaged,
    and accumulated with MI-FGSM momentum for the PGD update.

    This is particularly effective for transferability because different VLMs
    have different frequency sensitivities, and SSA implicitly optimizes
    across a distribution of frequency responses.
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
        rho = 0.5        # DCT coefficient scaling range
        sigma = config.epsilon  # Input noise std
        n_samples = 20   # Spectrum transforms per iteration
        mu = 1.0         # MI-FGSM momentum decay

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
            avg_grad_norm = avg_grad / (avg_grad.abs().mean(dim=[1, 2, 3], keepdim=True) + 1e-12)
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
