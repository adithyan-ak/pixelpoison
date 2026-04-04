"""Image quality metrics — PSNR and SSIM."""

from __future__ import annotations

import torch
import numpy as np


def compute_psnr(clean: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio between clean and adversarial images.

    Args:
        clean: Original image tensor (1, 3, H, W) in [0, 1].
        adversarial: Adversarial image tensor (1, 3, H, W) in [0, 1].

    Returns:
        PSNR in dB. Higher is better (less distortion).
    """
    mse = torch.mean((clean - adversarial) ** 2).item()
    if mse < 1e-10:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def compute_ssim(clean: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Compute Structural Similarity Index between clean and adversarial images.

    Uses scikit-image's implementation for accuracy.

    Args:
        clean: Original image tensor (1, 3, H, W) in [0, 1].
        adversarial: Adversarial image tensor (1, 3, H, W) in [0, 1].

    Returns:
        SSIM in [0, 1]. Higher is better (more similar).
    """
    from skimage.metrics import structural_similarity

    # Convert to numpy (H, W, C) in [0, 1]
    clean_np = clean.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    adv_np = adversarial.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # Determine appropriate win_size based on image dimensions
    min_dim = min(clean_np.shape[0], clean_np.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3

    ssim_val = structural_similarity(
        clean_np,
        adv_np,
        data_range=1.0,
        channel_axis=2,
        win_size=win_size,
    )
    return float(ssim_val)


def compute_linf(clean: torch.Tensor, adversarial: torch.Tensor) -> float:
    """Compute L-infinity norm of the perturbation.

    Returns:
        Maximum absolute pixel difference in [0, 1] scale.
    """
    return torch.max(torch.abs(clean - adversarial)).item()


def check_quality_gate(psnr: float, ssim: float) -> bool:
    """Check if image quality meets minimum thresholds.

    Thresholds: PSNR >= 36 dB, SSIM >= 0.95
    """
    return psnr >= 36.0 and ssim >= 0.95
