"""Input augmentation pipeline for adversarial robustness during optimization.

Stochastic differentiable augmentations applied during optimization to make
perturbations robust to unknown target VLM preprocessing.
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class AugmentationPipeline(nn.Module):
    """Stochastic differentiable augmentation for adversarial optimization.

    All augmentations use differentiable operations so gradients flow back
    to the perturbation. Each augmentation is applied with independent
    probability per forward call.

    Augmentations:
    1. Random resize: scale ∈ U(0.85, 1.15), p=0.5
    2. Gaussian blur: σ ∈ U(0.1, 1.0), p=0.3
    3. Color jitter: brightness ∈ U(-0.05, 0.05), p=0.2
    """

    def __init__(
        self,
        enable_resize: bool = True,
        enable_blur: bool = True,
        enable_jitter: bool = True,
    ):
        super().__init__()
        self.enable_resize = enable_resize
        self.enable_blur = enable_blur
        self.enable_jitter = enable_jitter

    def _random_resize(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable random resize."""
        _, _, h, w = x.shape
        scale = random.uniform(0.85, 1.15)
        new_h = max(16, int(h * scale))
        new_w = max(16, int(w * scale))
        x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        # Resize back to original
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable Gaussian blur."""
        sigma = random.uniform(0.1, 1.0)
        kernel_size = 2 * int(3 * sigma) + 1
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Create 1D Gaussian kernel
        coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Separable 2D convolution
        kernel_2d = kernel_1d.outer(kernel_1d)
        kernel_2d = kernel_2d.expand(3, 1, -1, -1)

        padding = kernel_size // 2
        return F.conv2d(x, kernel_2d, padding=padding, groups=3)

    def _color_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable brightness jitter."""
        brightness = random.uniform(-0.05, 0.05)
        return (x + brightness).clamp(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic augmentation chain.

        Args:
            x: Image tensor (B, 3, H, W) in [0, 1].

        Returns:
            Augmented image tensor (B, 3, H, W) in [0, 1].
        """
        if self.enable_resize and random.random() < 0.7:
            x = self._random_resize(x)

        if self.enable_blur and random.random() < 0.3:
            x = self._gaussian_blur(x)

        if self.enable_jitter and random.random() < 0.2:
            x = self._color_jitter(x)

        return x.clamp(0, 1)
