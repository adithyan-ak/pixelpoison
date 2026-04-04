"""Differentiable simulation of target VLM preprocessing pipelines."""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from pixelpoison.targets.profiles import TargetProfile, PROFILES, AUTO_WEIGHTS


class TargetPreprocessingSim(nn.Module):
    """Differentiable simulation of a target VLM's image preprocessing.

    When a specific profile is given, applies that VLM's known preprocessing
    (resize, tile extraction) using differentiable operations.

    When profile is None (auto mode), randomly samples a target profile each
    forward call, making the perturbation robust to diverse preprocessing.
    """

    def __init__(self, profile: Optional[TargetProfile] = None):
        super().__init__()
        self.profile = profile
        self.auto_mode = profile is None

        # Pre-compute auto mode sampling distribution
        if self.auto_mode:
            self._auto_names = list(AUTO_WEIGHTS.keys())
            self._auto_probs = list(AUTO_WEIGHTS.values())

    def _apply_profile(self, x: torch.Tensor, profile: TargetProfile) -> torch.Tensor:
        """Apply a specific target's preprocessing (differentiable).

        Args:
            x: Image tensor (B, 3, H, W) in [0, 1].
            profile: Target VLM profile.

        Returns:
            Preprocessed tensor, resized to 224x224 (CLIP input size).
        """
        _, _, h, w = x.shape
        max_res = profile.max_resolution

        # Step 1: Resize to target's max resolution
        if profile.preserve_aspect_ratio:
            # Scale so longest (or shortest for GPT-4o) edge matches max_res
            scale = max_res / max(h, w)
            if scale < 1.0:
                new_h = int(h * scale)
                new_w = int(w * scale)
                x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
        else:
            # Resize to square
            x = F.interpolate(x, size=(max_res, max_res), mode="bilinear", align_corners=False)

        # Step 2: Tile extraction (if applicable)
        if profile.uses_tiling and profile.tile_size:
            _, _, h_new, w_new = x.shape
            ts = profile.tile_size
            if h_new >= ts and w_new >= ts:
                # Extract center tile (differentiable crop via grid_sample)
                cy, cx = h_new // 2, w_new // 2
                y1 = max(0, cy - ts // 2)
                x1 = max(0, cx - ts // 2)
                x = x[:, :, y1 : y1 + ts, x1 : x1 + ts]

        # Step 3: Resize to 224x224 for CLIP/SigLIP input
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply target preprocessing simulation.

        In auto mode, randomly samples a profile each call for stochastic
        robustness across all target VLMs.

        Args:
            x: Image tensor (B, 3, H, W) in [0, 1].

        Returns:
            Preprocessed tensor (B, 3, 224, 224).
        """
        if not self.auto_mode:
            return self._apply_profile(x, self.profile)

        # Auto mode: random profile sampling
        chosen = random.choices(self._auto_names, weights=self._auto_probs, k=1)[0]

        if chosen is None:
            # No preprocessing — just resize to 224x224
            return F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        profile = PROFILES[chosen]
        return self._apply_profile(x, profile)
