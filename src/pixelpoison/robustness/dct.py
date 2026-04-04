"""DCT mid-frequency targeting for JPEG-robust perturbations.

Based on: "Invisible Injections" (Pathade et al., arXiv:2507.22304, July 2025)
and "Frequency Domain Model Augmentation" (Long et al., ECCV 2022).

Constrains perturbation to DCT mid-frequency band (indices 2-5 in zigzag order),
which survives JPEG quantization better than high frequencies while being less
perceptually visible than low frequencies.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Zigzag scan order for 8x8 block — maps linear index to (row, col)
ZIGZAG_ORDER = [
    (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2),
    (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3), (0, 4), (0, 5),
    (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2),
    (3, 3), (2, 4), (1, 5), (0, 6), (0, 7), (1, 6), (2, 5), (3, 4),
    (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4),
    (3, 5), (2, 6), (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3),
    (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
    (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7),
]


def _dct_matrix() -> torch.Tensor:
    """Compute the 8x8 DCT-II transformation matrix."""
    n = 8
    dct = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            if k == 0:
                dct[k, i] = 1.0 / math.sqrt(n)
            else:
                dct[k, i] = math.sqrt(2.0 / n) * math.cos(
                    math.pi * (2 * i + 1) * k / (2 * n)
                )
    return dct


class DCTMidFrequencyMask(nn.Module):
    """Masks perturbation to allow only mid-frequency DCT components.

    Frequencies are indexed by zigzag order within each 8x8 block:
    - Low frequencies (indices 0-1): affect overall color/brightness — too visible
    - Mid frequencies (indices 2-5): perceptually subtle, survive JPEG
    - High frequencies (indices 6-7): destroyed by JPEG quantization

    The mask is applied in DCT domain: transform to DCT, zero out non-mid
    frequencies, transform back to spatial domain.
    """

    def __init__(self, low_cutoff: int = 2, high_cutoff: int = 5):
        """
        Args:
            low_cutoff: Minimum zigzag frequency index to allow (inclusive).
            high_cutoff: Maximum zigzag frequency index to allow (inclusive).
                Indices are grouped by diagonal sum (row + col):
                0 = DC, 1 = first diagonal, ..., 14 = last.
        """
        super().__init__()
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

        # Pre-compute DCT matrices
        dct_mat = _dct_matrix()
        self.register_buffer("dct_matrix", dct_mat)
        self.register_buffer("idct_matrix", dct_mat.t())

        # Create the frequency band mask (8x8)
        mask = torch.zeros(8, 8)
        for idx, (r, c) in enumerate(ZIGZAG_ORDER):
            freq_band = r + c  # Diagonal sum as frequency measure
            if low_cutoff <= freq_band <= high_cutoff:
                mask[r, c] = 1.0
        self.register_buffer("mask", mask)

    def forward(self, perturbation: torch.Tensor) -> torch.Tensor:
        """Apply mid-frequency mask to perturbation.

        Args:
            perturbation: (B, C, H, W) tensor.

        Returns:
            Filtered perturbation with only mid-frequency components.
        """
        b, c, h, w = perturbation.shape

        # Pad to multiple of 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            perturbation = F.pad(perturbation, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, h_pad, w_pad = perturbation.shape

        # Process each channel
        result_channels = []
        for ch in range(c):
            channel = perturbation[:, ch:ch + 1]  # (B, 1, H, W)

            # Reshape into 8x8 blocks: (B*num_blocks, 8, 8)
            blocks = channel.reshape(b, 1, h_pad // 8, 8, w_pad // 8, 8)
            blocks = blocks.permute(0, 2, 4, 1, 3, 5).reshape(-1, 8, 8)

            # Forward DCT
            dct_blocks = torch.matmul(
                torch.matmul(self.dct_matrix, blocks), self.idct_matrix
            )

            # Apply frequency mask
            dct_blocks = dct_blocks * self.mask.unsqueeze(0)

            # Inverse DCT
            spatial = torch.matmul(
                torch.matmul(self.idct_matrix, dct_blocks), self.dct_matrix
            )

            # Reassemble
            spatial = spatial.reshape(b, h_pad // 8, w_pad // 8, 1, 8, 8)
            spatial = spatial.permute(0, 3, 1, 4, 2, 5).reshape(b, 1, h_pad, w_pad)
            result_channels.append(spatial[:, :, :h, :w])

        return torch.cat(result_channels, dim=1)
