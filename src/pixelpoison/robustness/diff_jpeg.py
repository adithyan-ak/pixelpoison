"""Differentiable JPEG simulation for adversarial robustness.

Based on: "Differentiable JPEG: The Devil is in the Details" (Reich et al., WACV 2024)

Replaces the non-differentiable rounding in JPEG quantization with a smooth
approximation: x - sin(2*pi*x) / (2*pi), allowing gradient flow through
JPEG compression during adversarial optimization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Standard JPEG luminance quantization table
LUMINANCE_Q = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float32)

# Standard JPEG chrominance quantization table
CHROMINANCE_Q = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=torch.float32)


def _quality_to_scale(quality: int) -> float:
    """Convert JPEG quality (1-100) to quantization scale factor."""
    if quality < 50:
        return 5000.0 / quality
    return 200.0 - 2.0 * quality


def _scaled_quant_table(base_table: torch.Tensor, quality: int) -> torch.Tensor:
    """Scale a quantization table by JPEG quality factor."""
    scale = _quality_to_scale(quality)
    table = torch.floor((base_table * scale + 50.0) / 100.0)
    return table.clamp(min=1.0)


def _dct_matrix() -> torch.Tensor:
    """Compute the 8x8 DCT-II transformation matrix."""
    n = 8
    dct = torch.zeros(n, n)
    for k in range(n):
        for i in range(n):
            if k == 0:
                dct[k, i] = 1.0 / math.sqrt(n)
            else:
                dct[k, i] = math.sqrt(2.0 / n) * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    return dct


def _smooth_round(x: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation of rounding.

    Uses: x - sin(2*pi*x) / (2*pi)
    This function closely approximates round(x) but has smooth gradients.
    """
    return x - torch.sin(2 * math.pi * x) / (2 * math.pi)


def _rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB image to YCbCr color space (differentiable).

    Args:
        rgb: (B, 3, H, W) in [0, 255] range.

    Returns:
        (B, 3, H, W) in YCbCr space.
    """
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return torch.cat([y, cb, cr], dim=1)


def _ycbcr_to_rgb(ycbcr: torch.Tensor) -> torch.Tensor:
    """Convert YCbCr image to RGB color space (differentiable).

    Args:
        ycbcr: (B, 3, H, W) in YCbCr space.

    Returns:
        (B, 3, H, W) in [0, 255] range.
    """
    y, cb, cr = ycbcr[:, 0:1], ycbcr[:, 1:2], ycbcr[:, 2:3]
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return torch.cat([r, g, b], dim=1)


class DiffJPEG(nn.Module):
    """Differentiable JPEG compression/decompression layer.

    Simulates JPEG compression with smooth rounding approximation
    so gradients can flow through the compression pipeline.

    Usage in adversarial optimization:
        diff_jpeg = DiffJPEG(quality=85)
        x_compressed = diff_jpeg(x_adv)  # Differentiable
        loss = compute_loss(x_compressed)
        loss.backward()  # Gradients flow through JPEG simulation
    """

    def __init__(self, quality: int = 85):
        super().__init__()
        self.quality = quality

        # Pre-compute DCT matrix
        dct_mat = _dct_matrix()
        self.register_buffer("dct_matrix", dct_mat)
        self.register_buffer("idct_matrix", dct_mat.t())

        # Pre-compute quantization tables
        lum_q = _scaled_quant_table(LUMINANCE_Q, quality)
        chrom_q = _scaled_quant_table(CHROMINANCE_Q, quality)
        self.register_buffer("lum_q", lum_q)
        self.register_buffer("chrom_q", chrom_q)

    def _block_dct(self, blocks: torch.Tensor) -> torch.Tensor:
        """Apply 2D DCT to 8x8 blocks.

        Args:
            blocks: (N, 8, 8) tensor of pixel blocks.

        Returns:
            (N, 8, 8) tensor of DCT coefficients.
        """
        # 2D DCT = D * block * D^T
        return torch.matmul(torch.matmul(self.dct_matrix, blocks), self.idct_matrix)

    def _block_idct(self, blocks: torch.Tensor) -> torch.Tensor:
        """Apply 2D inverse DCT to 8x8 blocks."""
        return torch.matmul(torch.matmul(self.idct_matrix, blocks), self.dct_matrix)

    def _image_to_blocks(self, x: torch.Tensor) -> torch.Tensor:
        """Split a single-channel image into 8x8 blocks.

        Args:
            x: (B, 1, H, W) tensor.

        Returns:
            (B*num_blocks, 8, 8) tensor.
        """
        b, c, h, w = x.shape
        # Pad to multiple of 8
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, h_pad, w_pad = x.shape
        # Reshape into blocks
        x = x.reshape(b, 1, h_pad // 8, 8, w_pad // 8, 8)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, 8, 8)
        return x

    def _blocks_to_image(self, blocks: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
        """Reassemble 8x8 blocks into a single-channel image.

        Args:
            blocks: (B*num_blocks, 8, 8) tensor.
            b: Batch size.
            h: Original height (before padding).
            w: Original width (before padding).

        Returns:
            (B, 1, H, W) tensor cropped to original size.
        """
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        h_pad = h + pad_h
        w_pad = w + pad_w

        blocks = blocks.reshape(b, h_pad // 8, w_pad // 8, 1, 8, 8)
        blocks = blocks.permute(0, 3, 1, 4, 2, 5).reshape(b, 1, h_pad, w_pad)
        return blocks[:, :, :h, :w]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply differentiable JPEG compression.

        Args:
            x: Image tensor (B, 3, H, W) in [0, 1] range.

        Returns:
            Compressed image tensor (B, 3, H, W) in [0, 1] range.
        """
        b, c, h, w = x.shape

        # Scale to [0, 255]
        x_255 = x * 255.0

        # RGB → YCbCr
        ycbcr = _rgb_to_ycbcr(x_255)

        channels = []
        for ch in range(3):
            channel = ycbcr[:, ch:ch + 1]  # (B, 1, H, W)

            # Shift by -128 (standard JPEG)
            channel = channel - 128.0

            # Split into 8x8 blocks
            blocks = self._image_to_blocks(channel)

            # Apply DCT
            dct_blocks = self._block_dct(blocks)

            # Quantize with smooth rounding
            q_table = self.lum_q if ch == 0 else self.chrom_q
            q_table = q_table.to(x.device).unsqueeze(0)  # (1, 8, 8)
            quantized = _smooth_round(dct_blocks / q_table)

            # Dequantize
            dequantized = quantized * q_table

            # Inverse DCT
            spatial = self._block_idct(dequantized)

            # Reassemble
            channel_out = self._blocks_to_image(spatial, b, h, w)

            # Shift back
            channel_out = channel_out + 128.0
            channels.append(channel_out)

        ycbcr_out = torch.cat(channels, dim=1)

        # YCbCr → RGB
        rgb_out = _ycbcr_to_rgb(ycbcr_out)

        # Scale back to [0, 1] and clamp
        return (rgb_out / 255.0).clamp(0, 1)
