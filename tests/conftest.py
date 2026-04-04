"""Shared test fixtures for PixelPoison."""

from __future__ import annotations

import torch
import pytest
from PIL import Image
import tempfile
from pathlib import Path


@pytest.fixture
def small_image_tensor() -> torch.Tensor:
    """A small 64x64 random image tensor in [0, 1] range, shape (1, 3, 64, 64)."""
    return torch.rand(1, 3, 64, 64)


@pytest.fixture
def small_image_path(tmp_path: Path) -> Path:
    """Save a small test image to disk and return its path."""
    img = Image.fromarray(
        (torch.rand(64, 64, 3).numpy() * 255).astype("uint8"), mode="RGB"
    )
    path = tmp_path / "test_image.png"
    img.save(path)
    return path


@pytest.fixture
def small_jpeg_path(tmp_path: Path) -> Path:
    """Save a small JPEG test image to disk."""
    img = Image.fromarray(
        (torch.rand(64, 64, 3).numpy() * 255).astype("uint8"), mode="RGB"
    )
    path = tmp_path / "test_image.jpg"
    img.save(path, quality=85)
    return path
