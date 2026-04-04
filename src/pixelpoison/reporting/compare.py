"""Image comparison — visual diff between original and adversarial."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image
from rich.console import Console
from rich.table import Table

from pixelpoison.scoring.quality import compute_psnr, compute_ssim, compute_linf
from pixelpoison.image.loader import load_image

console = Console()


@dataclass
class ComparisonResult:
    """Result of comparing original and adversarial images."""

    psnr: float
    ssim: float
    linf: float
    mean_abs_diff: float


def compare_images(original_path: str, adversarial_path: str) -> ComparisonResult:
    """Compare an original image with its adversarial version.

    Args:
        original_path: Path to original image.
        adversarial_path: Path to adversarial image.

    Returns:
        ComparisonResult with quality metrics.
    """
    clean_tensor, _ = load_image(original_path)
    adv_tensor, _ = load_image(adversarial_path)

    # Ensure same size
    if clean_tensor.shape != adv_tensor.shape:
        adv_tensor = torch.nn.functional.interpolate(
            adv_tensor,
            size=clean_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    psnr = compute_psnr(clean_tensor, adv_tensor)
    ssim = compute_ssim(clean_tensor, adv_tensor)
    linf = compute_linf(clean_tensor, adv_tensor)
    mean_abs = torch.mean(torch.abs(clean_tensor - adv_tensor)).item()

    return ComparisonResult(psnr=psnr, ssim=ssim, linf=linf, mean_abs_diff=mean_abs)


def print_comparison(result: ComparisonResult) -> None:
    """Print comparison results in a Rich table."""
    table = Table(title="Image Comparison", border_style="dim")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_column("Threshold")
    table.add_column("Status")

    psnr_ok = result.psnr >= 36.0
    ssim_ok = result.ssim >= 0.95

    table.add_row(
        "PSNR",
        f"{result.psnr:.1f} dB",
        ">= 36.0 dB",
        "[green]PASS[/green]" if psnr_ok else "[red]FAIL[/red]",
    )
    table.add_row(
        "SSIM",
        f"{result.ssim:.4f}",
        ">= 0.95",
        "[green]PASS[/green]" if ssim_ok else "[red]FAIL[/red]",
    )
    table.add_row(
        "L-inf",
        f"{result.linf:.4f} ({result.linf * 255:.1f}/255)",
        "<= 16/255",
        "[green]OK[/green]" if result.linf <= 16.0 / 255 else "[yellow]HIGH[/yellow]",
    )
    table.add_row(
        "Mean |diff|",
        f"{result.mean_abs_diff:.6f} ({result.mean_abs_diff * 255:.2f}/255)",
        "",
        "",
    )

    console.print(table)


def generate_diff_heatmap(
    original_path: str, adversarial_path: str, output_path: str, amplify: float = 10.0
) -> None:
    """Generate a pixel difference heatmap image.

    Args:
        original_path: Path to original image.
        adversarial_path: Path to adversarial image.
        output_path: Where to save the heatmap PNG.
        amplify: Amplification factor for visibility.
    """
    clean_tensor, _ = load_image(original_path)
    adv_tensor, _ = load_image(adversarial_path)

    if clean_tensor.shape != adv_tensor.shape:
        adv_tensor = torch.nn.functional.interpolate(
            adv_tensor, size=clean_tensor.shape[2:], mode="bilinear", align_corners=False
        )

    diff = torch.abs(clean_tensor - adv_tensor).squeeze(0)  # (3, H, W)
    # Convert to grayscale magnitude
    gray_diff = diff.mean(dim=0)  # (H, W)
    # Amplify and clamp
    gray_diff = (gray_diff * amplify).clamp(0, 1)

    # Convert to heatmap (red channel = high diff)
    heatmap = torch.zeros(3, *gray_diff.shape)
    heatmap[0] = gray_diff  # Red
    heatmap[1] = gray_diff * 0.3  # Slight green
    heatmap[2] = 0  # No blue

    arr = (heatmap.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(output_path)
