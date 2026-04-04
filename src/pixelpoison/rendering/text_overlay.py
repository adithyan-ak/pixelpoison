"""Covert text trigger rendering for CoTTA strategy.

Renders payload text onto an image at very low opacity and small font size,
creating a barely-visible "semantic anchor" that the perturbation amplifies.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class TextTriggerParams:
    """Parameters for a covert text trigger overlay."""

    position: tuple[int, int]  # (x, y) pixel coords
    font_size: int  # 8-14 pixels (below 8 causes rendering issues on some platforms)
    rotation: float  # 0-360 degrees
    opacity: float  # 0.02-0.10 (2-10%)
    color: tuple[int, int, int]  # RGB color


def _get_local_color(image_arr: np.ndarray, x: int, y: int, radius: int = 10) -> tuple[int, ...]:
    """Sample mean color from the local image region around (x, y)."""
    h, w = image_arr.shape[:2]
    y1 = max(0, y - radius)
    y2 = min(h, y + radius)
    x1 = max(0, x - radius)
    x2 = min(w, x + radius)
    region = image_arr[y1:y2, x1:x2]
    if region.size == 0:
        return (128, 128, 128)
    mean_color = region.mean(axis=(0, 1)).astype(int)
    return tuple(mean_color.tolist())


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a TrueType font at the specified size."""
    # Try common font paths across platforms
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",           # macOS
        "/System/Library/Fonts/SFNSText.ttf",             # macOS
        "/System/Library/Fonts/Supplemental/Arial.ttf",   # macOS
        "/Library/Fonts/Arial.ttf",                       # macOS
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/usr/share/fonts/TTF/DejaVuSans.ttf",            # Arch Linux
        "C:/Windows/Fonts/arial.ttf",                      # Windows
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    # Pillow >= 10.1 supports load_default(size=)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_text_trigger(
    image: torch.Tensor,
    text: str,
    params: TextTriggerParams,
) -> torch.Tensor:
    """Render a covert text trigger onto an image.

    Args:
        image: Image tensor (1, 3, H, W) in [0, 1].
        text: Text to render.
        params: Text trigger parameters (position, font, opacity, etc.).

    Returns:
        Image tensor with text trigger applied (1, 3, H, W) in [0, 1].
    """
    # Convert tensor to PIL
    arr = (image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    base_img = Image.fromarray(arr, mode="RGB")
    w, h = base_img.size

    # Create transparent overlay
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Get font
    font = _get_font(params.font_size)

    # Render text with rotation
    # Create a small image for the text, rotate it, then paste
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0] + 4
    text_h = text_bbox[3] - text_bbox[1] + 4

    text_img = Image.new("RGBA", (text_w, text_h), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_img)

    alpha = int(params.opacity * 255)
    r, g, b = params.color
    text_draw.text((2, 2), text, font=font, fill=(r, g, b, alpha))

    # Rotate
    if params.rotation != 0:
        text_img = text_img.rotate(params.rotation, expand=True, resample=Image.BILINEAR)

    # Paste onto overlay
    px = max(0, min(params.position[0], w - text_img.width))
    py = max(0, min(params.position[1], h - text_img.height))
    overlay.paste(text_img, (px, py))

    # Composite
    base_rgba = base_img.convert("RGBA")
    result = Image.alpha_composite(base_rgba, overlay).convert("RGB")

    # Convert back to tensor
    result_arr = np.array(result, dtype=np.float32) / 255.0
    result_tensor = torch.from_numpy(result_arr).permute(2, 0, 1).unsqueeze(0)
    return result_tensor.to(image.device)


def optimize_text_trigger(
    image: torch.Tensor,
    payload: str,
    ensemble,
    target_embeddings: dict[str, torch.Tensor],
    n_positions: int = 25,
    n_samples_per_position: int = 5,
) -> TextTriggerParams:
    """Find optimal text trigger parameters via grid search.

    Searches over a grid of positions and random samples of font/opacity/rotation,
    scoring each configuration by CLIP alignment minus visibility penalty.

    Args:
        image: Clean image tensor (1, 3, H, W) in [0, 1].
        payload: Text to embed.
        ensemble: CLIPEnsemble instance.
        target_embeddings: Target text embeddings.
        n_positions: Number of grid positions to test.
        n_samples_per_position: Random samples per position.

    Returns:
        Best TextTriggerParams found.
    """
    import torch.nn.functional as F

    _, _, h, w = image.shape
    image_arr = (image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

    # Grid of positions
    grid_n = int(math.sqrt(n_positions))
    x_positions = [int(w * (i + 0.5) / grid_n) for i in range(grid_n)]
    y_positions = [int(h * (i + 0.5) / grid_n) for i in range(grid_n)]

    best_score = -float("inf")
    best_params = TextTriggerParams(
        position=(w // 2, h // 2),
        font_size=10,
        rotation=0,
        opacity=0.05,
        color=(128, 128, 128),
    )

    for x in x_positions:
        for y in y_positions:
            local_color = _get_local_color(image_arr, x, y)

            for _ in range(n_samples_per_position):
                params = TextTriggerParams(
                    position=(x, y),
                    font_size=random.randint(8, 14),
                    rotation=random.uniform(0, 360),
                    opacity=random.uniform(0.03, 0.08),
                    color=local_color,
                )

                # Render and score
                triggered = render_text_trigger(image, payload, params)

                with torch.no_grad():
                    embs = ensemble.encode_image(triggered)
                    sims = []
                    for mid, emb in embs.items():
                        sim = F.cosine_similarity(emb, target_embeddings[mid], dim=-1).item()
                        sims.append(sim)
                    clip_score = sum(sims) / len(sims)

                # Visibility penalty: prefer lower opacity and smaller font
                visibility = params.opacity * (params.font_size / 12.0)
                score = clip_score - 0.5 * visibility

                if score > best_score:
                    best_score = score
                    best_params = params

    return best_params
