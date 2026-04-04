"""JSON report generation for adversarial image pipeline results."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pixelpoison import __version__


@dataclass
class StrategyResult:
    """Result from a single attack strategy."""

    name: str
    clip_score: float
    jpeg_survives: bool
    psnr: float
    ssim: float
    time_seconds: float
    per_model_scores: dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class PipelineReport:
    """Complete pipeline execution report."""

    # Input
    input_image: str
    payload: str
    target_vlm: str
    image_hash: str = ""

    # Hardware
    device: str = ""
    device_name: str = ""
    memory_gb: float = 0.0
    tier: int = 1

    # Config
    epsilon: float = 0.0627
    iterations: int = 500
    jpeg_robust: bool = True
    jpeg_quality: int = 85
    quick_mode: bool = False
    seed: Optional[int] = None

    # Results
    strategies: list[StrategyResult] = field(default_factory=list)
    best_strategy: str = ""
    warning: Optional[str] = None

    # Output
    output_image: str = ""
    output_hash: str = ""
    output_psnr: float = 0.0
    output_ssim: float = 0.0
    output_clip_score: float = 0.0
    output_jpeg_robust: bool = False


def _file_hash(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except FileNotFoundError:
        return ""


def generate_report(report: PipelineReport) -> dict:
    """Generate a JSON-serializable report dict matching PRD §3.4 schema."""
    # Compute file hashes
    input_hash = _file_hash(report.input_image)
    output_hash = _file_hash(report.output_image) if report.output_image else ""

    return {
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": {
            "image": report.input_image,
            "payload": report.payload,
            "image_hash_sha256": input_hash,
            "target_vlm": report.target_vlm,
        },
        "hardware": {
            "device": report.device,
            "device_name": report.device_name,
            "memory_gb": report.memory_gb,
            "tier": report.tier,
        },
        "config": {
            "epsilon": round(report.epsilon, 4),
            "iterations": report.iterations,
            "jpeg_robust": report.jpeg_robust,
            "jpeg_quality": report.jpeg_quality,
            "quick_mode": report.quick_mode,
            "seed": report.seed,
        },
        "strategies": [
            {
                "name": s.name,
                "clip_score": round(s.clip_score, 3),
                "jpeg_survives": s.jpeg_survives,
                "psnr": round(s.psnr, 1),
                "ssim": round(s.ssim, 3),
                "time_seconds": round(s.time_seconds, 1),
                "composite_score": round(s.composite_score, 3),
            }
            for s in report.strategies
        ],
        "best_strategy": report.best_strategy,
        "warning": report.warning,
        "output": {
            "image": report.output_image,
            "image_hash_sha256": output_hash,
            "psnr": round(report.output_psnr, 1),
            "ssim": round(report.output_ssim, 3),
            "clip_score": round(report.output_clip_score, 3),
            "jpeg_robust": report.output_jpeg_robust,
        },
    }


def save_report(report_dict: dict, path: str) -> None:
    """Save a report dict to a JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report_dict, f, indent=2)
