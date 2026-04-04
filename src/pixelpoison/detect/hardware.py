"""Hardware detection and tier selection."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import torch


@dataclass
class HardwareInfo:
    """Detected hardware capabilities."""

    device: str  # "cuda", "mps", "cpu"
    device_name: str  # Human-readable name
    memory_gb: float  # Available memory in GB
    tier: int  # 1, 2, or 3


def _get_system_memory_gb() -> float:
    """Get total system RAM in GB."""
    if platform.system() == "Darwin":
        try:
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return mem_bytes / (1024**3)
        except (ValueError, OSError):
            pass
    # Fallback: try /proc/meminfo on Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / (1024**2)
    except FileNotFoundError:
        pass
    return 8.0  # Conservative default


def _select_tier(device: str, memory_gb: float) -> int:
    """Select tier based on device type and available memory."""
    if device == "cuda":
        if memory_gb >= 20:
            return 3
        if memory_gb >= 8:
            return 2
        return 1
    if device == "mps":
        if memory_gb >= 32:
            return 3
        if memory_gb >= 16:
            return 2
        return 1
    # CPU
    return 1


def detect_hardware() -> HardwareInfo:
    """Detect available compute hardware and recommend a tier.

    Detection order: CUDA GPU → Apple MPS → CPU fallback.
    """
    # 1. Check CUDA
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        memory_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
        device_name = props.name
        tier = _select_tier("cuda", memory_gb)
        return HardwareInfo(
            device="cuda",
            device_name=device_name,
            memory_gb=round(memory_gb, 1),
            tier=tier,
        )

    # 2. Check Apple MPS
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        memory_gb = _get_system_memory_gb()
        # Try to get chip name on macOS
        device_name = "Apple Silicon"
        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                device_name = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        tier = _select_tier("mps", memory_gb)
        return HardwareInfo(
            device="mps",
            device_name=device_name,
            memory_gb=round(memory_gb, 1),
            tier=tier,
        )

    # 3. CPU fallback
    memory_gb = _get_system_memory_gb()
    return HardwareInfo(
        device="cpu",
        device_name=platform.processor() or "CPU",
        memory_gb=round(memory_gb, 1),
        tier=1,
    )


def get_device() -> torch.device:
    """Return the best available torch device."""
    info = detect_hardware()
    return torch.device(info.device)


TIER_DESCRIPTIONS = {
    1: "Lite (single CLIP, PGD + CoTTA)",
    2: "Standard (CLIP + SigLIP ensemble, + M-Attack)",
    3: "Full (ensemble + Q-Former + VLM proxy scoring, + IPGA)",
}
