"""Abstract base class for attack strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class AttackConfig:
    """Configuration for an attack run."""

    epsilon: float = 16.0 / 255.0  # Max perturbation (L-inf)
    iterations: int = 500  # Iterations per strategy
    step_size: float = 0.0  # Auto-computed if 0: epsilon / (iterations * 0.5)
    jpeg_robust: bool = True
    jpeg_quality: int = 85
    seed: Optional[int] = None
    quick: bool = False
    target_profile: Optional[str] = None  # Target VLM name or None for auto
    verbose: bool = False
    tier: int = 1

    def __post_init__(self):
        if self.step_size == 0.0:
            # Published MI-FGSM attacks (CoTTA, Qi et al. AAAI'24, InstructTA,
            # Chain of Attack CVPR'25) all use alpha=1/255 in [0,1] space.
            # With momentum, sign(accumulated_grad) moves exactly alpha per step.
            # After eps/alpha steps (~24 for eps=24/255), the perturbation hits
            # the boundary. Remaining iterations refine which pixels are +eps
            # vs -eps — this is where momentum's gradient smoothing matters most.
            # Floor at 1/255 to match the literature; cap at eps/10 for low-iter.
            self.step_size = max(self.epsilon / max(self.iterations, 10), 1.0 / 255.0)
        if self.quick:
            self.iterations = 100


@dataclass
class CandidateResult:
    """Result from a single attack strategy."""

    strategy_name: str
    adversarial_image: torch.Tensor  # (1, 3, H, W) in [0, 1]
    perturbation: torch.Tensor  # (1, 3, H, W), the delta
    per_model_scores: dict[str, float] = field(default_factory=dict)
    clip_score: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    jpeg_survives: bool = False
    iterations_used: int = 0
    time_seconds: float = 0.0
    metadata: dict = field(default_factory=dict)


class AttackStrategy(ABC):
    """Base class for all attack strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'pgd_baseline', 'cotta')."""

    @property
    @abstractmethod
    def required_tier(self) -> int:
        """Minimum tier needed (1, 2, or 3)."""

    @property
    @abstractmethod
    def required_models(self) -> list[str]:
        """Which model IDs this strategy needs loaded."""

    @abstractmethod
    def optimize(
        self,
        clean_image: torch.Tensor,
        target_embeddings: dict[str, torch.Tensor],
        ensemble,
        config: AttackConfig,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> CandidateResult:
        """Run the full optimization loop.

        Args:
            clean_image: Original image tensor (1, 3, H, W) in [0, 1].
            target_embeddings: {model_id: text_embedding_tensor}
            ensemble: CLIPEnsemble instance (loaded).
            config: Attack configuration.
            progress_callback: Optional callback(iteration, current_score) for progress display.

        Returns:
            CandidateResult with the adversarial image and metrics.
        """
