"""Strategy orchestrator — runs all tier-appropriate strategies, scores, and selects best."""

from __future__ import annotations

from typing import Optional

import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.attacks.cotta import CoTTAStrategy
from pixelpoison.attacks.cwa import CWAStrategy
from pixelpoison.attacks.m_attack import MAttackStrategy
from pixelpoison.attacks.pgd import PGDBaseline
from pixelpoison.attacks.soups import SoupsStrategy
from pixelpoison.attacks.ssa import SSAStrategy
from pixelpoison.attacks.vmi_fgsm import VMIFGSMStrategy
from pixelpoison.robustness import check_jpeg_survival, jpeg_harden
from pixelpoison.scoring.composite import CandidateScores, compute_composite
from pixelpoison.scoring.ensemble_scorer import compute_ensemble_agreement
from pixelpoison.scoring.quality import check_quality_gate

console = Console()


def _get_strategies_for_tier(tier: int) -> list[AttackStrategy]:
    """Return strategies available at the given tier."""
    strategies: list[AttackStrategy] = [
        PGDBaseline(),
        CoTTAStrategy(),
        SSAStrategy(),       # Tier 1: spectrum simulation for frequency-diverse transferability
        VMIFGSMStrategy(),   # Tier 1: variance-tuned momentum for stable gradients
        SoupsStrategy(),     # Tier 1: ensemble of hyperparameter configs
    ]

    if tier >= 2:
        strategies.append(MAttackStrategy())
        strategies.append(CWAStrategy())  # Tier 2: SAM + CSE for flat loss landscape

    if tier >= 3:
        try:
            from pixelpoison.attacks.ipga import IPGAStrategy
            strategies.append(IPGAStrategy())
        except ImportError:
            pass  # Tier 3 deps not installed

    return strategies


def _filter_strategies(
    strategies: list[AttackStrategy],
    requested: Optional[str],
) -> list[AttackStrategy]:
    """Filter strategies by user's --strategies flag."""
    if requested is None:
        return strategies
    names = [s.strip() for s in requested.split(",")]
    return [s for s in strategies if s.name in names]


class StrategyOrchestrator:
    """Runs all tier-appropriate strategies and selects the best result."""

    def __init__(
        self,
        tier: int,
        config: AttackConfig,
        ensemble,
        strategy_filter: Optional[str] = None,
    ):
        self.tier = tier
        self.config = config
        self.ensemble = ensemble
        self.strategies = _filter_strategies(_get_strategies_for_tier(tier), strategy_filter)

    def run_all(
        self,
        clean_image: torch.Tensor,
        target_embeddings: dict[str, torch.Tensor],
        payload: str,
    ) -> list[tuple[CandidateResult, CandidateScores]]:
        """Run all strategies sequentially, score each candidate.

        Returns:
            List of (CandidateResult, CandidateScores) tuples.
        """
        results = []

        # Store payload in config for CoTTA text trigger
        self.config._payload = payload

        console.print(
            f"\nRunning {len(self.strategies)} strategies "
            f"({self.config.iterations} iterations each)...\n"
        )

        for strategy in self.strategies:
            console.print(f"  [bold]{strategy.name}[/bold]", end="")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=30),
                TimeElapsedColumn(),
                console=console,
                transient=True,
            ) as progress:
                task = progress.add_task(
                    f"  {strategy.name}",
                    total=self.config.iterations,
                )

                def _progress_cb(iteration: int, score: float) -> None:
                    progress.update(task, completed=iteration + 1)

                try:
                    candidate = strategy.optimize(
                        clean_image,
                        target_embeddings,
                        self.ensemble,
                        self.config,
                        progress_callback=_progress_cb,
                    )
                except Exception as e:
                    error_msg = str(e)
                    if "out of memory" in error_msg.lower():
                        console.print(f"  [red]OOM during {strategy.name}[/red]")
                    else:
                        console.print(
                            f"  [red]Error during {strategy.name}: "
                            f"{type(e).__name__}: {error_msg[:200]}[/red]"
                        )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

            # JPEG hardening (skip in quick mode)
            if self.config.jpeg_robust and not self.config.quick:
                candidate = jpeg_harden(
                    candidate, clean_image, target_embeddings,
                    self.ensemble, self.config,
                    refinement_iterations=50 if self.tier == 1 else 100,
                )
            elif self.config.jpeg_robust:
                # Still check JPEG survival even in quick mode
                survives, _ = check_jpeg_survival(
                    candidate.adversarial_image, target_embeddings,
                    self.ensemble, self.config.jpeg_quality,
                )
                candidate.jpeg_survives = survives

            # Score the candidate
            scores = CandidateScores(
                clip_score=candidate.clip_score,
                jpeg_survival=1.0 if candidate.jpeg_survives else 0.0,
                image_quality=candidate.ssim,
                ensemble_agreement=compute_ensemble_agreement(candidate.per_model_scores),
                vlm_proxy=0.0,  # Set by Tier 3 VLM scorer later
            )
            compute_composite(scores, self.tier)

            results.append((candidate, scores))

            console.print(
                f"  {strategy.name:15s} | "
                f"score={candidate.clip_score:.3f} | "
                f"JPEG={'yes' if candidate.jpeg_survives else 'no ':3s} | "
                f"{candidate.time_seconds:.0f}s"
            )

            # Free memory between strategies
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    def select_best(
        self,
        results: list[tuple[CandidateResult, CandidateScores]],
    ) -> tuple[CandidateResult, CandidateScores, Optional[str]]:
        """Select the best candidate from all strategy results.

        Returns:
            Tuple of (best_candidate, best_scores, warning_message).
        """
        if not results:
            raise RuntimeError("No strategies produced results.")

        # Quality gate filter
        passing = [
            (c, s) for c, s in results
            if check_quality_gate(c.psnr, c.ssim)
        ]

        warning = None

        if not passing:
            # All failed quality gate — return least corrupted with warning
            warning = (
                "All candidates failed quality gates (PSNR < 36dB or SSIM < 0.95). "
                "Try increasing --iterations or --epsilon."
            )
            # Sort by SSIM (least corrupted)
            results.sort(key=lambda x: x[0].ssim, reverse=True)
            return results[0][0], results[0][1], warning

        # Sort passing candidates by composite score
        passing.sort(key=lambda x: x[1].composite, reverse=True)

        # Check for low confidence
        best_candidate, best_scores = passing[0]
        if best_scores.clip_score < 0.3:
            warning = "Low embedding alignment — attack unlikely to succeed."

        return best_candidate, best_scores, warning


def print_results_table(
    results: list[tuple[CandidateResult, CandidateScores]],
    best_name: str,
) -> None:
    """Print a summary table of all strategy results."""
    table = Table(border_style="dim")
    table.add_column("Strategy", style="bold")
    table.add_column("Ensemble Score", justify="right")
    table.add_column("JPEG Survives", justify="center")
    table.add_column("PSNR", justify="right")
    table.add_column("SSIM", justify="right")
    table.add_column("Time", justify="right")

    for candidate, scores in results:
        is_best = candidate.strategy_name == best_name
        style = "green" if is_best else ""
        marker = " *" if is_best else ""

        table.add_row(
            f"{candidate.strategy_name}{marker}",
            f"{candidate.clip_score:.3f}",
            "[green]yes[/green]" if candidate.jpeg_survives else "[red]no[/red]",
            f"{candidate.psnr:.1f} dB",
            f"{candidate.ssim:.3f}",
            f"{candidate.time_seconds:.0f}s",
            style=style,
        )

    console.print(table)
