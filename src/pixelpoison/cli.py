"""PixelPoison CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from pixelpoison import __version__
from pixelpoison.detect.hardware import TIER_DESCRIPTIONS, detect_hardware

app = typer.Typer(
    name="pixelpoison",
    help="Adversarial image generation for VLM security testing.",
    no_args_is_help=True,
)
console = Console()

NOTICE_PATH = Path.home() / ".pixelpoison" / ".notice_accepted"

RESPONSIBLE_USE_NOTICE = """\
[bold yellow]PixelPoison is a security research tool for authorized testing only.[/bold yellow]
Using this tool against systems without explicit authorization may
violate applicable laws including the Computer Fraud and Abuse Act.
By using this tool, you confirm you have authorization to test
the target systems.
"""


def _ensure_notice_accepted() -> None:
    """Show responsible use notice on first run."""
    if NOTICE_PATH.exists():
        return
    console.print()
    console.print(RESPONSIBLE_USE_NOTICE)
    confirmed = typer.confirm("Do you accept these terms?")
    if not confirmed:
        raise typer.Exit(1)
    NOTICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTICE_PATH.write_text("accepted\n")


@app.command()
def info() -> None:
    """Show detected hardware and recommended tier."""
    hw = detect_hardware()

    console.print(f"\n[bold]PixelPoison v{__version__}[/bold]")
    console.print()

    table = Table(title="Hardware Detection", show_header=False, border_style="dim")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Device", hw.device.upper())
    table.add_row("Device Name", hw.device_name)
    table.add_row("Memory", f"{hw.memory_gb} GB")
    table.add_row("Recommended Tier", f"{hw.tier} — {TIER_DESCRIPTIONS[hw.tier]}")

    console.print(table)
    console.print()

    # Show tier breakdown
    tier_table = Table(title="Tier Thresholds", border_style="dim")
    tier_table.add_column("Tier", style="bold")
    tier_table.add_column("Requirement")
    tier_table.add_column("Description")
    tier_table.add_row(
        "1", "Any hardware", TIER_DESCRIPTIONS[1], style="green" if hw.tier >= 1 else "dim"
    )
    tier_table.add_row(
        "2",
        "8GB+ VRAM or 16GB+ MPS",
        TIER_DESCRIPTIONS[2],
        style="green" if hw.tier >= 2 else "dim",
    )
    tier_table.add_row(
        "3",
        "20GB+ VRAM or 32GB+ MPS",
        TIER_DESCRIPTIONS[3],
        style="green" if hw.tier >= 3 else "dim",
    )
    console.print(tier_table)
    console.print()


@app.command()
def status() -> None:
    """Show version, installed models, and cache info."""
    console.print(f"\n[bold]PixelPoison v{__version__}[/bold]")

    cache_dir = Path.home() / ".pixelpoison" / "models"
    if cache_dir.exists():
        total_size = sum(f.stat().st_size for f in cache_dir.rglob("*") if f.is_file())
        size_mb = total_size / (1024 * 1024)
        console.print(f"Cache directory: {cache_dir}")
        console.print(f"Cache size: {size_mb:.1f} MB")
    else:
        console.print(f"Cache directory: {cache_dir} (not created yet)")

    hw = detect_hardware()
    console.print(f"Device: {hw.device_name} ({hw.device.upper()})")
    console.print(f"Tier: {hw.tier} — {TIER_DESCRIPTIONS[hw.tier]}")
    console.print()


@app.command()
def encode(
    image: str = typer.Option(..., "--image", help="Input image path (JPEG, PNG, WebP, BMP)"),
    payload: str = typer.Option(..., "--payload", help="Instruction to embed"),
    output: Optional[str] = typer.Option(None, "--output", help="Output image path"),
    target: str = typer.Option("auto", "--target", help="Target VLM (gpt4o, gpt5, claude, gemini, auto)"),
    tier: Optional[int] = typer.Option(None, "--tier", help="Override auto-detected tier (1, 2, 3)"),
    strategies: Optional[str] = typer.Option(None, "--strategies", help="Comma-separated strategy list"),
    iterations: Optional[int] = typer.Option(None, "--iterations", help="Iterations per strategy"),
    epsilon: float = typer.Option(16.0 / 255.0, "--epsilon", help="Perturbation budget (default: 16/255)"),
    jpeg_robust: bool = typer.Option(True, "--jpeg-robust/--no-jpeg-robust", help="JPEG hardening"),
    jpeg_quality: int = typer.Option(85, "--jpeg-quality", help="Target JPEG quality level"),
    format: Optional[str] = typer.Option(None, "--format", help="Output format (jpeg, png)"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Reproducibility seed"),
    quick: bool = typer.Option(False, "--quick", help="Quick mode: 100 iterations, skip JPEG refinement"),
    verbose: bool = typer.Option(False, "--verbose", help="Show per-iteration progress"),
    report: Optional[str] = typer.Option(None, "--report", help="Save JSON report to path"),
) -> None:
    """Generate an adversarial image that embeds a hidden payload."""
    _ensure_notice_accepted()

    if epsilon > 16.0 / 255.0:
        console.print("[yellow]Warning: epsilon > 16/255 may produce visible artifacts.[/yellow]")

    import torch
    from pixelpoison.detect.hardware import detect_hardware
    from pixelpoison.image.loader import load_image
    from pixelpoison.image.writer import save_image, generate_default_output_path
    from pixelpoison.models.registry import get_models_for_tier
    from pixelpoison.models.clip_ensemble import CLIPEnsemble
    from pixelpoison.attacks.base import AttackConfig
    from pixelpoison.attacks.composite import StrategyOrchestrator, print_results_table
    from pixelpoison.targets.profiles import get_profile
    from pixelpoison.reporting.report import PipelineReport, StrategyResult, generate_report, save_report

    # 1. Detect hardware
    hw = detect_hardware()
    effective_tier = tier if tier is not None else hw.tier
    effective_tier = min(effective_tier, 3)
    device = torch.device(hw.device)

    console.print(f"\n[bold]PixelPoison v{__version__}[/bold]{'  [dim][quick mode][/dim]' if quick else ''}")
    target_profile = get_profile(target)
    target_display = f"{target_profile.display_name} (preprocessing: {target_profile.max_resolution}px max)" if target_profile else "auto (optimizing for all VLMs)"
    console.print(f"Hardware: {hw.device_name} ({hw.memory_gb}GB) → Tier {effective_tier} ({TIER_DESCRIPTIONS[effective_tier]})")
    console.print(f"Target: {target_display}")

    # 2. Load image
    try:
        image_tensor, image_meta = load_image(image)
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    image_tensor = image_tensor.to(device)
    console.print(f"Image: {image_meta.original_size[1]}x{image_meta.original_size[0]} {image_meta.original_format}")

    # 3. Load models
    model_specs = get_models_for_tier(effective_tier)
    model_ids = [m.id for m in model_specs]
    console.print(f"Models: {', '.join(m.display_name for m in model_specs)}")

    ensemble = CLIPEnsemble(model_ids, device)
    ensemble.load()

    # 4. Encode payload
    console.print(f"Payload: \"{payload[:60]}{'...' if len(payload) > 60 else ''}\"")
    target_embeddings = ensemble.encode_text(payload)

    # 5. Configure attack
    # At 336px optimization resolution, each iteration is ~9x faster than 1024px.
    # Use higher iteration counts for better convergence at negligible time cost.
    effective_iterations = iterations if iterations is not None else (100 if quick else {1: 500, 2: 1000, 3: 1000}[effective_tier])

    attack_config = AttackConfig(
        epsilon=epsilon,
        iterations=effective_iterations,
        jpeg_robust=jpeg_robust,
        jpeg_quality=jpeg_quality,
        seed=seed,
        quick=quick,
        target_profile=target,
        verbose=verbose,
        tier=effective_tier,
    )

    # Seed
    if seed is not None:
        torch.manual_seed(seed)
        import numpy as np
        np.random.seed(seed)

    # 6. Run strategies
    orchestrator = StrategyOrchestrator(
        tier=effective_tier,
        config=attack_config,
        ensemble=ensemble,
        strategy_filter=strategies,
    )

    results = orchestrator.run_all(image_tensor, target_embeddings, payload)

    if not results:
        console.print("[red]All strategies failed. No adversarial image produced.[/red]")
        raise typer.Exit(1)

    # 7. Select best
    best_candidate, best_scores, warning = orchestrator.select_best(results)

    console.print()
    print_results_table(results, best_candidate.strategy_name)
    console.print()
    console.print(f"Best: [bold green]{best_candidate.strategy_name}[/bold green] (ensemble similarity: {best_candidate.clip_score:.3f})")
    console.print(f"Image quality: PSNR={best_candidate.psnr:.1f}dB, SSIM={best_candidate.ssim:.3f}")

    if warning:
        console.print(f"[yellow]⚠ {warning}[/yellow]")

    # 8. Save output
    output_path = output or generate_default_output_path(image)
    saved_path = save_image(
        best_candidate.adversarial_image,
        image_tensor,
        image_meta,
        output_path,
        output_format=format,
        jpeg_quality=jpeg_quality,
    )
    console.print(f"Saved: [bold]{saved_path}[/bold]")

    # 9. Generate report
    report_path = report or str(Path(saved_path).with_suffix(".json"))
    pipeline_report = PipelineReport(
        input_image=image,
        payload=payload,
        target_vlm=target,
        device=hw.device,
        device_name=hw.device_name,
        memory_gb=hw.memory_gb,
        tier=effective_tier,
        epsilon=epsilon,
        iterations=effective_iterations,
        jpeg_robust=jpeg_robust,
        jpeg_quality=jpeg_quality,
        quick_mode=quick,
        seed=seed,
        strategies=[
            StrategyResult(
                name=c.strategy_name,
                clip_score=c.clip_score,
                jpeg_survives=c.jpeg_survives,
                psnr=c.psnr,
                ssim=c.ssim,
                time_seconds=c.time_seconds,
                per_model_scores=c.per_model_scores,
                composite_score=s.composite,
            )
            for c, s in results
        ],
        best_strategy=best_candidate.strategy_name,
        warning=warning,
        output_image=saved_path,
        output_psnr=best_candidate.psnr,
        output_ssim=best_candidate.ssim,
        output_clip_score=best_candidate.clip_score,
        output_jpeg_robust=best_candidate.jpeg_survives,
    )
    report_dict = generate_report(pipeline_report)
    save_report(report_dict, report_path)
    console.print(f"Report: {report_path}")
    console.print()

    # Cleanup
    ensemble.unload()


@app.command()
def download(
    tier: int = typer.Option(..., "--tier", help="Download models for this tier (1, 2, 3)"),
) -> None:
    """Pre-download models for a specific tier."""
    _ensure_notice_accepted()
    import torch
    from pixelpoison.models.registry import get_models_for_tier, get_total_download_size
    from pixelpoison.models.clip_ensemble import CLIPEnsemble
    from pixelpoison.detect.hardware import detect_hardware

    hw = detect_hardware()
    device = torch.device(hw.device)

    specs = get_models_for_tier(tier)
    total_mb = get_total_download_size(tier)
    console.print(f"\n[bold]Downloading models for Tier {tier}[/bold] (~{total_mb} MB)")
    console.print(f"Models: {', '.join(m.display_name for m in specs)}")
    console.print()

    ensemble = CLIPEnsemble([m.id for m in specs], device)
    ensemble.load()
    console.print(f"\n[green]All {len(specs)} models downloaded and verified.[/green]")
    ensemble.unload()


@app.command()
def targets(
    detail: Optional[str] = typer.Option(None, "--detail", help="Show detailed profile for a target"),
) -> None:
    """List supported target VLM profiles."""
    from pixelpoison.targets.profiles import list_profiles, get_profile, PROFILES

    if detail:
        profile = get_profile(detail)
        if not profile:
            console.print(f"[red]Unknown target: {detail}[/red]")
            console.print(f"Available: {', '.join(PROFILES.keys())}")
            raise typer.Exit(1)
        console.print(f"\n[bold]{profile.display_name}[/bold]")
        console.print(f"  Max resolution: {profile.max_resolution}px")
        console.print(f"  Resize method: {profile.resize_method}")
        console.print(f"  Tiling: {'Yes (' + str(profile.tile_size) + 'px)' if profile.uses_tiling else 'No'}")
        console.print(f"  Encoder family: {profile.encoder_family}")
        console.print(f"  Confidence: {profile.confidence}")
        console.print(f"  Recommended surrogates: {', '.join(profile.recommended_surrogates)}")
        console.print(f"  Recommended strategies: {', '.join(profile.recommended_strategies)}")
        return

    table = Table(title="Target VLM Profiles", border_style="dim")
    table.add_column("Name", style="bold")
    table.add_column("Display Name")
    table.add_column("Encoder Family")
    table.add_column("Max Resolution")
    table.add_column("Confidence")

    for profile in list_profiles():
        tiling = f" (tiles: {profile.tile_size}px)" if profile.uses_tiling else ""
        table.add_row(
            profile.name,
            profile.display_name,
            profile.encoder_family,
            f"{profile.max_resolution}px{tiling}",
            profile.confidence,
        )

    console.print()
    console.print(table)
    console.print("\nUse: pixelpoison encode --image photo.jpg --payload \"...\" --target claude")
    console.print("Default: auto (optimizes for all targets simultaneously)")
    console.print()


@app.command()
def score(
    image: str = typer.Option(..., "--image", help="Adversarial image to score"),
    payload: str = typer.Option(..., "--payload", help="Original payload text"),
) -> None:
    """Validate an adversarial image against local VLM proxy (Tier 3 only)."""
    _ensure_notice_accepted()
    import torch
    from pixelpoison.detect.hardware import detect_hardware
    from pixelpoison.image.loader import load_image
    from pixelpoison.models.vlm_scorer import VLMProxyScorer

    hw = detect_hardware()
    if hw.tier < 3:
        console.print("[yellow]VLM proxy scoring requires Tier 3 hardware (20GB+ VRAM or 32GB+ MPS).[/yellow]")
        console.print(f"Your hardware was detected as Tier {hw.tier}.")
        raise typer.Exit(1)

    device = torch.device(hw.device)
    image_tensor, _ = load_image(image)
    image_tensor = image_tensor.to(device)

    scorer = VLMProxyScorer(device)
    scorer.load()

    if not scorer._loaded:
        raise typer.Exit(1)

    score_val = scorer.score(image_tensor, payload)
    console.print(f"\nVLM Proxy Score: [bold]{score_val:.3f}[/bold]")
    if score_val > 0.5:
        console.print("[green]High — VLM likely follows the payload.[/green]")
    elif score_val > 0.3:
        console.print("[yellow]Medium — partial payload compliance.[/yellow]")
    else:
        console.print("[red]Low — VLM unlikely to follow the payload.[/red]")
    scorer.unload()


@app.command()
def compare(
    original: str = typer.Option(..., "--original", help="Original image path"),
    adversarial: str = typer.Option(..., "--adversarial", help="Adversarial image path"),
) -> None:
    """Compare original and adversarial images (PSNR, SSIM, diff heatmap)."""
    from pixelpoison.reporting.compare import compare_images, print_comparison

    try:
        result = compare_images(original, adversarial)
        console.print()
        print_comparison(result)
        console.print()
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
