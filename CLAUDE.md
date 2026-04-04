# PixelPoison - Development Guide

## Project Overview

PixelPoison is a Python CLI tool that generates adversarial images for testing VLM (Vision-Language Model) pipelines against steganographic prompt injection. It takes a benign image + text payload and produces an imperceptible adversarial image that causes VLMs to follow the hidden instruction.

## Architecture

```
src/pixelpoison/
  cli.py               # Typer + Rich CLI. All commands defined here.
  detect/hardware.py   # Hardware detection (CUDA/MPS/CPU) and tier selection (1/2/3)
  models/
    registry.py        # Model specs: IDs, open_clip names, normalization constants, tier requirements
    clip_ensemble.py   # Loads CLIP + SigLIP models via open_clip, handles per-model normalization and dtype
    projector.py       # InstructBLIP Q-Former loading for Tier 3 IPGA attack
    vlm_scorer.py      # LLaVA-7B 4-bit proxy scorer for Tier 3 validation
  attacks/
    base.py            # AttackConfig, CandidateResult, AttackStrategy ABC
    pgd.py             # PGD baseline with X-Transfer dynamic ensemble weighting
    cotta.py           # CoTTA: covert text trigger (Phase 1) + dual-target alignment (Phase 2)
    m_attack.py        # M-Attack: differentiable random-crop via grid_sample + SGMA semantic guidance
    ipga.py            # IPGA: Q-Former projector-level attack with Residual Query Alignment (Tier 3)
    composite.py       # StrategyOrchestrator: runs all strategies, JPEG hardens, scores, selects best
  robustness/
    diff_jpeg.py       # DiffJPEG: differentiable JPEG compression (smooth rounding approximation)
    dct.py             # DCT mid-frequency band mask (zigzag indices 2-5)
    __init__.py        # jpeg_harden() post-optimization refinement + check_jpeg_survival()
  scoring/
    clip_scorer.py     # Mean cosine similarity across ensemble
    ensemble_scorer.py # Cross-model agreement: 1 - std/mean
    quality.py         # PSNR, SSIM, L-inf, quality gate check
    composite.py       # Tier-dependent weighted composite scoring
  targets/
    profiles.py        # Target VLM preprocessing profiles (GPT-4o, Claude, Gemini, etc.)
    preprocess_sim.py  # Differentiable preprocessing simulation for target-aware optimization
  augmentation/
    transforms.py      # Random resize, Gaussian blur, color jitter (all differentiable)
  rendering/
    text_overlay.py    # Covert text trigger rendering + grid-search optimization for CoTTA
  image/
    loader.py          # Image loading, validation (224-4096px), resize to 1024 max
    writer.py          # Image saving with perturbation upscaling and EXIF watermark
  reporting/
    report.py          # JSON report generation matching PRD schema
    compare.py         # Image comparison: PSNR, SSIM, L-inf, diff heatmap
```

## Key Technical Invariants

**CLIP vs SigLIP normalization must never be mixed.** CLIP uses mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]. SigLIP uses mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]. The CLIPEnsemble class handles this via per-model normalization stored in registry.py. If you add a new model, always specify the correct norm_mean/norm_std.

**Models load in fp32 on MPS, fp16 on CUDA.** MPS has known issues with half-precision conv operations. The ensemble casts embeddings back to float32 after encoding. Perturbation tensors are always float32.

**Differentiable crop in M-Attack uses `F.affine_grid` + `F.grid_sample`.** Never use PIL/OpenCV for cropping inside the optimization loop -- gradients must flow from CLIP loss through the crop back to full-image pixels.

**DiffJPEG smooth rounding:** `x - sin(2*pi*x) / (2*pi)` approximates `round(x)` with smooth gradients. Used in `robustness/diff_jpeg.py`.

**Strategies run sequentially**, not in parallel. Each strategy's full optimization loop completes before the next starts. This avoids OOM and allows per-strategy progress display.

**Font sizes below 8px cause rendering errors** on some platforms (Pillow + Helvetica.ttc). The text trigger renderer uses 8-14px range.

## Tier System

- Tier 1 (any hardware): CLIP ViT-B/32 only. PGD + CoTTA. ~500MB models.
- Tier 2 (8GB+ VRAM / 16GB+ MPS): + CLIP B/16, L/14, SigLIP SO400M. + M-Attack. ~4.5GB models.
- Tier 3 (20GB+ VRAM / 32GB+ MPS): + CLIP L/14@336, InstructBLIP Q-Former, LLaVA-7B. + IPGA. ~13GB models.

Tier selection logic is in `detect/hardware.py:_select_tier()`.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
```

## Testing

Tests are in `tests/`. Hardware detection tests use mocked torch backends. Attack tests use small synthetic images (64x64) with single CLIP models. Run with:

```bash
pytest tests/ -v               # All tests
pytest tests/ -k "pgd" -v      # Just PGD tests
```

For end-to-end testing with real models:
```bash
pixelpoison encode --image test_image.png --payload "test" --tier 1 --quick --seed 42
```

## Spec Documents

- `RESEARCH.md` -- Research papers and techniques underpinning each component (14 papers)
- `PRD.md` -- Product requirements: CLI interface, tier system, scoring, error handling
- `ATTACK_PIPELINE.md` -- Detailed technical specification of the 7-stage attack pipeline
- `TARGET_PROFILES.md` -- Known VLM preprocessing pipelines for target-aware optimization

When modifying attack strategies or adding new ones, read the relevant spec doc first. The algorithm pseudocode in ATTACK_PIPELINE.md is authoritative.

## Adding a New Attack Strategy

1. Create `src/pixelpoison/attacks/new_strategy.py`
2. Implement `AttackStrategy` ABC from `attacks/base.py` (name, required_tier, required_models, optimize)
3. Register in `attacks/composite.py:_get_strategies_for_tier()`
4. Add to the appropriate tier in the conditional chain

## Adding a New Surrogate Model

1. Add entry to `models/registry.py:MODELS` with correct open_clip ID, pretrained tag, family, and normalization constants
2. Update tier thresholds in `detect/hardware.py` if memory requirements change
3. The CLIPEnsemble handles loading automatically based on registry specs

## Common Pitfalls

- Forgetting to call `.detach()` on tensors stored in CandidateResult (causes memory leaks via retained graph)
- Running all ensemble models simultaneously instead of sequentially (OOM on Tier 2+)
- Using `torch.no_grad()` during the optimization loop (kills gradients for the perturbation)
- Not clearing CUDA cache between strategy runs (`torch.cuda.empty_cache()`)
