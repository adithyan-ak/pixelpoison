# PixelPoison: Target VLM Profiles

> **Version:** 1.0
> **Date:** 2026-04-03
> **Purpose:** Documents known preprocessing pipelines, vision encoder hypotheses, and recommended attack configurations for each target VLM. Enables the `--target` flag to simulate target-specific preprocessing during optimization.
> **Companion Docs:** [PRD.md](./PRD.md) | [ATTACK_PIPELINE.md](./ATTACK_PIPELINE.md) | [RESEARCH.md](./RESEARCH.md)

---

## 1. Why Target Profiles Matter

Different VLMs process images through different preprocessing pipelines before their vision encoder sees the pixels. An adversarial perturbation optimized at 224x224 may be destroyed when the target VLM downscales from 4096x4096 to 768x768 using bicubic interpolation, then tiles into 512x512 patches.

By simulating the target's preprocessing during optimization, perturbations are **pre-hardened** against the specific transformations they will encounter. This is analogous to DiffJPEG hardening against JPEG compression — but for the target's resize/tile/normalize pipeline.

When `--target auto` (default), PixelPoison randomly samples a target profile each iteration during optimization, producing perturbations robust to diverse preprocessing. When a specific target is selected, all iterations use that target's profile, producing more focused (but potentially less transferable) perturbations.

---

## 2. Profile Schema

Each target profile is defined as:

```python
@dataclass
class TargetProfile:
    name: str                           # CLI identifier (e.g., "claude", "gpt4o")
    display_name: str                   # Human-readable (e.g., "Claude (Anthropic)")
    
    # Image preprocessing
    max_resolution: int                 # Max dimension before downscale (longest edge)
    resize_method: str                  # 'bicubic' | 'bilinear' | 'lanczos'
    preserve_aspect_ratio: bool         # Whether resize preserves aspect ratio
    
    # Tiling (if applicable)
    uses_tiling: bool                   # Whether the model tiles the image into patches
    tile_size: int | None               # Tile dimensions (e.g., 512 for GPT-4o)
    tile_overlap: int                   # Overlap between tiles in pixels
    
    # Format handling
    supported_formats: list[str]        # ['jpeg', 'png', 'webp', 'gif']
    applies_jpeg_before_encoding: bool  # Whether JPEG compression happens in pipeline
    
    # Encoder hypothesis
    encoder_family: str                 # 'clip' | 'siglip' | 'proprietary' | 'unknown'
    encoder_notes: str                  # Free-text notes on encoder architecture
    
    # Recommended attack configuration
    recommended_surrogates: list[str]   # Prioritized surrogate model IDs
    recommended_strategies: list[str]   # Strategies expected to work best
    
    # Metadata
    last_verified: str                  # ISO date when this profile was last confirmed
    confidence: str                     # 'high' | 'medium' | 'low' (how certain we are)
    sources: list[str]                  # URLs/references for the preprocessing info
```

---

## 3. Target Profiles

### 3.1 GPT-4o / GPT-4o-mini (OpenAI)

```yaml
name: gpt4o
display_name: "GPT-4o (OpenAI)"

preprocessing:
  max_resolution: 768              # Shortest side scaled to 768px (high detail mode)
  resize_method: bicubic
  preserve_aspect_ratio: true
  uses_tiling: true
  tile_size: 512                   # Image divided into 512x512 tiles
  tile_overlap: 0                  # No documented overlap
  low_detail_mode: 512             # Low detail: fixed 512x512

format:
  supported: [jpeg, png, webp, gif]
  max_file_size: 20MB
  applies_jpeg_before_encoding: false  # No evidence of JPEG in pipeline

encoder:
  family: proprietary              # End-to-end unified multimodal architecture
  notes: >
    GPT-4o processes all modalities through unified attention layers.
    No separable CLIP-style encoder bottleneck is documented.
    The 512x512 tiling means perturbation must be effective
    at the tile level, not just globally.

attack_config:
  recommended_surrogates:
    - ViT-B-32/openai             # CLIP variants — historically high transfer to GPT
    - ViT-L-14/openai
    - ViT-SO400M-14-SigLIP2/webli  # Hedge against proprietary encoder
  recommended_strategies:
    - cotta                        # Text trigger survives tiling
    - m_attack                     # Random crops align with tiling behavior
  optimization_notes: >
    Optimize at 512x512 tile resolution to match GPT-4o's input tile size.
    Random crops in M-Attack naturally simulate the tiling effect.
    CoTTA's text trigger should be positioned to survive tiling —
    place text near center of 512x512 grid cells, not at tile boundaries.

metadata:
  last_verified: 2026-04-03
  confidence: medium
  sources:
    - "OpenAI API documentation: image input detail parameter"
    - "GPT-4o system card (May 2024)"
```

### 3.2 GPT-5 (OpenAI)

```yaml
name: gpt5
display_name: "GPT-5 (OpenAI)"

preprocessing:
  max_resolution: 768              # Assumed same as GPT-4o (not independently confirmed)
  resize_method: bicubic
  preserve_aspect_ratio: true
  uses_tiling: true                # Assumed same tiling architecture
  tile_size: 512
  tile_overlap: 0

format:
  supported: [jpeg, png, webp, gif]
  max_file_size: 50MB
  applies_jpeg_before_encoding: false

encoder:
  family: proprietary
  notes: >
    GPT-5 uses a "unified system" with router for fast/deep reasoning.
    Vision capabilities are natively multimodal from ground up.
    PSI (ICLR 2026) demonstrated successful transfer attacks against GPT-5,
    confirming it remains vulnerable to well-crafted adversarial images
    despite architectural improvements.

attack_config:
  recommended_surrogates:
    - ViT-B-32/openai
    - ViT-L-14/openai
    - ViT-SO400M-14-SigLIP2/webli
  recommended_strategies:
    - cotta                        # Best reported success on GPT-family
    - m_attack
    - ipga                         # Projector-level may bypass encoder defenses
  optimization_notes: >
    Same tiling considerations as GPT-4o. GPT-5 may have improved
    adversarial robustness — use full iteration budget (no --quick).
    PSI paper suggests diffusion-based attacks transfer well to GPT-5
    (post-MVP enhancement).

metadata:
  last_verified: 2026-04-03
  confidence: low                  # Preprocessing details not independently confirmed
  sources:
    - "PSI (ICLR 2026) attack results"
    - "OpenAI GPT-5 system card (August 2025)"
```

### 3.3 Claude (Anthropic)

```yaml
name: claude
display_name: "Claude (Anthropic)"

preprocessing:
  max_resolution: 1568             # Longest edge capped at 1568px
  resize_method: bicubic           # Aspect-ratio preserving bicubic
  preserve_aspect_ratio: true
  uses_tiling: false               # No tiling — tokenizes as single image
  tile_size: null
  tile_overlap: 0
  token_formula: "(width * height) / 750"  # Image token cost

format:
  supported: [jpeg, png, gif, webp]
  max_file_size: null              # No documented file size limit (API limit applies)
  max_pixels: 8000x8000           # Hard limit (2000x2000 if >20 images in conversation)
  applies_jpeg_before_encoding: false

encoder:
  family: unknown                  # Likely SigLIP or proprietary — NOT confirmed
  notes: >
    Anthropic does not disclose vision encoder architecture.
    Across all published papers, Claude shows consistently lower attack
    success rates than GPT-4o (M-Attack: 26% vs 95%; Invisible Injections: ~24%).
    
    Leading hypothesis: Claude uses SigLIP or a proprietary encoder with different
    feature geometry than CLIP. The sigmoid loss in SigLIP creates different
    representations that CLIP-optimized perturbations don't transfer to well.
    
    Alternative hypothesis: Claude's safety training may include adversarial
    robustness objectives (similar to AdPO defense, ICLR 2026).
    
    Claude cannot identify people, struggles with spatial reasoning,
    and has reduced accuracy on images <200px — suggesting a resolution-sensitive
    preprocessing pipeline.

attack_config:
  recommended_surrogates:
    - ViT-SO400M-14-SigLIP2/webli  # HIGHEST PRIORITY — addresses SigLIP hypothesis
    - ViT-L-14/openai              # Large CLIP for diversity
    - ViT-B-16/openai              # Different patch size
  recommended_strategies:
    - cotta                        # Text trigger provides non-perturbative signal
    - ipga                         # Projector-level bypasses encoder mismatch
    - m_attack                     # Spatial distribution covers 1568px preprocessing
  optimization_notes: >
    SigLIP surrogates are critical for Claude targeting. Optimize at higher
    resolution (1024x1024) to survive the 1568px downscale — perturbations
    in small images (<512px) may be distorted by the resize.
    
    CoTTA text trigger should use relatively larger font (6-10px) since
    Claude's preprocessing preserves more resolution than GPT-4o's tiling.
    
    IPGA is especially valuable here — if Claude uses a Q-Former-style
    projector, projector-level attacks bypass encoder-family mismatch entirely.

metadata:
  last_verified: 2026-04-03
  confidence: medium               # Preprocessing confirmed via API docs; encoder is hypothesis
  sources:
    - "Anthropic Claude API documentation: vision capabilities"
    - "M-Attack paper (NeurIPS 2025): Claude-3.5 results"
    - "AnyAttack (CVPR 2025): confirms transfer to Claude Sonnet"
```

### 3.4 Gemini 2.5 (Google DeepMind)

```yaml
name: gemini
display_name: "Gemini 2.5 (Google DeepMind)"

preprocessing:
  max_resolution: null             # Not publicly documented; 1M token context
  resize_method: unknown           # CNN+ViT hybrid — custom preprocessing
  preserve_aspect_ratio: true      # Assumed
  uses_tiling: false               # Uses CNN+ViT hierarchical encoding, not patch tiling
  tile_size: null
  tile_overlap: 0

format:
  supported: [jpeg, png, gif, webp]
  max_file_size: null
  applies_jpeg_before_encoding: false

encoder:
  family: proprietary
  notes: >
    Gemini 2.5 Pro uses a Vision CNN + 12-layer ViT combination with
    cross-attention fusion at every layer. This is architecturally distinct
    from both CLIP and SigLIP.
    
    Key architectural details (from technical reports):
    - ~150B parameters total (Mixture-of-Experts, 64 experts, top-2 routing)
    - Hierarchical temporal-pyramid encoder (frame CNN -> clip Transformer -> summary)
    - Two-tier attention: sliding-window local + 256 global memory tokens
    - Natively multimodal — parallel encoding of text/image/video embeddings
    
    The CNN+ViT combination means perturbations must work at both
    low-level (CNN features) and high-level (ViT attention). CLIP-only
    attacks target only the ViT level, missing the CNN preprocessing.
    
    Gemini 2.5 Flash (40B) and Flash-Lite (12B) use smaller versions
    of the same architecture.

attack_config:
  recommended_surrogates:
    - ViT-L-14/openai              # Best general CLIP for transfer
    - ViT-SO400M-14-SigLIP2/webli  # SigLIP for encoder diversity
    - ViT-B-32/openai              # Different patch size for spatial diversity
  recommended_strategies:
    - m_attack                     # Random crops test multi-scale features (CNN+ViT)
    - cotta                        # Text trigger provides modality-agnostic signal
    - ipga                         # Projector-level bypasses encoder architecture mismatch
  optimization_notes: >
    Gemini's CNN+ViT architecture is the most different from CLIP among
    the major VLMs. Expect lower transfer rates than GPT-4o.
    
    M-Attack's random cropping naturally exercises multi-scale features
    (CNN operates at fine scale, ViT at patch scale). Use wider crop
    range (scale 0.3-1.0 instead of 0.5-1.0) to capture more CNN-level features.
    
    The cross-attention architecture means Gemini processes visual and text
    tokens jointly — CoTTA's text trigger (which is text rendered visually)
    may be particularly effective since it creates a visual-textual hybrid signal.

metadata:
  last_verified: 2026-04-03
  confidence: low                  # Architecture from technical reports; preprocessing unconfirmed
  sources:
    - "Gemini 2.5 technical report (Google DeepMind)"
    - "AnyAttack (CVPR 2025): confirms transfer to Google Gemini"
```

### 3.5 Open-Source VLMs (LLaVA, InternVL, etc.)

```yaml
name: opensource
display_name: "Open-Source VLMs (LLaVA, InternVL, etc.)"

preprocessing:
  max_resolution: 336              # LLaVA-1.5: 336x336; InternVL-2: up to 448
  resize_method: bicubic
  preserve_aspect_ratio: false     # Most resize to fixed square
  uses_tiling: false               # LLaVA: single image; some newer models tile
  tile_size: null
  tile_overlap: 0

format:
  supported: [jpeg, png, webp, bmp, tiff]
  max_file_size: null              # Local — no API limits
  applies_jpeg_before_encoding: false

encoder:
  family: clip                     # LLaVA: CLIP ViT-L/14@336px; InternVL: InternViT
  notes: >
    Most open-source VLMs use CLIP variants:
    - LLaVA-1.5/1.6: CLIP ViT-L/14@336px
    - InstructBLIP: CLIP ViT-g/14 + Q-Former
    - InternVL-2: InternViT-6B (proprietary but CLIP-like architecture)
    - MiniGPT-4: CLIP ViT-g/14 + Q-Former
    
    CLIP-based surrogates have highest transfer rates to these models.
    Qwen2.5-VL is an exception — significantly more robust to basic
    gradient attacks (La Torre, March 2026: 6.5-15.5% ASR vs 52.6-66.9%
    for LLaVA-v1.5). Newer open-source VLMs may use SigLIP or
    custom encoders.

attack_config:
  recommended_surrogates:
    - ViT-L-14-336/openai          # Direct match for LLaVA-1.5
    - ViT-B-32/openai
    - ViT-B-16/openai
  recommended_strategies:
    - pgd_baseline                 # Sufficient for CLIP-encoder VLMs
    - m_attack                     # Improves on PGD for harder targets
    - ipga                         # Directly attacks Q-Former models
  optimization_notes: >
    Open-source VLMs are the easiest targets because the surrogate
    and target share the same encoder. PGD alone achieves >50% ASR
    on LLaVA-1.5. Use these as sanity checks before testing against
    commercial models.

metadata:
  last_verified: 2026-04-03
  confidence: high                 # Architectures are fully documented
  sources:
    - "LLaVA-1.5 paper (NeurIPS 2023)"
    - "InstructBLIP paper (NeurIPS 2023)"
    - "La Torre (arXiv, March 2026): Qwen2.5-VL robustness findings"
```

---

## 4. Auto Mode Behavior

When `--target auto` (the default), the optimization loop randomly selects a target profile at each iteration:

```python
def get_auto_preprocessing(profiles: list[TargetProfile], rng: Generator) -> Callable:
    """
    Returns a stochastic preprocessing function that samples a random
    target profile each call.
    
    Sampling weights:
      - GPT-4o/GPT-5: 30% (most common real-world target)
      - Claude:        30% (hardest target — needs proportional attention)
      - Gemini:        20%
      - No preprocessing (direct CLIP input): 20% (baseline)
    
    This stochastic approach produces perturbations that are reasonably
    robust across all targets, at the cost of not being optimal for any
    specific target. For targeted attacks, specifying --target is recommended.
    """
```

---

## 5. CLI Integration

```bash
# List all available target profiles
$ pixelpoison targets

Available target VLM profiles:

  Name        | Encoder Family | Max Resolution | Confidence
  ------------|----------------|----------------|------------
  gpt4o       | proprietary    | 768px (tiles)  | medium
  gpt5        | proprietary    | 768px (tiles)  | low
  claude      | unknown/siglip | 1568px         | medium
  gemini      | proprietary    | unknown        | low
  opensource   | clip           | 336px          | high

Use: pixelpoison encode --image photo.jpg --payload "..." --target claude
Default: auto (optimizes for all targets simultaneously)

# Show detailed profile
$ pixelpoison targets --detail claude
[prints full profile YAML above]
```

---

## 6. Profile Maintenance

Target profiles WILL become outdated as VLMs update their architectures and preprocessing. Maintenance protocol:

1. **After each evaluation round** (PRD §12, Week 4): Update profiles with observed attack success rates per target. Adjust `recommended_surrogates` and `recommended_strategies` based on empirical results.
2. **When a new model is released**: Add a new profile with `confidence: low`. Update as documentation and attack results become available.
3. **Quarterly**: Re-verify preprocessing details against current API documentation. Update `last_verified` dates.

Profiles are defined in `src/pixelpoison/targets/profiles.py` and can be extended by users via `~/.pixelpoison/custom_profiles.yaml`.
