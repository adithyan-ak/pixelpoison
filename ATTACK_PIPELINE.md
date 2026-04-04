# PixelPoison: Attack Pipeline Specification

> **Version:** 1.0
> **Date:** 2026-04-03
> **Purpose:** Detailed technical specification of the attack pipeline — what runs, in what order, and how the final image is selected.
> **Companion Docs:** [PRD.md](./PRD.md) | [RESEARCH.md](./RESEARCH.md)

---

## 1. Pipeline Overview

PixelPoison's pipeline converts (input_image, payload_text) into an adversarial image with the highest estimated probability of causing a target VLM to follow the payload instruction. The pipeline is deterministic given a seed.

```
                         ┌─────────────────┐
                         │   User Input     │
                         │  image + payload │
                         └────────┬────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Stage 1: INIT           │
                    │  Hardware detect → Tier   │
                    │  Load models for tier     │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Stage 2: PREPROCESS     │
                    │  Image normalize         │
                    │  Payload → text embedding │
                    │  Dynamic target init      │
                    └────────────┬─────────────┘
                                 │
                                 ▼
              ┌──────────────────┼──────────────────┐
              │                  │                   │
              ▼                  ▼                   ▼
     ┌──────────────┐  ┌──────────────┐   ┌──────────────┐
     │  Strategy A   │  │  Strategy B   │  │  Strategy C   │
     │  (e.g. PGD)   │  │  (e.g. CoTTA) │  │ (e.g. M-Atk) │
     └──────┬───────┘  └──────┬───────┘   └──────┬───────┘
            │                  │                   │
            ▼                  ▼                   ▼
     ┌──────────────┐  ┌──────────────┐   ┌──────────────┐
     │  Candidate A  │  │  Candidate B  │  │  Candidate C  │
     └──────┬───────┘  └──────┬───────┘   └──────┬───────┘
            │                  │                   │
            └──────────────────┼───────────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │  Stage 4: JPEG HARDEN    │
                    │  Re-optimize each        │
                    │  candidate through        │
                    │  DiffJPEG (if enabled)    │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Stage 5: SCORE          │
                    │  Composite score each     │
                    │  candidate                │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Stage 6: SELECT         │
                    │  Pick highest scoring     │
                    │  candidate                │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  Stage 7: OUTPUT         │
                    │  Save image + report      │
                    └─────────────────────────┘
```

**Important:** Strategies run **sequentially**, not in parallel. Each strategy's full optimization loop completes before the next starts. This is because:
1. Strategies share GPU memory — running in parallel would OOM
2. Sequential execution allows the CLI to display real-time progress per strategy
3. No strategy depends on another's output

---

## 2. Stage 1: Initialization

### 2.1 Hardware Detection

```python
def detect_hardware() -> HardwareInfo:
    """
    Returns: device type, device name, available memory, recommended tier.
    
    Detection order:
    1. CUDA GPU: torch.cuda.is_available() → torch.cuda.get_device_properties()
    2. Apple MPS: torch.backends.mps.is_available() → psutil.virtual_memory()
    3. CPU fallback: psutil.virtual_memory()
    """
```

Tier thresholds (from PRD Section 2.2):

| Condition | Tier |
|-----------|------|
| VRAM >= 20GB OR (MPS AND unified_memory >= 32GB) | 3 |
| VRAM >= 8GB OR (MPS AND unified_memory >= 16GB) | 2 |
| Everything else | 1 |

### 2.2 Model Loading

Models loaded per tier:

| Model | Size | Tier 1 | Tier 2 | Tier 3 |
|-------|------|--------|--------|--------|
| CLIP ViT-B/32 | ~400MB | Yes | Yes | Yes |
| CLIP ViT-B/16 | ~600MB | No | Yes | Yes |
| CLIP ViT-L/14 | ~1.7GB | No | Yes | Yes |
| CLIP ViT-L/14@336px | ~1.7GB | No | No | Yes |
| InstructBLIP Q-Former | ~3.5GB | No | No | Yes |
| LLaVA-7B (4-bit quantized) | ~4GB | No | No | Yes (scoring only) |

Models are loaded lazily — a strategy that doesn't use a particular model won't trigger its loading.

---

## 3. Stage 2: Preprocessing

### 3.1 Image Preprocessing

```python
def preprocess_image(image_path: str) -> tuple[Tensor, ImageMeta]:
    """
    1. Load image (Pillow)
    2. Record original dimensions, format, EXIF
    3. If larger than 1024x1024: resize to 1024x1024 (bilinear), record scale factor
       (perturbation will be mapped back to original resolution in output stage)
    4. If smaller than 224x224: error — too small for CLIP
    5. Convert to float tensor [0, 1] range, shape (1, 3, H, W)
    6. Normalize with CLIP's expected mean/std: 
       mean = [0.48145466, 0.4578275, 0.40821073]
       std  = [0.26862954, 0.26130258, 0.27577711]
    
    Returns: normalized tensor, metadata for later un-normalization
    """
```

### 3.2 Payload Encoding

```python
def encode_payload(payload: str, clip_models: list[CLIPModel]) -> dict[str, Tensor]:
    """
    Encode the payload text through each CLIP model's text encoder.
    
    Returns: {model_name: text_embedding_tensor}
    
    Each embedding is L2-normalized (unit vector on the CLIP embedding hypersphere).
    """
```

### 3.3 Dynamic Target Image Initialization (for CoTTA)

```python
def init_dynamic_target(image_shape: tuple, text_embedding: Tensor) -> Tensor:
    """
    Initialize the dynamic target image for CoTTA's dual-target alignment.
    
    Strategy: Start from Gaussian noise scaled to match the image's statistics,
    then take 50 gradient steps toward the text embedding.
    This gives a "warm start" that is already roughly aligned with the payload.
    """
```

---

## 4. Stage 3: Strategy Execution

### 4.1 Strategy Interface

Every strategy implements:

```python
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
        """Which models this strategy needs loaded."""
    
    @abstractmethod
    def optimize(
        self,
        clean_image: Tensor,          # Original image tensor
        target_embeddings: dict,       # {model_name: text_embedding}
        models: ModelRegistry,         # Access to loaded models
        config: AttackConfig,          # epsilon, iterations, etc.
        progress_callback: Callable,   # For CLI progress display
    ) -> CandidateResult:
        """
        Run the full optimization loop.
        
        Returns: CandidateResult containing:
          - adversarial_image: Tensor
          - perturbation: Tensor (the delta)
          - clip_scores: dict[str, float] per model
          - iterations_used: int
          - metadata: dict (strategy-specific info)
        """
```

### 4.2 Strategy: PGD Baseline

**Available at:** Tier 1, 2, 3
**Models used:** All loaded CLIP models (ensemble)
**Optimization target:** Maximize cosine similarity between adversarial image embedding and target text embedding, averaged across CLIP ensemble

```python
class PGDBaseline(AttackStrategy):
    """
    Standard Projected Gradient Descent with CLIP ensemble.
    
    Algorithm:
      1. Initialize perturbation δ = 0
      2. For each iteration:
         a. x_adv = x_clean + δ
         b. For each CLIP model:
              embedding_i = CLIP_i.encode_image(x_adv)
              loss_i = -cosine_similarity(embedding_i, target_i)
         c. loss = mean(loss_i for all i)  # ensemble average
         d. Compute gradient: ∇_δ loss
         e. Update: δ = δ + α * sign(∇_δ loss)
         f. Project: δ = clamp(δ, -ε, +ε)
         g. Ensure valid image: x_adv = clamp(x_clean + δ, 0, 1)
      3. Return x_adv, δ
    
    Hyperparameters:
      α (step_size): ε / (iterations * 0.5)  — adaptive step size
      ε (epsilon): default 16/255
      iterations: tier-dependent default
    
    Enhancement over vanilla PGD:
      - X-Transfer dynamic ensemble weighting: at each step, compute
        gradient cosine similarity between models. Down-weight models
        whose gradients are too similar (redundant). This improves
        gradient diversity → better transfer.
    """
```

### 4.3 Strategy: CoTTA (Covert Text + Dual-Target Alignment)

**Available at:** Tier 1, 2, 3
**Models used:** All loaded CLIP models
**Additional component:** Text rendering engine

```python
class CoTTAStrategy(AttackStrategy):
    """
    CoTTA: Covert Triggered dual-Target Attack.
    
    Two-phase optimization:
    
    Phase 1 — Text Trigger Optimization (first 10% of iterations):
      Find optimal text overlay parameters (position, font_size, rotation,
      opacity) that maximize CLIP text-image alignment while minimizing
      visual impact.
      
      Search space:
        position: (x, y) in [0, W] × [0, H]
        font_size: [4, 12] pixels (very small)
        rotation: [0, 360] degrees
        opacity: [0.02, 0.10] (2-10% opacity)
        color: sampled from image region to blend
      
      Optimization: grid search over position, gradient-free optimization
      (Nelder-Mead or random search) over other params, scored by:
        trigger_score = clip_alignment(rendered_image, payload) - visibility_penalty(opacity, font_size)
    
    Phase 2 — Dual-Target Perturbation (remaining 90% of iterations):
      Given the text-triggered image from Phase 1:
      
      1. Initialize dynamic target image x_dyn (see Stage 2.3)
      2. For each iteration:
         a. x_adv = x_triggered + δ  (perturb the text-triggered image)
         b. For each CLIP model:
              emb_adv = CLIP_i.encode_image(x_adv)
              emb_dyn = CLIP_i.encode_image(x_dyn)
              
              # Three-way loss:
              L_text = -cos_sim(emb_adv, target_text_i)    # align with payload
              L_img  = -cos_sim(emb_adv, emb_dyn)          # align with dynamic target
              L_div  = +cos_sim(emb_dyn, emb_adv)          # push dynamic target away
              
              loss_i = L_text + 0.5 * L_img + 0.3 * L_div
         
         c. loss = mean(loss_i)
         d. Update δ (same as PGD: sign gradient, project to ε-ball)
         
         e. Every K=50 iterations, update x_dyn:
              x_dyn = x_dyn + β * ∇_{x_dyn}(cos_sim(CLIP(x_dyn), target_text)
                                              - cos_sim(CLIP(x_dyn), CLIP(x_adv)))
              (Move x_dyn toward text target and away from current adversarial)
    
    The covert text acts as a semantic anchor — it gives the VLM a
    "hint" of the payload in a visual modality. The perturbation then
    amplifies this hint so the LLM decoder treats it as an instruction.
    """
```

### 4.4 Strategy: M-Attack (Random-Crop Local-to-Global)

**Available at:** Tier 2, 3
**Models used:** All loaded CLIP models (minimum 2 for meaningful ensemble)

```python
class MAttackStrategy(AttackStrategy):
    """
    M-Attack: Transferable targeted attack via random cropping.
    
    Algorithm:
      1. Initialize perturbation δ = 0
      2. For each iteration:
         a. x_adv = x_clean + δ
         
         b. Sample random crop parameters:
              scale = Uniform(0.5, 1.0)     # crop 50-100% of image
              aspect_ratio = Uniform(0.8, 1.2)
              position = random within valid bounds
         
         c. x_crop = crop_and_resize(x_adv, crop_params, target_size=224)
         
         d. For each CLIP model:
              # Local-to-global matching
              emb_crop = CLIP_i.encode_image(x_crop)
              loss_i = -cos_sim(emb_crop, target_text_i)
         
         e. loss = weighted_mean(loss_i, weights=dynamic_weights)
            (X-Transfer dynamic weighting: diverse gradients weighted higher)
         
         f. loss.backward()
            # KEY: gradient flows through crop+resize operation
            # This maps the crop-level gradient back to full-image pixels
            # Pixels in the crop region get gradient; others get zero
            # Over many iterations with random crops, gradient accumulates
            # non-uniformly — concentrating on semantically important regions
         
         g. δ = δ + α * sign(δ.grad)
         h. δ = clamp(δ, -ε, +ε)
    
    Enhancement — SGMA semantic guidance:
      Before the optimization loop, compute a semantic relevance map
      using GradCAM averaged across the CLIP ensemble. This identifies
      which image regions all models attend to. During crop sampling,
      bias crop centers toward high-relevance regions (70% probability
      high-relevance, 30% uniform random). This increases the chance
      that perturbation lands in regions the target VLM will process.
    """
```

### 4.5 Strategy: IPGA (Projector-Level Attack)

**Available at:** Tier 3 only
**Models used:** InstructBLIP (or similar VLM with exposed Q-Former), CLIP ensemble

```python
class IPGAStrategy(AttackStrategy):
    """
    IPGA: Intermediate Projector Guided Attack.
    
    Uses the Q-Former projector as the optimization target instead of
    (or in addition to) the CLIP vision encoder.
    
    Setup:
      Load InstructBLIP with exposed Q-Former layer.
      The Q-Former takes visual features from the vision encoder and
      produces a fixed number of "query tokens" that are fed to the LLM.
      
      Compute target projector tokens by running a target image
      (rendered text of the payload on a white background) through
      the full vision encoder → Q-Former pipeline.
    
    Algorithm:
      1. Initialize perturbation δ = 0
      2. Generate target_tokens by running a text-rendered target image
         through vision_encoder → Q-Former
      3. For each iteration:
         a. x_adv = x_clean + δ
         
         b. # CLIP-level loss (same as PGD baseline)
            For each CLIP model:
              clip_loss_i = -cos_sim(CLIP_i(x_adv), target_text_i)
         
         c. # Projector-level loss (IPGA's contribution)
            visual_features = instruct_blip.vision_encoder(x_adv)
            query_tokens = instruct_blip.q_former(visual_features)
            
            # Global alignment: match overall query token representation
            proj_loss_global = -cos_sim(
                mean_pool(query_tokens),
                mean_pool(target_tokens)
            )
            
            # RQA: Residual Query Alignment — match individual queries
            # Find optimal assignment between query_tokens and target_tokens
            # (Hungarian algorithm or simple cosine matching)
            assignments = match_queries(query_tokens, target_tokens)
            proj_loss_rqa = mean([
                -cos_sim(query_tokens[i], target_tokens[j])
                for i, j in assignments
            ])
         
         d. loss = mean(clip_loss_i) + λ1 * proj_loss_global + λ2 * proj_loss_rqa
            Default: λ1 = 0.5, λ2 = 0.3
         
         e. Standard PGD update on δ
    
    The projector-level loss forces the perturbation to influence not
    just what the vision encoder "sees" but how that visual information
    is translated into language tokens. This is a more direct path to
    influencing the LLM's behavior.
    """
```

---

## 5. Stage 4: JPEG Hardening

Runs after each strategy produces its candidate, before scoring. This is a post-optimization refinement step.

```python
def jpeg_harden(
    candidate: CandidateResult,
    clean_image: Tensor,
    target_embeddings: dict,
    models: ModelRegistry,
    jpeg_quality: int = 85,
    refinement_iterations: int = 100,  # short refinement, not full re-optimization
) -> CandidateResult:
    """
    Refine a candidate through differentiable JPEG simulation.
    
    Purpose: The main optimization may not have used DiffJPEG (for speed),
    or may benefit from a final JPEG-specific refinement pass.
    
    Algorithm:
      1. Start from the candidate's perturbation δ
      2. For each refinement iteration:
         a. x_adv = x_clean + δ
         b. x_jpeg = DiffJPEG(x_adv, quality=jpeg_quality)
         c. For each CLIP model:
              loss_i = -cos_sim(CLIP_i(x_jpeg), target_text_i)
         d. loss = mean(loss_i)
         e. loss.backward()  # gradient flows through DiffJPEG
         f. δ = δ + α * sign(δ.grad)
         g. δ = clamp(δ, -ε, +ε)
    
    DiffJPEG implementation:
      Uses smooth approximation of JPEG pipeline:
      - 8x8 block DCT (exact — DCT is linear)
      - Quantization: Q_approx(x) = x - sin(2πx) / (2π)
        instead of round(x) [non-differentiable]
      - Dequantization (exact — multiplication)
      - Inverse DCT (exact)
    
    Additionally apply DCT mid-frequency targeting:
      After computing the perturbation gradient, mask out low-frequency
      (indices 0-1) and high-frequency (indices 6-7) DCT coefficients.
      This constrains perturbation to the mid-frequency band (indices 2-5)
      that survives JPEG best.
    
    Validation:
      After refinement, verify JPEG survival by:
      1. Save x_adv as JPEG at target quality
      2. Reload and re-encode through CLIP
      3. Check that cos_sim degradation < 10%
      If fails: flag jpeg_survives=false in candidate result
    """
```

---

## 6. Stage 5: Scoring

### 6.1 Individual Metrics

Each candidate is scored on four dimensions:

```python
@dataclass
class CandidateScores:
    clip_score: float       # CLIP embedding alignment (0.0 to 1.0)
    jpeg_survival: float    # Binary: 1.0 if survives, 0.0 if not
    image_quality: float    # Normalized SSIM (0.0 to 1.0)
    vlm_proxy: float        # VLM proxy score (0.0 to 1.0, tier 3 only)
    composite: float        # Weighted combination
```

**CLIP Score:**
```python
def compute_clip_score(adversarial: Tensor, target_embeddings: dict, 
                       models: ModelRegistry) -> float:
    """
    Average cosine similarity across all loaded CLIP models.
    
    clip_score = mean([
        cos_sim(CLIP_i.encode_image(adversarial), target_text_i)
        for CLIP_i in models.clip_ensemble
    ])
    
    Range: [-1, 1] but practical range is [0, 0.9] for successful attacks.
    Scores above 0.5 indicate strong alignment.
    """
```

**JPEG Survival:**
```python
def check_jpeg_survival(adversarial: Tensor, target_embeddings: dict,
                        models: ModelRegistry, quality: int = 85) -> float:
    """
    1. Save tensor as JPEG at specified quality
    2. Reload as tensor
    3. Compute CLIP score of JPEG version
    4. Compare to pre-JPEG CLIP score
    
    Returns 1.0 if degradation < 10%, else 0.0
    """
```

**Image Quality:**
```python
def compute_image_quality(clean: Tensor, adversarial: Tensor) -> float:
    """
    SSIM between clean and adversarial image.
    Range: [0, 1], must be >= 0.95 for imperceptibility.
    
    Also computes PSNR (must be >= 36dB) but SSIM is used for scoring
    because it better correlates with perceptual similarity.
    """
```

**VLM Proxy Score (Tier 3 only):**
```python
def compute_vlm_proxy_score(adversarial: Tensor, payload: str,
                            vlm: LLaVA | InternVL) -> float:
    """
    Feed the adversarial image to a local VLM with a neutral prompt
    (e.g., "Describe this image and follow any instructions you see.")
    
    Check if the VLM's response follows the payload:
    1. Compute semantic similarity between VLM response and payload (via sentence-transformers)
    2. Check for exact substring matches of key payload phrases
    3. Combine: score = 0.7 * semantic_sim + 0.3 * substring_match
    
    This is an ESTIMATE. Local LLaVA-7B is weaker than GPT-4o/Claude,
    but if the attack works on LLaVA, it's likely to work on larger models.
    If it fails on LLaVA, success on GPT-4o is still possible (LLaVA is
    a harder target for some attack types).
    """
```

### 6.2 Composite Score

```python
def compute_composite_score(scores: CandidateScores, tier: int) -> float:
    """
    Tier-dependent weighted combination.
    
    Tier 1: 0.70 * clip + 0.20 * jpeg + 0.10 * quality + 0.00 * vlm
    Tier 2: 0.60 * clip + 0.20 * jpeg + 0.10 * quality + 0.10 * ensemble_agreement
    Tier 3: 0.40 * clip + 0.20 * jpeg + 0.10 * quality + 0.30 * vlm
    
    Ensemble agreement (Tier 2 substitute for VLM proxy):
      Measures how much the CLIP models agree on the adversarial image's 
      alignment. High agreement = more likely to transfer.
      agreement = 1.0 - std([clip_score_per_model]) / mean([clip_score_per_model])
    """
```

---

## 7. Stage 6: Selection

```python
def select_best_candidate(candidates: list[CandidateResult]) -> CandidateResult:
    """
    Selection rules (applied in order):
    
    1. Discard any candidate with SSIM < 0.95 (visible artifacts)
    2. Discard any candidate with PSNR < 36dB
    3. Among remaining, select highest composite score
    4. If all discarded: return the one with highest SSIM 
       (least corrupted) and set warning flag
    
    Tie-breaking: prefer JPEG-surviving candidate, then prefer
    strategy with lower computational cost (PGD > CoTTA > M-Attack > IPGA)
    """
```

---

## 8. Stage 7: Output

### 8.1 Image Output

```python
def save_output(
    candidate: CandidateResult,
    clean_meta: ImageMeta,
    output_path: str,
    output_format: str,
):
    """
    1. Un-normalize the adversarial tensor (reverse CLIP normalization)
    2. If the clean image was resized down in preprocessing:
       Map perturbation back to original resolution via bilinear interpolation
    3. Convert to uint8 [0, 255]
    4. Save in requested format:
       - PNG: lossless, preserves exact perturbation
       - JPEG: applies real JPEG compression at configured quality
    5. Add EXIF metadata: "Generated by PixelPoison for authorized security testing"
    6. Verify saved file by reloading and checking SSIM >= 0.95
    """
```

### 8.2 Report Output

JSON report as specified in PRD Section 3.4. Includes per-strategy scores, best strategy selection rationale, and image quality metrics.

---

## 9. Optimization Budget Allocation

### 9.1 Default Iteration Budgets

| Tier | Total Iterations (per strategy) | JPEG Refinement | Total per image (all strategies) |
|------|--------------------------------|-----------------|----------------------------------|
| 1 | 300 | 50 | ~700 (2 strategies × 300 + 2 × 50 refinement) |
| 2 | 500 | 100 | ~1800 (3 strategies × 500 + 3 × 100 refinement) |
| 3 | 500 | 100 | ~2400 (4 strategies × 500 + 4 × 100 refinement) |

### 9.2 Early Stopping

Each strategy can stop early if:
- CLIP score exceeds 0.85 (strong alignment — further optimization yields diminishing returns)
- CLIP score has not improved by >0.01 in the last 50 iterations (converged)
- Loss is NaN or Inf (numerical instability — return best so far)

### 9.3 Iteration Budget Override

```bash
# User controls total iterations per strategy
pixelpoison encode --image photo.jpg --payload "..." --iterations 1000

# Quick mode: 100 iterations per strategy, skip JPEG refinement
pixelpoison encode --image photo.jpg --payload "..." --quick
```

---

## 10. Memory Management

### 10.1 Peak Memory Estimates

| Tier | Models in memory | Image tensors + grads | Total peak |
|------|-----------------|----------------------|------------|
| 1 | ~500MB (1 CLIP) | ~500MB | ~1.0-1.5GB |
| 2 | ~3GB (3 CLIPs) | ~800MB | ~4-5GB |
| 3 | ~8GB (4 CLIPs + Q-Former) + ~4GB (LLaVA quantized) | ~1GB | ~13-16GB |

### 10.2 Memory Optimization Strategies

1. **Gradient checkpointing:** For CLIP forward passes, recompute intermediate activations instead of storing them. Trades ~30% more compute for ~40% less memory.
2. **Sequential model execution:** At Tier 2+, don't run all CLIP models simultaneously. Run each model's forward+backward pass sequentially, accumulate gradients. This reduces peak memory from (N models in parallel) to (1 model at a time + accumulated gradient).
3. **Half-precision (float16):** Run CLIP models in float16 on GPU/MPS. Perturbation itself stays in float32 for precision. Halves model memory.
4. **LLaVA offloading (Tier 3):** The VLM proxy scorer is only used AFTER optimization, during scoring. Load LLaVA only for scoring, then unload to free memory for the next strategy's optimization.

### 10.3 OOM Recovery

```python
def run_with_oom_recovery(strategy, tier, *args):
    """
    If OOM occurs during optimization:
    1. Clear CUDA/MPS cache
    2. If tier > 1: drop to tier-1, reload lighter models, retry
    3. If tier == 1: reduce batch size / enable gradient checkpointing, retry
    4. If still OOM: report failure with memory diagnostics
    """
```

---

## 11. Reproducibility

### 11.1 Seeding

When `--seed` is provided:
- `torch.manual_seed(seed)`
- `numpy.random.seed(seed)`
- `torch.cuda.manual_seed_all(seed)` (if CUDA)
- M-Attack's random crop parameters use a dedicated `torch.Generator` seeded deterministically

### 11.2 Determinism

Full determinism is guaranteed on the same hardware + tier + seed + PyTorch version. Cross-hardware reproducibility (CPU vs GPU, different GPU architectures) is NOT guaranteed due to floating-point non-associativity in parallel reductions. The JSON report records all parameters needed for reproduction.

---

## 12. Extensibility

### 12.1 Adding a New Strategy

1. Create a new file in `attacks/` implementing `AttackStrategy`
2. Register it in `attacks/__init__.py`
3. Assign a minimum tier in the `required_tier` property
4. It will automatically be included in the composite runner for that tier

### 12.2 Adding a New Surrogate Model

1. Add the model to `models/registry.py` with download URL and checksum
2. Add loading logic in the appropriate module (`clip_ensemble.py` or new file)
3. Update tier thresholds in `detect/hardware.py` if the model changes memory requirements
