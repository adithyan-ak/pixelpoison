"""SigLIP model loading documentation and utilities.

SigLIP models are loaded via open_clip (same as CLIP models) but have
critical differences that must be respected:

1. Normalization: SigLIP uses mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
   vs CLIP's ImageNet-derived mean/std. Using the wrong normalization
   silently destroys attack effectiveness.

2. Text tokenizer: SigLIP uses GemmaTokenizer (max 64 tokens)
   vs CLIP's BPE tokenizer (max 77 tokens).

3. Loss function: SigLIP was trained with sigmoid loss (binary per-pair)
   vs CLIP's contrastive softmax (NxN matrix). This creates different
   feature geometries — perturbations optimized against CLIP may not
   transfer to SigLIP-based VLMs.

4. Training data: SigLIP 2 uses WebLI + additional objectives (bbox
   prediction, region captioning, self-distillation) making it more
   localization-aware than CLIP.

The CLIPEnsemble class in clip_ensemble.py handles both families
transparently — the per-model normalization is stored in registry.py
and applied automatically during encode_image().

Loading SigLIP via open_clip:
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-SO400M-14-SigLIP2',
        pretrained='webli',
    )

Available SigLIP variants in open_clip v3.0+:
    - ViT-SO400M-14-SigLIP2 / webli  (400M params, 14px patches)
    - ViT-B-16-SigLIP / webli  (86M params, 16px patches)
    - ViT-L-16-SigLIP-256 / webli  (303M params, 16px patches)
"""

# SigLIP normalization constants — used by registry.py
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

# CLIP normalization constants — for reference
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
