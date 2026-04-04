"""CLIP and SigLIP ensemble loader — load, normalize, encode."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from rich.console import Console

from pixelpoison.models.registry import ModelSpec, MODELS, get_models_for_tier

console = Console()


class CLIPEnsemble:
    """Manages an ensemble of CLIP/SigLIP models for adversarial optimization.

    Models are loaded sequentially and share the same device. Each model
    applies its own normalization (CLIP vs SigLIP have different constants).
    """

    def __init__(self, model_ids: list[str], device: torch.device):
        self.model_ids = model_ids
        self.device = device
        self._models: dict[str, object] = {}
        self._tokenizers: dict[str, object] = {}
        self._specs: dict[str, ModelSpec] = {}
        self._dtypes: dict[str, torch.dtype] = {}
        self._loaded = False

    @classmethod
    def for_tier(cls, tier: int, device: torch.device) -> CLIPEnsemble:
        """Create an ensemble with all models appropriate for the given tier."""
        specs = get_models_for_tier(tier)
        return cls([s.id for s in specs], device)

    def load(self) -> None:
        """Download (if needed) and load all models into memory."""
        import open_clip

        # fp16 only on CUDA — MPS has known issues with half-precision conv ops
        precision = "fp16" if self.device.type == "cuda" else "fp32"

        for model_id in self.model_ids:
            spec = MODELS[model_id]
            self._specs[model_id] = spec

            console.print(f"  Loading {spec.display_name}...", style="dim")
            model, _, _preprocess = open_clip.create_model_and_transforms(
                spec.open_clip_model,
                pretrained=spec.pretrained,
                device=self.device,
                precision=precision,
            )
            model.eval()
            self._models[model_id] = model
            self._tokenizers[model_id] = open_clip.get_tokenizer(spec.open_clip_model)
            # Store model dtype for input casting
            try:
                p = next(model.parameters())
                self._dtypes[model_id] = p.dtype
            except StopIteration:
                self._dtypes[model_id] = torch.float32

        self._loaded = True

    def _normalize(self, x: torch.Tensor, model_id: str) -> torch.Tensor:
        """Apply model-specific normalization to image tensor.

        Args:
            x: Image tensor in [0, 1] range, shape (B, 3, H, W).
            model_id: Which model's normalization to apply.
        """
        spec = self._specs[model_id]
        mean = torch.tensor(spec.norm_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(spec.norm_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std

    def _resize_for_model(self, x: torch.Tensor, model_id: str) -> torch.Tensor:
        """Resize image to model's expected input size (224 or 336)."""
        target_size = 336 if "336" in model_id else 224
        if x.shape[-1] != target_size or x.shape[-2] != target_size:
            return F.interpolate(x, size=(target_size, target_size), mode="bicubic", align_corners=False)
        return x

    def encode_image(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode an image through all models in the ensemble.

        Args:
            x: Image tensor in [0, 1] range, shape (B, 3, H, W).
               NOT normalized — each model applies its own normalization.

        Returns:
            Dict mapping model_id → L2-normalized embedding tensor.
        """
        assert self._loaded, "Call load() before encode_image()"
        embeddings = {}
        for model_id, model in self._models.items():
            x_norm = self._normalize(x, model_id)
            x_resized = self._resize_for_model(x_norm, model_id)
            # Cast to model dtype (models may be fp16, input is fp32)
            x_cast = x_resized.to(dtype=self._dtypes[model_id])
            emb = model.encode_image(x_cast)
            emb = F.normalize(emb.float(), dim=-1)  # Scores always in fp32
            embeddings[model_id] = emb
        return embeddings

    def encode_image_single(self, x: torch.Tensor, model_id: str) -> torch.Tensor:
        """Encode image through a single model. Returns L2-normalized embedding."""
        assert self._loaded, "Call load() before encode_image_single()"
        model = self._models[model_id]
        x_norm = self._normalize(x, model_id)
        x_resized = self._resize_for_model(x_norm, model_id)
        x_cast = x_resized.to(dtype=self._dtypes[model_id])
        emb = model.encode_image(x_cast)
        return F.normalize(emb.float(), dim=-1)

    def encode_text(self, text: str) -> dict[str, torch.Tensor]:
        """Encode a text string through all models' text encoders.

        Returns:
            Dict mapping model_id → L2-normalized text embedding tensor.
        """
        assert self._loaded, "Call load() before encode_text()"
        embeddings = {}
        for model_id, model in self._models.items():
            tokenizer = self._tokenizers[model_id]
            tokens = tokenizer([text]).to(self.device)
            with torch.no_grad():
                emb = model.encode_text(tokens)
                emb = F.normalize(emb.float(), dim=-1)
            embeddings[model_id] = emb
        return embeddings

    def get_model(self, model_id: str):
        """Get a raw model by ID (for GradCAM or other introspection)."""
        return self._models[model_id]

    @property
    def model_count(self) -> int:
        return len(self._models)

    @property
    def loaded_models(self) -> list[str]:
        return list(self._models.keys())

    def unload(self, model_id: Optional[str] = None) -> None:
        """Unload a specific model or all models to free memory."""
        if model_id:
            if model_id in self._models:
                del self._models[model_id]
                del self._tokenizers[model_id]
        else:
            self._models.clear()
            self._tokenizers.clear()
            self._loaded = False

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
