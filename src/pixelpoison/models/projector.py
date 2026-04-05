"""Q-Former/projector model loading for IPGA strategy (Tier 3).

Loads InstructBLIP's Q-Former as a surrogate for projector-level attacks.
The get_projected_tokens method is fully differentiable so gradients flow
from the Q-Former output back to the input image pixels.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

console = Console()

_BLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_BLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Get a TrueType font, trying common paths across platforms."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


class ProjectorModel:
    """InstructBLIP Q-Former projector for Tier 3 IPGA attack.

    Provides access to the intermediate projection layer that bridges
    the vision encoder and LLM decoder in BLIP-2 family VLMs.

    get_projected_tokens() is fully differentiable — it applies preprocessing
    directly to the input tensor (no PIL conversion) so gradients flow back
    to the perturbation delta during IPGA optimization.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._model = None
        self._processor = None
        self._loaded = False
        self._qformer_input_ids: Optional[torch.Tensor] = None
        self._qformer_attention_mask: Optional[torch.Tensor] = None
        self._norm_mean: Optional[torch.Tensor] = None
        self._norm_std: Optional[torch.Tensor] = None
        self._image_size = 224

    def load(self) -> None:
        """Load InstructBLIP model with exposed Q-Former layer."""
        console.print("  Loading InstructBLIP Q-Former (Tier 3)...", style="dim")

        from transformers import InstructBlipModel, InstructBlipProcessor

        self._processor = InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-vicuna-7b"
        )
        self._model = InstructBlipModel.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            torch_dtype=torch.float16 if self.device.type != "cpu" else torch.float32,
        ).to(self.device)
        self._model.eval()

        # Precompute Q-Former text input tokens from a dummy instruction.
        # The Q-Former's forward() requires input_ids — these are the tokenized
        # instruction text that conditions the query extraction.
        dummy_img = Image.new("RGB", (224, 224), (128, 128, 128))
        dummy_inputs = self._processor(
            images=dummy_img,
            text="Describe this image.",
            return_tensors="pt",
        )
        self._qformer_input_ids = dummy_inputs["qformer_input_ids"].to(self.device)
        qformer_attn = dummy_inputs.get("qformer_attention_mask")
        if qformer_attn is not None:
            self._qformer_attention_mask = qformer_attn.to(self.device)
        else:
            self._qformer_attention_mask = torch.ones_like(self._qformer_input_ids)

        # Image preprocessing constants (InstructBLIP uses EVA-CLIP vision encoder
        # with the same normalization as OpenAI CLIP)
        self._norm_mean = torch.tensor(_BLIP_MEAN, device=self.device).view(1, 3, 1, 1)
        self._norm_std = torch.tensor(_BLIP_STD, device=self.device).view(1, 3, 1, 1)

        # Resolve image size from processor config
        if hasattr(self._processor, "image_processor"):
            ip = self._processor.image_processor
            size_cfg = getattr(ip, "size", None)
            if isinstance(size_cfg, dict):
                self._image_size = size_cfg.get("height", 224)
            elif isinstance(size_cfg, int):
                self._image_size = size_cfg

        self._loaded = True
        console.print("  InstructBLIP Q-Former loaded.", style="dim")

    def get_projected_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Run image through vision encoder → Q-Former to get projected tokens.

        Fully differentiable: preprocessing is applied directly to the tensor
        so gradients flow back to the input pixels. The caller controls the
        gradient context (use torch.no_grad() externally if not optimizing).

        Args:
            image: Image tensor (B, 3, H, W) in [0, 1].

        Returns:
            Projected query tokens (B, num_queries, hidden_dim) in float32.
        """
        assert self._loaded, "Call load() first"

        # Differentiable preprocessing (matches InstructBLIP's BlipImageProcessor)
        x = F.interpolate(
            image, size=(self._image_size, self._image_size),
            mode="bicubic", align_corners=False,
        )
        x = (x - self._norm_mean) / self._norm_std

        # Cast to model dtype (fp16 on CUDA) — this op is differentiable
        model_dtype = next(self._model.vision_model.parameters()).dtype
        x = x.to(dtype=model_dtype)

        # Vision encoder forward pass
        vision_outputs = self._model.vision_model(pixel_values=x)
        image_embeds = vision_outputs[0]

        # Build attention masks for Q-Former
        batch_size = image_embeds.shape[0]
        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=self.device,
        )

        query_tokens = self._model.query_tokens.expand(batch_size, -1, -1)
        query_attention_mask = torch.ones(
            query_tokens.size()[:-1], dtype=torch.long, device=self.device,
        )

        # Expand precomputed text tokens to batch size
        qformer_ids = self._qformer_input_ids.expand(batch_size, -1)
        qformer_attn = self._qformer_attention_mask.expand(batch_size, -1)

        # Full attention mask = [query_tokens_mask, text_tokens_mask]
        full_attention_mask = torch.cat([query_attention_mask, qformer_attn], dim=1)

        # Q-Former forward pass
        query_outputs = self._model.qformer(
            input_ids=qformer_ids,
            attention_mask=full_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )

        # Return in float32 for stable gradient computation
        return query_outputs[0].float()

    def generate_target_tokens(self, payload: str) -> torch.Tensor:
        """Generate target Q-Former tokens from a text-rendered payload image.

        Renders the payload as visible text on a white background, then runs
        it through the vision encoder → Q-Former pipeline. This is the
        "target representation" that IPGA tries to push the adversarial
        image toward in projector space.

        No gradients needed — this computes a fixed target.
        """
        img = Image.new("RGB", (384, 384), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = _get_font(16)

        words = payload.split()
        lines, current = [], ""
        for word in words:
            test = f"{current} {word}".strip()
            bbox = draw.textbbox((0, 0), test, font=font)
            if bbox[2] > 360:
                lines.append(current)
                current = word
            else:
                current = test
        if current:
            lines.append(current)

        y = 20
        for line in lines:
            draw.text((20, y), line, fill=(0, 0, 0), font=font)
            y += 24

        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return self.get_projected_tokens(tensor)

    def unload(self) -> None:
        """Free model memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._qformer_input_ids = None
        self._qformer_attention_mask = None
        self._norm_mean = None
        self._norm_std = None
        self._loaded = False

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
