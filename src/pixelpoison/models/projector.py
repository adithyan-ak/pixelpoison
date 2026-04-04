"""Q-Former/projector model loading for IPGA strategy (Tier 3).

Loads InstructBLIP's Q-Former as a surrogate for projector-level attacks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

console = Console()


class ProjectorModel:
    """InstructBLIP Q-Former projector for Tier 3 IPGA attack.

    Provides access to the intermediate projection layer that bridges
    the vision encoder and LLM decoder in BLIP-2 family VLMs.
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._model = None
        self._processor = None
        self._loaded = False

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
        self._loaded = True

    def get_projected_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """Run image through vision encoder → Q-Former to get projected tokens.

        Args:
            image: Image tensor (1, 3, H, W) in [0, 1].

        Returns:
            Projected query tokens from the Q-Former.
        """
        assert self._loaded, "Call load() first"

        # Convert tensor to PIL for processor
        arr = (image.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)

        inputs = self._processor(images=pil_img, text="", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get vision outputs through the Q-Former
            vision_outputs = self._model.vision_model(pixel_values=inputs["pixel_values"])
            image_embeds = vision_outputs[0]

            # Q-Former processing
            query_tokens = self._model.qformer(
                query_embeds=self._model.query_tokens.expand(image_embeds.shape[0], -1, -1),
                encoder_hidden_states=image_embeds,
            )[0]

        return query_tokens

    def generate_target_tokens(self, payload: str) -> torch.Tensor:
        """Generate target Q-Former tokens from a text-rendered payload image.

        Renders the payload as text on a white background, then runs it
        through the full vision encoder → Q-Former pipeline.

        Args:
            payload: Text payload to render.

        Returns:
            Target projected query tokens.
        """
        # Render payload as image
        img = Image.new("RGB", (384, 384), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Word wrap
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

        # Process through model
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        return self.get_projected_tokens(tensor)

    def unload(self) -> None:
        """Free model memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._loaded = False

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
