"""CoTTA: Covert Triggered dual-Target Attack.

Research basis: arXiv:2603.29418 (March 2026)

Two-phase optimization:
- Phase 1 (10%): Optimize covert text trigger placement
- Phase 2 (90%): Dual-target perturbation with dynamic target refinement
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.rendering.text_overlay import (
    TextTriggerParams,
    optimize_text_trigger,
    render_text_trigger,
)
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


class CoTTAStrategy(AttackStrategy):
    """CoTTA: Covert Triggered dual-Target Attack.

    Phase 1 — Text Trigger Optimization:
        Find optimal text overlay parameters that maximize CLIP alignment
        while minimizing visual impact. The text acts as a semantic anchor.

    Phase 2 — Dual-Target Perturbation:
        Optimize perturbation against two targets simultaneously:
        1. Target text embedding (the payload)
        2. A dynamic target image that evolves toward the payload
        This prevents convergence to local minima that satisfy CLIP loss
        but don't cause the LLM decoder to follow the instruction.
    """

    @property
    def name(self) -> str:
        return "cotta"

    @property
    def required_tier(self) -> int:
        return 1

    @property
    def required_models(self) -> list[str]:
        return []  # Uses whatever models are in the ensemble

    def _init_dynamic_target(
        self,
        image_shape: tuple,
        target_embeddings: dict[str, torch.Tensor],
        ensemble,
        device: torch.device,
        warmup_steps: int = 50,
    ) -> torch.Tensor:
        """Initialize the dynamic target image with warmup toward text embedding.

        Starts from Gaussian noise matching the image statistics, then takes
        gradient steps toward the text embedding for a warm start.
        """
        x_dyn = torch.randn(image_shape, device=device) * 0.1 + 0.5
        x_dyn = x_dyn.clamp(0, 1).requires_grad_(True)

        optimizer = torch.optim.Adam([x_dyn], lr=0.01)

        for _ in range(warmup_steps):
            optimizer.zero_grad()
            embs = ensemble.encode_image(x_dyn)
            loss = torch.tensor(0.0, device=device)
            for mid, emb in embs.items():
                loss = loss - F.cosine_similarity(emb, target_embeddings[mid], dim=-1).mean()
            loss = loss / len(embs)
            loss.backward()
            optimizer.step()
            x_dyn.data.clamp_(0, 1)

        return x_dyn.detach()

    def optimize(
        self,
        clean_image: torch.Tensor,
        target_embeddings: dict[str, torch.Tensor],
        ensemble,
        config: AttackConfig,
        progress_callback: Optional[Callable[[int, float], None]] = None,
    ) -> CandidateResult:
        start_time = time.time()
        device = clean_image.device

        if config.seed is not None:
            torch.manual_seed(config.seed)

        total_iterations = config.iterations
        phase1_iters = max(1, total_iterations // 10)
        phase2_iters = total_iterations - phase1_iters

        # ---- Phase 1: Text Trigger Optimization ----
        n_pos = 9 if config.quick else 25
        n_samples = 2 if config.quick else 5

        best_trigger = optimize_text_trigger(
            clean_image, config._payload if hasattr(config, '_payload') else "test",
            ensemble, target_embeddings,
            n_positions=n_pos, n_samples_per_position=n_samples,
        )

        # Apply text trigger to create the base image
        # We need the payload text — store it in metadata
        payload_text = getattr(config, '_payload', 'test')
        x_triggered = render_text_trigger(clean_image, payload_text, best_trigger)
        x_triggered = x_triggered.to(device)

        # ---- Phase 2: Dual-Target Perturbation ----
        # Initialize dynamic target
        x_dyn = self._init_dynamic_target(
            clean_image.shape, target_embeddings, ensemble, device,
            warmup_steps=10 if config.quick else 50,
        )

        delta = torch.zeros_like(clean_image, requires_grad=True, device=device)
        step_size = config.step_size

        best_score = -float("inf")
        best_delta = delta.data.clone()
        no_improve_count = 0
        dyn_update_interval = 50

        for iteration in range(phase2_iters):
            if delta.grad is not None:
                delta.grad.zero_()

            x_adv = (x_triggered + delta).clamp(0, 1)

            total_loss = torch.tensor(0.0, device=device)

            for model_id in ensemble.loaded_models:
                emb_adv = ensemble.encode_image_single(x_adv, model_id)
                emb_dyn = ensemble.encode_image_single(x_dyn.detach(), model_id)
                target_emb = target_embeddings[model_id]

                # Three-way loss
                l_text = -F.cosine_similarity(emb_adv, target_emb, dim=-1).mean()
                l_img = -F.cosine_similarity(emb_adv, emb_dyn, dim=-1).mean()
                l_div = F.cosine_similarity(emb_dyn, emb_adv.detach(), dim=-1).mean()

                loss_i = l_text + 0.5 * l_img + 0.3 * l_div
                total_loss = total_loss + loss_i

            total_loss = total_loss / ensemble.model_count
            total_loss.backward()

            # PGD update
            with torch.no_grad():
                delta.data = delta.data - step_size * delta.grad.sign()
                delta.data = delta.data.clamp(-config.epsilon, config.epsilon)
                delta.data = (x_triggered + delta.data).clamp(0, 1) - x_triggered

            # Track score (use text alignment only for monitoring)
            with torch.no_grad():
                embs = ensemble.encode_image((x_triggered + delta.data).clamp(0, 1))
                sims = [
                    F.cosine_similarity(embs[mid], target_embeddings[mid], dim=-1).item()
                    for mid in embs
                ]
                current_score = sum(sims) / len(sims)

            if current_score > best_score:
                best_score = current_score
                best_delta = delta.data.clone()
                no_improve_count = 0
            else:
                no_improve_count += 1

            if progress_callback:
                progress_callback(phase1_iters + iteration, current_score)

            # Update dynamic target periodically
            if (iteration + 1) % dyn_update_interval == 0:
                x_dyn = x_dyn.requires_grad_(True)
                dyn_opt = torch.optim.Adam([x_dyn], lr=0.005)
                for _ in range(5):
                    dyn_opt.zero_grad()
                    dyn_embs = ensemble.encode_image(x_dyn)
                    adv_embs = ensemble.encode_image((x_triggered + best_delta).clamp(0, 1))
                    dyn_loss = torch.tensor(0.0, device=device)
                    for mid in dyn_embs:
                        # Move toward text, away from current adversarial
                        dyn_loss -= F.cosine_similarity(
                            dyn_embs[mid], target_embeddings[mid], dim=-1
                        ).mean()
                        dyn_loss += 0.3 * F.cosine_similarity(
                            dyn_embs[mid], adv_embs[mid].detach(), dim=-1
                        ).mean()
                    dyn_loss = dyn_loss / len(dyn_embs)
                    dyn_loss.backward()
                    dyn_opt.step()
                    x_dyn.data.clamp_(0, 1)
                x_dyn = x_dyn.detach()

            # Early stopping
            early_stop_threshold = 0.75 if config.quick else 0.85
            if current_score > early_stop_threshold:
                break
            if no_improve_count >= 50:
                break

        # Final result
        adversarial = (x_triggered + best_delta).clamp(0, 1)

        final_per_model = {}
        with torch.no_grad():
            embeddings = ensemble.encode_image(adversarial)
            for model_id, emb in embeddings.items():
                sim = F.cosine_similarity(emb, target_embeddings[model_id], dim=-1).item()
                final_per_model[model_id] = sim

        mean_clip = sum(final_per_model.values()) / len(final_per_model)

        return CandidateResult(
            strategy_name=self.name,
            adversarial_image=adversarial.detach(),
            perturbation=best_delta.detach(),
            per_model_scores=final_per_model,
            clip_score=mean_clip,
            psnr=compute_psnr(clean_image, adversarial.detach()),
            ssim=compute_ssim(clean_image, adversarial.detach()),
            iterations_used=total_iterations,
            time_seconds=time.time() - start_time,
            metadata={
                "trigger_params": {
                    "position": best_trigger.position,
                    "font_size": best_trigger.font_size,
                    "opacity": best_trigger.opacity,
                    "rotation": best_trigger.rotation,
                },
            },
        )
