"""IPGA: Intermediate Projector Guided Attack (Tier 3) with MI-FGSM + DIM.

Research basis:
- IPGA: arXiv:2508.13739 (August 2025)
- MI-FGSM momentum: Dong et al. (CVPR 2018)
- DIM (Diverse Input Method): Xie et al. (CVPR 2019)
- TIM (Translation-Invariant Method): Dong et al. (CVPR 2019)

Targets the Q-Former projector layer rather than just the vision encoder,
providing a more direct path to influencing the LLM's perception.
MI-FGSM momentum and DIM improve transferability to black-box targets.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from pixelpoison.attacks.base import AttackConfig, AttackStrategy, CandidateResult
from pixelpoison.augmentation.transforms import AugmentationPipeline
from pixelpoison.models.projector import ProjectorModel
from pixelpoison.scoring.quality import compute_psnr, compute_ssim


def _get_gaussian_kernel_2d(kernel_size: int = 7, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2D Gaussian kernel for TIM gradient smoothing."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = g / g.sum()
    kernel_2d = kernel_1d.outer(kernel_1d)
    return kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)


def _greedy_query_matching(
    queries: torch.Tensor, targets: torch.Tensor
) -> list[tuple[int, int]]:
    """Greedy approximation of Hungarian matching for query alignment."""
    n = queries.shape[0]
    m = targets.shape[0]
    k = min(n, m)

    sims = F.cosine_similarity(
        queries.unsqueeze(1), targets.unsqueeze(0), dim=-1
    )

    assignments = []
    used_targets = set()

    for _ in range(k):
        mask = torch.ones_like(sims)
        for t in used_targets:
            mask[:, t] = 0
        masked_sims = sims * mask

        flat_idx = masked_sims.argmax().item()
        qi, ti = flat_idx // m, flat_idx % m
        assignments.append((qi, ti))
        used_targets.add(ti)
        sims[qi, :] = -1

    return assignments


class IPGAStrategy(AttackStrategy):
    """IPGA with MI-FGSM momentum + DIM + TIM for maximum transferability.

    Combines CLIP-level loss with Q-Former projector-level losses:
    - Global alignment: match mean-pooled query representations
    - RQA (Residual Query Alignment): match individual query-target pairs

    MI-FGSM momentum stabilizes gradient direction for better convergence.
    DIM prevents overfitting to surrogate spatial features.
    TIM smooths gradients for position-invariant perturbations.
    """

    @property
    def name(self) -> str:
        return "ipga"

    @property
    def required_tier(self) -> int:
        return 3

    @property
    def required_models(self) -> list[str]:
        return []

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

        projector = ProjectorModel(device)
        projector.load()

        try:
            return self._run_optimization(
                projector, clean_image, target_embeddings,
                ensemble, config, progress_callback, start_time,
            )
        finally:
            projector.unload()

    def _run_optimization(
        self,
        projector: ProjectorModel,
        clean_image: torch.Tensor,
        target_embeddings: dict[str, torch.Tensor],
        ensemble,
        config: AttackConfig,
        progress_callback,
        start_time: float,
    ) -> CandidateResult:
        device = clean_image.device

        # Setup DIM (input diversity)
        augmenter = AugmentationPipeline(
            enable_resize=True, enable_blur=True, enable_jitter=True,
        ).to(device)

        # Setup TIM (Gaussian kernel for gradient smoothing)
        tim_kernel = _get_gaussian_kernel_2d(kernel_size=7, sigma=1.5).to(device)
        tim_padding = 7 // 2

        # Generate target projector tokens from rendered payload text
        payload_text = getattr(config, '_payload', 'test')
        target_tokens = projector.generate_target_tokens(payload_text)

        # Initialize perturbation and MI-FGSM momentum
        delta = torch.zeros_like(clean_image, requires_grad=True, device=device)
        momentum = torch.zeros_like(clean_image, device=device)
        mu = 1.0  # MI-FGSM momentum decay
        alpha = config.step_size

        best_score = -float("inf")
        best_delta = delta.data.clone()
        no_improve_count = 0
        iteration = 0

        lambda1 = 0.5
        lambda2 = 0.3

        for iteration in range(config.iterations):
            if delta.grad is not None:
                delta.grad.zero_()

            x_adv = (clean_image + delta).clamp(0, 1)

            # --- DIM: apply input diversity ---
            x_adv_div = augmenter(x_adv)

            # CLIP-level losses
            clip_loss = torch.tensor(0.0, device=device)
            for model_id in ensemble.loaded_models:
                emb = ensemble.encode_image_single(x_adv_div, model_id)
                target_emb = target_embeddings[model_id]
                clip_loss = clip_loss - F.cosine_similarity(emb, target_emb, dim=-1).mean()
            clip_loss = clip_loss / ensemble.model_count

            # Projector-level losses (use un-augmented for projector consistency)
            query_tokens = projector.get_projected_tokens(x_adv)

            # Global alignment
            query_mean = query_tokens.mean(dim=1)
            target_mean = target_tokens.mean(dim=1)
            proj_loss_global = -F.cosine_similarity(query_mean, target_mean, dim=-1).mean()

            # RQA: Residual Query Alignment
            q = query_tokens.squeeze(0)
            t = target_tokens.squeeze(0)
            assignments = _greedy_query_matching(q.detach(), t.detach())

            rqa_loss = torch.tensor(0.0, device=device)
            for qi, ti in assignments:
                rqa_loss = rqa_loss - F.cosine_similarity(
                    q[qi].unsqueeze(0), t[ti].unsqueeze(0), dim=-1
                ).mean()
            rqa_loss = rqa_loss / max(len(assignments), 1)

            total_loss = clip_loss + lambda1 * proj_loss_global + lambda2 * rqa_loss
            total_loss.backward()

            # --- TIM: smooth gradient ---
            grad = delta.grad.data
            grad = F.conv2d(grad, tim_kernel, padding=tim_padding, groups=3)

            # --- MI-FGSM: momentum update ---
            grad_norm = grad / (torch.mean(torch.abs(grad)) + 1e-12)
            momentum = mu * momentum + grad_norm

            with torch.no_grad():
                delta.data = delta.data - alpha * momentum.sign()
                delta.data = delta.data.clamp(-config.epsilon, config.epsilon)
                delta.data = (clean_image + delta.data).clamp(0, 1) - clean_image

            if (iteration + 1) % 25 == 0 or iteration == config.iterations - 1:
                with torch.no_grad():
                    full_adv = (clean_image + delta.data).clamp(0, 1)
                    embs = ensemble.encode_image(full_adv)
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
                    progress_callback(iteration, current_score)

                # Early stopping (relaxed)
                if current_score > 0.95:
                    break
                if no_improve_count >= 20:  # Checked every 25 = 500 stale iters
                    break

        adversarial = (clean_image + best_delta).clamp(0, 1)

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
            iterations_used=iteration + 1,
            time_seconds=time.time() - start_time,
            metadata={
                "lambda1": lambda1,
                "lambda2": lambda2,
                "momentum_decay": mu,
            },
        )
