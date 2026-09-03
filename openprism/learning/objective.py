"""Training objectives and reproducibility-auditable metric components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .model import EGTCFOutput


@dataclass(frozen=True, slots=True)
class EGTCFLossConfig:
    gradient_weight: float = 1.0
    task_intensity_weight: float = 0.7
    abstention_weight: float = 0.8
    calibration_weight: float = 0.5
    task_classification_weight: float = 0.25
    contribution_entropy_weight: float = 0.02


def _gradient(value: Tensor) -> Tensor:
    horizontal = F.pad(value[..., :, 1:] - value[..., :, :-1], (0, 1, 0, 0))
    vertical = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(horizontal.square() + vertical.square() + 1e-8)


def proxy_targets(
    rgb: Tensor,
    thermal: Tensor,
    evidence_support: Tensor,
    task_ids: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return declared reference-free gradient and intensity proxy targets."""

    visible_y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    visible_gradient = _gradient(visible_y)
    thermal_gradient = _gradient(thermal)
    target_gradient = torch.maximum(
        visible_gradient, thermal_gradient * evidence_support
    )
    thermal_saliency = (
        thermal - F.avg_pool2d(thermal, 7, stride=1, padding=3)
    ).abs()
    saliency_scale = (
        thermal_saliency.flatten(1).quantile(0.9, dim=1).clamp_min(1e-4)
    )
    saliency = (
        thermal_saliency / saliency_scale[:, None, None, None]
    ).clamp(0.0, 1.0)
    night = ((0.45 - visible_y.mean(dim=(2, 3), keepdim=True)) / 0.35).clamp(
        0.0, 1.0
    )
    task_weight = torch.where(
        (task_ids == 1)[:, None, None, None],
        0.25 + 0.70 * saliency,
        torch.where(
            (task_ids == 2)[:, None, None, None],
            0.15 + 0.25 * saliency,
            0.10 + 0.35 * torch.maximum(saliency, night),
        ),
    )
    task_weight = (task_weight * evidence_support).clamp(0.0, 0.90)
    intensity_target = (1.0 - task_weight) * visible_y + task_weight * thermal
    return target_gradient, intensity_target


def fusion_proxy_metrics(
    fused_luminance: Tensor,
    rgb: Tensor,
    thermal: Tensor,
    evidence_support: Tensor,
    task_ids: Tensor,
) -> dict[str, Tensor]:
    target_gradient, intensity_target = proxy_targets(
        rgb, thermal, evidence_support, task_ids
    )
    return {
        "gradient_l1": F.l1_loss(_gradient(fused_luminance), target_gradient),
        "task_intensity_smooth_l1": F.smooth_l1_loss(
            fused_luminance, intensity_target
        ),
    }


class EGTCFLoss(nn.Module):
    """Reference-free fusion loss plus corruption-supervised abstention.

    The targets are explicitly proxies, not fused-image ground truth.  Paper
    experiments must therefore prioritize downstream task performance,
    selective risk, calibration, and corruption robustness over aesthetic
    no-reference fusion scores.
    """

    def __init__(self, config: EGTCFLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or EGTCFLossConfig()

    def forward(
        self,
        output: EGTCFOutput,
        rgb: Tensor,
        thermal: Tensor,
        task_ids: Tensor,
        *,
        corruption_target: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        support = output.evidence_support
        proxy = fusion_proxy_metrics(
            output.fused_luminance, rgb, thermal, support, task_ids
        )
        gradient_loss = proxy["gradient_l1"]
        intensity_loss = proxy["task_intensity_smooth_l1"]

        if corruption_target is None:
            corruption_target = 1.0 - support
        corruption_target = corruption_target.clamp(0.0, 1.0)
        abstention_loss = F.binary_cross_entropy(
            output.abstention.clamp(1e-5, 1.0 - 1e-5), corruption_target
        )

        _, intensity_target = proxy_targets(rgb, thermal, support, task_ids)
        proxy_error = (output.fused_luminance.detach() - intensity_target).abs()
        proxy_error = torch.maximum(proxy_error, corruption_target)
        calibration_loss = F.mse_loss(output.predictive_uncertainty, proxy_error)
        task_loss = F.cross_entropy(output.task_logits, task_ids)

        alpha = output.thermal_contribution.clamp(1e-5, 1.0 - 1e-5)
        entropy = -(alpha * alpha.log() + (1.0 - alpha) * (1.0 - alpha).log()).mean()
        # A small negative entropy term discourages immediate all-zero collapse;
        # the evidence envelope still makes unsafe contribution impossible.
        total = (
            self.config.gradient_weight * gradient_loss
            + self.config.task_intensity_weight * intensity_loss
            + self.config.abstention_weight * abstention_loss
            + self.config.calibration_weight * calibration_loss
            + self.config.task_classification_weight * task_loss
            - self.config.contribution_entropy_weight * entropy
        )
        components = {
            "loss": total.detach(),
            "gradient": gradient_loss.detach(),
            "task_intensity": intensity_loss.detach(),
            "abstention_bce": abstention_loss.detach(),
            "calibration_brier": calibration_loss.detach(),
            "task_cross_entropy": task_loss.detach(),
            "contribution_entropy": entropy.detach(),
            "mean_thermal_contribution": output.thermal_contribution.detach().mean(),
            "mean_abstention": output.abstention.detach().mean(),
        }
        return total, components
