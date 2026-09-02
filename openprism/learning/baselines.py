"""Dependency-light, executable luminance baselines for PRISM-EGT."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


BASELINE_NAMES = (
    "rgb_only",
    "thermal_only",
    "average",
    "maximum",
    "deterministic_openprism",
)


def _visible_luminance(rgb: Tensor) -> Tensor:
    return 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]


def _gradient_magnitude(value: Tensor) -> Tensor:
    horizontal = F.pad(value[..., :, 1:] - value[..., :, :-1], (0, 1, 0, 0))
    vertical = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(horizontal.square() + vertical.square() + 1e-8)


def fuse_baseline(
    name: str,
    rgb: Tensor,
    thermal: Tensor,
    evidence: Tensor,
) -> Tensor:
    """Return one evidence-bounded baseline luminance in ``[0, 1]``."""

    if name not in BASELINE_NAMES:
        raise ValueError(f"unknown baseline: {name}")
    visible = _visible_luminance(rgb)
    support = torch.prod(evidence.clamp(0.0, 1.0), dim=1, keepdim=True)
    if name == "rgb_only":
        return visible
    if name == "thermal_only":
        # Unsupported thermal pixels fail back to visible rather than becoming
        # artificial black image content.
        return visible * (1.0 - support) + thermal * support
    if name == "average":
        alpha = 0.5 * support
        return visible * (1.0 - alpha) + thermal * alpha
    if name == "maximum":
        selected = torch.maximum(visible, thermal)
        return visible * (1.0 - support) + selected * support

    visible_detail = _gradient_magnitude(visible)
    thermal_detail = _gradient_magnitude(thermal)
    visible_scale = visible_detail.flatten(1).quantile(0.99, dim=1).clamp_min(1e-5)
    thermal_scale = thermal_detail.flatten(1).quantile(0.99, dim=1).clamp_min(1e-5)
    visible_detail = (visible_detail / visible_scale[:, None, None, None]).clamp(0.0, 1.0)
    thermal_detail = (thermal_detail / thermal_scale[:, None, None, None]).clamp(0.0, 1.0)
    local = F.avg_pool2d(thermal, 7, stride=1, padding=3)
    thermal_contrast = (thermal - local).abs()
    contrast_scale = thermal_contrast.flatten(1).quantile(0.99, dim=1).clamp_min(1e-5)
    thermal_contrast = (
        thermal_contrast / contrast_scale[:, None, None, None]
    ).clamp(0.0, 1.0)
    saliency = (0.55 * thermal_detail + 0.45 * thermal_contrast).clamp(0.0, 1.0)
    night = ((0.48 - visible.mean(dim=(2, 3), keepdim=True)) / 0.38).clamp(0.0, 1.0)
    alpha = (0.08 + 0.22 * night + 0.58 * saliency).clamp(0.0, 0.86) * support
    return ((1.0 - alpha) * visible + alpha * thermal).clamp(0.0, 1.0)
