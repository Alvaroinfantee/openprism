"""PRISM-EGT: evidence-gated, task-conditioned learned fusion.

The model is intentionally selective.  It predicts how much registered
thermal evidence to use, but a non-learned evidence envelope bounds that
prediction at every pixel.  Invalid or unsupported measurements therefore
cannot be amplified by learned weights.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


TASK_NAMES = ("navigate", "search", "terrain")


def _group_count(channels: int) -> int:
    groups = max(1, min(8, channels // 4))
    while channels % groups:
        groups -= 1
    return groups


@dataclass(frozen=True, slots=True)
class EGTCFConfig:
    """Serializable architecture definition for PRISM-EGT."""

    base_channels: int = 24
    task_embedding_dim: int = 16
    pose_features: int = 8
    dropout: float = 0.05
    tasks: tuple[str, ...] = TASK_NAMES

    def __post_init__(self) -> None:
        if self.base_channels < 8:
            raise ValueError("base_channels must be at least 8")
        if self.task_embedding_dim < 4:
            raise ValueError("task_embedding_dim must be at least 4")
        if self.pose_features < 0:
            raise ValueError("pose_features cannot be negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("tasks must be non-empty and unique")

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["tasks"] = list(self.tasks)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EGTCFConfig":
        values = dict(payload)
        if "tasks" in values:
            values["tasks"] = tuple(str(item) for item in values["tasks"])
        return cls(**values)


class EGTCFOutput(NamedTuple):
    """Tensor outputs retained for losses, runtime rendering, and audits."""

    fused_luminance: Tensor
    thermal_contribution: Tensor
    visible_reliability: Tensor
    thermal_reliability: Tensor
    abstention: Tensor
    predictive_uncertainty: Tensor
    task_logits: Tensor
    task_probabilities: Tensor
    evidence_support: Tensor


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, value: Tensor) -> Tensor:
        return self.activation(value + self.body(value))


class _Encoder(nn.Module):
    def __init__(self, input_channels: int, channels: int, dropout: float) -> None:
        super().__init__()
        groups = _group_count(channels)
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, channels, 5, padding=2, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            _ResidualBlock(channels, dropout),
            _ResidualBlock(channels, dropout),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class EGTCF(nn.Module):
    """Evidence-gated task-conditioned fusion with calibrated abstention.

    Inputs are co-registered visible RGB, normalized thermal intensity, and
    three non-learned evidence maps: sensor validity, registration support,
    and timing support.  The following invariant holds by construction::

        0 <= thermal_contribution <= validity * registration * timing

    An explicit abstention map can reduce the learned contribution further.
    A task may be supplied by an operator/mission planner, or inferred by the
    model when ``task_ids`` is omitted.  Optional pose context is intended for
    inter-frame motion/timing features, not raw absolute GPS coordinates.
    """

    def __init__(self, config: EGTCFConfig | None = None) -> None:
        super().__init__()
        self.config = config or EGTCFConfig()
        channels = self.config.base_channels
        task_dim = self.config.task_embedding_dim

        self.visible_encoder = _Encoder(3, channels, self.config.dropout)
        self.thermal_encoder = _Encoder(1, channels, self.config.dropout)
        self.task_head = nn.Sequential(
            nn.Linear(channels * 4 + 3, channels * 2),
            nn.SiLU(inplace=True),
            nn.Dropout(self.config.dropout),
            nn.Linear(channels * 2, len(self.config.tasks)),
        )
        self.task_embedding = nn.Embedding(len(self.config.tasks), task_dim)
        self.pose_embedding = (
            nn.Sequential(
                nn.Linear(self.config.pose_features, task_dim),
                nn.SiLU(inplace=True),
                nn.Linear(task_dim, task_dim),
            )
            if self.config.pose_features
            else None
        )
        self.condition = nn.Sequential(
            nn.Linear(task_dim * 2, channels * 2),
            nn.SiLU(inplace=True),
            nn.Linear(channels * 2, channels * 2),
        )
        # Shared and disagreement features let the selector distinguish
        # complementary evidence from cross-modal conflicts.
        selector_channels = channels * 4 + 3
        self.selector = nn.Sequential(
            nn.Conv2d(selector_channels, channels * 2, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels * 2), channels * 2),
            nn.SiLU(inplace=True),
            _ResidualBlock(channels * 2, self.config.dropout),
            nn.Conv2d(channels * 2, 5, 1),
        )

    @staticmethod
    def _validate_inputs(rgb: Tensor, thermal: Tensor, evidence: Tensor) -> None:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape Bx3xHxW")
        if thermal.ndim != 4 or thermal.shape[1] != 1:
            raise ValueError("thermal must have shape Bx1xHxW")
        if evidence.ndim != 4 or evidence.shape[1] != 3:
            raise ValueError("evidence must contain validity, registration, and timing")
        if rgb.shape[0] != thermal.shape[0] or rgb.shape[0] != evidence.shape[0]:
            raise ValueError("batch dimensions must match")
        if rgb.shape[2:] != thermal.shape[2:] or rgb.shape[2:] != evidence.shape[2:]:
            raise ValueError("spatial dimensions must match")
        for name, value in (("rgb", rgb), ("thermal", thermal), ("evidence", evidence)):
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")

    def _task_condition(
        self,
        fused_features: Tensor,
        evidence: Tensor,
        task_ids: Tensor | None,
        pose_context: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        pooled = torch.cat(
            (
                F.adaptive_avg_pool2d(fused_features, 1).flatten(1),
                F.adaptive_avg_pool2d(evidence, 1).flatten(1),
            ),
            dim=1,
        )
        task_logits = self.task_head(pooled)
        task_probabilities = torch.softmax(task_logits, dim=1)
        if task_ids is None:
            task_vector = task_probabilities @ self.task_embedding.weight
        else:
            task_ids = task_ids.to(device=fused_features.device, dtype=torch.long)
            if task_ids.shape != (fused_features.shape[0],):
                raise ValueError("task_ids must have shape B")
            task_vector = self.task_embedding(task_ids)

        if self.pose_embedding is None:
            pose_vector = torch.zeros_like(task_vector)
        elif pose_context is None:
            pose_vector = torch.zeros_like(task_vector)
        else:
            if pose_context.shape != (
                fused_features.shape[0],
                self.config.pose_features,
            ):
                raise ValueError(
                    f"pose_context must have shape Bx{self.config.pose_features}"
                )
            pose_vector = self.pose_embedding(pose_context.to(fused_features.dtype))
        return self.condition(torch.cat((task_vector, pose_vector), dim=1)), task_logits, task_probabilities

    def forward(
        self,
        rgb: Tensor,
        thermal: Tensor,
        evidence: Tensor,
        *,
        task_ids: Tensor | None = None,
        pose_context: Tensor | None = None,
    ) -> EGTCFOutput:
        self._validate_inputs(rgb, thermal, evidence)
        rgb = rgb.clamp(0.0, 1.0)
        thermal = thermal.clamp(0.0, 1.0)
        evidence = evidence.clamp(0.0, 1.0)

        visible_features = self.visible_encoder(rgb)
        thermal_features = self.thermal_encoder(thermal)
        paired = torch.cat(
            (
                visible_features,
                thermal_features,
                torch.abs(visible_features - thermal_features),
                visible_features * thermal_features,
            ),
            dim=1,
        )
        conditioning, task_logits, task_probabilities = self._task_condition(
            paired, evidence, task_ids, pose_context
        )
        scale, bias = conditioning.chunk(2, dim=1)
        paired = paired.view(paired.shape[0], 4, -1, *paired.shape[2:])
        scale = torch.tanh(scale)[:, None, :, None, None]
        bias = torch.tanh(bias)[:, None, :, None, None]
        paired = paired * (1.0 + 0.25 * scale)
        paired = paired + 0.25 * bias
        paired = paired.flatten(1, 2)

        raw = self.selector(torch.cat((paired, evidence), dim=1))
        modality_logits = raw[:, 0:2]
        learned_abstention = torch.sigmoid(raw[:, 2:3])
        learned_uncertainty = torch.sigmoid(raw[:, 3:4])
        thermal_gate = torch.sigmoid(raw[:, 4:5])

        evidence_support = torch.prod(evidence, dim=1, keepdim=True)
        # The explicit lower bound means missing evidence always becomes an
        # abstention, regardless of network parameters.
        abstention = torch.maximum(learned_abstention, 1.0 - evidence_support)
        reliabilities = torch.softmax(modality_logits, dim=1)
        visible_reliability = reliabilities[:, 0:1]
        thermal_reliability = reliabilities[:, 1:2]
        thermal_contribution = (
            thermal_reliability
            * thermal_gate
            * evidence_support
            * (1.0 - abstention)
        )
        visible_y = (
            0.299 * rgb[:, 0:1]
            + 0.587 * rgb[:, 1:2]
            + 0.114 * rgb[:, 2:3]
        )
        fused_luminance = (
            (1.0 - thermal_contribution) * visible_y
            + thermal_contribution * thermal
        ).clamp(0.0, 1.0)
        predictive_uncertainty = torch.maximum(
            learned_uncertainty,
            1.0 - evidence_support * (1.0 - abstention),
        )
        return EGTCFOutput(
            fused_luminance=fused_luminance,
            thermal_contribution=thermal_contribution,
            visible_reliability=visible_reliability,
            thermal_reliability=thermal_reliability,
            abstention=abstention,
            predictive_uncertainty=predictive_uncertainty,
            task_logits=task_logits,
            task_probabilities=task_probabilities,
            evidence_support=evidence_support,
        )

    def invariant_report(self, output: EGTCFOutput, tolerance: float = 1e-6) -> dict[str, float | bool]:
        violation = torch.relu(output.thermal_contribution - output.evidence_support)
        maximum = float(violation.detach().max().cpu())
        return {
            "thermal_contribution_bounded_by_evidence": maximum <= tolerance,
            "maximum_support_violation": maximum,
            "mean_abstention": float(output.abstention.detach().mean().cpu()),
            "mean_predictive_uncertainty": float(
                output.predictive_uncertainty.detach().mean().cpu()
            ),
        }
