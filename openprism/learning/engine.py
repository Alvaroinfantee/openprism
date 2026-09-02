"""Runtime adapter from PRISM-EGT tensors to the OpenPRISM evidence contract."""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import torch

from ..contracts import FusionOutput, PrismFrame
from ..fusion import (
    EvidenceFusionEngine,
    FusionConfig,
    _replace_luminance,
    _semantic_view,
)
from .checkpoint import CheckpointMetadata, load_checkpoint
from .model import EGTCF, EGTCFOutput, TASK_NAMES


POSE_FEATURES = (
    "delta_east_m",
    "delta_north_m",
    "delta_up_m",
    "delta_roll_rad",
    "delta_pitch_rad",
    "delta_yaw_rad",
    "delta_time_s",
    "timing_uncertainty_s",
)
_POSE_SCALES = np.asarray((50.0, 50.0, 20.0, np.pi, np.pi, np.pi, 2.0, 0.05))


def normalize_pose_context(
    value: Mapping[str, float] | Sequence[float] | None,
) -> np.ndarray | None:
    """Normalize relative Pixhawk/camera motion features for the network."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        array = np.asarray([float(value.get(name, 0.0)) for name in POSE_FEATURES])
    else:
        array = np.asarray(value, dtype=np.float64)
    if array.shape != (len(POSE_FEATURES),) or not np.all(np.isfinite(array)):
        raise ValueError(f"pose context must contain {len(POSE_FEATURES)} finite values")
    return np.clip(array / _POSE_SCALES, -4.0, 4.0).astype(np.float32)


def _channel(output: FusionOutput, name: str) -> np.ndarray:
    return output.machine_tensor[output.channel_names.index(name)]


def _bypass_output(probe: FusionOutput, reason: str) -> FusionOutput:
    provenance = dict(probe.provenance)
    provenance.update(
        {
            "learned_fusion_requested": True,
            "learned_fusion_applied": False,
            "learned_fusion_bypass_reason": reason,
            "safety_contract": "deterministic_evidence_gate_authoritative",
        }
    )
    return replace(probe, provenance=provenance)


class LearnedFusionEngine:
    """Apply a trained PRISM-EGT model inside deterministic evidence gates."""

    def __init__(
        self,
        model: EGTCF,
        *,
        checkpoint: CheckpointMetadata | None = None,
        base_engine: EvidenceFusionEngine | None = None,
        device: str | torch.device | None = None,
        allow_unvalidated: bool = False,
        tile_size: int = 512,
        tile_overlap: int = 48,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()
        self.checkpoint = checkpoint
        self.base_engine = base_engine or EvidenceFusionEngine()
        if tile_size < 64 or not 0 <= tile_overlap < tile_size:
            raise ValueError("tile_size must be >=64 and overlap must be smaller")
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        if checkpoint is None and not allow_unvalidated:
            raise ValueError(
                "an explicit checkpoint is required; set allow_unvalidated=True only for tests"
            )

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        base_engine: EvidenceFusionEngine | None = None,
        device: str | torch.device | None = None,
    ) -> "LearnedFusionEngine":
        target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, metadata = load_checkpoint(path, device=target)
        return cls(
            model,
            checkpoint=metadata,
            base_engine=base_engine,
            device=target,
        )

    @staticmethod
    def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
        if length <= tile_size:
            return [0]
        step = tile_size - overlap
        starts = list(range(0, max(1, length - tile_size + 1), step))
        final = length - tile_size
        if starts[-1] != final:
            starts.append(final)
        return starts

    def _infer(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        evidence: torch.Tensor,
        task_ids: torch.Tensor | None,
        pose_context: torch.Tensor | None,
    ) -> tuple[EGTCFOutput, bool]:
        height, width = rgb.shape[2:]
        if height <= self.tile_size and width <= self.tile_size:
            return (
                self.model(
                    rgb,
                    thermal,
                    evidence,
                    task_ids=task_ids,
                    pose_context=pose_context,
                ),
                False,
            )

        context = None
        fixed_tasks = task_ids
        if fixed_tasks is None:
            scale = min(1.0, 256.0 / max(height, width))
            context_size = (
                max(32, int(round(height * scale))),
                max(32, int(round(width * scale))),
            )
            context = self.model(
                torch.nn.functional.interpolate(
                    rgb, context_size, mode="bilinear", align_corners=False
                ),
                torch.nn.functional.interpolate(
                    thermal, context_size, mode="bilinear", align_corners=False
                ),
                torch.nn.functional.interpolate(
                    evidence, context_size, mode="bilinear", align_corners=False
                ),
                task_ids=None,
                pose_context=pose_context,
            )
            fixed_tasks = context.task_probabilities.argmax(dim=1)

        fields = (
            "fused_luminance",
            "thermal_contribution",
            "visible_reliability",
            "thermal_reliability",
            "abstention",
            "predictive_uncertainty",
        )
        accumulators = {
            name: torch.zeros(
                (rgb.shape[0], 1, height, width), device=rgb.device, dtype=rgb.dtype
            )
            for name in fields
        }
        weights = torch.zeros_like(accumulators["fused_luminance"])
        first_output = None
        for top in self._tile_starts(height, self.tile_size, self.tile_overlap):
            bottom = min(height, top + self.tile_size)
            for left in self._tile_starts(width, self.tile_size, self.tile_overlap):
                right = min(width, left + self.tile_size)
                output = self.model(
                    rgb[..., top:bottom, left:right],
                    thermal[..., top:bottom, left:right],
                    evidence[..., top:bottom, left:right],
                    task_ids=fixed_tasks,
                    pose_context=pose_context,
                )
                if first_output is None:
                    first_output = output
                tile_height, tile_width = bottom - top, right - left
                vertical = torch.hann_window(
                    tile_height, periodic=False, device=rgb.device, dtype=rgb.dtype
                ).clamp_min(1e-3)
                horizontal = torch.hann_window(
                    tile_width, periodic=False, device=rgb.device, dtype=rgb.dtype
                ).clamp_min(1e-3)
                weight = (vertical[:, None] * horizontal[None, :])[None, None]
                weights[..., top:bottom, left:right] += weight
                for name in fields:
                    accumulators[name][..., top:bottom, left:right] += (
                        getattr(output, name) * weight
                    )
        if first_output is None:  # pragma: no cover - guarded by positive geometry
            raise RuntimeError("tiled inference produced no tiles")
        normalized = {name: value / weights.clamp_min(1e-8) for name, value in accumulators.items()}
        evidence_support = torch.prod(evidence, dim=1, keepdim=True)
        task_logits = context.task_logits if context is not None else first_output.task_logits
        task_probabilities = (
            context.task_probabilities
            if context is not None
            else first_output.task_probabilities
        )
        return (
            EGTCFOutput(
                fused_luminance=normalized["fused_luminance"],
                thermal_contribution=torch.minimum(
                    normalized["thermal_contribution"], evidence_support
                ),
                visible_reliability=normalized["visible_reliability"],
                thermal_reliability=normalized["thermal_reliability"],
                abstention=torch.maximum(
                    normalized["abstention"], 1.0 - evidence_support
                ),
                predictive_uncertainty=torch.maximum(
                    normalized["predictive_uncertainty"], 1.0 - evidence_support
                ),
                task_logits=task_logits,
                task_probabilities=task_probabilities,
                evidence_support=evidence_support,
            ),
            True,
        )

    def fuse(
        self,
        frame: PrismFrame,
        *,
        task: str = "automatic",
        pose_context: Mapping[str, float] | Sequence[float] | None = None,
    ) -> FusionOutput:
        if task != "automatic" and task not in TASK_NAMES:
            raise ValueError(f"task must be automatic or one of {TASK_NAMES}")
        probe = self.base_engine.fuse(frame, FusionConfig(thermal_gain=1.0))
        if not probe.pixel_fusion_applied:
            return _bypass_output(probe, "upstream evidence gate rejected pixel fusion")

        available = set(probe.channel_names)
        required = {
            "visible_r_srgb",
            "visible_g_srgb",
            "visible_b_srgb",
            "thermal_radiometric_norm",
            "sensor_validity",
            "registration_support_score",
        }
        if missing := required - available:
            return _bypass_output(probe, f"missing machine evidence channels: {sorted(missing)}")

        rgb = np.stack(
            tuple(_channel(probe, name) for name in (
                "visible_r_srgb", "visible_g_srgb", "visible_b_srgb"
            )),
            axis=0,
        ).astype(np.float32)
        thermal = _channel(probe, "thermal_radiometric_norm")[None].astype(np.float32)
        timing_support = np.ones_like(thermal, dtype=np.float32)
        evidence = np.stack(
            (
                _channel(probe, "sensor_validity"),
                _channel(probe, "registration_support_score"),
                timing_support[0],
            ),
            axis=0,
        ).astype(np.float32)
        task_ids = None
        if task != "automatic":
            task_ids = torch.tensor([TASK_NAMES.index(task)], device=self.device)
        normalized_pose = normalize_pose_context(pose_context)
        pose_tensor = (
            None
            if normalized_pose is None
            else torch.from_numpy(normalized_pose[None]).to(self.device)
        )

        with torch.inference_mode():
            learned, tiled_inference = self._infer(
                torch.from_numpy(rgb[None]).to(self.device),
                torch.from_numpy(thermal[None]).to(self.device),
                torch.from_numpy(evidence[None]).to(self.device),
                task_ids,
                pose_tensor,
            )
        arrays = {
            name: tensor[0, 0].detach().float().cpu().numpy().astype(np.float32)
            for name, tensor in (
                ("fused_luminance", learned.fused_luminance),
                ("thermal_contribution", learned.thermal_contribution),
                ("visible_reliability", learned.visible_reliability),
                ("thermal_reliability", learned.thermal_reliability),
                ("abstention", learned.abstention),
                ("predictive_uncertainty", learned.predictive_uncertainty),
                ("evidence_support", learned.evidence_support),
            )
        }
        operator_float = _replace_luminance(
            np.moveaxis(rgb, 0, -1), arrays["fused_luminance"], preserve=0.88
        )
        operator_rgb = np.rint(operator_float * 255.0).astype(np.uint8)
        semantic_view = _semantic_view(operator_rgb, frame.semantic_mask, 0.42)

        channel_names = list(probe.channel_names)
        machine = np.array(probe.machine_tensor, copy=True)
        machine[channel_names.index("thermal_contribution")] = arrays[
            "thermal_contribution"
        ]
        learned_support = np.clip(
            arrays["evidence_support"] * (1.0 - arrays["predictive_uncertainty"]),
            0.0,
            1.0,
        )
        machine[channel_names.index("fusion_support_score")] = learned_support
        additions = (
            ("learned_visible_reliability", arrays["visible_reliability"]),
            ("learned_thermal_reliability", arrays["thermal_reliability"]),
            ("learned_abstention_probability", arrays["abstention"]),
            ("learned_predictive_uncertainty", arrays["predictive_uncertainty"]),
            ("learned_fused_luminance", arrays["fused_luminance"]),
        )
        channel_names.extend(name for name, _ in additions)
        machine = np.concatenate(
            (machine, np.stack(tuple(value for _, value in additions), axis=0)), axis=0
        ).astype(np.float32)

        task_probabilities = learned.task_probabilities[0].detach().float().cpu().numpy()
        selected_task = task if task != "automatic" else TASK_NAMES[int(task_probabilities.argmax())]
        invariant = self.model.invariant_report(learned)
        checkpoint = self.checkpoint
        provenance = dict(probe.provenance)
        provenance.update(
            {
                "algorithm": "prism_egt_evidence_gated_task_conditioned_fusion",
                "fusion_mode": "learned_selective_pixel_fusion",
                "learned_fusion_requested": True,
                "learned_fusion_applied": True,
                "non_hallucinatory": True,
                "operator_task": task,
                "selected_task": selected_task,
                "task_probabilities": {
                    name: float(value)
                    for name, value in zip(TASK_NAMES, task_probabilities)
                },
                "pose_context_features": POSE_FEATURES if normalized_pose is not None else (),
                "tiled_inference": tiled_inference,
                "tile_size": self.tile_size,
                "tile_overlap": self.tile_overlap,
                "model_id": checkpoint.model_id if checkpoint else "unvalidated_in_memory",
                "model_artifact_sha256": (
                    checkpoint.artifact_sha256 if checkpoint else None
                ),
                "training_provenance": (
                    checkpoint.training_provenance if checkpoint else "untrained_test_only"
                ),
                "validation_scope": (
                    checkpoint.validation_scope if checkpoint else "none"
                ),
                "selective_fusion_invariant": invariant,
                "safety_contract": {
                    "model_may_override_hard_evidence_gates": False,
                    "thermal_contribution_is_evidence_bounded": True,
                    "automatic_task_is_operator_overridable": True,
                    "source_modalities_preserved": True,
                },
                "machine_channels": tuple(channel_names),
            }
        )
        return FusionOutput(
            frame_id=probe.frame_id,
            operator_rgb=operator_rgb,
            machine_tensor=machine,
            channel_names=tuple(channel_names),
            fusion_support=learned_support,
            registration_support=probe.registration_support,
            thermal_view=probe.thermal_view,
            visible_view=probe.visible_view,
            semantic_view=semantic_view,
            provenance=MappingProxyType(provenance),
            pixel_fusion_applied=bool(np.any(arrays["thermal_contribution"] > 1e-6)),
            synchronization_state=probe.synchronization_state,
            physical_timing_uncertainty_ns=probe.physical_timing_uncertainty_ns,
        )
