"""Explainable automatic fusion control and compact AI scene digests.

The controller is deliberately constrained: a small, versioned policy model
may recommend an operator preset and thermal gain, but hard evidence gates in
the fusion engine remain authoritative.  The bundled coefficients are expert
initialized, not represented as trained or benchmark-optimal weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .contracts import FusionOutput, PrismFrame
from .fusion import EvidenceFusionEngine, FusionConfig


POLICY_SCHEMA_VERSION = "openprism.fusion-policy/1.0"
DIGEST_SCHEMA_VERSION = "openprism.ai-scene-digest/1.0"
_ALLOWED_PRESETS = ("navigate", "search", "terrain", "integrity")


def _finite_vector(name: str, value: Any, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite values")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


def _finite_matrix(name: str, value: Any, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must have shape {shape} and finite values")
    array = np.array(array, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FusionPolicyModel:
    """Validated, immutable linear policy artifact."""

    model_id: str
    model_kind: str
    training_provenance: str
    validation_scope: str
    feature_order: tuple[str, ...]
    preset_order: tuple[str, ...]
    preset_bias: np.ndarray
    preset_weights: np.ndarray
    thermal_gain_bias: float
    thermal_gain_weights: np.ndarray
    thermal_gain_limits: tuple[float, float]
    minimum_registration_support: float
    minimum_valid_fraction: float
    artifact_sha256: str

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any], raw: bytes) -> "FusionPolicyModel":
        if payload.get("schema_version") != POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported fusion policy schema")
        features = tuple(str(item).strip() for item in payload["feature_order"])
        presets = tuple(str(item).strip() for item in payload["preset_order"])
        if not features or len(set(features)) != len(features):
            raise ValueError("fusion policy feature names must be unique")
        if set(presets) != set(_ALLOWED_PRESETS) or len(presets) != len(_ALLOWED_PRESETS):
            raise ValueError("fusion policy must define every supported preset exactly once")
        limits = tuple(float(item) for item in payload["thermal_gain_limits"])
        if len(limits) != 2 or not 0.0 <= limits[0] < limits[1] <= 2.5:
            raise ValueError("thermal gain limits must be ordered within [0, 2.5]")
        minimum_registration = float(payload["minimum_registration_support"])
        minimum_valid = float(payload["minimum_valid_fraction"])
        if not 0.0 <= minimum_registration <= 1.0:
            raise ValueError("minimum registration support must be within [0, 1]")
        if not 0.0 <= minimum_valid <= 1.0:
            raise ValueError("minimum valid fraction must be within [0, 1]")
        thermal_bias = float(payload["thermal_gain_bias"])
        if not math.isfinite(thermal_bias):
            raise ValueError("thermal gain bias must be finite")
        return cls(
            model_id=str(payload["model_id"]).strip(),
            model_kind=str(payload["model_kind"]).strip(),
            training_provenance=str(payload["training_provenance"]).strip(),
            validation_scope=str(payload["validation_scope"]).strip(),
            feature_order=features,
            preset_order=presets,
            preset_bias=_finite_vector("preset_bias", payload["preset_bias"], len(presets)),
            preset_weights=_finite_matrix(
                "preset_weights", payload["preset_weights"], (len(presets), len(features))
            ),
            thermal_gain_bias=thermal_bias,
            thermal_gain_weights=_finite_vector(
                "thermal_gain_weights", payload["thermal_gain_weights"], len(features)
            ),
            thermal_gain_limits=(limits[0], limits[1]),
            minimum_registration_support=minimum_registration,
            minimum_valid_fraction=minimum_valid,
            artifact_sha256=hashlib.sha256(raw).hexdigest(),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FusionPolicyModel":
        source = Path(path)
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        return cls._from_payload(payload, raw)

    @classmethod
    def bundled(cls) -> "FusionPolicyModel":
        resource = files("openprism").joinpath("policy", "fusion_policy_v1.json")
        with resource.open("rb") as stream:
            raw = stream.read()
        # Parsing bytes works for normal and zipped resources.
        payload = json.loads(raw.decode("utf-8"))
        return cls._from_payload(payload, raw)


@dataclass(frozen=True, slots=True)
class FusionControlRecommendation:
    operator_preset: str
    thermal_gain: float
    confidence: float
    status: str
    reasons: tuple[str, ...]
    scores: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.operator_preset not in _ALLOWED_PRESETS:
            raise ValueError("unsupported operator preset")
        if not 0.0 <= self.thermal_gain <= 2.5:
            raise ValueError("thermal_gain must be within [0, 2.5]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if self.status not in {"recommended", "safety_override"}:
            raise ValueError("unsupported recommendation status")
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator_preset": self.operator_preset,
            "thermal_gain": self.thermal_gain,
            "thermal_strength_percent": round(self.thermal_gain / 2.5 * 100.0, 1),
            "confidence": self.confidence,
            "status": self.status,
            "reasons": list(self.reasons),
            "preset_scores": dict(self.scores),
        }


def _channel(output: FusionOutput, name: str) -> np.ndarray:
    return output.machine_tensor[output.channel_names.index(name)]


def _mean_channel(output: FusionOutput, name: str) -> float:
    return float(np.mean(_channel(output, name), dtype=np.float64))


class AdaptiveFusionController:
    """Recommend fusion controls, with deterministic safety overrides."""

    def __init__(self, model: FusionPolicyModel | None = None) -> None:
        self.model = model or FusionPolicyModel.bundled()

    def _features(self, frame: PrismFrame, probe: FusionOutput) -> dict[str, float]:
        rgb = np.moveaxis(probe.machine_tensor[:3], 0, -1)
        luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        available = set(probe.channel_names)
        feature_values = {
            "visible_luminance_mean": float(np.mean(luminance, dtype=np.float64)),
            "visible_detail_mean": _mean_channel(probe, "visible_detail"),
            "thermal_detail_mean": _mean_channel(probe, "thermal_detail"),
            "thermal_saliency_mean": _mean_channel(probe, "thermal_saliency"),
            "registration_support_mean": float(
                np.mean(probe.registration_support, dtype=np.float64)
            ),
            "fusion_support_mean": float(np.mean(probe.fusion_support, dtype=np.float64)),
            "valid_fraction": (
                _mean_channel(probe, "sensor_validity")
                if "sensor_validity" in available
                else 0.0
            ),
            "night_factor": float(probe.provenance.get("night_factor", 0.0)),
            "has_detections": 1.0 if frame.detections else 0.0,
            "has_semantics": 1.0 if frame.semantic_mask is not None else 0.0,
        }
        missing = set(self.model.feature_order) - feature_values.keys()
        if missing:
            raise ValueError(f"policy requests unsupported features: {sorted(missing)}")
        return {name: feature_values[name] for name in self.model.feature_order}

    def recommend_from_probe(
        self, frame: PrismFrame, probe: FusionOutput
    ) -> tuple[FusionControlRecommendation, Mapping[str, float]]:
        features = self._features(frame, probe)
        vector = np.asarray([features[name] for name in self.model.feature_order])
        registration = features["registration_support_mean"]
        valid_fraction = features["valid_fraction"]
        if (
            not probe.pixel_fusion_applied
            or registration < self.model.minimum_registration_support
            or valid_fraction < self.model.minimum_valid_fraction
        ):
            reasons = ["hard evidence gate selected the integrity view"]
            if not probe.pixel_fusion_applied:
                reasons.append(str(probe.provenance.get("fallback_reason", "pixel fusion unavailable")))
            if registration < self.model.minimum_registration_support:
                reasons.append("registration support is below policy minimum")
            if valid_fraction < self.model.minimum_valid_fraction:
                reasons.append("valid multisensor coverage is below policy minimum")
            return (
                FusionControlRecommendation(
                    operator_preset="integrity",
                    thermal_gain=0.0,
                    confidence=1.0,
                    status="safety_override",
                    reasons=tuple(reasons),
                    scores={preset: 0.0 for preset in self.model.preset_order},
                ),
                MappingProxyType(features),
            )

        logits = self.model.preset_bias + self.model.preset_weights @ vector
        shifted = logits - float(np.max(logits))
        probabilities = np.exp(shifted) / float(np.sum(np.exp(shifted)))
        selected_index = int(np.argmax(probabilities))
        selected = self.model.preset_order[selected_index]
        raw_gain = self.model.thermal_gain_bias + float(
            self.model.thermal_gain_weights @ vector
        )
        # Registration/validity form a safety envelope around the model's
        # continuous output. The policy cannot amplify unsupported evidence.
        support_envelope = min(
            1.0,
            registration / max(self.model.minimum_registration_support, 1e-9),
            valid_fraction / max(self.model.minimum_valid_fraction, 1e-9),
        )
        low, high = self.model.thermal_gain_limits
        gain = float(np.clip(raw_gain, low, high * support_envelope))
        reasons = (
            f"{selected} has the highest constrained policy score",
            f"registration support={registration:.3f}",
            f"valid multisensor fraction={valid_fraction:.3f}",
            f"thermal saliency={features['thermal_saliency_mean']:.3f}",
        )
        return (
            FusionControlRecommendation(
                operator_preset=selected,
                thermal_gain=round(gain, 4),
                confidence=float(probabilities[selected_index]),
                status="recommended",
                reasons=reasons,
                scores={
                    preset: float(probability)
                    for preset, probability in zip(self.model.preset_order, probabilities)
                },
            ),
            MappingProxyType(features),
        )

    def recommend(
        self, frame: PrismFrame, engine: EvidenceFusionEngine
    ) -> tuple[FusionControlRecommendation, Mapping[str, float], FusionOutput]:
        probe = engine.fuse(frame, FusionConfig(thermal_gain=1.0))
        recommendation, features = self.recommend_from_probe(frame, probe)
        return recommendation, features, probe

    def scene_digest(
        self,
        frame: PrismFrame,
        output: FusionOutput,
        recommendation: FusionControlRecommendation,
        features: Mapping[str, float],
        *,
        automatic_control: bool,
        applied_thermal_gain: float,
    ) -> dict[str, Any]:
        statistics: dict[str, dict[str, float]] = {}
        for name, channel in zip(output.channel_names, output.machine_tensor):
            percentiles = np.percentile(channel, (5.0, 50.0, 95.0))
            statistics[name] = {
                "mean": float(np.mean(channel, dtype=np.float64)),
                "p05": float(percentiles[0]),
                "p50": float(percentiles[1]),
                "p95": float(percentiles[2]),
            }
        return {
            "schema_version": DIGEST_SCHEMA_VERSION,
            "frame_id": frame.frame_id,
            "summary": (
                f"{recommendation.status}: {recommendation.operator_preset} view, "
                f"thermal gain {applied_thermal_gain:.2f}; "
                f"fusion support {features['fusion_support_mean']:.3f}."
            ),
            "control": {
                "mode": "automatic" if automatic_control else "manual",
                "applied_thermal_gain": applied_thermal_gain,
                "recommendation": recommendation.as_dict(),
            },
            "evidence_features": dict(features),
            "machine_projection": {
                "shape_chw": list(output.machine_tensor.shape),
                "dtype": str(output.machine_tensor.dtype),
                "channel_names": list(output.channel_names),
                "channel_statistics": statistics,
            },
            "scene_evidence": {
                "detection_count": len(frame.detections),
                "semantic_mask_available": frame.semantic_mask is not None,
                "pixel_fusion_applied": output.pixel_fusion_applied,
                "synchronization_state": output.synchronization_state,
                "physical_timing_uncertainty_ns": output.physical_timing_uncertainty_ns,
            },
            "policy_model": {
                "model_id": self.model.model_id,
                "model_kind": self.model.model_kind,
                "artifact_sha256": self.model.artifact_sha256,
                "training_provenance": self.model.training_provenance,
                "validation_scope": self.model.validation_scope,
            },
            "safety_contract": {
                "model_may_override_hard_evidence_gates": False,
                "operator_can_disable_automatic_control": True,
                "generated_pixels": False,
                "decision_is_accuracy_certificate": False,
            },
        }


__all__ = [
    "AdaptiveFusionController",
    "DIGEST_SCHEMA_VERSION",
    "FusionControlRecommendation",
    "FusionPolicyModel",
    "POLICY_SCHEMA_VERSION",
]
