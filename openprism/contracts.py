"""Core OpenPRISM data contracts.

The contracts deliberately separate measured evidence from derived views. A
``PrismFrame`` owns immutable sensor observations. A ``FusionOutput`` contains
derived machine and operator projections and can always point back to those
observations through its provenance graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _readonly(array: np.ndarray) -> np.ndarray:
    # Own the storage. Merely clearing the write flag on ``np.asarray(array)``
    # still lets a caller mutate the observation through its original handle.
    value = np.array(array, copy=True)
    value.setflags(write=False)
    return value


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_deep_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly(value)
    return value


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return _deep_freeze(dict(value or {}))


_SYNCHRONIZATION_STATES = frozenset(
    {
        "unknown",
        "exact",
        "bounded_skew",
        "interpolated",
        "declared_replay_aligned",
        "late",
        "future",
        "missing",
        "stale",
        "incompatible_clock_domain",
        "unsynchronized",
    }
)
_ELIGIBLE_SYNCHRONIZATION_STATES = frozenset(
    {"exact", "bounded_skew", "interpolated", "declared_replay_aligned"}
)


@dataclass(frozen=True, slots=True)
class SynchronizationStatus:
    """Explicit temporal eligibility carried by a :class:`PrismFrame`.

    Timing uncertainty is a physical quantity and deliberately remains
    separate from registration or fusion support scores. ``basis`` distinguishes
    a live measured result from publisher-declared replay alignment; an archive
    timestamp such as zero is never, by itself, evidence of synchronization.
    """

    state: str = "unknown"
    pixel_fusion_eligible: bool = False
    basis: str = "unknown"
    clock_domain: str | None = None
    measured_max_skew_ns: int | None = None
    effective_max_skew_ns: int | None = None
    physical_timing_uncertainty_ns: int | None = None
    sensor_states: Mapping[str, str] = field(default_factory=dict)
    sensor_skew_ns: Mapping[str, int | None] = field(default_factory=dict)
    sensor_effective_skew_ns: Mapping[str, int | None] = field(default_factory=dict)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        state = str(self.state)
        if state not in _SYNCHRONIZATION_STATES:
            raise ValueError(f"unsupported synchronization state: {state}")
        if self.basis not in {"unknown", "measured", "declared"}:
            raise ValueError("synchronization basis must be unknown, measured, or declared")
        if self.pixel_fusion_eligible and state not in _ELIGIBLE_SYNCHRONIZATION_STATES:
            raise ValueError(
                "pixel fusion eligibility requires an exact, bounded, interpolated, "
                "or declared replay-aligned state"
            )
        if self.pixel_fusion_eligible and self.basis == "unknown":
            raise ValueError("pixel fusion eligibility requires a measured or declared basis")
        if state == "declared_replay_aligned" and self.basis != "declared":
            raise ValueError("declared replay alignment requires basis='declared'")
        if (
            state == "interpolated"
            and self.pixel_fusion_eligible
            and not bool(self.details.get("interpolation_performed", False))
        ):
            raise ValueError(
                "an eligible interpolated state requires interpolation_performed=True"
            )

        for name in (
            "measured_max_skew_ns",
            "effective_max_skew_ns",
            "physical_timing_uncertainty_ns",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative when supplied")
        if (
            self.measured_max_skew_ns is not None
            and self.effective_max_skew_ns is not None
            and self.effective_max_skew_ns < self.measured_max_skew_ns
        ):
            raise ValueError("effective skew cannot be smaller than measured skew")

        sensor_states = {str(key): str(value) for key, value in self.sensor_states.items()}
        invalid_states = set(sensor_states.values()) - _SYNCHRONIZATION_STATES
        if invalid_states:
            raise ValueError(f"unsupported per-sensor synchronization states: {invalid_states}")

        sensor_skew: dict[str, int | None] = {}
        for key, value in self.sensor_skew_ns.items():
            sensor_skew[str(key)] = None if value is None else int(value)
        sensor_effective_skew: dict[str, int | None] = {}
        for key, value in self.sensor_effective_skew_ns.items():
            normalized = None if value is None else int(value)
            if normalized is not None and normalized < 0:
                raise ValueError("effective per-sensor skew must be non-negative")
            sensor_effective_skew[str(key)] = normalized

        object.__setattr__(self, "state", state)
        object.__setattr__(self, "sensor_states", MappingProxyType(sensor_states))
        object.__setattr__(self, "sensor_skew_ns", MappingProxyType(sensor_skew))
        object.__setattr__(
            self,
            "sensor_effective_skew_ns",
            MappingProxyType(sensor_effective_skew),
        )
        object.__setattr__(self, "details", _frozen_mapping(self.details))

    @classmethod
    def declared_replay_aligned(
        cls,
        sensor_ids: tuple[str, ...],
        *,
        clock_domain: str | None,
        declaration: str,
    ) -> "SynchronizationStatus":
        """Create an eligible replay status without claiming measured timing."""

        return cls(
            state="declared_replay_aligned",
            pixel_fusion_eligible=True,
            basis="declared",
            clock_domain=clock_domain,
            physical_timing_uncertainty_ns=None,
            sensor_states={sensor_id: "declared_replay_aligned" for sensor_id in sensor_ids},
            details={"declaration": declaration, "measured_timing": False},
        )


@dataclass(frozen=True, slots=True)
class Timestamp:
    """Hardware-oriented timestamp with explicit uncertainty.

    ``tai_ns`` is used instead of a floating-point Unix timestamp so a live
    adapter can preserve sub-millisecond timing and avoid leap-second ambiguity.
    It is ``None`` when the source does not provide capture time; unknown time is
    never encoded as a plausible numeric instant.
    """

    tai_ns: int | None
    clock_id: str = "dataset"
    uncertainty_ns: int | None = 0
    clock_domain: str | None = None

    def __post_init__(self) -> None:
        if self.tai_ns is not None and self.tai_ns < 0:
            raise ValueError("timestamp must be non-negative when supplied")
        if self.uncertainty_ns is not None and self.uncertainty_ns < 0:
            raise ValueError("timestamp uncertainty must be non-negative when supplied")
        if self.tai_ns is None and self.uncertainty_ns is not None:
            raise ValueError("unknown capture time must have unknown uncertainty")
        if not self.clock_id:
            raise ValueError("clock_id is required")
        if self.clock_domain is not None and not self.clock_domain:
            raise ValueError("clock_domain must be non-empty when supplied")

    @property
    def effective_clock_domain(self) -> str:
        """Return the comparison domain, falling back to legacy ``clock_id``."""

        return self.clock_domain or self.clock_id


@dataclass(frozen=True, slots=True)
class SensorObservation:
    """An immutable measurement from one physical or recorded sensor."""

    sensor_id: str
    modality: str
    frame_id: str
    timestamp: Timestamp
    data: np.ndarray
    encoding: str
    units: str = "unitless"
    validity: np.ndarray | None = None
    uncertainty: np.ndarray | float | None = None
    source_path: Path | None = None
    calibration_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sensor_id or not self.modality or not self.frame_id:
            raise ValueError("sensor_id, modality, and frame_id are required")
        data = _readonly(np.asarray(self.data))
        if not 1 <= data.ndim <= 4:
            raise ValueError("sensor data must have between one and four dimensions")
        object.__setattr__(self, "data", data)

        if self.validity is not None:
            validity = np.asarray(self.validity, dtype=bool)
            if data.ndim == 1:
                if validity.shape != data.shape:
                    raise ValueError("one-dimensional validity must match sample geometry")
            else:
                height, width = data.shape[:2]
                if validity.shape == (height,):
                    validity = np.broadcast_to(validity[:, None], (height, width)).copy()
                elif validity.shape[:2] == (height, width):
                    if validity.ndim > 2:
                        validity = np.all(
                            validity,
                            axis=tuple(range(2, validity.ndim)),
                        )
                else:
                    raise ValueError(
                        "image validity must match rows or begin with HxW geometry"
                    )
                if validity.shape != (height, width):
                    raise ValueError("image validity could not be normalized to HxW")
            object.__setattr__(self, "validity", _readonly(validity))

        if self.uncertainty is not None:
            uncertainty = np.asarray(self.uncertainty, dtype=np.float32)
            if not np.all(np.isfinite(uncertainty)) or np.any(uncertainty < 0.0):
                raise ValueError("measurement uncertainty must be finite and non-negative")
            if uncertainty.ndim == 0:
                object.__setattr__(self, "uncertainty", float(uncertainty))
            else:
                allowed_shapes = {data.shape, data.shape[:1]}
                if data.ndim >= 2:
                    allowed_shapes.add(data.shape[:2])
                if uncertainty.shape not in allowed_shapes:
                    raise ValueError(
                        "uncertainty must match the sample, row, or image geometry"
                    )
                object.__setattr__(self, "uncertainty", _readonly(uncertainty))

        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def height(self) -> int:
        if self.data.ndim < 2:
            raise AttributeError("one-dimensional observations have no image height")
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        if self.data.ndim < 2:
            raise AttributeError("one-dimensional observations have no image width")
        return int(self.data.shape[1])


@dataclass(frozen=True, slots=True)
class Detection:
    """A normalized evidence box, from ground truth or a model plugin."""

    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float
    source: str
    track_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be in [0, 1]")
        for name in ("x", "y", "width", "height"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"normalized {name} must be in [0, 1]")
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("detection box extends beyond normalized image bounds")

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "w": self.width,
            "h": self.height,
            "source": self.source,
            "track_id": self.track_id,
        }


@dataclass(frozen=True, slots=True)
class PrismFrame:
    """Immutable multisensor evidence at one synchronization watermark."""

    frame_id: str
    timestamp: Timestamp
    reference_frame: str
    observations: Mapping[str, SensorObservation]
    transforms: Mapping[str, np.ndarray] = field(default_factory=dict)
    detections: tuple[Detection, ...] = ()
    semantic_mask: np.ndarray | None = None
    semantic_classes: Mapping[int, str] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    synchronization: SynchronizationStatus = field(default_factory=SynchronizationStatus)

    def __post_init__(self) -> None:
        observations = dict(self.observations)
        if not observations:
            raise ValueError("a PrismFrame requires at least one observation")
        if any(key != obs.sensor_id for key, obs in observations.items()):
            raise ValueError("observation keys must match sensor_id")
        object.__setattr__(self, "observations", MappingProxyType(observations))

        frozen_transforms: dict[str, np.ndarray] = {}
        for name, transform in self.transforms.items():
            matrix = np.asarray(transform, dtype=np.float64)
            if matrix.shape != (3, 3) and matrix.shape != (4, 4):
                raise ValueError("transforms must be 3x3 image or 4x4 spatial matrices")
            frozen_transforms[name] = _readonly(matrix)
        object.__setattr__(self, "transforms", MappingProxyType(frozen_transforms))

        if self.semantic_mask is not None:
            mask = np.asarray(self.semantic_mask)
            reference = next(
                (
                    observation
                    for observation in observations.values()
                    if observation.frame_id == self.reference_frame
                ),
                None,
            )
            if reference is None:
                raise ValueError(
                    "reference_frame must identify an observation when a semantic mask is present"
                )
            if mask.shape != (reference.height, reference.width):
                raise ValueError("semantic mask must match the reference observation")
            object.__setattr__(self, "semantic_mask", _readonly(mask))

        object.__setattr__(
            self, "semantic_classes", MappingProxyType(dict(self.semantic_classes))
        )

        provenance = dict(self.provenance)
        synchronization = self.synchronization
        if not isinstance(synchronization, SynchronizationStatus):
            if not isinstance(synchronization, Mapping):
                raise TypeError("synchronization must be a SynchronizationStatus or mapping")
            synchronization = SynchronizationStatus(**dict(synchronization))

        # Compatibility for the staged datasets is deliberately based on their
        # explicit publisher-rectification declaration, never on an absent or
        # placeholder timestamp. New adapters should pass
        # SynchronizationStatus.declared_replay_aligned(...) directly.
        if (
            synchronization.state == "unknown"
            and not synchronization.pixel_fusion_eligible
            and provenance.get("alignment") == "publisher_provided_rectification"
            and provenance.get("dataset")
        ):
            domains = {
                observation.timestamp.effective_clock_domain
                for observation in observations.values()
            }
            archive_declared = bool(domains) and all(
                domain.endswith("_archive") for domain in domains
            )
            if archive_declared and len(domains) == 1:
                synchronization = SynchronizationStatus.declared_replay_aligned(
                    tuple(observations),
                    clock_domain=next(iter(domains)),
                    declaration="provenance.alignment=publisher_provided_rectification",
                )

        object.__setattr__(self, "synchronization", synchronization)
        object.__setattr__(self, "provenance", _frozen_mapping(provenance))


@dataclass(frozen=True, slots=True)
class FusionOutput:
    """Synchronized operator and machine projections of a ``PrismFrame``."""

    frame_id: str
    operator_rgb: np.ndarray
    machine_tensor: np.ndarray
    channel_names: tuple[str, ...]
    fusion_support: np.ndarray
    registration_support: np.ndarray
    thermal_view: np.ndarray
    visible_view: np.ndarray
    semantic_view: np.ndarray | None
    provenance: Mapping[str, Any]
    pixel_fusion_applied: bool = False
    synchronization_state: str = "unknown"
    physical_timing_uncertainty_ns: int | None = None

    def __post_init__(self) -> None:
        operator = np.asarray(self.operator_rgb, dtype=np.uint8)
        if operator.ndim != 3 or operator.shape[2] != 3:
            raise ValueError("operator view must be HxWx3 RGB")
        tensor = np.asarray(self.machine_tensor, dtype=np.float32)
        if tensor.ndim != 3 or tensor.shape[0] != len(self.channel_names):
            raise ValueError("machine tensor must be CxHxW with named channels")
        if tensor.shape[1:] != operator.shape[:2]:
            raise ValueError("machine and operator projections must share geometry")

        maps = (self.fusion_support, self.registration_support)
        for score_map in maps:
            score_array = np.asarray(score_map)
            if score_array.shape != operator.shape[:2]:
                raise ValueError("support/confidence maps must match operator geometry")
            if not np.all(np.isfinite(score_array)) or np.any(
                (score_array < 0.0) | (score_array > 1.0)
            ):
                raise ValueError("support/confidence values must be within [0, 1]")

        for name, view in (
            ("thermal", self.thermal_view),
            ("visible", self.visible_view),
            ("semantic", self.semantic_view),
        ):
            if view is not None and np.asarray(view).shape[:2] != operator.shape[:2]:
                raise ValueError(f"{name} view must match operator geometry")

        object.__setattr__(self, "operator_rgb", _readonly(operator))
        object.__setattr__(self, "machine_tensor", _readonly(tensor))
        object.__setattr__(
            self, "fusion_support", _readonly(np.asarray(maps[0], dtype=np.float32))
        )
        object.__setattr__(
            self,
            "registration_support",
            _readonly(np.asarray(maps[1], dtype=np.float32)),
        )
        object.__setattr__(self, "thermal_view", _readonly(self.thermal_view))
        object.__setattr__(self, "visible_view", _readonly(self.visible_view))
        if self.semantic_view is not None:
            object.__setattr__(self, "semantic_view", _readonly(self.semantic_view))
        if self.synchronization_state not in _SYNCHRONIZATION_STATES:
            raise ValueError("unsupported output synchronization state")
        if (
            self.physical_timing_uncertainty_ns is not None
            and self.physical_timing_uncertainty_ns < 0
        ):
            raise ValueError("physical timing uncertainty must be non-negative")
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))
