"""Watermark synchronization for asynchronous multisensor observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

from .contracts import SensorObservation, SynchronizationStatus


class SensorAdapter(Protocol):
    """Minimal live-adapter interface for cameras, IMU, lidar, radar, or GNSS."""

    sensor_id: str
    modality: str

    def poll(self) -> SensorObservation | None:
        """Return the next hardware-timestamped observation when available."""


@dataclass(frozen=True, slots=True)
class SynchronizationPolicy:
    required_sensor_ids: tuple[str, ...]
    exact_tolerance_ns: int = 1_000_000
    pixel_fusion_tolerance_ns: int = 5_000_000
    max_staleness_ns: int = 100_000_000
    buffer_size: int = 32

    def __post_init__(self) -> None:
        if not self.required_sensor_ids:
            raise ValueError("at least one required sensor is needed")
        if len(set(self.required_sensor_ids)) != len(self.required_sensor_ids):
            raise ValueError("required sensor IDs must be unique")
        if not (
            0 <= self.exact_tolerance_ns
            <= self.pixel_fusion_tolerance_ns
            <= self.max_staleness_ns
        ):
            raise ValueError("synchronization tolerances must be ordered")
        if self.buffer_size < 2:
            raise ValueError("buffer_size must be at least two")


@dataclass(frozen=True, slots=True)
class SynchronizedSelection:
    reference_tai_ns: int
    reference_clock_domain: str
    observations: dict[str, SensorObservation]
    states: dict[str, str]
    skew_ns: dict[str, int | None]
    effective_skew_ns: dict[str, int | None]
    timing_uncertainty_ns: dict[str, int | None]
    max_skew_ns: int | None
    max_effective_skew_ns: int | None
    physical_timing_uncertainty_ns: int | None
    pixel_fusion_eligible: bool
    synchronization: SynchronizationStatus


class WatermarkSynchronizer:
    """Select nearest observations without silently pretending they are exact."""

    def __init__(self, policy: SynchronizationPolicy) -> None:
        self.policy = policy
        self._buffers: dict[str, deque[SensorObservation]] = {}

    def push(self, observation: SensorObservation) -> None:
        queue = self._buffers.setdefault(
            observation.sensor_id, deque(maxlen=self.policy.buffer_size)
        )
        if observation.timestamp.tai_ns is not None:
            previous_measured = next(
                (
                    item
                    for item in reversed(queue)
                    if item.timestamp.effective_clock_domain
                    == observation.timestamp.effective_clock_domain
                    and item.timestamp.tai_ns is not None
                ),
                None,
            )
            if (
                previous_measured is not None
                and observation.timestamp.tai_ns
                < previous_measured.timestamp.tai_ns
            ):
                raise ValueError(
                    f"out-of-order timestamp for sensor {observation.sensor_id}"
                )
        queue.append(observation)

    def assemble(self, reference_sensor_id: str) -> SynchronizedSelection:
        reference_queue = self._buffers.get(reference_sensor_id)
        if not reference_queue:
            raise KeyError(f"no observations for reference sensor {reference_sensor_id}")
        reference = reference_queue[-1]
        if reference.timestamp.tai_ns is None:
            raise ValueError("cannot assemble a watermark from an unknown capture time")
        watermark = reference.timestamp.tai_ns
        reference_domain = reference.timestamp.effective_clock_domain
        observations: dict[str, SensorObservation] = {}
        states: dict[str, str] = {}
        skew: dict[str, int | None] = {}
        effective_skew: dict[str, int | None] = {}
        timing_uncertainty: dict[str, int | None] = {}

        sensor_ids = tuple(
            dict.fromkeys((*self.policy.required_sensor_ids, *self._buffers.keys()))
        )
        for sensor_id in sensor_ids:
            queue = self._buffers.get(sensor_id)
            if not queue:
                states[sensor_id] = "missing"
                skew[sensor_id] = None
                effective_skew[sensor_id] = None
                timing_uncertainty[sensor_id] = None
                continue

            compatible = tuple(
                item
                for item in queue
                if item.timestamp.effective_clock_domain == reference_domain
                and item.timestamp.tai_ns is not None
            )
            if not compatible:
                # Preserve the newest raw observation for degraded/operator use,
                # but never compare its numeric timestamp across an unrelated
                # clock domain or admit it to pixel fusion.
                observations[sensor_id] = queue[-1]
                same_domain = any(
                    item.timestamp.effective_clock_domain == reference_domain
                    for item in queue
                )
                states[sensor_id] = (
                    "unknown" if same_domain else "incompatible_clock_domain"
                )
                skew[sensor_id] = None
                effective_skew[sensor_id] = None
                timing_uncertainty[sensor_id] = None
                continue

            candidate = min(
                compatible,
                key=lambda item: (
                    abs(item.timestamp.tai_ns - watermark)
                    + (
                        0
                        if item is reference
                        else (
                            self.policy.max_staleness_ns + 1
                            if item.timestamp.uncertainty_ns is None
                            or reference.timestamp.uncertainty_ns is None
                            else item.timestamp.uncertainty_ns
                            + reference.timestamp.uncertainty_ns
                        )
                    )
                ),
            )
            delta = candidate.timestamp.tai_ns - watermark
            absolute_delta = abs(delta)
            pair_uncertainty = 0 if candidate is reference else (
                None
                if candidate.timestamp.uncertainty_ns is None
                or reference.timestamp.uncertainty_ns is None
                else candidate.timestamp.uncertainty_ns
                + reference.timestamp.uncertainty_ns
            )
            if pair_uncertainty is None:
                observations[sensor_id] = candidate
                states[sensor_id] = "unsynchronized"
                skew[sensor_id] = delta
                effective_skew[sensor_id] = None
                timing_uncertainty[sensor_id] = None
                continue
            worst_case_delta = absolute_delta + pair_uncertainty
            timing_uncertainty[sensor_id] = pair_uncertainty
            effective_skew[sensor_id] = worst_case_delta
            if worst_case_delta > self.policy.max_staleness_ns:
                states[sensor_id] = "missing"
                skew[sensor_id] = delta
                continue
            observations[sensor_id] = candidate
            skew[sensor_id] = delta
            if worst_case_delta <= self.policy.exact_tolerance_ns:
                states[sensor_id] = "exact"
            elif delta < 0:
                states[sensor_id] = "late"
            elif delta > 0:
                # This is a nearest future sample, not an interpolated value.
                states[sensor_id] = "future"
            else:
                # Nominal capture instants coincide, but the uncertainty bounds
                # are too wide to call the pairing exact.
                states[sensor_id] = "bounded_skew"

        required_present = all(
            sensor_id in observations
            and states.get(sensor_id) not in {"missing", "incompatible_clock_domain"}
            for sensor_id in self.policy.required_sensor_ids
        )
        required_skews = [
            abs(skew[sensor_id])
            for sensor_id in self.policy.required_sensor_ids
            if skew.get(sensor_id) is not None
        ]
        required_effective_skews = [
            effective_skew[sensor_id]
            for sensor_id in self.policy.required_sensor_ids
            if effective_skew.get(sensor_id) is not None
        ]
        required_uncertainties = [
            timing_uncertainty[sensor_id]
            for sensor_id in self.policy.required_sensor_ids
            if timing_uncertainty.get(sensor_id) is not None
        ]
        required_count = len(self.policy.required_sensor_ids)
        max_skew = (
            max(required_skews) if len(required_skews) == required_count else None
        )
        max_effective_skew = (
            max(required_effective_skews)
            if len(required_effective_skews) == required_count
            else None
        )
        physical_uncertainty = (
            max(required_uncertainties)
            if len(required_uncertainties) == required_count
            else None
        )
        eligible = bool(
            required_present
            and max_effective_skew is not None
            and max_effective_skew <= self.policy.pixel_fusion_tolerance_ns
        )

        required_states = {
            sensor_id: states.get(sensor_id, "missing")
            for sensor_id in self.policy.required_sensor_ids
        }
        if any(value == "incompatible_clock_domain" for value in required_states.values()):
            synchronization_state = "incompatible_clock_domain"
        elif not required_present:
            synchronization_state = "missing"
        elif eligible and all(value == "exact" for value in required_states.values()):
            synchronization_state = "exact"
        elif eligible:
            synchronization_state = "bounded_skew"
        else:
            synchronization_state = "unsynchronized"

        synchronization = SynchronizationStatus(
            state=synchronization_state,
            pixel_fusion_eligible=eligible,
            basis="measured",
            clock_domain=reference_domain,
            measured_max_skew_ns=max_skew,
            effective_max_skew_ns=max_effective_skew,
            physical_timing_uncertainty_ns=physical_uncertainty,
            sensor_states=states,
            sensor_skew_ns=skew,
            sensor_effective_skew_ns=effective_skew,
            details={
                "reference_sensor_id": reference_sensor_id,
                "reference_clock_id": reference.timestamp.clock_id,
                "uncertainty_policy": "worst_case_sum",
            },
        )
        return SynchronizedSelection(
            reference_tai_ns=watermark,
            reference_clock_domain=reference_domain,
            observations=observations,
            states=states,
            skew_ns=skew,
            effective_skew_ns=effective_skew,
            timing_uncertainty_ns=timing_uncertainty,
            max_skew_ns=max_skew,
            max_effective_skew_ns=max_effective_skew,
            physical_timing_uncertainty_ns=physical_uncertainty,
            pixel_fusion_eligible=eligible,
            synchronization=synchronization,
        )
