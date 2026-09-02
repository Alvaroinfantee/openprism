"""Deterministic, confidence-aware reference fusion engine.

This is intentionally a transparent baseline rather than a claim of learned
state of the art. It preserves source tensors for machine consumers and derives
an operator rendering without inventing measurements. Learned registrars,
encoders, detectors, and segmenters attach around the same contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .contracts import FusionOutput, PrismFrame, SensorObservation, SynchronizationStatus
from .registration import (
    IdentityRegistrar,
    Registrar,
    gradient_energy,
    luminance,
    shift_with_validity,
)


CALTECH_PALETTE: dict[int, tuple[int, int, int]] = {
    0: (255, 36, 0),
    1: (0, 0, 0),
    2: (242, 216, 196),
    3: (89, 70, 54),
    4: (166, 166, 166),
    5: (82, 89, 90),
    6: (155, 230, 0),
    7: (0, 138, 53),
    8: (0, 216, 245),
    9: (13, 127, 252),
    10: (255, 249, 0),
    11: (254, 0, 170),
}


@dataclass(frozen=True, slots=True)
class FusionConfig:
    thermal_gain: float = 1.0
    thermal_floor: float = 0.08
    thermal_ceiling: float = 0.86
    semantic_alpha: float = 0.42
    min_registration_support: float = 0.12
    preserve_color: float = 0.88

    def __post_init__(self) -> None:
        if not 0.0 <= self.thermal_gain <= 2.5:
            raise ValueError("thermal_gain must be in [0, 2.5]")
        if not 0.0 <= self.thermal_floor <= self.thermal_ceiling <= 1.0:
            raise ValueError("thermal floor/ceiling must be ordered within [0, 1]")
        if not 0.0 <= self.semantic_alpha <= 1.0:
            raise ValueError("semantic_alpha must be in [0, 1]")
        if not 0.0 <= self.preserve_color <= 1.0:
            raise ValueError("preserve_color must be in [0, 1]")


def robust_normalize(
    value: np.ndarray,
    validity: np.ndarray | None = None,
    percentiles: tuple[float, float] = (1.0, 99.0),
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    selection = array[np.asarray(validity, dtype=bool)] if validity is not None else array.ravel()
    selection = selection[np.isfinite(selection)]
    if selection.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low, high = np.percentile(selection, percentiles)
    if high <= low + 1e-8:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - float(low)) / float(high - low), 0.0, 1.0)


def box_blur(value: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0:
        return np.asarray(value, dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    height, width = array.shape
    padded = np.pad(array, ((radius, radius), (radius, radius)), mode="reflect")
    total = np.zeros((height, width), dtype=np.float32)
    size = radius * 2 + 1
    for dy in range(size):
        for dx in range(size):
            total += padded[dy : dy + height, dx : dx + width]
    return total / float(size * size)


def _rgb_float(observation: SensorObservation) -> np.ndarray:
    image = np.asarray(observation.data)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] > 3:
        image = image[..., :3]
    maximum = float(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else 1.0
    return np.clip(image.astype(np.float32) / maximum, 0.0, 1.0)


def _image_validity(
    observation: SensorObservation,
    shape: tuple[int, int],
) -> np.ndarray:
    """Return the observation validity in canonical HxW form."""

    if observation.validity is None:
        return np.ones(shape, dtype=bool)
    validity = np.asarray(observation.validity, dtype=bool)
    if validity.shape != shape:
        # SensorObservation normalizes new image masks to HxW.  Keep this
        # defensive reduction for observations deserialized by older adapters.
        if validity.shape[:2] == shape and validity.ndim > 2:
            validity = np.all(validity, axis=tuple(range(2, validity.ndim)))
        elif validity.shape == (shape[0],):
            validity = np.broadcast_to(validity[:, None], shape)
        else:
            raise ValueError(
                f"validity for {observation.sensor_id} does not match HxW geometry"
            )
    return np.asarray(validity, dtype=bool)


def _shift_validity(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    shifted, geometric_support = shift_with_validity(
        np.asarray(mask, dtype=np.uint8), dy, dx
    )
    return np.asarray(shifted, dtype=bool) & geometric_support


def _synchronization_provenance(status: SynchronizationStatus) -> dict[str, object]:
    return {
        "state": status.state,
        "pixel_fusion_eligible": status.pixel_fusion_eligible,
        "basis": status.basis,
        "clock_domain": status.clock_domain,
        "measured_max_skew_ns": status.measured_max_skew_ns,
        "effective_max_skew_ns": status.effective_max_skew_ns,
        "physical_timing_uncertainty_ns": status.physical_timing_uncertainty_ns,
        "sensor_states": dict(status.sensor_states),
        "sensor_skew_ns": dict(status.sensor_skew_ns),
        "sensor_effective_skew_ns": dict(status.sensor_effective_skew_ns),
        "details": dict(status.details),
    }


def _normalize_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    scale = float(np.percentile(array, 99.0))
    return np.clip(array / max(scale, 1e-7), 0.0, 1.0)


def _replace_luminance(rgb: np.ndarray, target_y: np.ndarray, preserve: float) -> np.ndarray:
    red, green, blue = np.moveaxis(rgb, -1, 0)
    source_y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = (blue - source_y) * 0.564
    cr = (red - source_y) * 0.713
    cb *= preserve
    cr *= preserve
    out_r = target_y + 1.403 * cr
    out_g = target_y - 0.344 * cb - 0.714 * cr
    out_b = target_y + 1.773 * cb
    return np.clip(np.stack((out_r, out_g, out_b), axis=-1), 0.0, 1.0)


def _interpolate_palette(
    value: np.ndarray, stops: Iterable[tuple[float, tuple[int, int, int]]]
) -> np.ndarray:
    array = np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)
    result = np.zeros(array.shape + (3,), dtype=np.float32)
    ordered = list(stops)
    for (left_x, left_rgb), (right_x, right_rgb) in zip(ordered, ordered[1:]):
        selected = (array >= left_x) & (array <= right_x)
        factor = np.clip((array - left_x) / max(right_x - left_x, 1e-8), 0.0, 1.0)
        left = np.asarray(left_rgb, dtype=np.float32) / 255.0
        right = np.asarray(right_rgb, dtype=np.float32) / 255.0
        color = left + factor[..., None] * (right - left)
        result[selected] = color[selected]
    result[array < ordered[0][0]] = np.asarray(ordered[0][1], dtype=np.float32) / 255.0
    result[array > ordered[-1][0]] = np.asarray(ordered[-1][1], dtype=np.float32) / 255.0
    return np.rint(result * 255.0).astype(np.uint8)


def thermal_colormap(value: np.ndarray) -> np.ndarray:
    return _interpolate_palette(
        value,
        (
            (0.00, (4, 5, 18)),
            (0.22, (42, 30, 92)),
            (0.48, (139, 45, 94)),
            (0.70, (229, 93, 45)),
            (0.88, (250, 190, 65)),
            (1.00, (255, 247, 190)),
        ),
    )


def support_colormap(value: np.ndarray) -> np.ndarray:
    # Magenta = do not trust overlay; blue = limited; green = strong evidence.
    return _interpolate_palette(
        value,
        (
            (0.00, (178, 36, 126)),
            (0.45, (45, 105, 180)),
            (0.72, (39, 163, 151)),
            (1.00, (190, 231, 120)),
        ),
    )


def semantic_color(class_id: int) -> tuple[int, int, int]:
    if int(class_id) in CALTECH_PALETTE:
        return CALTECH_PALETTE[int(class_id)]
    seed = int(class_id) * 2654435761 & 0xFFFFFF
    return (
        96 + (seed & 0x7F),
        96 + ((seed >> 8) & 0x7F),
        96 + ((seed >> 16) & 0x7F),
    )


def _semantic_view(
    operator_rgb: np.ndarray,
    mask: np.ndarray | None,
    alpha: float,
) -> np.ndarray | None:
    if mask is None:
        return None
    labels = np.asarray(mask)
    colors = np.zeros(labels.shape + (3,), dtype=np.uint8)
    for class_id in np.unique(labels):
        colors[labels == class_id] = semantic_color(int(class_id))
    base = operator_rgb.astype(np.float32)
    mixed = (1.0 - alpha) * base + alpha * colors.astype(np.float32)
    return np.rint(np.clip(mixed, 0.0, 255.0)).astype(np.uint8)


class EvidenceFusionEngine:
    """Create reversible machine evidence and a deterministic operator view."""

    def __init__(self, registrar: Registrar | None = None) -> None:
        self.registrar = registrar or IdentityRegistrar()

    @staticmethod
    def _find(
        frame: PrismFrame,
        keys: tuple[str, ...],
        modalities: tuple[str, ...],
    ) -> SensorObservation:
        for key in keys:
            if key in frame.observations:
                return frame.observations[key]
        for observation in frame.observations.values():
            if observation.modality in modalities:
                return observation
        raise KeyError(f"missing required modality: {modalities}")

    @staticmethod
    def _find_optional(
        frame: PrismFrame,
        keys: tuple[str, ...],
        modalities: tuple[str, ...],
    ) -> SensorObservation | None:
        try:
            return EvidenceFusionEngine._find(frame, keys, modalities)
        except KeyError:
            return None

    @staticmethod
    def _synchronization_gate(
        frame: PrismFrame,
        contributors: tuple[SensorObservation, ...],
    ) -> tuple[bool, str | None, str]:
        status = frame.synchronization
        if not status.pixel_fusion_eligible:
            return False, f"synchronization_{status.state}", status.state

        domains = {
            observation.timestamp.effective_clock_domain
            for observation in contributors
        }
        if len(domains) != 1:
            return (
                False,
                "incompatible_observation_clock_domains",
                "incompatible_clock_domain",
            )
        observed_domain = next(iter(domains))
        if status.clock_domain is not None and status.clock_domain != observed_domain:
            return (
                False,
                "frame_clock_domain_mismatch",
                "incompatible_clock_domain",
            )
        return True, None, status.state

    @staticmethod
    def _no_fusion_zone(
        frame: PrismFrame,
        visible: SensorObservation,
        thermal: SensorObservation | None,
        settings: FusionConfig,
        reason: str,
        synchronization_state: str,
    ) -> FusionOutput:
        """Preserve the reference view while contributing zero thermal pixels."""

        rgb = _rgb_float(visible)
        height, width = rgb.shape[:2]
        geometry = (height, width)
        visible_validity = _image_validity(visible, geometry)
        visible_detail = _normalize_map(gradient_energy(rgb))
        zeros = np.zeros(geometry, dtype=np.float32)

        # A same-sized, normalized thermal preview remains available as a
        # separate operator rail. It is not placed in the registered machine
        # tensor or used in the RGB canvas while synchronization is ineligible.
        thermal_preview = zeros
        thermal_preview_kind = "unavailable"
        if thermal is not None and thermal.data.shape[:2] == geometry:
            raw_signal = luminance(thermal.data)
            raw_validity = _image_validity(thermal, geometry)
            thermal_preview = robust_normalize(raw_signal, raw_validity)
            thermal_preview = np.where(raw_validity, thermal_preview, 0.0).astype(
                np.float32
            )
            thermal_preview_kind = "normalized_unfused"

        operator_rgb = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        fusion_support = zeros.copy()
        registration_support = zeros.copy()
        channel_names = (
            "visible_r_srgb",
            "visible_g_srgb",
            "visible_b_srgb",
            "thermal_radiometric_norm",
            "visible_detail",
            "thermal_detail",
            "thermal_saliency",
            "sensor_validity",
            "registration_support_score",
            "thermal_contribution",
            "fusion_support_score",
        )
        machine_tensor = np.stack(
            (
                rgb[..., 0],
                rgb[..., 1],
                rgb[..., 2],
                zeros,
                visible_detail,
                zeros,
                zeros,
                visible_validity.astype(np.float32),
                registration_support,
                zeros,
                fusion_support,
            ),
            axis=0,
        ).astype(np.float32)
        semantic_view = _semantic_view(
            operator_rgb, frame.semantic_mask, settings.semantic_alpha
        )
        provenance = {
            "openprism_version": "0.1.0",
            "algorithm": "no_fusion_zone_reference_only",
            "fusion_mode": "no_fusion_zone",
            "pixel_fusion_applied": False,
            "non_hallucinatory": True,
            "fallback_reason": reason,
            "source_frame_id": frame.frame_id,
            "source_sensors": tuple(frame.observations.keys()),
            "thermal_view_kind": thermal_preview_kind,
            "thermal_contribution": 0.0,
            "unmodeled_measurement_uncertainty_sensors": tuple(
                observation.sensor_id
                for observation in frame.observations.values()
                if observation.uncertainty is not None
            ),
            "physical_timing_uncertainty_ns": (
                frame.synchronization.physical_timing_uncertainty_ns
            ),
            "synchronization": _synchronization_provenance(frame.synchronization),
            "machine_channels": channel_names,
            "upstream": dict(frame.provenance),
        }
        return FusionOutput(
            frame_id=frame.frame_id,
            operator_rgb=operator_rgb,
            machine_tensor=machine_tensor,
            channel_names=channel_names,
            fusion_support=fusion_support,
            registration_support=registration_support,
            thermal_view=thermal_colormap(thermal_preview),
            visible_view=operator_rgb,
            semantic_view=semantic_view,
            provenance=provenance,
            pixel_fusion_applied=False,
            synchronization_state=synchronization_state,
            physical_timing_uncertainty_ns=(
                frame.synchronization.physical_timing_uncertainty_ns
            ),
        )

    def fuse(self, frame: PrismFrame, config: FusionConfig | None = None) -> FusionOutput:
        settings = config or FusionConfig()
        visible = self._find(frame, ("visible", "rgb"), ("visible_rgb", "rgb"))
        thermal = self._find_optional(
            frame,
            ("thermal", "thermal8", "infrared"),
            ("thermal_lwir_8", "thermal_lwir", "infrared"),
        )
        radiometric_source = None
        for key in ("thermal16", "radiometric"):
            if key in frame.observations:
                radiometric_source = frame.observations[key]
                break

        if thermal is None:
            return self._no_fusion_zone(
                frame,
                visible,
                None,
                settings,
                "missing_thermal_observation",
                "missing",
            )

        contributors = (
            (visible, thermal, radiometric_source)
            if radiometric_source is not None
            else (visible, thermal)
        )
        eligible, gate_reason, output_sync_state = self._synchronization_gate(
            frame,
            contributors,
        )
        if not eligible:
            return self._no_fusion_zone(
                frame,
                visible,
                thermal,
                settings,
                gate_reason or "synchronization_ineligible",
                output_sync_state,
            )

        unmodeled_uncertainty = tuple(
            observation.sensor_id
            for observation in contributors
            if observation.uncertainty is not None
        )
        if unmodeled_uncertainty:
            return self._no_fusion_zone(
                frame,
                visible,
                thermal,
                settings,
                "measurement_uncertainty_model_unavailable",
                frame.synchronization.state,
            )

        rgb = _rgb_float(visible)
        registration = self.registrar.register(visible.data, thermal.data)
        thermal_source = registration.aligned
        dx = int(round(registration.diagnostics.get("dx_px", 0.0)))
        dy = int(round(registration.diagnostics.get("dy_px", 0.0)))

        visible_validity = _image_validity(visible, rgb.shape[:2])
        display_thermal_validity = _shift_validity(
            _image_validity(thermal, thermal.data.shape[:2]), dy, dx
        )
        if radiometric_source is not None:
            shifted, shifted_validity = shift_with_validity(radiometric_source.data, dy, dx)
            thermal_signal = luminance(shifted)
            radiometric_validity = shifted_validity & _shift_validity(
                _image_validity(
                    radiometric_source,
                    radiometric_source.data.shape[:2],
                ),
                dy,
                dx,
            )
        else:
            thermal_signal = luminance(thermal_source)
            radiometric_validity = np.ones(thermal_signal.shape, dtype=bool)

        valid = (
            registration.validity
            & visible_validity
            & display_thermal_validity
            & radiometric_validity
        )

        thermal_norm = robust_normalize(thermal_signal, valid)
        visible_y = luminance(rgb)
        visible_detail = _normalize_map(gradient_energy(rgb))
        thermal_detail = _normalize_map(gradient_energy(thermal_norm))
        thermal_local = box_blur(thermal_norm, radius=3)
        thermal_contrast = _normalize_map(np.abs(thermal_norm - thermal_local))
        thermal_saliency = np.clip(0.55 * thermal_detail + 0.45 * thermal_contrast, 0.0, 1.0)

        exposure = np.clip(1.0 - 1.7 * np.abs(visible_y - 0.5), 0.08, 1.0)
        visible_reliability = np.clip(0.45 * exposure + 0.55 * visible_detail, 0.0, 1.0)
        thermal_reliability = np.clip(0.30 + 0.70 * thermal_saliency, 0.0, 1.0)
        night_factor = float(np.clip((0.48 - np.mean(visible_y)) / 0.38, 0.0, 1.0))

        adaptive_weight = (
            settings.thermal_floor
            + 0.22 * night_factor
            + 0.58 * thermal_saliency
        )
        adaptive_weight = np.clip(
            settings.thermal_gain * adaptive_weight, 0.0, settings.thermal_ceiling
        )
        trusted_registration = np.where(
            registration.confidence >= settings.min_registration_support,
            registration.confidence,
            0.0,
        )
        adaptive_weight *= trusted_registration * valid.astype(np.float32)

        tone_visible = np.clip((visible_y - 0.5) * 1.08 + 0.5, 0.0, 1.0)
        fused_y = (1.0 - adaptive_weight) * tone_visible + adaptive_weight * thermal_norm
        operator = _replace_luminance(rgb, fused_y, settings.preserve_color)

        # A restrained amber cue makes strong thermal-only evidence perceptible
        # while keeping the source measurement and the cue visually separable.
        hot_evidence = np.clip((thermal_norm - 0.58) / 0.42, 0.0, 1.0)
        hot_evidence *= thermal_saliency * trusted_registration * valid.astype(np.float32)
        amber = np.array((1.0, 0.56, 0.16), dtype=np.float32)
        cue_alpha = (0.20 * settings.thermal_gain * hot_evidence)[..., None]
        operator = operator * (1.0 - cue_alpha) + amber * cue_alpha
        operator_rgb = np.rint(np.clip(operator, 0.0, 1.0) * 255.0).astype(np.uint8)

        fusion_support = np.clip(
            np.maximum(visible_reliability, thermal_reliability)
            * trusted_registration
            * valid.astype(np.float32),
            0.0,
            1.0,
        )
        validity = valid.astype(np.float32)
        channel_names = (
            "visible_r_srgb",
            "visible_g_srgb",
            "visible_b_srgb",
            "thermal_radiometric_norm",
            "visible_detail",
            "thermal_detail",
            "thermal_saliency",
            "sensor_validity",
            "registration_support_score",
            "thermal_contribution",
            "fusion_support_score",
        )
        machine_tensor = np.stack(
            (
                rgb[..., 0],
                rgb[..., 1],
                rgb[..., 2],
                thermal_norm,
                visible_detail,
                thermal_detail,
                thermal_saliency,
                validity,
                registration.confidence,
                adaptive_weight,
                fusion_support,
            ),
            axis=0,
        ).astype(np.float32)

        semantic_view = _semantic_view(
            operator_rgb, frame.semantic_mask, settings.semantic_alpha
        )
        provenance = {
            "openprism_version": "0.1.0",
            "algorithm": "deterministic_confidence_aware_luminance_fusion",
            "fusion_mode": "pixel_fusion",
            "pixel_fusion_applied": bool(np.any(adaptive_weight > 0.0)),
            "non_hallucinatory": True,
            "source_frame_id": frame.frame_id,
            "source_sensors": tuple(frame.observations.keys()),
            "registration": {
                "method": registration.method,
                **registration.diagnostics,
            },
            "thermal_gain": settings.thermal_gain,
            "night_factor": night_factor,
            "physical_timing_uncertainty_ns": (
                frame.synchronization.physical_timing_uncertainty_ns
            ),
            "measurement_uncertainty_policy": "none_supplied",
            "synchronization": _synchronization_provenance(frame.synchronization),
            "machine_channels": channel_names,
            "upstream": dict(frame.provenance),
        }
        return FusionOutput(
            frame_id=frame.frame_id,
            operator_rgb=operator_rgb,
            machine_tensor=machine_tensor,
            channel_names=channel_names,
            fusion_support=fusion_support,
            registration_support=registration.confidence,
            thermal_view=thermal_colormap(thermal_norm),
            visible_view=np.rint(rgb * 255.0).astype(np.uint8),
            semantic_view=semantic_view,
            provenance=provenance,
            pixel_fusion_applied=bool(np.any(adaptive_weight > 0.0)),
            synchronization_state=frame.synchronization.state,
            physical_timing_uncertainty_ns=(
                frame.synchronization.physical_timing_uncertainty_ns
            ),
        )
