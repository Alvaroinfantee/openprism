"""Registration primitives with explicit validity and uncertainty.

The reference implementation includes identity registration for already
rectified datasets and edge-domain phase correlation for residual translation.
Production adapters are expected to provide calibrated geometric priors and may
replace the residual registrar with XoFTR/MINIMA or a depth-aware flow plugin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    aligned: np.ndarray
    validity: np.ndarray
    confidence: np.ndarray
    transform: np.ndarray
    method: str
    diagnostics: dict[str, float]


class Registrar(Protocol):
    def register(self, reference: np.ndarray, moving: np.ndarray) -> RegistrationResult:
        """Align ``moving`` to ``reference`` geometry."""


def luminance(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image, dtype=np.float32)
    if value.ndim == 2:
        result = value
    elif value.ndim == 3 and value.shape[2] >= 3:
        result = (
            0.2126 * value[..., 0]
            + 0.7152 * value[..., 1]
            + 0.0722 * value[..., 2]
        )
    else:
        raise ValueError("expected a grayscale or RGB image")
    if result.max(initial=0.0) > 1.0:
        result = result / float(np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else 255.0)
    return np.clip(result, 0.0, 1.0)


def gradient_energy(image: np.ndarray) -> np.ndarray:
    gray = luminance(image)
    padded = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    gx = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    gy = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    return np.sqrt(gx * gx + gy * gy, dtype=np.float32)


def shift_with_validity(image: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Translate an array without treating wrapped pixels as valid."""

    shifted = np.roll(image, shift=(dy, dx), axis=(0, 1))
    valid = np.ones(image.shape[:2], dtype=bool)
    if dy > 0:
        valid[:dy, :] = False
    elif dy < 0:
        valid[dy:, :] = False
    if dx > 0:
        valid[:, :dx] = False
    elif dx < 0:
        valid[:, dx:] = False
    if shifted.ndim == 2:
        shifted = np.where(valid, shifted, 0)
    else:
        shifted = np.where(valid[..., None], shifted, 0)
    return shifted, valid


class IdentityRegistrar:
    """Use publisher-rectified inputs without pretending alignment was measured."""

    def __init__(self, declared_confidence: float = 0.95) -> None:
        if not 0.0 <= declared_confidence < 1.0:
            raise ValueError("declared confidence must be in [0, 1)")
        self.declared_confidence = float(declared_confidence)

    def register(self, reference: np.ndarray, moving: np.ndarray) -> RegistrationResult:
        if reference.shape[:2] != moving.shape[:2]:
            raise ValueError("identity registration requires equal image geometry")
        height, width = reference.shape[:2]
        return RegistrationResult(
            aligned=np.asarray(moving),
            validity=np.ones((height, width), dtype=bool),
            confidence=np.full(
                (height, width), self.declared_confidence, dtype=np.float32
            ),
            transform=np.eye(3, dtype=np.float64),
            method="publisher_rectified_declared",
            diagnostics={
                "dx_px": 0.0,
                "dy_px": 0.0,
                "peak_ratio": 0.0,
                "measured": 0.0,
                "declared_confidence": self.declared_confidence,
            },
        )


class PhaseCorrelationRegistrar:
    """Estimate residual integer translation using cross-modal edge structure."""

    def __init__(self, max_shift_px: int = 96) -> None:
        if max_shift_px < 0:
            raise ValueError("max_shift_px must be non-negative")
        self.max_shift_px = int(max_shift_px)

    def register(self, reference: np.ndarray, moving: np.ndarray) -> RegistrationResult:
        if reference.shape[:2] != moving.shape[:2]:
            raise ValueError("phase registration currently requires equal image geometry")
        reference_edge = gradient_energy(reference)
        moving_edge = gradient_energy(moving)
        edge_energy = float(
            min(np.mean(reference_edge), np.mean(moving_edge))
        )
        if edge_energy < 1e-4:
            height, width = reference.shape[:2]
            return RegistrationResult(
                aligned=np.asarray(moving),
                validity=np.ones((height, width), dtype=bool),
                confidence=np.zeros((height, width), dtype=np.float32),
                transform=np.eye(3, dtype=np.float64),
                method="edge_phase_correlation_abstained",
                diagnostics={
                    "dx_px": 0.0,
                    "dy_px": 0.0,
                    "peak_ratio": 0.0,
                    "edge_energy": edge_energy,
                },
            )
        reference_edge -= reference_edge.mean()
        moving_edge -= moving_edge.mean()

        window_y = np.hanning(reference_edge.shape[0]).astype(np.float32)
        window_x = np.hanning(reference_edge.shape[1]).astype(np.float32)
        window = window_y[:, None] * window_x[None, :]
        ref_fft = np.fft.rfft2(reference_edge * window)
        mov_fft = np.fft.rfft2(moving_edge * window)
        cross_power = ref_fft * np.conj(mov_fft)
        cross_power /= np.maximum(np.abs(cross_power), 1e-8)
        correlation = np.abs(np.fft.irfft2(cross_power, s=reference_edge.shape))

        height, width = correlation.shape
        allowed = np.zeros_like(correlation, dtype=bool)
        radius = self.max_shift_px
        allowed[: radius + 1, : radius + 1] = True
        if radius:
            allowed[-radius:, : radius + 1] = True
            allowed[: radius + 1, -radius:] = True
            allowed[-radius:, -radius:] = True
        restricted = np.where(allowed, correlation, -np.inf)
        peak_y, peak_x = np.unravel_index(np.argmax(restricted), restricted.shape)
        dy = int(peak_y if peak_y <= height // 2 else peak_y - height)
        dx = int(peak_x if peak_x <= width // 2 else peak_x - width)

        peak = float(correlation[peak_y, peak_x])
        finite = correlation[np.isfinite(correlation)]
        background = float(np.percentile(finite, 99.0)) if finite.size else 0.0
        peak_ratio = peak / max(background, 1e-8)
        global_confidence = float(np.clip((peak_ratio - 1.0) / 4.0, 0.05, 1.0))

        aligned, validity = shift_with_validity(moving, dy, dx)
        confidence = validity.astype(np.float32) * global_confidence
        transform = np.array(
            [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return RegistrationResult(
            aligned=aligned,
            validity=validity,
            confidence=confidence,
            transform=transform,
            method="edge_phase_correlation",
            diagnostics={
                "dx_px": float(dx),
                "dy_px": float(dy),
                "peak_ratio": peak_ratio,
                "edge_energy": edge_energy,
            },
        )
