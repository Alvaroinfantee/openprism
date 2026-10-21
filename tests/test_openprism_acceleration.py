from __future__ import annotations

import importlib.util
import os
import unittest
from unittest.mock import patch

import numpy as np

from openprism.acceleration import _load_native, backend_info, native_kernels
from openprism.contracts import PrismFrame, SensorObservation, SynchronizationStatus, Timestamp
from openprism.fusion import EvidenceFusionEngine, FusionConfig, box_blur, support_colormap, thermal_colormap
from openprism.registration import IdentityRegistrar, PhaseCorrelationRegistrar


def with_backend(name, function, *args, **kwargs):
    with patch.dict(os.environ, OPENPRISM_BACKEND=name):
        return function(*args, **kwargs)


def synthetic_frame(height=48, width=64, *, seed=7, synchronized=True, uncertainty=None, flat=False):
    rng = np.random.default_rng(seed)
    timestamp = Timestamp(1000, clock_id="synthetic")
    visible = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    thermal = rng.integers(0, 65536, (height, width), dtype=np.uint16)
    if flat:
        visible.fill(96)
        thermal.fill(32000)
    observations = {}
    for name, modality, data, encoding in (
        ("visible", "visible_rgb", visible, "rgb8"),
        ("thermal", "thermal_lwir", thermal, "mono16"),
    ):
        validity = rng.random((height, width)) > 0.03
        observations[name] = SensorObservation(
            sensor_id=name, modality=modality, frame_id=name,
            timestamp=timestamp, data=data, encoding=encoding, validity=validity,
            uncertainty=uncertainty,
        )
    status = (
        SynchronizationStatus.declared_replay_aligned(
            tuple(observations), clock_domain="synthetic", declaration="synthetic test fixture"
        ) if synchronized else SynchronizationStatus()
    )
    return PrismFrame(
        frame_id="synthetic", timestamp=timestamp, reference_frame="visible",
        observations=observations, synchronization=status,
        semantic_mask=rng.integers(0, 12, (height, width), dtype=np.uint8),
    )


class BackendSelectionTests(unittest.TestCase):
    def test_numpy_default_does_not_import_extension(self):
        with patch.dict(os.environ, {}, clear=True), patch("openprism.acceleration._load_native") as load:
            self.assertIsNone(native_kernels())
            self.assertEqual(backend_info()["name"], "numpy")
            load.assert_not_called()

    def test_invalid_backend_is_not_silently_ignored(self):
        with self.assertRaisesRegex(ValueError, "OPENPRISM_BACKEND"):
            with_backend("typo", native_kernels)

    def test_missing_and_incompatible_extension_fail_explicitly(self):
        for replacement, message in ((ImportError("missing"), "unavailable"), (type("Module", (), {"API_VERSION": 999})(), "API version")):
            _load_native.cache_clear()
            kwargs = {"side_effect": replacement} if isinstance(replacement, Exception) else {"return_value": replacement}
            with patch("openprism.acceleration.importlib.import_module", **kwargs):
                with self.assertRaisesRegex(RuntimeError, message):
                    with_backend("rust", native_kernels)
        _load_native.cache_clear()


@unittest.skipUnless(importlib.util.find_spec("openprism_native"), "optional Rust extension not installed")
class NativeParityTests(unittest.TestCase):
    def test_palette_is_byte_exact_at_stops_and_over_dense_domain(self):
        stops = np.array([0, .22, .45, .48, .70, .72, .88, 1], dtype=np.float32)
        values = np.concatenate((
            np.linspace(-0.1, 1.1, 200001, dtype=np.float32), stops,
            np.nextafter(stops, np.float32(-np.inf)),
            np.nextafter(stops, np.float32(np.inf)),
            np.array([np.nan, np.inf, -np.inf], dtype=np.float32),
        ))
        for colormap in (thermal_colormap, support_colormap):
            for value in (values, values[::-2], values.astype(np.float64), np.array(0.7), np.empty((0, 3))):
                expected = with_backend("numpy", colormap, value)
                actual = with_backend("rust", colormap, value)
                np.testing.assert_array_equal(actual, expected)

    def test_blur_reflection_strides_dtypes_and_accuracy(self):
        rng = np.random.default_rng(31)
        for shape in ((1, 1), (1, 9), (8, 1), (7, 11), (64, 96)):
            base = rng.random(shape, dtype=np.float32)
            for value in (base, base[::-1, ::-1], base.T, np.asfortranarray(base), base.astype(np.float64)):
                original = value.copy()
                for radius in (0, 1, 2, 3, 7, 15):
                    expected = with_backend("numpy", box_blur, value, radius)
                    actual = with_backend("rust", box_blur, value, radius)
                    # Reference float32 accumulation error grows with area;
                    # the native result is independently checked against f64.
                    tolerance = 8e-7 if radius <= 3 else (2 * radius + 1) ** 2 * np.finfo(np.float32).eps
                    np.testing.assert_allclose(actual, expected, atol=tolerance, rtol=0)
                    self.assertEqual(actual.dtype, np.float32)
                np.testing.assert_array_equal(value, original)

    def test_blur_matches_float64_oracle(self):
        rng = np.random.default_rng(81)
        source = rng.random((13, 19), dtype=np.float32)
        for radius in (1, 3, 15, 64):
            padded = np.pad(source.astype(np.float64), radius, mode="reflect")
            oracle = np.zeros(source.shape, dtype=np.float64)
            size = 2 * radius + 1
            for dy in range(size):
                for dx in range(size):
                    oracle += padded[dy:dy + 13, dx:dx + 19]
            oracle /= size * size
            result = with_backend("rust", box_blur, source, radius)
            np.testing.assert_array_equal(result, oracle.astype(np.float32))

    def test_general_blur_domain_uses_reference_semantics(self):
        for special in (np.nan, np.inf, -np.inf, -10.0, 1e30):
            value = np.full((8, 9), 0.25, dtype=np.float32)
            value[3, 4] = special
            with np.errstate(invalid="ignore", over="ignore"):
                expected = with_backend("numpy", box_blur, value, 3)
                actual = with_backend("rust", box_blur, value, 3)
            np.testing.assert_array_equal(actual, expected)
        value = np.ones((1, 1), dtype=np.float32)
        np.testing.assert_array_equal(with_backend("rust", box_blur, value, 65), [[1]])

    def test_native_entrypoints_reject_unsupported_inputs(self):
        native = with_backend("rust", native_kernels)
        for value, radius in ((np.empty((0, 1), np.float32), 3), (np.ones((1, 1), np.float32), 65), (np.array([[np.nan]], np.float32), 3)):
            with self.assertRaises(ValueError):
                native.box_blur(value, radius)
        with self.assertRaises(ValueError):
            native.box_blur(np.ones((3, 4), np.float32)[:, ::2], 3)
        with self.assertRaises(ValueError):
            native.palette(np.ones(1, np.float32), [0, 0], [(0, 0, 0)] * 2, [1])

    def test_unaligned_inputs_are_copied_at_python_boundary(self):
        value = np.ndarray((7, 9), dtype=np.float32, buffer=bytearray(7 * 9 * 4 + 1), offset=1)
        value[:] = np.linspace(0, 1, 63, dtype=np.float32).reshape(7, 9)
        self.assertFalse(value.flags.aligned)
        for function in (box_blur, thermal_colormap):
            expected = with_backend("numpy", function, value)
            actual = with_backend("rust", function, value)
            np.testing.assert_allclose(actual, expected, atol=8e-7, rtol=0)
        native = with_backend("rust", native_kernels)
        with self.assertRaisesRegex(ValueError, "aligned"):
            native.box_blur(value, 3)
        with self.assertRaisesRegex(ValueError, "aligned"):
            native.palette(value, [0, 1], [(0, 0, 0), (255, 255, 255)], [1])

    def test_complete_fusion_parity_and_exact_evidence_gates(self):
        frames = [synthetic_frame(seed=i) for i in range(4)] + [
            synthetic_frame(synchronized=False), synthetic_frame(uncertainty=0.1),
            synthetic_frame(flat=True),
        ]
        for frame in frames:
            for registrar in (IdentityRegistrar(), IdentityRegistrar(0.05), PhaseCorrelationRegistrar(8)):
                engine = EvidenceFusionEngine(registrar)
                for gain in (0.0, 1.0, 2.5):
                    config = FusionConfig(thermal_gain=gain)
                    reference = with_backend("numpy", engine.fuse, frame, config)
                    actual = with_backend("rust", engine.fuse, frame, config)
                    self.assertEqual(actual.channel_names, reference.channel_names)
                    self.assertEqual(actual.pixel_fusion_applied, reference.pixel_fusion_applied)
                    self.assertEqual(actual.synchronization_state, reference.synchronization_state)
                    self.assertEqual(actual.provenance["compute_backend"]["name"], "rust")
                    np.testing.assert_allclose(actual.machine_tensor, reference.machine_tensor, atol=2e-6, rtol=0)
                    np.testing.assert_allclose(actual.fusion_support, reference.fusion_support, atol=2e-6, rtol=0)
                    np.testing.assert_array_equal(actual.registration_support, reference.registration_support)
                    np.testing.assert_array_equal(actual.thermal_view, reference.thermal_view)
                    for name in ("operator_rgb", "semantic_view"):
                        difference = np.abs(getattr(actual, name).astype(int) - getattr(reference, name).astype(int))
                        self.assertLessEqual(int(difference.max()), 1)
                    for name in ("sensor_validity", "registration_support_score", "thermal_contribution"):
                        index = actual.channel_names.index(name)
                        zero = reference.machine_tensor[index] == 0
                        np.testing.assert_array_equal(actual.machine_tensor[index][zero], 0)
                    for name in frame.observations:
                        self.assertFalse(frame.observations[name].data.flags.writeable)


if __name__ == "__main__":
    unittest.main()
