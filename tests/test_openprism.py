from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from openprism.contracts import (
    PrismFrame,
    SensorObservation,
    SynchronizationStatus,
    Timestamp,
)
from openprism.datasets import DatasetCatalog
from openprism.fusion import EvidenceFusionEngine, FusionConfig
from openprism.registration import PhaseCorrelationRegistrar
from openprism.server import OperatorApplication
from openprism.synchronization import SynchronizationPolicy, WatermarkSynchronizer


REPO_ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_unknown_time_is_explicit_not_a_numeric_sentinel(self) -> None:
        timestamp = Timestamp(
            tai_ns=None,
            clock_id="archive",
            uncertainty_ns=None,
        )
        self.assertIsNone(timestamp.tai_ns)
        self.assertIsNone(timestamp.uncertainty_ns)
        with self.assertRaises(ValueError):
            Timestamp(tai_ns=None, clock_id="archive", uncertainty_ns=0)

    def test_measurement_uncertainty_must_be_finite_and_non_negative(self) -> None:
        with self.assertRaises(ValueError):
            SensorObservation(
                sensor_id="thermal",
                modality="thermal_lwir_8",
                frame_id="thermal_optical",
                timestamp=Timestamp(0),
                data=np.zeros((4, 5), dtype=np.uint8),
                encoding="mono8",
                uncertainty=-0.1,
            )

    def test_observation_owns_immutable_storage(self) -> None:
        source = np.zeros((8, 9), dtype=np.uint8)
        observation = SensorObservation(
            sensor_id="thermal",
            modality="thermal_lwir",
            frame_id="thermal_optical",
            timestamp=Timestamp(0),
            data=source,
            encoding="mono8",
            metadata={"nested": {"value": 3}},
        )
        source[0, 0] = 255
        self.assertEqual(int(observation.data[0, 0]), 0)
        with self.assertRaises(ValueError):
            observation.data[0, 0] = 1
        with self.assertRaises(TypeError):
            observation.metadata["nested"]["value"] = 4

    def test_semantic_mask_resolves_explicit_reference(self) -> None:
        timestamp = Timestamp(0)
        thermal = SensorObservation(
            sensor_id="thermal",
            modality="thermal_lwir",
            frame_id="thermal_optical",
            timestamp=timestamp,
            data=np.zeros((4, 5), dtype=np.uint8),
            encoding="mono8",
        )
        visible = SensorObservation(
            sensor_id="visible",
            modality="visible_rgb",
            frame_id="visible_optical",
            timestamp=timestamp,
            data=np.zeros((6, 7, 3), dtype=np.uint8),
            encoding="rgb8",
        )
        frame = PrismFrame(
            frame_id="synthetic",
            timestamp=timestamp,
            reference_frame="visible_optical",
            observations={"thermal": thermal, "visible": visible},
            semantic_mask=np.zeros((6, 7), dtype=np.uint8),
        )
        self.assertEqual(frame.semantic_mask.shape, (6, 7))

    def test_channel_validity_reduces_conservatively_to_image_geometry(self) -> None:
        validity = np.ones((8, 9, 3), dtype=bool)
        validity[2, 4, 1] = False
        observation = SensorObservation(
            sensor_id="visible",
            modality="visible_rgb",
            frame_id="visible_optical",
            timestamp=Timestamp(0),
            data=np.zeros((8, 9, 3), dtype=np.uint8),
            encoding="rgb8",
            validity=validity,
        )
        self.assertEqual(observation.validity.shape, (8, 9))
        self.assertFalse(bool(observation.validity[2, 4]))


class RegistrationTests(unittest.TestCase):
    def test_known_integer_shift_and_sign(self) -> None:
        reference = np.zeros((96, 128), dtype=np.float32)
        reference[15:50, 22:30] = 1.0
        reference[58:72, 60:108] = 0.7
        reference[25:31, 74:91] = 0.4
        moving = np.roll(reference, shift=(7, -11), axis=(0, 1))
        result = PhaseCorrelationRegistrar(max_shift_px=20).register(reference, moving)
        self.assertEqual(result.diagnostics["dy_px"], -7.0)
        self.assertEqual(result.diagnostics["dx_px"], 11.0)
        self.assertGreater(float(np.mean(result.confidence)), 0.05)
        np.testing.assert_allclose(
            result.aligned[result.validity], reference[result.validity], atol=1e-5
        )

    def test_low_texture_abstains(self) -> None:
        flat = np.full((64, 64), 0.5, dtype=np.float32)
        result = PhaseCorrelationRegistrar(max_shift_px=10).register(flat, flat)
        self.assertEqual(result.method, "edge_phase_correlation_abstained")
        self.assertEqual(float(result.confidence.max()), 0.0)


class SynchronizationTests(unittest.TestCase):
    @staticmethod
    def observation(
        sensor_id: str,
        tai_ns: int,
        *,
        clock_id: str = "ptp-domain-0",
        uncertainty_ns: int = 100,
    ) -> SensorObservation:
        return SensorObservation(
            sensor_id=sensor_id,
            modality="imu" if sensor_id == "imu" else "visible_rgb",
            frame_id=f"{sensor_id}_frame",
            timestamp=Timestamp(
                tai_ns,
                clock_id=clock_id,
                uncertainty_ns=uncertainty_ns,
            ),
            data=(
                np.zeros(6, dtype=np.float32)
                if sensor_id == "imu"
                else np.zeros((4, 5, 3), dtype=np.uint8)
            ),
            encoding="float32x6" if sensor_id == "imu" else "rgb8",
        )

    def test_exact_and_delayed_sensor_gating(self) -> None:
        synchronizer = WatermarkSynchronizer(
            SynchronizationPolicy(
                required_sensor_ids=("rgb", "thermal"),
                exact_tolerance_ns=1_000,
                pixel_fusion_tolerance_ns=5_000,
                max_staleness_ns=20_000,
            )
        )
        synchronizer.push(self.observation("rgb", 100_000))
        missing = synchronizer.assemble("rgb")
        self.assertEqual(missing.states["thermal"], "missing")
        self.assertFalse(missing.pixel_fusion_eligible)

        synchronizer.push(self.observation("thermal", 97_000))
        eligible = synchronizer.assemble("rgb")
        self.assertEqual(eligible.states["thermal"], "late")
        self.assertEqual(eligible.max_skew_ns, 3_000)
        self.assertTrue(eligible.pixel_fusion_eligible)

        synchronizer.push(self.observation("rgb", 130_000))
        stale = synchronizer.assemble("rgb")
        self.assertEqual(stale.states["thermal"], "missing")
        self.assertFalse(stale.pixel_fusion_eligible)

    def test_auxiliary_one_dimensional_sensor_is_supported(self) -> None:
        imu = self.observation("imu", 42)
        self.assertEqual(imu.data.shape, (6,))
        with self.assertRaises(AttributeError):
            _ = imu.width

    def test_clock_domain_mismatch_is_never_compared_or_fused(self) -> None:
        synchronizer = WatermarkSynchronizer(
            SynchronizationPolicy(
                required_sensor_ids=("rgb", "thermal"),
                exact_tolerance_ns=1_000,
                pixel_fusion_tolerance_ns=5_000,
                max_staleness_ns=20_000,
            )
        )
        synchronizer.push(self.observation("rgb", 100_000, clock_id="ptp-a"))
        synchronizer.push(self.observation("thermal", 100_000, clock_id="ptp-b"))
        selection = synchronizer.assemble("rgb")
        self.assertEqual(selection.states["thermal"], "incompatible_clock_domain")
        self.assertEqual(selection.synchronization.state, "incompatible_clock_domain")
        self.assertFalse(selection.pixel_fusion_eligible)
        self.assertIsNone(selection.skew_ns["thermal"])

    def test_timestamp_uncertainty_participates_in_fusion_gate(self) -> None:
        synchronizer = WatermarkSynchronizer(
            SynchronizationPolicy(
                required_sensor_ids=("rgb", "thermal"),
                exact_tolerance_ns=100,
                pixel_fusion_tolerance_ns=500,
                max_staleness_ns=2_000,
            )
        )
        synchronizer.push(
            self.observation("rgb", 100_000, uncertainty_ns=300)
        )
        synchronizer.push(
            self.observation("thermal", 100_000, uncertainty_ns=300)
        )
        selection = synchronizer.assemble("rgb")
        self.assertEqual(selection.max_skew_ns, 0)
        self.assertEqual(selection.max_effective_skew_ns, 600)
        self.assertFalse(selection.pixel_fusion_eligible)
        self.assertEqual(selection.synchronization.state, "unsynchronized")

    def test_nearest_future_sample_is_truthfully_labeled(self) -> None:
        synchronizer = WatermarkSynchronizer(
            SynchronizationPolicy(
                required_sensor_ids=("rgb", "thermal"),
                exact_tolerance_ns=0,
                pixel_fusion_tolerance_ns=5_000,
                max_staleness_ns=20_000,
            )
        )
        synchronizer.push(self.observation("rgb", 100_000, uncertainty_ns=0))
        synchronizer.push(
            self.observation("thermal", 103_000, uncertainty_ns=0)
        )
        selection = synchronizer.assemble("rgb")
        self.assertEqual(selection.states["thermal"], "future")
        self.assertEqual(selection.synchronization.state, "bounded_skew")
        self.assertTrue(selection.pixel_fusion_eligible)

    def test_unknown_candidate_time_is_preserved_but_never_fused(self) -> None:
        synchronizer = WatermarkSynchronizer(
            SynchronizationPolicy(required_sensor_ids=("rgb", "thermal"))
        )
        synchronizer.push(self.observation("rgb", 100_000))
        synchronizer.push(
            SensorObservation(
                sensor_id="thermal",
                modality="thermal_lwir_8",
                frame_id="thermal_frame",
                timestamp=Timestamp(
                    tai_ns=None,
                    clock_id="ptp-domain-0",
                    uncertainty_ns=None,
                ),
                data=np.zeros((4, 5), dtype=np.uint8),
                encoding="mono8",
            )
        )
        selection = synchronizer.assemble("rgb")
        self.assertEqual(selection.states["thermal"], "unknown")
        self.assertFalse(selection.pixel_fusion_eligible)
        self.assertEqual(selection.synchronization.state, "unsynchronized")


class FusionSafetyTests(unittest.TestCase):
    @staticmethod
    def observations() -> dict[str, SensorObservation]:
        timestamp = Timestamp(1_000, clock_id="test-domain")
        visible_validity = np.ones((8, 9, 3), dtype=bool)
        visible_validity[1, 2, 0] = False
        thermal_validity = np.ones((8, 9), dtype=bool)
        thermal_validity[3, 4] = False
        radiometric_validity = np.ones((8, 9), dtype=bool)
        radiometric_validity[5, 6] = False
        return {
            "visible": SensorObservation(
                sensor_id="visible",
                modality="visible_rgb",
                frame_id="visible_optical",
                timestamp=timestamp,
                data=np.full((8, 9, 3), 96, dtype=np.uint8),
                encoding="rgb8",
                validity=visible_validity,
            ),
            "thermal8": SensorObservation(
                sensor_id="thermal8",
                modality="thermal_lwir_8",
                frame_id="thermal_optical",
                timestamp=timestamp,
                data=np.arange(72, dtype=np.uint8).reshape(8, 9),
                encoding="mono8",
                validity=thermal_validity,
            ),
            "thermal16": SensorObservation(
                sensor_id="thermal16",
                modality="thermal_lwir_16",
                frame_id="thermal_optical",
                timestamp=timestamp,
                data=(np.arange(72, dtype=np.uint16).reshape(8, 9) * 100),
                encoding="mono16",
                validity=radiometric_validity,
            ),
        }

    def test_unknown_synchronization_enters_no_fusion_zone(self) -> None:
        observations = self.observations()
        frame = PrismFrame(
            frame_id="unsynchronized",
            timestamp=observations["visible"].timestamp,
            reference_frame="visible_optical",
            observations=observations,
        )
        output = EvidenceFusionEngine().fuse(frame)
        thermal_index = output.channel_names.index("thermal_radiometric_norm")
        contribution_index = output.channel_names.index("thermal_contribution")
        self.assertFalse(output.pixel_fusion_applied)
        self.assertEqual(output.provenance["fusion_mode"], "no_fusion_zone")
        self.assertEqual(float(output.machine_tensor[thermal_index].max()), 0.0)
        self.assertEqual(float(output.machine_tensor[contribution_index].max()), 0.0)

    def test_all_sensor_validity_masks_reach_machine_projection(self) -> None:
        observations = self.observations()
        synchronization = SynchronizationStatus.declared_replay_aligned(
            tuple(observations),
            clock_domain="test-domain",
            declaration="test fixture",
        )
        frame = PrismFrame(
            frame_id="declared-aligned",
            timestamp=observations["visible"].timestamp,
            reference_frame="visible_optical",
            observations=observations,
            synchronization=synchronization,
        )
        output = EvidenceFusionEngine().fuse(frame)
        validity = output.machine_tensor[
            output.channel_names.index("sensor_validity")
        ]
        self.assertEqual(float(validity[1, 2]), 0.0)
        self.assertEqual(float(validity[3, 4]), 0.0)
        self.assertEqual(float(validity[5, 6]), 0.0)
        self.assertTrue(output.pixel_fusion_applied)

    def test_zero_thermal_gain_means_zero_thermal_contribution(self) -> None:
        observations = self.observations()
        frame = PrismFrame(
            frame_id="zero-thermal-gain",
            timestamp=observations["visible"].timestamp,
            reference_frame="visible_optical",
            observations=observations,
            synchronization=SynchronizationStatus.declared_replay_aligned(
                tuple(observations),
                clock_domain="test-domain",
                declaration="test fixture",
            ),
        )
        output = EvidenceFusionEngine().fuse(frame, FusionConfig(thermal_gain=0.0))
        contribution = output.machine_tensor[
            output.channel_names.index("thermal_contribution")
        ]
        self.assertEqual(float(np.max(contribution)), 0.0)
        self.assertFalse(output.pixel_fusion_applied)

    def test_unmodeled_measurement_uncertainty_forces_safe_fallback(self) -> None:
        observations = self.observations()
        thermal = observations["thermal8"]
        observations["thermal8"] = SensorObservation(
            sensor_id=thermal.sensor_id,
            modality=thermal.modality,
            frame_id=thermal.frame_id,
            timestamp=thermal.timestamp,
            data=thermal.data,
            encoding=thermal.encoding,
            validity=thermal.validity,
            uncertainty=0.25,
        )
        frame = PrismFrame(
            frame_id="uncertainty-unmodeled",
            timestamp=observations["visible"].timestamp,
            reference_frame="visible_optical",
            observations=observations,
            synchronization=SynchronizationStatus.declared_replay_aligned(
                tuple(observations),
                clock_domain="test-domain",
                declaration="test fixture",
            ),
        )
        output = EvidenceFusionEngine().fuse(frame)
        self.assertFalse(output.pixel_fusion_applied)
        self.assertEqual(
            output.provenance["fallback_reason"],
            "measurement_uncertainty_model_unavailable",
        )
        self.assertIn(
            "thermal8",
            output.provenance["unmodeled_measurement_uncertainty_sensors"],
        )


@unittest.skipUnless(
    (REPO_ROOT / "data" / "LLVIP").is_dir()
    and (REPO_ROOT / "data" / "MSRS").is_dir()
    and (REPO_ROOT / "data" / "Caltech_Aerial_RGBT").is_dir(),
    "requires separately downloaded LLVIP, MSRS, and Caltech Aerial RGB-T archives",
)
class DatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = DatasetCatalog(REPO_ROOT / "data")
        cls.engine = EvidenceFusionEngine()

    def test_catalog_counts(self) -> None:
        self.assertEqual(self.catalog.count("llvip", "train"), 12025)
        self.assertEqual(self.catalog.count("llvip", "test"), 3463)
        self.assertEqual(self.catalog.count("msrs", "train"), 1083)
        self.assertEqual(self.catalog.count("msrs", "test"), 361)
        self.assertEqual(self.catalog.count("msrs", "detection"), 80)
        self.assertEqual(self.catalog.count("caltech", "all"), 2282)

    def test_content_sniff_and_thermal16_preservation(self) -> None:
        frame = self.catalog.load("caltech", "all", 0)
        self.assertEqual(frame.observations["visible"].data.shape, (600, 960, 3))
        self.assertEqual(frame.observations["visible"].data.dtype, np.uint8)
        self.assertEqual(frame.observations["thermal16"].data.dtype, np.uint16)
        self.assertEqual(frame.observations["thermal16"].units, "raw_sensor_count")
        self.assertIsNotNone(frame.semantic_mask)
        self.assertIsNone(frame.timestamp.tai_ns)
        self.assertIsNone(frame.timestamp.uncertainty_ns)
        self.assertEqual(frame.synchronization.state, "declared_replay_aligned")
        self.assertEqual(frame.synchronization.basis, "declared")
        self.assertIsNone(frame.synchronization.physical_timing_uncertainty_ns)

    def test_ground_truth_is_attributed(self) -> None:
        llvip = self.catalog.load("llvip", "train", 0)
        msrs = self.catalog.load("msrs", "detection", 0)
        self.assertGreater(len(llvip.detections), 0)
        self.assertGreater(len(msrs.detections), 0)
        self.assertTrue(
            all(item.source == "dataset_ground_truth" for item in llvip.detections)
        )
        self.assertTrue(
            all(item.source == "dataset_ground_truth" for item in msrs.detections)
        )

    def test_fusion_is_bounded_deterministic_and_dual_rail(self) -> None:
        for dataset, split in (
            ("llvip", "train"),
            ("msrs", "train"),
            ("caltech", "all"),
        ):
            frame = self.catalog.load(dataset, split, 0)
            first = self.engine.fuse(frame, FusionConfig(thermal_gain=1.0))
            second = self.engine.fuse(frame, FusionConfig(thermal_gain=1.0))
            self.assertEqual(first.machine_tensor.shape[0], len(first.channel_names))
            self.assertEqual(first.operator_rgb.shape[:2], first.machine_tensor.shape[1:])
            self.assertTrue(np.isfinite(first.machine_tensor).all())
            self.assertGreaterEqual(float(first.machine_tensor.min()), 0.0)
            self.assertLessEqual(float(first.machine_tensor.max()), 1.0)
            self.assertGreaterEqual(float(first.fusion_support.min()), 0.0)
            self.assertLessEqual(float(first.fusion_support.max()), 1.0)
            np.testing.assert_array_equal(first.operator_rgb, second.operator_rgb)
            np.testing.assert_array_equal(first.machine_tensor, second.machine_tensor)

    def test_operator_api_payload(self) -> None:
        application = OperatorApplication(REPO_ROOT / "data")
        payload = application.frame_payload("msrs", "detection", 0, 1.0)
        self.assertEqual(payload["meta"]["annotation_source"], "dataset_ground_truth")
        self.assertEqual(len(payload["meta"]["machine_channels"]), 11)
        self.assertIn(
            "registration_support_score", payload["meta"]["machine_channels"]
        )
        self.assertIsNone(payload["meta"]["registration_confidence"])
        self.assertEqual(
            payload["meta"]["registration_evidence_kind"],
            "publisher_declared_prior",
        )
        self.assertEqual(payload["meta"]["source_mode"], "dataset_replay")
        self.assertTrue(payload["meta"]["pixel_fusion_applied"])
        self.assertEqual(
            payload["meta"]["synchronization_state"],
            "declared_replay_aligned",
        )
        self.assertEqual(payload["meta"]["synchronization_basis"], "declared")
        self.assertIsNone(payload["meta"]["physical_timing_uncertainty_ns"])
        self.assertEqual(payload["meta"]["fusion_mode"], "pixel_fusion")
        self.assertTrue(payload["images"]["fused"].startswith("data:image/jpeg;base64,"))
        self.assertTrue(payload["images"]["support"].startswith("data:image/png;base64,"))
        self.assertGreater(len(payload["detections"]), 0)
        self.assertIsNone(payload["images"]["semantic"])
        self.assertEqual(payload["terrain_classes"], [])
        self.assertEqual(payload["ai"]["schema_version"], "openprism.ai-scene-digest/1.0")
        self.assertEqual(payload["ai"]["control"]["mode"], "manual")
        self.assertEqual(
            payload["ai"]["policy_model"]["training_provenance"],
            "expert_initialized_not_fitted",
        )

        automatic = application.frame_payload(
            "msrs", "detection", 0, 1.0, automatic_control=True
        )
        self.assertEqual(automatic["ai"]["control"]["mode"], "automatic")
        self.assertAlmostEqual(
            automatic["ai"]["control"]["applied_thermal_gain"],
            automatic["ai"]["control"]["recommendation"]["thermal_gain"],
        )

        context = application.ai_context_payload("msrs", "detection", 0)
        self.assertEqual(
            context["schema_version"], "openprism.ai-context-envelope/1.0"
        )
        self.assertNotIn("images", context)
        self.assertEqual(context["digest"]["frame_id"], payload["meta"]["frame_id"])

        terrain = application.frame_payload("caltech", "all", 0, 1.0)
        self.assertTrue(terrain["images"]["semantic"].startswith("data:image/jpeg;base64,"))
        self.assertGreater(len(terrain["terrain_classes"]), 0)
        self.assertAlmostEqual(
            sum(item["coverage"] for item in terrain["terrain_classes"]), 1.0
        )


class SchemaTests(unittest.TestCase):
    def test_schema_is_valid_json_with_required_contract_sections(self) -> None:
        schema_path = REPO_ROOT / "openprism" / "spec" / "prism-frame.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertIn("observations", schema["properties"])
        self.assertIn("quality", schema["properties"])
        self.assertIn("provenance", schema["properties"])
        modalities = schema["$defs"]["observation"]["properties"]["modality"][
            "oneOf"
        ][0]["enum"]
        self.assertIn("thermal_lwir_8", modalities)
        self.assertIn("thermal_lwir_16", modalities)
        matrix_variants = schema["$defs"]["transform"]["properties"]["matrix"][
            "oneOf"
        ]
        self.assertEqual(
            {(item["minItems"], item["maxItems"]) for item in matrix_variants},
            {(9, 9), (16, 16)},
        )
        pairing_states = schema["$defs"]["capture"]["properties"][
            "pairing_state"
        ]["enum"]
        self.assertIn("declared_replay_aligned", pairing_states)
        self.assertIn("incompatible_clock_domain", pairing_states)
        quality_required = schema["$defs"]["quality"]["required"]
        self.assertIn("time_basis", quality_required)
        self.assertIn("pixel_fusion_eligible", quality_required)


if __name__ == "__main__":
    unittest.main()
