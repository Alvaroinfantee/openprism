from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from openprism.autonomy import (
    AdaptiveFusionController,
    DIGEST_SCHEMA_VERSION,
    FusionPolicyModel,
)
from openprism.contracts import (
    Detection,
    PrismFrame,
    SensorObservation,
    SynchronizationStatus,
    Timestamp,
)
from openprism.fusion import EvidenceFusionEngine, FusionConfig


def frame(*, synchronized: bool = True, detections: bool = True) -> PrismFrame:
    timestamp = Timestamp(1_000_000, clock_id="test-clock", uncertainty_ns=100)
    visible = np.zeros((16, 16, 3), dtype=np.uint8)
    visible[:, 8:] = 80
    thermal = np.zeros((16, 16), dtype=np.uint8)
    thermal[4:12, 4:12] = 255
    observations = {
        "visible": SensorObservation(
            sensor_id="visible",
            modality="visible_rgb",
            frame_id="visible_optical",
            timestamp=timestamp,
            data=visible,
            encoding="rgb8",
        ),
        "thermal": SensorObservation(
            sensor_id="thermal",
            modality="thermal_lwir_8",
            frame_id="thermal_optical",
            timestamp=timestamp,
            data=thermal,
            encoding="mono8",
        ),
    }
    synchronization = (
        SynchronizationStatus(
            state="exact",
            pixel_fusion_eligible=True,
            basis="measured",
            clock_domain="test-clock",
            measured_max_skew_ns=0,
            effective_max_skew_ns=200,
            physical_timing_uncertainty_ns=200,
        )
        if synchronized
        else SynchronizationStatus()
    )
    declared_detections = (
        (
            Detection(
                label="person",
                confidence=0.9,
                x=0.25,
                y=0.25,
                width=0.25,
                height=0.5,
                source="test_model",
            ),
        )
        if detections
        else ()
    )
    return PrismFrame(
        frame_id="autonomy-test",
        timestamp=timestamp,
        reference_frame="visible_optical",
        observations=observations,
        detections=declared_detections,
        synchronization=synchronization,
    )


class FusionPolicyTests(unittest.TestCase):
    def test_bundled_policy_is_explicitly_not_claimed_as_trained(self) -> None:
        policy = FusionPolicyModel.bundled()
        self.assertEqual(policy.training_provenance, "expert_initialized_not_fitted")
        self.assertEqual(len(policy.artifact_sha256), 64)
        self.assertEqual(set(policy.preset_order), {"navigate", "search", "terrain", "integrity"})

    def test_invalid_policy_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps({"schema_version": "wrong"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported fusion policy schema"):
                FusionPolicyModel.from_json(path)

    def test_controller_recommends_and_emits_json_safe_digest(self) -> None:
        engine = EvidenceFusionEngine()
        controller = AdaptiveFusionController()
        source = frame()
        recommendation, features, _ = controller.recommend(source, engine)
        self.assertEqual(recommendation.status, "recommended")
        self.assertIn(recommendation.operator_preset, {"navigate", "search", "terrain"})
        self.assertGreater(recommendation.thermal_gain, 0.0)
        output = engine.fuse(source, FusionConfig(thermal_gain=recommendation.thermal_gain))
        digest = controller.scene_digest(
            source,
            output,
            recommendation,
            features,
            automatic_control=True,
            applied_thermal_gain=recommendation.thermal_gain,
        )
        self.assertEqual(digest["schema_version"], DIGEST_SCHEMA_VERSION)
        self.assertEqual(digest["control"]["mode"], "automatic")
        self.assertFalse(
            digest["safety_contract"]["model_may_override_hard_evidence_gates"]
        )
        self.assertEqual(
            digest["machine_projection"]["shape_chw"], list(output.machine_tensor.shape)
        )
        self.assertEqual(
            set(digest["machine_projection"]["channel_statistics"]),
            set(output.channel_names),
        )
        json.dumps(digest, allow_nan=False)

    def test_unsynchronized_scene_forces_integrity_and_zero_gain(self) -> None:
        engine = EvidenceFusionEngine()
        recommendation, _, _ = AdaptiveFusionController().recommend(
            frame(synchronized=False), engine
        )
        self.assertEqual(recommendation.operator_preset, "integrity")
        self.assertEqual(recommendation.thermal_gain, 0.0)
        self.assertEqual(recommendation.status, "safety_override")


if __name__ == "__main__":
    unittest.main()
