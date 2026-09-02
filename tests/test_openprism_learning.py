from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency in minimal installs
    torch = None

from openprism.contracts import (
    PrismFrame,
    SensorObservation,
    SynchronizationStatus,
    Timestamp,
)
if torch is not None:
    from openprism.learning.data import _caltech_partition, _llvip_test_partition


@unittest.skipIf(torch is None, "PyTorch learning extra is not installed")
class LearnedFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        from openprism.learning import EGTCF, EGTCFConfig

        torch.manual_seed(7)
        self.model = EGTCF(EGTCFConfig(base_channels=8, dropout=0.0))

    def test_shapes_probabilities_and_evidence_bound(self) -> None:
        rgb = torch.rand(2, 3, 32, 40)
        thermal = torch.rand(2, 1, 32, 40)
        evidence = torch.rand(2, 3, 32, 40)
        result = self.model(
            rgb, thermal, evidence, task_ids=torch.tensor([0, 2])
        )
        self.assertEqual(result.fused_luminance.shape, (2, 1, 32, 40))
        self.assertEqual(result.task_logits.shape, (2, 3))
        self.assertTrue(
            torch.allclose(result.task_probabilities.sum(dim=1), torch.ones(2))
        )
        self.assertTrue(
            bool(torch.all(result.thermal_contribution <= result.evidence_support + 1e-7))
        )
        self.assertTrue(self.model.invariant_report(result)[
            "thermal_contribution_bounded_by_evidence"
        ])

    def test_invalid_pixels_force_abstention_and_zero_thermal(self) -> None:
        rgb = torch.rand(1, 3, 32, 32)
        thermal = torch.rand(1, 1, 32, 32)
        evidence = torch.ones(1, 3, 32, 32)
        evidence[:, 0, 8:20, 5:17] = 0.0
        result = self.model(rgb, thermal, evidence)
        region = (..., slice(8, 20), slice(5, 17))
        self.assertTrue(bool(torch.all(result.abstention[region] == 1.0)))
        self.assertTrue(bool(torch.all(result.thermal_contribution[region] == 0.0)))
        self.assertTrue(bool(torch.all(result.predictive_uncertainty[region] == 1.0)))

    def test_loss_is_finite_and_backpropagates(self) -> None:
        from openprism.learning import EGTCFLoss

        rgb = torch.rand(2, 3, 32, 32)
        thermal = torch.rand(2, 1, 32, 32)
        evidence = torch.ones(2, 3, 32, 32)
        tasks = torch.tensor([0, 1])
        result = self.model(rgb, thermal, evidence, task_ids=tasks)
        loss, components = EGTCFLoss()(result, rgb, thermal, tasks)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIn("calibration_brier", components)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in self.model.parameters()))

    def test_checkpoint_round_trip_preserves_output(self) -> None:
        from openprism.learning import load_checkpoint, save_checkpoint

        rgb = torch.rand(1, 3, 32, 32)
        thermal = torch.rand(1, 1, 32, 32)
        evidence = torch.ones(1, 3, 32, 32)
        self.model.eval()
        expected = self.model(rgb, thermal, evidence).fused_luminance
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            saved = save_checkpoint(
                path,
                self.model,
                model_id="unit-test",
                training_provenance="synthetic unit test",
                validation_scope="unit test only",
                epoch=1,
                metrics={"loss": 1.0},
            )
            loaded, metadata = load_checkpoint(path)
            actual = loaded(rgb, thermal, evidence).fused_luminance
            self.assertEqual(saved.artifact_sha256, metadata.artifact_sha256)
            self.assertTrue(torch.allclose(expected, actual))

    @staticmethod
    def _frame(*, synchronized: bool) -> PrismFrame:
        timestamp = Timestamp(1_000_000, clock_id="clock", uncertainty_ns=100)
        visible = np.zeros((32, 32, 3), dtype=np.uint8)
        visible[:, 16:] = 80
        thermal = np.zeros((32, 32), dtype=np.uint8)
        thermal[8:24, 8:24] = 255
        observations = {
            "visible": SensorObservation(
                "visible", "visible_rgb", "visible_optical", timestamp, visible, "rgb8"
            ),
            "thermal": SensorObservation(
                "thermal", "thermal_lwir_8", "thermal_optical", timestamp, thermal, "mono8"
            ),
        }
        status = (
            SynchronizationStatus(
                state="exact",
                pixel_fusion_eligible=True,
                basis="measured",
                clock_domain="clock",
                measured_max_skew_ns=0,
                effective_max_skew_ns=200,
                physical_timing_uncertainty_ns=200,
            )
            if synchronized
            else SynchronizationStatus()
        )
        return PrismFrame(
            "learned-test",
            timestamp,
            "visible_optical",
            observations,
            synchronization=status,
        )

    def test_runtime_engine_preserves_hard_gate_and_provenance(self) -> None:
        from openprism.learning import LearnedFusionEngine

        engine = LearnedFusionEngine(self.model, allow_unvalidated=True, device="cpu")
        rejected = engine.fuse(self._frame(synchronized=False), task="search")
        self.assertFalse(rejected.provenance["learned_fusion_applied"])
        self.assertFalse(rejected.pixel_fusion_applied)

        accepted = engine.fuse(self._frame(synchronized=True), task="search")
        self.assertTrue(accepted.provenance["learned_fusion_applied"])
        self.assertIn("learned_predictive_uncertainty", accepted.channel_names)
        self.assertFalse(
            accepted.provenance["safety_contract"][
                "model_may_override_hard_evidence_gates"
            ]
        )
        alpha = accepted.machine_tensor[
            accepted.channel_names.index("thermal_contribution")
        ]
        evidence = accepted.machine_tensor[
            accepted.channel_names.index("sensor_validity")
        ] * accepted.registration_support
        self.assertTrue(np.all(alpha <= evidence + 1e-6))

    def test_tiled_runtime_preserves_full_geometry_and_bound(self) -> None:
        from openprism.learning import LearnedFusionEngine

        engine = LearnedFusionEngine(
            self.model,
            allow_unvalidated=True,
            device="cpu",
            tile_size=64,
            tile_overlap=16,
        )
        source = self._frame(synchronized=True)
        # Expand the immutable observations into a frame larger than one tile.
        timestamp = source.timestamp
        visible = np.tile(source.observations["visible"].data, (3, 3, 1))
        thermal = np.tile(source.observations["thermal"].data, (3, 3))
        expanded = PrismFrame(
            "tiled-test",
            timestamp,
            "visible_optical",
            {
                "visible": SensorObservation(
                    "visible", "visible_rgb", "visible_optical", timestamp, visible, "rgb8"
                ),
                "thermal": SensorObservation(
                    "thermal", "thermal_lwir_8", "thermal_optical", timestamp, thermal, "mono8"
                ),
            },
            synchronization=source.synchronization,
        )
        output = engine.fuse(expanded, task="automatic")
        self.assertEqual(output.operator_rgb.shape, (96, 96, 3))
        self.assertTrue(output.provenance["tiled_inference"])
        alpha = output.machine_tensor[
            output.channel_names.index("thermal_contribution")
        ]
        self.assertTrue(np.all(alpha <= output.registration_support + 1e-6))

    def test_selective_metrics_reward_correct_uncertainty_ranking(self) -> None:
        from openprism.learning.evaluation import selective_metrics

        target = torch.tensor([0.0, 0.0, 1.0, 1.0])
        perfect = selective_metrics(torch.tensor([0.0, 0.1, 0.9, 1.0]), target)
        reversed_scores = selective_metrics(torch.tensor([1.0, 0.9, 0.1, 0.0]), target)
        self.assertEqual(perfect["uncertainty_auroc"], 1.0)
        self.assertEqual(reversed_scores["uncertainty_auroc"], 0.0)
        self.assertLess(perfect["brier"], reversed_scores["brier"])

    def test_baselines_are_bounded_and_preserve_invalid_visible_pixels(self) -> None:
        from openprism.learning import BASELINE_NAMES, fuse_baseline

        rgb = torch.rand(2, 3, 32, 32)
        thermal = torch.rand(2, 1, 32, 32)
        evidence = torch.ones(2, 3, 32, 32)
        evidence[:, 0, 8:16, 8:16] = 0.0
        visible = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        for name in BASELINE_NAMES:
            fused = fuse_baseline(name, rgb, thermal, evidence)
            self.assertTrue(bool(torch.all((0.0 <= fused) & (fused <= 1.0))))
            self.assertTrue(torch.allclose(fused[..., 8:16, 8:16], visible[..., 8:16, 8:16]))

    def test_detection_probe_metrics_reward_perfect_localization(self) -> None:
        from openprism.learning.detection_evaluation import detection_metrics

        truth = {
            "a": [[0.0, 0.0, 10.0, 10.0]],
            "b": [[5.0, 5.0, 15.0, 15.0]],
        }
        perfect = detection_metrics(
            truth,
            [
                ("a", 0.9, [0.0, 0.0, 10.0, 10.0]),
                ("b", 0.8, [5.0, 5.0, 15.0, 15.0]),
            ],
        )
        false_first = detection_metrics(
            truth,
            [
                ("a", 0.99, [20.0, 20.0, 30.0, 30.0]),
                ("a", 0.9, [0.0, 0.0, 10.0, 10.0]),
                ("b", 0.8, [5.0, 5.0, 15.0, 15.0]),
            ],
        )
        self.assertAlmostEqual(perfect["ap50"], 1.0)
        self.assertAlmostEqual(perfect["ap_50_95"], 1.0)
        self.assertLess(perfect["log_average_miss_rate"], 1e-8)
        self.assertLess(false_first["ap50"], perfect["ap50"])

    def test_detection_probe_keeps_final_test_locked(self) -> None:
        from openprism.learning.detection_evaluation import evaluate_llvip_detection

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "final test is locked"):
                evaluate_llvip_detection(
                    Path(directory),
                    Path(directory) / "report.json",
                    partition="test",
                    views=(),
                )

    def test_external_runner_minmax_is_finite_for_constant_input(self) -> None:
        from tooling.run_external_fusion import _LOADERS, _minmax

        constant = torch.full((1, 1, 4, 4), 7.0)
        self.assertTrue(torch.equal(_minmax(constant), torch.zeros_like(constant)))
        self.assertEqual(set(_LOADERS), {"seafusion", "cddfuse", "paif", "c2rf"})


@unittest.skipIf(torch is None, "PyTorch learning extra is not installed")
class ProtocolPartitionTests(unittest.TestCase):
    def test_llvip_sequences_never_cross_validation_and_test(self) -> None:
        validation = {prefix for prefix in ("19", "21", "23") if _llvip_test_partition(prefix + "0001") == "validation"}
        test = {prefix for prefix in ("20", "22", "24", "26") if _llvip_test_partition(prefix + "0001") == "test"}
        self.assertFalse(validation & test)

    def test_caltech_scene_group_has_one_stable_partition(self) -> None:
        name = "2022-05-15_ColoradoRiver_flight1"
        self.assertEqual(_caltech_partition(name), _caltech_partition(name))


if __name__ == "__main__":
    unittest.main()
