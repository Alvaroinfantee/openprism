from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

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
    from openprism.learning.data import (
        _caltech_partition,
        _llvip_test_partition,
        _msrs_train_partition,
        msrs_scene_group,
    )


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

    def test_preregistered_model_ablations_are_explicit(self) -> None:
        from openprism.learning import EGTCF, EGTCFConfig

        rgb = torch.rand(2, 3, 32, 32)
        thermal = torch.rand(2, 1, 32, 32)
        evidence = torch.ones(2, 3, 32, 32)
        evidence[0] = 0.0

        no_task = EGTCF(
            EGTCFConfig(
                base_channels=8,
                dropout=0.0,
                pose_features=0,
                use_task_conditioning=False,
            )
        ).eval()
        repeated_rgb = rgb[:1].repeat(2, 1, 1, 1)
        repeated_thermal = thermal[:1].repeat(2, 1, 1, 1)
        repeated_evidence = torch.ones(2, 3, 32, 32)
        no_task_output = no_task(
            repeated_rgb,
            repeated_thermal,
            repeated_evidence,
            task_ids=torch.tensor([0, 2]),
        )
        self.assertTrue(
            torch.allclose(
                no_task_output.fused_luminance[0],
                no_task_output.fused_luminance[1],
            )
        )

        no_abstention = EGTCF(
            EGTCFConfig(
                base_channels=8,
                dropout=0.0,
                pose_features=0,
                use_learned_abstention=False,
            )
        ).eval()
        supported = no_abstention(
            rgb[1:], thermal[1:], evidence[1:], task_ids=torch.tensor([1])
        )
        self.assertTrue(bool(torch.all(supported.abstention == 0.0)))

        soft = EGTCF(
            EGTCFConfig(
                base_channels=8,
                dropout=0.0,
                pose_features=0,
                hard_evidence_envelope=False,
            )
        ).eval()
        soft_output = soft(rgb[:1], thermal[:1], evidence[:1])
        self.assertGreater(
            float(soft_output.thermal_contribution.detach().max()), 0.0
        )
        self.assertFalse(
            soft.invariant_report(soft_output)[
                "thermal_contribution_bounded_by_evidence"
            ]
        )

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
        self.assertAlmostEqual(
            perfect["generalized_risk_coverage_area"], 0.125
        )
        self.assertAlmostEqual(
            reversed_scores["generalized_risk_coverage_area"], 0.375
        )
        self.assertAlmostEqual(
            perfect["oracle_generalized_risk_coverage_area"], 0.125
        )
        self.assertAlmostEqual(
            perfect["random_order_expected_generalized_risk_coverage_area"],
            0.25,
        )

    def test_selective_diagnostic_curves_are_complete_and_plot_ready(self) -> None:
        from openprism.learning.evaluation import selective_diagnostic_curves

        score = torch.tensor([0.0, 0.1, 0.4, 0.8, 1.0])
        risk = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0])
        report = selective_diagnostic_curves(
            score, risk, bins=5, coverage_points=5
        )

        self.assertEqual(report["sampled_pixels"], 5)
        self.assertEqual(
            sum(item["count"] for item in report["reliability_bins"]), 5
        )
        curve = report["risk_coverage_curve"]
        self.assertEqual(len(curve), 5)
        self.assertEqual(curve[-1]["realized_coverage"], 1.0)
        self.assertAlmostEqual(curve[-1]["conditional_risk"], 0.6)
        self.assertLessEqual(
            curve[0]["oracle_conditional_risk"], curve[0]["conditional_risk"]
        )

    def test_selective_aurc_never_splits_equal_score_blocks(self) -> None:
        from openprism.learning.evaluation import (
            selective_diagnostic_curves,
            selective_metrics,
        )

        score = torch.tensor([0.0, 0.0, 1.0, 1.0])
        risk = torch.tensor([0.0, 1.0, 0.0, 1.0])
        metrics = selective_metrics(score, risk)
        diagnostics = selective_diagnostic_curves(
            score, risk, bins=2, coverage_points=4
        )

        # Both two-pixel tie blocks enter at once: 0.5*0.5 + 0.5*0.5.
        self.assertAlmostEqual(metrics["risk_coverage_area"], 0.5)
        # AUGRC trapezoids complete tied blocks: (0,0), (0.5,0.25), (1,0.5).
        self.assertAlmostEqual(metrics["generalized_risk_coverage_area"], 0.25)
        self.assertAlmostEqual(
            metrics["realized_coverage_at_80_percent_requested_coverage"], 1.0
        )
        self.assertAlmostEqual(
            metrics["selective_risk_at_80_percent_requested_coverage"], 0.5
        )
        realized = [
            item["realized_coverage"]
            for item in diagnostics["risk_coverage_curve"]
        ]
        self.assertEqual(realized, [0.5, 0.5, 1.0, 1.0])
        self.assertIn("every equal-score pixel", diagnostics["risk_coverage_tie_policy"])

    def test_model_evaluation_emits_pre_specified_score_comparators(self) -> None:
        from torch.utils.data import DataLoader

        from openprism.learning import EGTCFLoss
        from openprism.learning.evaluation import evaluate_model_pass

        examples = []
        for index in range(2):
            examples.append(
                {
                    "rgb": torch.rand(3, 32, 32),
                    "thermal": torch.rand(1, 32, 32),
                    "evidence": torch.ones(3, 32, 32),
                    "task_id": torch.tensor(index, dtype=torch.long),
                    "corruption_target": torch.zeros(1, 32, 32),
                    "sample_id": f"sample-{index}",
                    "dataset": "synthetic",
                    "scene_group": f"group-{index}",
                }
            )

        report = evaluate_model_pass(
            self.model,
            DataLoader(examples, batch_size=2, shuffle=False),
            EGTCFLoss(),
            torch.device("cpu"),
            bootstrap_replicates=16,
            spatial_stride=8,
        )

        self.assertEqual(report["runtime_and_failures"]["successful_examples"], 2)
        self.assertEqual(
            set(report["failure_score_comparators"]),
            {
                "evidence_insufficiency",
                "visible_thermal_disagreement",
                "learned_abstention",
            },
        )
        self.assertEqual(
            report["failure_score_metrics_pixel_weighted"]["samples"], 32.0
        )
        self.assertEqual(
            report["runtime_and_failures"]["latency"]["recorded_batches"], 1
        )

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

    def test_detection_automatic_views_use_and_cache_automatic_task(self) -> None:
        from openprism.learning.detection_evaluation import _view_tensor

        class RecordingEngine:
            def __init__(self) -> None:
                self.tasks: list[str] = []

            def fuse(self, frame, *, task: str):
                self.tasks.append(task)
                luminance = np.full((4, 5), 0.4, dtype=np.float32)
                return SimpleNamespace(
                    operator_rgb=np.full((4, 5, 3), 102, dtype=np.uint8),
                    machine_tensor=np.stack((luminance,)),
                    channel_names=("learned_fused_luminance",),
                )

        frame = SimpleNamespace(
            observations={
                "visible": SimpleNamespace(data=np.zeros((4, 5, 3), dtype=np.uint8)),
                "thermal": SimpleNamespace(
                    data=np.arange(20, dtype=np.uint8).reshape(4, 5)
                ),
            },
            provenance={"sample_id": "sample"},
        )
        engine = RecordingEngine()
        cache: dict[str, object] = {}
        for view in (
            "prism_egt_operator",
            "prism_egt_luminance",
            "prism_egt_operator_automatic",
            "prism_egt_luminance_automatic",
        ):
            result = _view_tensor(
                view,
                frame,
                learned_engine=engine,
                external_fused={},
                cache=cache,
            )
            self.assertEqual(tuple(result.shape), (3, 4, 5))
        self.assertEqual(engine.tasks, ["search", "automatic"])

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

    def test_msrs_contiguous_blocks_never_cross_partitions(self) -> None:
        self.assertEqual(msrs_scene_group("00301D"), msrs_scene_group("00399D"))
        self.assertEqual(
            _msrs_train_partition("00301D"), _msrs_train_partition("00399D")
        )
        self.assertNotEqual(msrs_scene_group("00399D"), msrs_scene_group("00400D"))
        with self.assertRaisesRegex(ValueError, "unsupported MSRS"):
            msrs_scene_group("frame")


if __name__ == "__main__":
    unittest.main()
