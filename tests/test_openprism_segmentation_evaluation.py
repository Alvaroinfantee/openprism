from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from openprism.learning.segmentation_evaluation import (
    FrozenModelProvenance,
    SegmentationCase,
    evaluate_msrs_semantics,
    load_prediction_cases,
    paired_scene_group_comparison,
    summarize_latency_and_failures,
)


def _provenance(*, frozen: bool = True, checksum: bool = True) -> FrozenModelProvenance:
    return FrozenModelProvenance(
        model_id="frozen-msrs-probe-v1",
        artifact_sha256="a" * 64 if checksum else None,
        source="https://example.invalid/model-card",
        source_revision="unit-test-revision",
        preprocessing_id="rgbt-letterbox-v1",
        pretrained=True,
        frozen=frozen,
        training_data="MSRS train only",
    )


def _case(
    sample_id: str,
    group: str,
    truth: np.ndarray,
    prediction: np.ndarray | None,
    *,
    latency_ms: float | None = 10.0,
    failure_reason: str | None = None,
) -> SegmentationCase:
    return SegmentationCase(
        sample_id,
        group,
        truth,
        prediction,
        latency_ms=latency_ms,
        failure_reason=failure_reason,
    )


class MSRSSegmentationMetricTests(unittest.TestCase):
    def test_miou_and_per_class_iou_use_frozen_ignore_policy(self) -> None:
        truth = np.array([[0, 1, 1], [2, 2, 255]], dtype=np.uint8)
        prediction = np.array([[8, 1, 2], [2, 0, 7]], dtype=np.uint8)
        report = evaluate_msrs_semantics(
            [_case("frame-1", "scene-a", truth, prediction)],
            _provenance(),
            bootstrap_replicates=40,
        )
        metrics = report["metrics"]

        self.assertAlmostEqual(metrics["per_class"]["1"]["iou"], 0.5)
        self.assertAlmostEqual(metrics["per_class"]["2"]["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["mean_iou"], (0.5 + 1.0 / 3.0) / 2.0)
        self.assertIsNone(metrics["per_class"]["8"]["iou"])
        self.assertEqual(metrics["evaluated_pixels"], 4)
        self.assertEqual(report["evaluator"]["ignore_ground_truth_labels"], [0, 255])

    def test_scene_group_bootstrap_is_deterministic_and_reports_group_unit(self) -> None:
        truth = np.array([[1, 1], [2, 2]], dtype=np.uint8)
        cases = [
            _case("a-1", "scene-a", truth, truth),
            _case("a-2", "scene-a", truth, truth),
            _case("b-1", "scene-b", truth, np.zeros_like(truth)),
        ]
        first = evaluate_msrs_semantics(
            cases,
            _provenance(),
            bootstrap_replicates=300,
            bootstrap_seed=17,
        )
        second = evaluate_msrs_semantics(
            cases,
            _provenance(),
            bootstrap_replicates=300,
            bootstrap_seed=17,
        )
        interval = first["confidence_intervals_95"]

        self.assertEqual(interval, second["confidence_intervals_95"])
        self.assertEqual(interval["method"], "percentile_scene_group_bootstrap")
        self.assertEqual(interval["scene_group_count"], 2)
        self.assertLess(interval["mean_iou"]["lower"], interval["mean_iou"]["upper"])

    def test_invalid_taxonomy_label_is_rejected_instead_of_wrapped(self) -> None:
        truth = np.array([[1]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "outside the frozen taxonomy"):
            evaluate_msrs_semantics(
                [_case("bad", "scene", truth, np.array([[99]], dtype=np.int16))],
                _provenance(),
                bootstrap_replicates=10,
            )


class PairedStatisticsAndRuntimeTests(unittest.TestCase):
    def test_paired_group_bootstrap_reports_candidate_improvement(self) -> None:
        truth_a = np.array([[1, 1], [2, 2]], dtype=np.uint8)
        truth_b = np.array([[1, 2], [1, 2]], dtype=np.uint8)
        baseline = [
            _case("a", "scene-a", truth_a, np.zeros_like(truth_a)),
            _case("b", "scene-b", truth_b, np.zeros_like(truth_b)),
        ]
        candidate = [
            _case("a", "scene-a", truth_a, truth_a),
            _case("b", "scene-b", truth_b, truth_b),
        ]
        result = paired_scene_group_comparison(
            baseline, candidate, replicates=200, seed=9
        )

        self.assertAlmostEqual(result["mean_iou"]["delta"], 1.0)
        self.assertEqual(
            result["mean_iou"]["bootstrap_probability_candidate_better"], 1.0
        )
        self.assertEqual(result["complete_pair_scene_groups"], 2)
        self.assertEqual(result["incomplete_pairs"], 0)

    def test_paired_comparison_discloses_failures_and_requires_same_truth(self) -> None:
        truth = np.array([[1]], dtype=np.uint8)
        baseline = [_case("a", "scene", truth, truth)]
        failed = [
            _case(
                "a",
                "scene",
                truth,
                None,
                failure_reason="out_of_memory",
            )
        ]
        with self.assertRaisesRegex(ValueError, "no samples completed"):
            paired_scene_group_comparison(baseline, failed, replicates=10)

        changed_truth = [_case("a", "scene", np.array([[2]]), np.array([[2]]))]
        with self.assertRaisesRegex(ValueError, "ground_truth mismatch"):
            paired_scene_group_comparison(baseline, changed_truth, replicates=10)

    def test_latency_and_failures_have_quantiles_missingness_and_reasons(self) -> None:
        truth = np.array([[1]], dtype=np.uint8)
        cases = [
            _case("a", "g", truth, truth, latency_ms=10.0),
            _case("b", "g", truth, truth, latency_ms=30.0),
            _case(
                "c", "g", truth, None, latency_ms=50.0, failure_reason="timeout"
            ),
            _case(
                "d", "g", truth, None, latency_ms=None, failure_reason="timeout"
            ),
        ]
        summary = summarize_latency_and_failures(cases)

        self.assertEqual(summary["attempted"], 4)
        self.assertEqual(summary["failed"], 2)
        self.assertEqual(summary["failure_rate"], 0.5)
        self.assertEqual(summary["failures_by_reason"], {"timeout": 2})
        self.assertEqual(summary["latency_ms_all_attempts"]["recorded"], 3)
        self.assertAlmostEqual(summary["latency_ms_successful"]["p50"], 20.0)


class ProvenanceAndTestLockTests(unittest.TestCase):
    def test_final_test_requires_explicit_unlock_and_frozen_hashed_artifact(self) -> None:
        truth = np.array([[1]], dtype=np.uint8)
        cases = [_case("a", "scene", truth, truth)]
        with self.assertRaisesRegex(ValueError, "final test is locked"):
            evaluate_msrs_semantics(
                cases, _provenance(), partition="test", bootstrap_replicates=10
            )
        with self.assertRaisesRegex(ValueError, "frozen model provenance"):
            evaluate_msrs_semantics(
                cases,
                _provenance(frozen=False),
                partition="test",
                unlock_final_test=True,
                bootstrap_replicates=10,
            )
        report = evaluate_msrs_semantics(
            cases,
            _provenance(),
            partition="test",
            unlock_final_test=True,
            bootstrap_replicates=10,
        )
        self.assertTrue(report["final_test_unlocked"])
        self.assertEqual(report["model_provenance"]["artifact_sha256"], "a" * 64)

    def test_model_artifact_hash_is_computed_and_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "weights.bin"
            artifact.write_bytes(b"frozen pretrained weights")
            provenance = FrozenModelProvenance.from_artifact(
                artifact,
                model_id="model",
                source="publisher",
                source_revision="tag-v1",
                preprocessing_id="pre-v1",
                pretrained=True,
            )
        self.assertEqual(
            provenance.artifact_sha256,
            hashlib.sha256(b"frozen pretrained weights").hexdigest(),
        )
        self.assertTrue(provenance.frozen)
        self.assertTrue(provenance.pretrained)

    def test_prediction_loader_turns_missing_output_into_a_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth_dir = root / "truth"
            prediction_dir = root / "prediction"
            truth_dir.mkdir()
            prediction_dir.mkdir()
            Image.fromarray(np.array([[1]], dtype=np.uint8)).save(truth_dir / "a.png")
            Image.fromarray(np.array([[2]], dtype=np.uint8)).save(truth_dir / "b.png")
            Image.fromarray(np.array([[1]], dtype=np.uint8)).save(prediction_dir / "a.png")

            cases = load_prediction_cases(
                truth_dir,
                prediction_dir,
                {"a": "scene-1", "b": "scene-2"},
            )
        self.assertIsNotNone(cases[0].prediction)
        self.assertIsNone(cases[1].prediction)
        self.assertEqual(cases[1].failure_reason, "missing_prediction")


if __name__ == "__main__":
    unittest.main()
