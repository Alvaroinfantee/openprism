from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from openprism.learning.caltech_segmentation_evaluation import (
    evaluate_caltech_semantics,
    paired_caltech_scene_group_comparison,
)
from openprism.learning.caltech_segmentation_probe import (
    CaltechProbeConfig,
    CaltechVisibleTerrainProbe,
    _save_checkpoint,
    _target_indices,
    evaluate_caltech_frozen_probe,
    load_caltech_probe_checkpoint,
    train_caltech_visible_probe,
)
from openprism.learning.data import _caltech_partition
from openprism.learning.segmentation_evaluation import (
    FrozenModelProvenance,
    SegmentationCase,
)


def _provenance() -> FrozenModelProvenance:
    return FrozenModelProvenance(
        "caltech-unit-probe",
        "c" * 64,
        "unit-test-source",
        "unit-test-revision",
        "unit-test-preprocessing",
        False,
        True,
        "Caltech protocol train only",
    )


def _case(
    sample_id: str,
    group: str,
    truth: np.ndarray,
    prediction: np.ndarray | None,
    failure: str | None = None,
) -> SegmentationCase:
    return SegmentationCase(
        sample_id,
        group,
        truth,
        prediction,
        latency_ms=5.0,
        failure_reason=failure,
    )


def _group_for(partition: str) -> str:
    for index in range(10_000):
        group = f"synthetic-flight-{index:04d}"
        if _caltech_partition(group) == partition:
            return group
    raise AssertionError(f"no group found for {partition}")


def _scaffold_other_datasets(root: Path) -> None:
    for split in ("train", "test"):
        (root / "LLVIP" / "visible" / split).mkdir(parents=True, exist_ok=True)
        (root / "LLVIP" / "infrared" / split).mkdir(parents=True, exist_ok=True)
        for directory in ("vi", "ir", "Segmentation_labels"):
            (root / "MSRS" / split / directory).mkdir(parents=True, exist_ok=True)
    for directory in ("vi", "ir", "labels"):
        (root / "MSRS" / "detection" / directory).mkdir(parents=True, exist_ok=True)
    for directory in ("color", "thermal8", "thermal16", "annotations"):
        (root / "Caltech_Aerial_RGBT" / directory).mkdir(parents=True, exist_ok=True)


def _write_caltech_sample(root: Path, group: str, frame_index: int, labels: np.ndarray) -> None:
    _scaffold_other_datasets(root)
    dataset = root / "Caltech_Aerial_RGBT"
    suffix = f"{frame_index:04d}"
    visible = np.zeros((*labels.shape, 3), dtype=np.uint8)
    visible[..., 0] = labels * 20
    visible[..., 1] = labels * 11
    visible[..., 2] = labels * 7
    thermal = (labels * 18).astype(np.uint8)
    Image.fromarray(visible).save(dataset / "color" / f"{group}_eo-{suffix}.png")
    Image.fromarray(thermal).save(dataset / "thermal8" / f"{group}_thermal-{suffix}.png")
    Image.fromarray(labels.astype(np.uint8)).save(
        dataset / "annotations" / f"{group}_mask-{suffix}.png"
    )


class CaltechEvaluationTests(unittest.TestCase):
    def test_all_terrain_and_object_classes_are_in_frozen_metrics(self) -> None:
        truth = np.array([[0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 255]])
        prediction = truth.copy()
        prediction[0, 1] = 3
        report = evaluate_caltech_semantics(
            [_case("a", "flight-a", truth, prediction)],
            _provenance(),
            bootstrap_replicates=20,
        )

        self.assertEqual(set(report["metrics"]["per_class"]), {str(i) for i in range(1, 12)})
        self.assertEqual(report["metrics"]["per_class"]["2"]["name"], "bare ground")
        self.assertEqual(report["metrics"]["per_class"]["10"]["name"], "vehicles")
        self.assertEqual(report["metrics"]["per_class"]["11"]["name"], "person")
        self.assertAlmostEqual(report["metrics"]["per_class"]["2"]["iou"], 0.0)
        self.assertAlmostEqual(report["metrics"]["per_class"]["3"]["iou"], 0.5)
        self.assertEqual(report["metrics"]["evaluated_pixels"], 10)

    def test_bootstrap_and_paired_comparison_use_complete_scene_groups(self) -> None:
        truth = np.array([[2, 5], [10, 11]], dtype=np.uint8)
        baseline = [
            _case("a", "flight-a", truth, np.zeros_like(truth)),
            _case("b", "flight-b", truth, np.zeros_like(truth)),
        ]
        candidate = [
            _case("a", "flight-a", truth, truth),
            _case("b", "flight-b", truth, truth),
        ]
        report = evaluate_caltech_semantics(
            candidate, _provenance(), bootstrap_replicates=100, bootstrap_seed=4
        )
        comparison = paired_caltech_scene_group_comparison(
            baseline, candidate, replicates=100, seed=4
        )

        self.assertEqual(report["confidence_intervals_95"]["scene_group_count"], 2)
        self.assertEqual(
            report["confidence_intervals_95"]["method"],
            "percentile_complete_scene_group_bootstrap",
        )
        self.assertAlmostEqual(comparison["mean_iou"]["delta"], 1.0)
        self.assertEqual(comparison["complete_pair_scene_groups"], 2)

    def test_final_test_lock_requires_explicit_unlock_and_frozen_hash(self) -> None:
        truth = np.array([[2]], dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "final test is locked"):
            evaluate_caltech_semantics(
                [_case("a", "flight", truth, truth)],
                _provenance(),
                partition="test",
                bootstrap_replicates=10,
            )


class CaltechProbeTests(unittest.TestCase):
    def test_model_and_label_contract_cover_exactly_classes_1_through_11(self) -> None:
        model = CaltechVisibleTerrainProbe(CaltechProbeConfig(4, 16))
        output = model(torch.rand(1, 3, 17, 19))
        self.assertEqual(output.shape, (1, 11, 17, 19))
        np.testing.assert_array_equal(
            _target_indices(np.array([[0, 1, 11, 255]], dtype=np.uint8)),
            np.array([[255, 0, 10, 255]], dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "outside 0..11/255"):
            _target_indices(np.array([[12]], dtype=np.int16))

    def test_checkpoint_roundtrip_preserves_grouped_selection_contract(self) -> None:
        model = CaltechVisibleTerrainProbe(CaltechProbeConfig(4, 16))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "caltech.pt"
            saved = _save_checkpoint(
                path,
                model,
                model_id="caltech-unit",
                training_run_id="run-1",
                epoch=3,
                validation_mean_iou=0.2,
                development_subset=False,
                class_weights=np.ones(11, dtype=np.float32),
                training_ids=["train"],
                validation_ids=["validation"],
            )
            loaded, metadata = load_caltech_probe_checkpoint(path)

        self.assertEqual(saved.artifact_sha256, metadata.artifact_sha256)
        self.assertEqual(metadata.best_epoch, 3)
        self.assertEqual(loaded.config.classes, 11)

    def test_training_and_frozen_multiview_validation_execute_without_test_access(self) -> None:
        labels = np.tile(np.arange(1, 12, dtype=np.uint8), (16, 2))
        train_group = _group_for("train")
        validation_group = _group_for("validation")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_caltech_sample(root, train_group, 1, labels)
            _write_caltech_sample(root, validation_group, 1, labels[:, ::-1])
            checkpoint = root / "caltech.pt"
            training = train_caltech_visible_probe(
                root,
                checkpoint,
                config=CaltechProbeConfig(4, 16),
                epochs=1,
                batch_size=1,
                device_name="cpu",
                seed=7,
            )
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            report = evaluate_caltech_frozen_probe(
                checkpoint,
                root,
                root / "evaluation.json",
                views=("visible_rgb", "thermal_grayscale"),
                external_fused={"missing-system": root / "missing-external"},
                device_name="cpu",
                bootstrap_replicates=20,
            )
            after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

        self.assertEqual(before, after)
        self.assertFalse(training["data_use"]["final_test_accessed"])
        self.assertEqual(training["data_use"]["training_scene_groups"], 1)
        self.assertEqual(training["data_use"]["validation_scene_groups"], 1)
        self.assertEqual(report["sample_manifest"]["scene_groups"], 1)
        self.assertEqual(
            report["views"],
            ["visible_rgb", "thermal_grayscale", "external:missing-system"],
        )
        self.assertEqual(
            report["view_reports"]["visible_rgb"]["model_provenance"]["artifact_sha256"],
            before,
        )
        self.assertIn("p95", report["view_reports"]["visible_rgb"]["runtime_and_failures"]["latency_ms_successful"])
        external_runtime = report["view_reports"]["external:missing-system"]["runtime_and_failures"]
        self.assertEqual(external_runtime["failed"], 1)
        self.assertFalse(report["external_fused_inputs"]["missing-system"]["directory_available"])
        self.assertFalse(report["paired_against_visible"]["external:missing-system"]["available"])

    def test_probe_final_lock_rejects_partial_test_before_file_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "final test is locked"):
                evaluate_caltech_frozen_probe(
                    root / "missing.pt",
                    root / "missing-data",
                    root / "report.json",
                    partition="test",
                    views=("visible_rgb",),
                )
            with self.assertRaisesRegex(ValueError, "complete partition"):
                evaluate_caltech_frozen_probe(
                    root / "missing.pt",
                    root / "missing-data",
                    root / "report.json",
                    partition="test",
                    unlock_final_test=True,
                    max_samples=1,
                    views=("visible_rgb",),
                )


if __name__ == "__main__":
    unittest.main()
