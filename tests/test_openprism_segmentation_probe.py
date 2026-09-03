from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from openprism.datasets import SampleRecord
from openprism.learning.data import _msrs_train_partition
from openprism.learning.segmentation_probe import (
    MSRSVisibleSegmentationProbe,
    ProbeConfig,
    _save_checkpoint,
    _target_indices,
    _view_rgb,
    class_weights_from_training_records,
    evaluate_frozen_probe,
    load_probe_checkpoint,
    train_visible_probe,
)


def _partition_name(partition: str) -> str:
    for index in range(1, 20_000):
        sample_id = f"{index:05d}D"
        if _msrs_train_partition(sample_id) == partition:
            return sample_id
    raise AssertionError(f"no synthetic MSRS identifier found for {partition}")


def _write_msrs_sample(root: Path, sample_id: str, labels: np.ndarray) -> None:
    # protocol_items indexes all configured families even though this probe
    # selects only MSRS; provide empty peer roots in the minimal fixture.
    for split in ("train", "test"):
        (root / "LLVIP" / "visible" / split).mkdir(parents=True, exist_ok=True)
        (root / "LLVIP" / "infrared" / split).mkdir(parents=True, exist_ok=True)
        for directory in ("vi", "ir", "Segmentation_labels"):
            (root / "MSRS" / split / directory).mkdir(parents=True, exist_ok=True)
    for directory in ("vi", "ir", "labels"):
        (root / "MSRS" / "detection" / directory).mkdir(parents=True, exist_ok=True)
    for directory in ("color", "thermal8", "thermal16", "annotations"):
        (root / "Caltech_Aerial_RGBT" / directory).mkdir(parents=True, exist_ok=True)
    for directory in ("vi", "ir", "Segmentation_labels"):
        (root / "MSRS" / "train" / directory).mkdir(parents=True, exist_ok=True)
    visible = np.zeros((*labels.shape, 3), dtype=np.uint8)
    visible[..., 0] = labels * 28
    visible[..., 1] = labels * 17
    visible[..., 2] = labels * 9
    thermal = (labels * 25).astype(np.uint8)
    Image.fromarray(visible).save(root / "MSRS" / "train" / "vi" / f"{sample_id}.png")
    Image.fromarray(thermal).save(root / "MSRS" / "train" / "ir" / f"{sample_id}.png")
    Image.fromarray(labels.astype(np.uint8)).save(
        root / "MSRS" / "train" / "Segmentation_labels" / f"{sample_id}.png"
    )


class ProbeModelTests(unittest.TestCase):
    def test_compact_unet_shape_and_repeatability(self) -> None:
        torch.manual_seed(4)
        model = MSRSVisibleSegmentationProbe(ProbeConfig(base_channels=4, image_size=16))
        value = torch.rand(2, 3, 17, 19)
        model.eval()
        first = model(value)
        second = model(value)

        self.assertEqual(first.shape, (2, 8, 17, 19))
        self.assertTrue(torch.equal(first, second))
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 100_000)

    def test_labels_zero_and_255_are_ignored_and_class_ids_are_remapped(self) -> None:
        labels = np.array([[0, 1, 8, 255]], dtype=np.uint8)
        np.testing.assert_array_equal(
            _target_indices(labels),
            np.array([[255, 0, 7, 255]], dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "outside 0..8/255"):
            _target_indices(np.array([[9]], dtype=np.int16))

    def test_class_weights_are_derived_only_from_supplied_training_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.png"
            Image.fromarray(np.array([[1, 1], [1, 2]], dtype=np.uint8)).save(first)
            Image.fromarray(np.array([[2, 2], [0, 255]], dtype=np.uint8)).save(second)
            records = [
                SampleRecord("msrs", "train", "a", first, first, annotation_path=first, annotation_kind="semantic_mask"),
                SampleRecord("msrs", "train", "b", second, second, annotation_path=second, annotation_kind="semantic_mask"),
            ]
            weights, counts = class_weights_from_training_records(records)

        np.testing.assert_array_equal(counts[:2], [3, 3])
        np.testing.assert_array_equal(counts[2:], np.zeros(6, dtype=np.int64))
        np.testing.assert_allclose(weights[:2], [1.0, 1.0])
        np.testing.assert_array_equal(weights[2:], np.zeros(6, dtype=np.float32))

    def test_learned_view_uses_explicit_task_and_reuses_frame_output(self) -> None:
        class RecordingEngine:
            def __init__(self) -> None:
                self.tasks: list[str] = []

            def fuse(self, frame, *, task: str):
                self.tasks.append(task)
                luminance = np.full((4, 5), 0.25, dtype=np.float32)
                return SimpleNamespace(
                    provenance={"learned_fusion_applied": True},
                    operator_rgb=np.full((4, 5, 3), 64, dtype=np.uint8),
                    machine_tensor=np.stack((luminance,)),
                    channel_names=("learned_fused_luminance",),
                )

        frame = SimpleNamespace(
            observations={
                "visible": SimpleNamespace(data=np.zeros((4, 5, 3), dtype=np.uint8)),
                "thermal": SimpleNamespace(data=np.zeros((4, 5), dtype=np.uint8)),
            },
            provenance={"sample_id": "sample"},
        )
        engine = RecordingEngine()
        cache: dict[str, object] = {}
        operator = _view_rgb(
            "prism_egt_operator",
            frame,
            deterministic_engine=SimpleNamespace(),
            learned_engine=engine,
            external={},
            learned_task="terrain",
            cache=cache,
        )
        luminance = _view_rgb(
            "prism_egt_luminance",
            frame,
            deterministic_engine=SimpleNamespace(),
            learned_engine=engine,
            external={},
            learned_task="terrain",
            cache=cache,
        )
        automatic_operator = _view_rgb(
            "prism_egt_operator_automatic",
            frame,
            deterministic_engine=SimpleNamespace(),
            learned_engine=engine,
            external={},
            learned_task="terrain",
            cache=cache,
        )
        _view_rgb(
            "prism_egt_luminance_automatic",
            frame,
            deterministic_engine=SimpleNamespace(),
            learned_engine=engine,
            external={},
            learned_task="terrain",
            cache=cache,
        )

        self.assertEqual(engine.tasks, ["terrain", "automatic"])
        self.assertEqual(operator.shape, (4, 5, 3))
        self.assertEqual(automatic_operator.shape, (4, 5, 3))
        np.testing.assert_allclose(luminance, 0.25)


class ProbeCheckpointAndExecutionTests(unittest.TestCase):
    def test_checkpoint_roundtrip_records_visible_train_and_validation_selection(self) -> None:
        config = ProbeConfig(base_channels=4, image_size=16)
        torch.manual_seed(5)
        model = MSRSVisibleSegmentationProbe(config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.pt"
            saved = _save_checkpoint(
                path,
                model,
                model_id="unit-probe",
                training_run_id="run-1",
                epoch=2,
                validation_mean_iou=0.25,
                development_subset=False,
                class_weights=np.ones(8, dtype=np.float32),
                training_ids=["train-a"],
                validation_ids=["validation-a"],
            )
            loaded, metadata = load_probe_checkpoint(path)
            expected = model(torch.rand(1, 3, 16, 16))
            actual = loaded(torch.rand(1, 3, 16, 16))

            self.assertEqual(saved.artifact_sha256, metadata.artifact_sha256)
            self.assertEqual(metadata.best_epoch, 2)
            self.assertEqual(metadata.validation_mean_iou, 0.25)
            self.assertEqual(loaded.config, config)
            self.assertEqual(expected.shape, actual.shape)

    def test_publisher_test_lock_fires_before_checkpoint_or_data_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "publisher test is locked"):
                evaluate_frozen_probe(
                    root / "missing.pt",
                    root / "missing-data",
                    root / "report.json",
                    partition="test",
                    views=("visible_rgb",),
                )
            with self.assertRaisesRegex(ValueError, "complete partition"):
                evaluate_frozen_probe(
                    root / "missing.pt",
                    root / "missing-data",
                    root / "report.json",
                    partition="test",
                    unlock_final_test=True,
                    views=("visible_rgb",),
                    max_samples=1,
                )

    def test_training_and_multiview_validation_are_executable_and_checkpoint_is_unchanged(self) -> None:
        labels = np.tile(np.arange(1, 9, dtype=np.uint8), (16, 2))
        train_id = _partition_name("train")
        validation_id = _partition_name("validation")
        self.assertNotEqual(train_id, validation_id)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_msrs_sample(root, train_id, labels)
            _write_msrs_sample(root, validation_id, labels[:, ::-1])
            checkpoint = root / "probe.pt"
            training = train_visible_probe(
                root,
                checkpoint,
                config=ProbeConfig(base_channels=4, image_size=16),
                epochs=1,
                batch_size=1,
                device_name="cpu",
                seed=13,
            )
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            report = evaluate_frozen_probe(
                checkpoint,
                root,
                root / "evaluation.json",
                partition="validation",
                views=("visible_rgb", "thermal_grayscale"),
                device_name="cpu",
                bootstrap_replicates=20,
            )
            after = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            self.assertEqual(before, after)
            self.assertFalse(training["data_use"]["publisher_test_accessed"])
            self.assertEqual(training["data_use"]["training_samples"], 1)
            self.assertEqual(training["data_use"]["validation_samples"], 1)
            self.assertEqual(report["views"], ["visible_rgb", "thermal_grayscale"])
            self.assertEqual(report["sample_manifest"]["count"], 1)
            self.assertEqual(
                report["view_reports"]["visible_rgb"]["model_provenance"]["artifact_sha256"],
                before,
            )
            self.assertFalse(
                report["view_reports"]["visible_rgb"]["model_provenance"]["pretrained"]
            )
            self.assertIn("thermal_grayscale", report["paired_against_visible"])
            self.assertIn("p95", report["view_reports"]["visible_rgb"]["runtime_and_failures"]["latency_ms_successful"])

    def test_missing_external_fused_output_is_reported_as_failure(self) -> None:
        labels = np.ones((16, 16), dtype=np.uint8)
        validation_id = _partition_name("validation")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_msrs_sample(root, validation_id, labels)
            checkpoint = root / "probe.pt"
            model = MSRSVisibleSegmentationProbe(ProbeConfig(base_channels=4, image_size=16))
            _save_checkpoint(
                checkpoint,
                model,
                model_id="unit-probe",
                training_run_id="run-1",
                epoch=1,
                validation_mean_iou=0.1,
                development_subset=False,
                class_weights=np.ones(8, dtype=np.float32),
                training_ids=["train"],
                validation_ids=[validation_id],
            )
            external = root / "empty-external"
            external.mkdir()
            report = evaluate_frozen_probe(
                checkpoint,
                root,
                root / "external-report.json",
                views=("visible_rgb",),
                external_fused={"method": external},
                device_name="cpu",
                bootstrap_replicates=10,
            )

        failures = report["view_reports"]["external:method"]["runtime_and_failures"]
        self.assertEqual(failures["failure_rate"], 1.0)
        self.assertEqual(failures["failed"], 1)
        self.assertFalse(report["paired_against_visible"]["external:method"]["available"])


if __name__ == "__main__":
    unittest.main()
