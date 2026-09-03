from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import torch

from openprism.fusion import _replace_luminance
from tooling import benchmark_fusion_latency as latency


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FusionLatencyContractTests(unittest.TestCase):
    def test_operator_tensor_matches_published_numpy_renderer(self) -> None:
        rng = np.random.default_rng(20260903)
        rgb = rng.random((2, 11, 13, 3), dtype=np.float32)
        target = rng.random((2, 11, 13), dtype=np.float32)
        observed = latency._replace_luminance_tensor(
            torch.from_numpy(np.moveaxis(rgb, -1, 1)),
            torch.from_numpy(target[:, None]),
        ).numpy()
        expected = np.stack(
            [_replace_luminance(rgb[index], target[index], preserve=0.88) for index in range(2)]
        )
        np.testing.assert_allclose(np.moveaxis(observed, 1, -1), expected, atol=2e-7)

    def test_latency_summary_reports_quantiles_and_true_sequential_throughput(self) -> None:
        report = latency._latency_summary([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(report["recorded"], 4)
        self.assertEqual(report["p50_ms"], 2.5)
        self.assertAlmostEqual(report["p90_ms"], 3.7)
        self.assertAlmostEqual(report["p95_ms"], 3.85)
        self.assertAlmostEqual(report["throughput_images_per_second"], 400.0)
        empty = latency._latency_summary([])
        self.assertEqual(empty["recorded"], 0)
        self.assertIsNone(empty["p95_ms"])

    def test_final_authorization_fails_before_any_artifact_access(self) -> None:
        token = "a" * 64
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("artifact accessed")):
            with self.assertRaisesRegex(latency.BenchmarkError, "one-shot controller"):
                latency._partition_authorization(
                    "test",
                    unlock_final_test=True,
                    final_suite_token=latency.FINAL_SUITE_TOKEN,
                    environment={},
                )
            with self.assertRaisesRegex(latency.BenchmarkError, "literal controller"):
                latency._partition_authorization(
                    "test",
                    unlock_final_test=True,
                    final_suite_token="not-the-handoff-token",
                    environment={
                        "OPENPRISM_FINAL_SUITE": "1",
                        "OPENPRISM_FINAL_SUITE_MANIFEST_SHA256": token,
                    },
                )
        authorized = latency._partition_authorization(
            "test",
            unlock_final_test=True,
            final_suite_token=latency.FINAL_SUITE_TOKEN,
            environment={
                "OPENPRISM_FINAL_SUITE": "1",
                "OPENPRISM_FINAL_SUITE_MANIFEST_SHA256": token,
            },
        )
        self.assertTrue(authorized["one_shot_controller_authorized"])
        self.assertTrue(authorized["final_test_unlocked"])

    def test_validation_rejects_final_unlock_material(self) -> None:
        with self.assertRaisesRegex(latency.BenchmarkError, "must not carry"):
            latency._partition_authorization(
                "validation",
                unlock_final_test=True,
                final_suite_token=None,
            )
        report = latency._partition_authorization(
            "validation",
            unlock_final_test=False,
            final_suite_token=None,
        )
        self.assertFalse(report["test_data_accessed"])

    def test_run_rejects_missing_persistent_claim_before_stage_access(self) -> None:
        token = "a" * 64
        with (
            mock.patch.object(
                latency,
                "_attest_one_shot_claim",
                side_effect=latency.BenchmarkError("no persistent claim"),
            ),
            mock.patch.object(
                latency,
                "_load_stage_manifest",
                side_effect=AssertionError("stage data accessed"),
            ),
        ):
            with self.assertRaisesRegex(latency.BenchmarkError, "persistent claim"):
                latency.run_benchmark(
                    Path("never-open-checkpoint.pt"),
                    {name: Path(f"never-open-{name}.json") for name in latency.DATASETS},
                    Path("never-create-output.json"),
                    partition="test",
                    unlock_final_test=True,
                    final_suite_token=latency.FINAL_SUITE_TOKEN,
                    environment={
                        "OPENPRISM_FINAL_SUITE": "1",
                        "OPENPRISM_FINAL_SUITE_MANIFEST_SHA256": token,
                    },
                )

    def test_method_failures_are_counted_without_dropping_attempts(self) -> None:
        inputs = [
            {
                "key": "fixture:good",
                "rgb": torch.zeros((1, 3, 192, 192)),
                "thermal": torch.zeros((1, 1, 192, 192)),
                "evidence": torch.ones((1, 3, 192, 192)),
            },
            {
                "key": "fixture:bad",
                "rgb": torch.zeros((1, 3, 192, 192)),
                "thermal": torch.zeros((1, 1, 192, 192)),
                "evidence": torch.ones((1, 3, 192, 192)),
            },
        ]

        def fixture(sample):
            if sample["key"] == "fixture:bad":
                raise RuntimeError("declared fixture failure")
            return sample["thermal"]

        report = latency._benchmark_callable(
            "fixture",
            fixture,
            inputs,
            torch.device("cpu"),
            expected_outputs=("luminance",),
        )
        self.assertEqual(report["attempted"], 2 * latency.MEASURED_PASSES)
        self.assertEqual(report["failed"], latency.MEASURED_PASSES)
        self.assertEqual(report["successful"], latency.MEASURED_PASSES)
        self.assertEqual(report["latency"]["recorded"], latency.MEASURED_PASSES)
        self.assertEqual(report["failure_reasons"][0]["count"], latency.MEASURED_PASSES)
        self.assertEqual(report["warmup"]["failed"], latency.WARMUP_PASSES)


class FusionLatencyManifestTests(unittest.TestCase):
    def _stage_fixture(self, root: Path, *, count: int = 2) -> tuple[Path, dict[str, object]]:
        stage = root / "stage"
        visible_dir = stage / "visible"
        thermal_dir = stage / "thermal"
        visible_dir.mkdir(parents=True)
        thermal_dir.mkdir()
        items = []
        for index in range(count):
            sample_id = f"frame-{index:03d}"
            visible = visible_dir / f"{sample_id}.png"
            thermal = thermal_dir / f"{sample_id}.png"
            Image.fromarray(np.full((6, 7, 3), 40 + index, dtype=np.uint8)).save(visible)
            Image.fromarray(np.full((6, 7), 80 + index, dtype=np.uint8)).save(thermal)
            items.append(
                {
                    "sample_id": sample_id,
                    "source_split": "fixture",
                    "scene_group": "fixture-group",
                    "visible": {
                        "staged": visible.name,
                        "sha256": _sha256(visible),
                    },
                    "thermal": {
                        "staged": thermal.name,
                        "sha256": _sha256(thermal),
                    },
                }
            )
        document: dict[str, object] = {
            "schema_version": latency.STAGE_SCHEMA_VERSION,
            "dataset": "llvip",
            "partition": "validation",
            "final_test_unlocked": False,
            "output_directory": str(stage.resolve()),
            "protocol_count": count,
            "protocol_sample_manifest_sha256": "c" * 64,
            "items": items,
        }
        document["manifest_payload_sha256"] = latency._canonical_sha256(document)
        path = stage / "stage_manifest.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path, document

    def test_stage_manifest_requires_exact_payload_and_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, document = self._stage_fixture(Path(temporary))
            expected = latency.EXPECTED_COUNTS["validation"]["llvip"]
            with mock.patch.dict(
                latency.EXPECTED_COUNTS["validation"], {"llvip": 2}, clear=False
            ):
                loaded = latency._load_stage_manifest(path, "llvip", "validation")
                self.assertEqual(loaded["sample_ids"], {"frame-000", "frame-001"})
                document["items"][0]["sample_id"] = "mutated"  # type: ignore[index]
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.assertRaisesRegex(latency.BenchmarkError, "payload digest"):
                    latency._load_stage_manifest(path, "llvip", "validation")
            self.assertEqual(latency.EXPECTED_COUNTS["validation"]["llvip"], expected)

    def test_external_timing_is_explicitly_noncomparable_and_hash_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_path, _ = self._stage_fixture(root)
            with mock.patch.dict(
                latency.EXPECTED_COUNTS["validation"], {"llvip": 2}, clear=False
            ):
                stage = latency._load_stage_manifest(stage_path, "llvip", "validation")
            output_dir = root / "external"
            output_dir.mkdir()
            outputs = []
            for sample_id in sorted(stage["sample_ids"]):
                image = output_dir / f"{sample_id}.png"
                Image.fromarray(np.full((6, 7, 3), 127, dtype=np.uint8)).save(image)
                outputs.append(
                    {
                        "sample_id": sample_id,
                        "path": image.name,
                        "sha256": _sha256(image),
                        "elapsed_seconds": 0.01,
                    }
                )
            report = {
                "schema_version": "openprism.external-fusion-run/1.1",
                "baseline": "seafusion",
                "revision": "d" * 40,
                "weights_sha256": "e" * 64,
                "adapter_source_sha256": "f" * 64,
                "inputs": {
                    "visible_directory": str(stage_path.parent / "visible"),
                    "thermal_directory": str(stage_path.parent / "thermal"),
                    "paired_ids_sorted": sorted(stage["sample_ids"]),
                },
                "input_count": 2,
                "outputs": outputs,
                "runtime": {"device": "cuda"},
                "runtime_resources": {
                    "peak_cuda_allocated_bytes": 12345,
                    "incremental_peak_cuda_allocated_bytes": 2345,
                },
                "adapter": {},
            }
            manifest = output_dir / "run_manifest.json"
            manifest.write_text(json.dumps(report), encoding="utf-8")
            audited = latency._external_latency_report(
                "llvip", "seafusion", manifest, stage
            )
            self.assertEqual(audited["status"], "complete")
            self.assertEqual(audited["historical_latency"]["p50_ms"], 10.0)
            self.assertIn(
                "noncomparable", audited["timing_scope"]["comparison_class"]
            )
            self.assertFalse(audited["parameter_count"]["available"])
            self.assertEqual(audited["cuda_memory"]["peak_allocated_bytes"], 12345)
            outputs[0]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(report), encoding="utf-8")
            failed = latency._external_latency_report(
                "llvip", "seafusion", manifest, stage
            )
            self.assertEqual(failed["status"], "failed_integrity_or_completeness")
            self.assertGreater(failed["failed"], 0)

    def test_external_declared_runtime_failure_preserves_reason(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stage_path, _ = self._stage_fixture(root)
            with mock.patch.dict(
                latency.EXPECTED_COUNTS["validation"], {"llvip": 2}, clear=False
            ):
                stage = latency._load_stage_manifest(stage_path, "llvip", "validation")
            output_dir = root / "external"
            output_dir.mkdir()
            successful_id, failed_id = sorted(stage["sample_ids"])
            image = output_dir / f"{successful_id}.png"
            Image.fromarray(np.full((6, 7, 3), 127, dtype=np.uint8)).save(image)
            report = {
                "schema_version": "openprism.external-fusion-run/1.1",
                "baseline": "seafusion",
                "inputs": {
                    "visible_directory": str(stage_path.parent / "visible"),
                    "thermal_directory": str(stage_path.parent / "thermal"),
                    "paired_ids_sorted": sorted(stage["sample_ids"]),
                },
                "input_count": 2,
                "outputs": [
                    {
                        "sample_id": successful_id,
                        "path": image.name,
                        "sha256": _sha256(image),
                        "elapsed_seconds": 0.01,
                    }
                ],
                "failure_accounting": {
                    "attempted": 2,
                    "successful": 1,
                    "failed": 1,
                    "failure_rate": 0.5,
                    "failures": [
                        {
                            "sample_id": failed_id,
                            "stage": "adapter_inference_and_png_save",
                            "reason": "RuntimeError: out of memory",
                        }
                    ],
                },
            }
            manifest = output_dir / "run_manifest.json"
            manifest.write_text(json.dumps(report), encoding="utf-8")
            audited = latency._external_latency_report(
                "llvip", "seafusion", manifest, stage
            )
            self.assertEqual(audited["status"], "completed_with_runtime_failures")
            self.assertEqual(audited["failed"], 1)
            self.assertIn("out of memory", audited["failure_reasons"][0]["reason"])
            self.assertIn(
                audited["status"], latency.SCIENTIFICALLY_COMPLETE_EXTERNAL_STATUSES
            )

    def test_integrity_failure_is_not_a_scientifically_complete_external_status(self) -> None:
        self.assertNotIn(
            "failed_integrity_or_completeness",
            latency.SCIENTIFICALLY_COMPLETE_EXTERNAL_STATUSES,
        )


if __name__ == "__main__":
    unittest.main()
