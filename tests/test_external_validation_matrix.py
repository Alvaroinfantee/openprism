from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tooling import run_external_validation_matrix as matrix


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


class ExternalValidationMatrixTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        repository = root / "repo"
        repository.mkdir()
        stages = {}
        for dataset in matrix.DATASETS:
            stage = root / "stages" / dataset
            visible, thermal = stage / "visible", stage / "thermal"
            visible.mkdir(parents=True)
            thermal.mkdir()
            items = []
            for sample_id in (f"{dataset}-a", f"{dataset}-b"):
                visible_file, thermal_file = visible / f"{sample_id}.png", thermal / f"{sample_id}.png"
                visible_file.write_bytes(f"visible:{sample_id}".encode())
                thermal_file.write_bytes(f"thermal:{sample_id}".encode())
                items.append({
                    "sample_id": sample_id,
                    "visible": {"staged": visible_file.name, "sha256": _sha256(visible_file)},
                    "thermal": {"staged": thermal_file.name, "sha256": _sha256(thermal_file)},
                })
            stages[dataset] = str(_write(stage / "stage_manifest.json", {
                "schema_version": "openprism.protocol-pair-stage/1.0",
                "dataset": dataset,
                "partition": "validation",
                "final_test_unlocked": False,
                "protocol_count": 2,
                "items": items,
            }).resolve())
        baselines = {}
        candidates = []
        for index, baseline in enumerate(matrix.BASELINES, 1):
            source = root / "source" / baseline
            source.mkdir(parents=True)
            (source / "model.py").write_text("# frozen\n", encoding="utf-8")
            weights = root / "weights" / baseline
            if baseline == "c2rf":
                weights.mkdir(parents=True)
                (weights / "weights.pth").write_bytes(b"weights")
            else:
                weights.parent.mkdir(parents=True, exist_ok=True)
                weights = weights.with_suffix(".pth")
                weights.write_bytes(b"weights")
            baselines[baseline] = {
                "source_root": str(source.resolve()), "weights": str(weights.resolve())
            }
            if baseline == "seafusion":
                baselines[baseline]["allowed_case_collision_paths"] = ["README.md"]
            candidates.append({
                "id": baseline,
                "revision": str(index) * 40,
                "weights_status": f"aggregate sha256={str(index + 4) * 64}",
            })
        lock = _write(root / "baselines.lock.json", {
            "schema_version": "openprism.external-baselines-lock/1.0",
            "candidates": candidates,
        })
        output_root = root / "outputs"
        spec = _write(root / "spec.json", {
            "schema_version": matrix.SPEC_SCHEMA,
            "python_executable": str(Path(sys.executable).resolve()),
            "repository_root": str(repository.resolve()),
            "baselines_lock": str(lock.resolve()),
            "stage_manifests": stages,
            "baselines": baselines,
            "output_root": str(output_root.resolve()),
            "device": "cuda:0",
        })
        return spec, root / "matrix.json"

    def test_complete_matrix_is_validation_only_verified_and_restart_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, output_manifest = self._fixture(root)
            calls = []

            def fake_run(argv, **kwargs):
                calls.append(list(argv))
                self.assertIs(kwargs["shell"], False)
                self.assertEqual(argv[argv.index("--partition") + 1], "validation")
                self.assertNotIn("--unlock-final-test", argv)
                baseline = argv[argv.index("--baseline") + 1]
                dataset = argv[argv.index("--dataset") + 1]
                collision_values = [
                    argv[index + 1]
                    for index, token in enumerate(argv[:-1])
                    if token == "--allow-nonexecuted-case-collision"
                ]
                self.assertEqual(
                    collision_values, ["README.md"] if baseline == "seafusion" else []
                )
                visible = Path(argv[argv.index("--visible-dir") + 1])
                output = Path(argv[argv.index("--output-dir") + 1])
                output.mkdir()
                outputs = []
                for source in sorted(visible.iterdir()):
                    destination = output / f"{source.stem}.png"
                    destination.write_bytes(b"fused:" + source.read_bytes())
                    outputs.append({
                        "sample_id": source.stem,
                        "path": destination.name,
                        "sha256": _sha256(destination),
                    })
                _write(output / "run_manifest.json", {
                    "schema_version": matrix.RUN_SCHEMA,
                    "baseline": baseline,
                    "dataset": dataset,
                    "partition": "validation",
                    "final_test_unlocked": False,
                    "one_shot_controller_authorized": False,
                    "run_complete": True,
                    "input_count": len(outputs),
                    "outputs": outputs,
                    "failure_accounting": {"failed": 0, "failures": []},
                })
                return subprocess.CompletedProcess(argv, 0)

            with (
                patch.dict(matrix.EXPECTED_COUNTS, {name: 2 for name in matrix.DATASETS}),
                patch.object(matrix.subprocess, "run", side_effect=fake_run),
            ):
                report = matrix.run(spec, output_manifest)
                restarted = matrix.run(spec, output_manifest)

        self.assertTrue(report["matrix_complete"])
        self.assertEqual(len(calls), 12)
        self.assertEqual(len(restarted["runs"]), 12)
        self.assertTrue(all(item["skipped_existing_complete_run"] for item in restarted["runs"]))

    def test_partial_existing_directory_fails_without_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, output_manifest = self._fixture(root)
            partial = root / "outputs" / "seafusion" / "llvip"
            partial.mkdir(parents=True)
            (partial / "orphan.png").write_bytes(b"partial")
            with (
                patch.dict(matrix.EXPECTED_COUNTS, {name: 2 for name in matrix.DATASETS}),
                patch.object(matrix.subprocess, "run") as runner,
                self.assertRaisesRegex(matrix.ExternalValidationMatrixError, "partial"),
            ):
                matrix.run(spec, output_manifest)
            runner.assert_not_called()


if __name__ == "__main__":
    unittest.main()
