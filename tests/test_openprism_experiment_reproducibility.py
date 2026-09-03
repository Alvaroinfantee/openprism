from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

from openprism.datasets import SampleRecord
from openprism.learning.data import ProtocolItem, item_scene_group
from tooling import run_external_fusion as external
from tooling import stage_protocol_pairs as staging


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _repository(root: Path, files: dict[str, bytes]) -> str:
    root.mkdir()
    _git(root, "init", "--quiet")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "OpenPRISM Test")
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    _git(root, "add", ".")
    _git(root, "commit", "--quiet", "-m", "fixture")
    return _git(root, "rev-parse", "HEAD")


class ExternalSourceAttestationTests(unittest.TestCase):
    def test_test_partition_requires_persistent_one_shot_controller(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OPENPRISM_FINAL_SUITE": "",
                "OPENPRISM_FINAL_SUITE_MANIFEST_SHA256": "",
                "OPENPRISM_FINAL_SUITE_LEDGER": "",
                "OPENPRISM_FINAL_SUITE_STEP_ID": "",
                "OPENPRISM_FINAL_SUITE_CONTROLLER_SHA256": "",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "one-shot final-suite controller"):
                external.run_external_fusion(
                    "seafusion",
                    Path("source"),
                    Path("weights"),
                    Path("visible"),
                    Path("thermal"),
                    Path("output"),
                    dataset="llvip",
                    partition="test",
                    unlock_final_test=True,
                    expected_revision="0" * 40,
                    device_name="cpu",
                )

    def test_external_cli_exits_nonzero_after_persisted_sample_failures(self) -> None:
        report = {
            "baseline": "seafusion",
            "partition": "validation",
            "revision": "a" * 40,
            "weights_sha256": "b" * 64,
            "input_count": 2,
            "elapsed_seconds": 1.0,
            "failure_accounting": {
                "successful": 1,
                "failed": 1,
            },
        }
        argv = [
            "--baseline", "seafusion",
            "--dataset", "llvip",
            "--source-root", "source",
            "--weights", "weights",
            "--visible-dir", "visible",
            "--thermal-dir", "thermal",
            "--output-dir", "output",
            "--expected-revision", "a" * 40,
        ]
        with (
            patch.object(external, "run_external_fusion", return_value=report),
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaisesRegex(SystemExit, "2"):
                external.main(argv)

    def test_external_cli_returns_after_accounted_final_test_failures(self) -> None:
        report = {
            "baseline": "seafusion",
            "partition": "test",
            "revision": "a" * 40,
            "weights_sha256": "b" * 64,
            "input_count": 2,
            "elapsed_seconds": 1.0,
            "failure_accounting": {
                "successful": 1,
                "failed": 1,
            },
        }
        argv = [
            "--baseline", "seafusion",
            "--dataset", "llvip",
            "--source-root", "source",
            "--weights", "weights",
            "--visible-dir", "visible",
            "--thermal-dir", "thermal",
            "--output-dir", "output",
            "--expected-revision", "a" * 40,
            "--partition", "test",
            "--unlock-final-test",
        ]
        with (
            patch.object(external, "run_external_fusion", return_value=report),
            redirect_stdout(io.StringIO()),
        ):
            external.main(argv)

    def test_parameter_inventory_deduplicates_shared_parameter_objects(self) -> None:
        shared = nn.Linear(3, 2)
        independent = nn.Linear(2, 1, bias=False)
        inventory = external._parameter_inventory(
            {
                "first_reference": shared,
                "second_reference": shared,
                "independent": independent,
            }
        )
        expected = sum(parameter.numel() for parameter in shared.parameters()) + sum(
            parameter.numel() for parameter in independent.parameters()
        )
        self.assertEqual(inventory["total_parameters"], expected)
        self.assertEqual(inventory["trainable_parameters"], expected)
        self.assertEqual(
            inventory["shared_parameter_references_deduplicated"],
            len(tuple(shared.parameters())),
        )
        self.assertIn("object identities", inventory["counting_policy"])

    def test_dirty_tracked_or_unrelated_untracked_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            _repository(root, {"model.py": b"VALUE = 1\n"})
            (root / "model.py").write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "worktree is not clean"):
                external._source_worktree_attestation(root)

            _git(root, "restore", "model.py")
            (root / "notes.bin").write_bytes(b"not an allowed cache")
            with self.assertRaisesRegex(ValueError, "notes.bin"):
                external._source_worktree_attestation(root)

    def test_python_cache_is_allowed_but_source_is_compiled_and_hashed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            _repository(root, {"model.py": b"VALUE = 7\n"})
            cache = root / "__pycache__" / "model.cpython-test.pyc"
            cache.parent.mkdir()
            cache.write_bytes(b"untrusted cache bytes")

            attestation, tracked = external._source_worktree_attestation(root)
            self.assertEqual(
                attestation["safe_python_cache_entries"][0]["path"],
                "__pycache__/model.cpython-test.pyc",
            )
            with external._track_upstream_source_execution(root) as executed:
                module = external._module(root / "model.py", "openprism_fixture_model")
            self.assertEqual(module.VALUE, 7)
            manifest = external._executed_source_manifest(root, executed, tracked)
            self.assertEqual(set(manifest), {"model.py"})
            self.assertEqual(manifest["model.py"], external._sha256(root / "model.py"))

    def test_case_collision_exception_is_narrow_and_documentation_only(self) -> None:
        tracked = frozenset({"README.md", "readme.md", "model.py", "Model.py"})
        allowed = external._documentation_collision_exceptions(
            tracked, ("README.md",)
        )
        self.assertEqual(allowed["readme.md"], ("README.md", "readme.md"))
        with self.assertRaisesRegex(ValueError, "documentation-only"):
            external._documentation_collision_exceptions(tracked, ("model.py",))
        with self.assertRaisesRegex(ValueError, "no differently-cased tracked peer"):
            external._documentation_collision_exceptions(
                frozenset({"README.md"}), ("README.md",)
            )

    def test_run_manifest_attests_executed_source_weights_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            revision = _repository(source, {"FusionNet.py": b"VALUE = 1\n"})
            weights = root / "weights.bin"
            weights.write_bytes(b"frozen weights")
            visible_dir = root / "visible"
            thermal_dir = root / "thermal"
            visible_dir.mkdir()
            thermal_dir.mkdir()
            Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8)).save(
                visible_dir / "frame.png"
            )
            Image.fromarray(np.full((4, 5), 63, dtype=np.uint8)).save(
                thermal_dir / "frame.png"
            )

            def loader(source_root, _weights, _device):
                module = external._module(
                    source_root / "FusionNet.py", "openprism_fixture_fusion"
                )
                self.assertEqual(module.VALUE, 1)
                fixture_model = nn.Conv2d(3, 3, 1, bias=False)
                return (lambda visible, _thermal: visible), {
                    "fixture": True,
                    "parameter_inventory": external._parameter_inventory(
                        {"fixture": fixture_model}
                    ),
                }

            with patch.dict(external._LOADERS, {"seafusion": loader}):
                report = external.run_external_fusion(
                    "seafusion",
                    source,
                    weights,
                    visible_dir,
                    thermal_dir,
                    root / "output",
                    dataset="llvip",
                    expected_revision=revision,
                    expected_weights_sha256=external._sha256(weights),
                    device_name="cpu",
                )
            self.assertEqual(report["schema_version"], "openprism.external-fusion-run/1.3")
            self.assertEqual(report["dataset"], "llvip")
            self.assertEqual(report["partition"], "validation")
            self.assertFalse(report["final_test_unlocked"])
            self.assertFalse(report["one_shot_controller_authorized"])
            attestation = report["source_attestation"]
            self.assertTrue(attestation["verified_before_and_after_inference"])
            self.assertEqual(
                set(attestation["executed_source_files_sha256"]), {"FusionNet.py"}
            )
            self.assertEqual(report["weight_files"], {"weights.bin": external._sha256(weights)})
            self.assertEqual(report["weights_total_bytes"], len(b"frozen weights"))
            self.assertEqual(
                report["weight_file_bytes"], {"weights.bin": len(b"frozen weights")}
            )
            self.assertEqual(report["parameter_inventory"]["total_parameters"], 9)
            self.assertTrue(report["run_complete"])
            self.assertIsNone(report["runtime_resources"]["peak_cuda_allocated_bytes"])
            self.assertEqual(
                report["failure_accounting"],
                {
                    "attempted": 1,
                    "successful": 1,
                    "failed": 0,
                    "failure_rate": 0.0,
                    "failures": [],
                },
            )
            self.assertEqual(report["runtime"]["numpy"], np.__version__)
            self.assertEqual(report["runtime"]["device"], "cpu")

    def test_external_sample_failure_is_persisted_without_hiding_other_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            revision = _repository(source, {"FusionNet.py": b"VALUE = 1\n"})
            weights = root / "weights.bin"
            weights.write_bytes(b"frozen weights")
            visible_dir = root / "visible"
            thermal_dir = root / "thermal"
            visible_dir.mkdir()
            thermal_dir.mkdir()
            for sample_id in ("bad", "good"):
                Image.fromarray(np.full((4, 5, 3), 127, dtype=np.uint8)).save(
                    visible_dir / f"{sample_id}.png"
                )
                Image.fromarray(np.full((4, 5), 63, dtype=np.uint8)).save(
                    thermal_dir / f"{sample_id}.png"
                )

            calls = 0

            def loader(source_root, _weights, _device):
                external._module(source_root / "FusionNet.py", "openprism_fixture_failure")
                module = nn.Conv2d(3, 3, 1, bias=False)

                def infer(visible, _thermal):
                    nonlocal calls
                    calls += 1
                    if calls == 1:
                        raise RuntimeError("fixture inference failure")
                    return visible

                return infer, {
                    "parameter_inventory": external._parameter_inventory({"fixture": module})
                }

            with patch.dict(external._LOADERS, {"seafusion": loader}):
                report = external.run_external_fusion(
                    "seafusion",
                    source,
                    weights,
                    visible_dir,
                    thermal_dir,
                    root / "output",
                    dataset="llvip",
                    expected_revision=revision,
                    expected_weights_sha256=external._sha256(weights),
                    device_name="cpu",
                )
            accounting = report["failure_accounting"]
            self.assertTrue(report["run_complete"])
            self.assertEqual(accounting["attempted"], 2)
            self.assertEqual(accounting["successful"], 1)
            self.assertEqual(accounting["failed"], 1)
            self.assertEqual(accounting["failures"][0]["sample_id"], "bad")
            self.assertIn("fixture inference failure", accounting["failures"][0]["reason"])
            self.assertTrue((root / "output" / "good.png").is_file())
            self.assertFalse((root / "output" / "bad.png").exists())


def _protocol_item(
    root: Path,
    sample_id: str,
    *,
    partition: str = "validation",
    dataset: str = "llvip",
) -> ProtocolItem:
    visible = root / "sources" / "visible" / f"{sample_id}.png"
    thermal = root / "sources" / "thermal" / f"{sample_id}.png"
    visible.parent.mkdir(parents=True, exist_ok=True)
    thermal.parent.mkdir(parents=True, exist_ok=True)
    visible.write_bytes(f"visible:{sample_id}".encode())
    thermal.write_bytes(f"thermal:{sample_id}".encode())
    return ProtocolItem(
        SampleRecord(
            dataset=dataset,
            split="test" if partition != "train" else "train",
            sample_id=sample_id,
            visible_path=visible,
            thermal_path=thermal,
        ),
        "search",
        partition,
    )


def _protocol_manifest(
    items: list[ProtocolItem], dataset: str, partition: str, *, count: int | None = None
) -> dict[str, object]:
    return {
        "schema_version": "fixture-protocol/1.0",
        "counts": {partition: {dataset: len(items) if count is None else count}},
        "scene_groups": {
            partition: {dataset: sorted({item_scene_group(item) for item in items})}
        },
        "sample_manifest_sha256": {
            partition: {dataset: staging._selection_sha256(items)}
        },
    }


class ProtocolPairStagingTests(unittest.TestCase):
    def test_validation_stage_is_complete_hashed_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = [_protocol_item(root, "190001"), _protocol_item(root, "190002")]
            protocol = _protocol_manifest(items, "llvip", "validation")
            output = root / "stage"
            with (
                patch.object(staging, "protocol_items", return_value=items),
                patch.object(staging, "protocol_manifest", return_value=protocol),
            ):
                report = staging.stage_protocol_pairs(
                    root / "data", "llvip", "validation", output
                )

            self.assertEqual(report["protocol_count"], 2)
            self.assertEqual(
                {path.stem for path in (output / "visible").iterdir()},
                {"190001", "190002"},
            )
            self.assertEqual(
                {path.stem for path in (output / "thermal").iterdir()},
                {"190001", "190002"},
            )
            persisted = json.loads((output / "stage_manifest.json").read_text())
            payload_digest = persisted.pop("manifest_payload_sha256")
            self.assertEqual(payload_digest, staging._canonical_sha256(persisted))
            for entry in report["items"]:
                for modality in ("visible", "thermal"):
                    artifact = entry[modality]
                    self.assertIn(artifact["method"], {"hardlink", "copy_fallback"})
                    self.assertEqual(
                        artifact["sha256"],
                        staging._sha256(output / modality / artifact["staged"]),
                    )

    def test_copy_fallback_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = [_protocol_item(root, "190001")]
            protocol = _protocol_manifest(items, "llvip", "validation")
            with (
                patch.object(staging, "protocol_items", return_value=items),
                patch.object(staging, "protocol_manifest", return_value=protocol),
                patch.object(staging.os, "link", side_effect=OSError(18, "cross-device")),
            ):
                report = staging.stage_protocol_pairs(
                    root / "data", "llvip", "validation", root / "stage"
                )
            self.assertTrue(
                all(
                    entry[modality]["method"] == "copy_fallback"
                    for entry in report["items"]
                    for modality in ("visible", "thermal")
                )
            )

    def test_final_test_is_locked_before_protocol_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, (
            patch.object(staging, "protocol_manifest", side_effect=AssertionError("accessed"))
        ):
            with self.assertRaisesRegex(ValueError, "final-test pair staging is locked"):
                staging.stage_protocol_pairs(
                    Path(temporary) / "data",
                    "llvip",
                    "test",
                    Path(temporary) / "stage",
                )

    def test_unlocked_final_test_rejects_partial_or_unfrozen_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = [_protocol_item(root, "200001", partition="test")]
            digest = staging._selection_sha256(items)
            partial_manifest = _protocol_manifest(
                items, "llvip", "test", count=2
            )
            with (
                patch.object(staging, "protocol_items", return_value=items),
                patch.object(
                    staging, "protocol_manifest", return_value=partial_manifest
                ),
            ):
                with self.assertRaisesRegex(ValueError, "partial or inconsistent"):
                    staging.stage_protocol_pairs(
                        root / "data",
                        "llvip",
                        "test",
                        root / "stage",
                        unlock_final_test=True,
                        expected_sample_manifest_sha256=digest,
                    )
            self.assertFalse((root / "stage").exists())

            complete_manifest = _protocol_manifest(items, "llvip", "test")
            with (
                patch.object(staging, "protocol_items", return_value=items),
                patch.object(
                    staging, "protocol_manifest", return_value=complete_manifest
                ),
            ):
                with self.assertRaisesRegex(ValueError, "explicitly frozen"):
                    staging.stage_protocol_pairs(
                        root / "data",
                        "llvip",
                        "test",
                        root / "stage",
                        unlock_final_test=True,
                        expected_sample_manifest_sha256="0" * 64,
                    )

    def test_case_insensitive_sample_collision_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            items = [_protocol_item(root, "Frame"), _protocol_item(root, "frame")]
            protocol = _protocol_manifest(items, "llvip", "validation")
            with (
                patch.object(staging, "protocol_items", return_value=items),
                patch.object(staging, "protocol_manifest", return_value=protocol),
            ):
                with self.assertRaisesRegex(ValueError, "case-insensitive"):
                    staging.stage_protocol_pairs(
                        root / "data", "llvip", "validation", root / "stage"
                    )


if __name__ == "__main__":
    unittest.main()
