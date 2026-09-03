"""Run the frozen external RGB--thermal fusion baselines on validation only.

The matrix is restart-safe only for already complete, content-verified runs.
Partial directories are never deleted or overwritten automatically.  Test
partitions and the final-test unlock flag are unconditionally rejected here;
test baseline generation belongs exclusively to the one-shot final controller.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Mapping


SPEC_SCHEMA = "openprism.external-validation-matrix-spec/1.0"
OUTPUT_SCHEMA = "openprism.external-validation-matrix/1.0"
RUN_SCHEMA = "openprism.external-fusion-run/1.3"
DATASETS = ("llvip", "msrs", "caltech")
BASELINES = ("seafusion", "cddfuse", "paif", "c2rf")
EXPECTED_COUNTS = {"llvip": 1730, "msrs": 270, "caltech": 177}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ExternalValidationMatrixError(RuntimeError):
    """The validation matrix is malformed, incomplete, or unsafe to run."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExternalValidationMatrixError(f"cannot load JSON {path}: {error}") from error
    if type(value) is not dict:
        raise ExternalValidationMatrixError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _absolute(raw: object, label: str, *, kind: str) -> Path:
    if not isinstance(raw, (str, os.PathLike)) or not str(raw):
        raise ExternalValidationMatrixError(f"{label} must be a non-empty path")
    path = Path(raw)
    if not path.is_absolute():
        raise ExternalValidationMatrixError(f"{label} must be absolute: {path}")
    if path.is_symlink():
        raise ExternalValidationMatrixError(f"{label} must not be a symlink: {path}")
    if kind == "file" and not path.is_file():
        raise ExternalValidationMatrixError(f"{label} file is missing: {path}")
    if kind == "directory" and not path.is_dir():
        raise ExternalValidationMatrixError(f"{label} directory is missing: {path}")
    return path.resolve()


def _stage(path: Path, dataset: str) -> dict[str, object]:
    document = _load(path)
    expected_count = EXPECTED_COUNTS[dataset]
    if (
        document.get("schema_version") != "openprism.protocol-pair-stage/1.0"
        or document.get("dataset") != dataset
        or document.get("partition") != "validation"
        or document.get("final_test_unlocked") is not False
        or document.get("protocol_count") != expected_count
    ):
        raise ExternalValidationMatrixError(f"invalid validation stage manifest: {path}")
    items = document.get("items")
    if type(items) is not list or len(items) != expected_count:
        raise ExternalValidationMatrixError(f"partial validation stage manifest: {path}")
    visible_dir, thermal_dir = path.parent / "visible", path.parent / "thermal"
    if not visible_dir.is_dir() or not thermal_dir.is_dir():
        raise ExternalValidationMatrixError(f"stage pair directories are missing: {path.parent}")
    ids: set[str] = set()
    for index, item in enumerate(items):
        if type(item) is not dict or not isinstance(item.get("sample_id"), str):
            raise ExternalValidationMatrixError(f"invalid stage item {index}: {path}")
        sample_id = str(item["sample_id"])
        if not sample_id or sample_id in ids:
            raise ExternalValidationMatrixError(f"duplicate/empty stage ID {sample_id!r}")
        ids.add(sample_id)
        for modality, directory in (("visible", visible_dir), ("thermal", thermal_dir)):
            record = item.get(modality)
            if type(record) is not dict:
                raise ExternalValidationMatrixError(f"stage item lacks {modality}: {sample_id}")
            relative, digest = record.get("staged"), record.get("sha256")
            if not isinstance(relative, str) or Path(relative).name != relative:
                raise ExternalValidationMatrixError(f"unsafe staged filename: {sample_id}")
            image = directory / relative
            if not image.is_file() or image.is_symlink() or _sha256(image) != digest:
                raise ExternalValidationMatrixError(
                    f"staged {modality} hash mismatch: {dataset}:{sample_id}"
                )
    return {
        "manifest": str(path),
        "manifest_sha256": _sha256(path),
        "visible_dir": str(visible_dir.resolve()),
        "thermal_dir": str(thermal_dir.resolve()),
        "sample_ids": sorted(ids),
        "sample_ids_sha256": hashlib.sha256(
            "\n".join(sorted(ids)).encode("utf-8")
        ).hexdigest(),
    }


def _lock(path: Path) -> dict[str, dict[str, str]]:
    document = _load(path)
    if document.get("schema_version") != "openprism.external-baselines-lock/1.0":
        raise ExternalValidationMatrixError("unsupported external baseline lock")
    candidates = document.get("candidates")
    if type(candidates) is not list:
        raise ExternalValidationMatrixError("baseline lock lacks candidates")
    result: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        if type(candidate) is not dict or candidate.get("id") not in BASELINES:
            continue
        identifier = str(candidate["id"])
        revision = candidate.get("revision")
        status = candidate.get("weights_status")
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ExternalValidationMatrixError(f"invalid locked revision: {identifier}")
        matches = SHA256_RE.findall(str(status))
        if not matches:
            raise ExternalValidationMatrixError(f"no aggregate weight hash: {identifier}")
        result[identifier] = {"revision": revision, "weights_sha256": matches[0]}
    if set(result) != set(BASELINES):
        raise ExternalValidationMatrixError("baseline lock is incomplete")
    return result


def _verified_run(
    directory: Path, *, baseline: str, dataset: str, expected_ids: set[str]
) -> dict[str, object] | None:
    manifest = directory / "run_manifest.json"
    if not manifest.is_file():
        return None
    document = _load(manifest)
    if (
        document.get("schema_version") != RUN_SCHEMA
        or document.get("baseline") != baseline
        or document.get("dataset") != dataset
        or document.get("partition") != "validation"
        or document.get("final_test_unlocked") is not False
        or document.get("one_shot_controller_authorized") is not False
        or document.get("run_complete") is not True
        or document.get("input_count") != len(expected_ids)
    ):
        return None
    outputs = document.get("outputs")
    failures = document.get("failure_accounting")
    if (
        type(outputs) is not list
        or type(failures) is not dict
        or failures.get("failed") != 0
        or failures.get("failures") != []
    ):
        return None
    observed: set[str] = set()
    for item in outputs:
        if type(item) is not dict or not isinstance(item.get("sample_id"), str):
            return None
        sample_id, relative, digest = item["sample_id"], item.get("path"), item.get("sha256")
        if sample_id in observed or not isinstance(relative, str) or Path(relative).name != relative:
            return None
        image = directory / relative
        if not image.is_file() or image.is_symlink() or _sha256(image) != digest:
            return None
        observed.add(sample_id)
    actual_files = {item.name for item in directory.iterdir() if item.is_file()}
    declared_files = {str(item["path"]) for item in outputs} | {"run_manifest.json"}
    if observed != expected_ids or actual_files != declared_files:
        return None
    return {
        "directory": str(directory.resolve()),
        "run_manifest": str(manifest.resolve()),
        "run_manifest_sha256": _sha256(manifest),
        "output_count": len(observed),
    }


def run(spec_path: Path, output_manifest: Path) -> dict[str, object]:
    spec_path = _absolute(spec_path, "spec", kind="file")
    spec = _load(spec_path)
    if spec.get("schema_version") != SPEC_SCHEMA:
        raise ExternalValidationMatrixError(f"spec schema must be {SPEC_SCHEMA}")
    python = _absolute(spec.get("python_executable"), "python_executable", kind="file")
    repository = _absolute(spec.get("repository_root"), "repository_root", kind="directory")
    lock_path = _absolute(spec.get("baselines_lock"), "baselines_lock", kind="file")
    output_root_raw = spec.get("output_root")
    if not isinstance(output_root_raw, str) or not Path(output_root_raw).is_absolute():
        raise ExternalValidationMatrixError("output_root must be absolute")
    output_root = Path(output_root_raw).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stages_raw, baselines_raw = spec.get("stage_manifests"), spec.get("baselines")
    if type(stages_raw) is not dict or set(stages_raw) != set(DATASETS):
        raise ExternalValidationMatrixError("stage_manifests must contain all three datasets")
    if type(baselines_raw) is not dict or set(baselines_raw) != set(BASELINES):
        raise ExternalValidationMatrixError("baselines must contain all four methods")
    stages = {
        dataset: _stage(
            _absolute(stages_raw[dataset], f"stage_manifests.{dataset}", kind="file"),
            dataset,
        )
        for dataset in DATASETS
    }
    locked = _lock(lock_path)
    baseline_paths: dict[str, dict[str, object]] = {}
    for baseline in BASELINES:
        raw = baselines_raw[baseline]
        if (
            type(raw) is not dict
            or not {"source_root", "weights"}.issubset(raw)
            or set(raw) - {"source_root", "weights", "allowed_case_collision_paths"}
        ):
            raise ExternalValidationMatrixError(f"invalid baseline paths: {baseline}")
        collision_paths = raw.get("allowed_case_collision_paths", [])
        if (
            type(collision_paths) is not list
            or any(type(item) is not str or not item.strip() for item in collision_paths)
            or len(set(collision_paths)) != len(collision_paths)
        ):
            raise ExternalValidationMatrixError(
                f"invalid allowed case-collision paths: {baseline}"
            )
        weights_path = Path(str(raw["weights"]))
        baseline_paths[baseline] = {
            "source_root": _absolute(
                raw["source_root"], f"baselines.{baseline}.source_root", kind="directory"
            ),
            "weights": _absolute(
                raw["weights"], f"baselines.{baseline}.weights",
                kind="directory" if weights_path.is_dir() else "file",
            ),
            "allowed_case_collision_paths": tuple(collision_paths),
        }
    device = spec.get("device", "cuda:0")
    if not isinstance(device, str) or not device or device.casefold() == "cpu":
        raise ExternalValidationMatrixError("the frozen external matrix requires a CUDA device")

    report: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "partition": "validation",
        "final_test_unlocked": False,
        "spec": {"path": str(spec_path), "sha256": _sha256(spec_path)},
        "baselines_lock": {"path": str(lock_path), "sha256": _sha256(lock_path)},
        "stages": stages,
        "runs": [],
        "matrix_complete": False,
    }
    logs = output_root / "logs"
    logs.mkdir(exist_ok=True)
    for dataset in DATASETS:
        expected_ids = set(stages[dataset]["sample_ids"])
        for baseline in BASELINES:
            destination = output_root / baseline / dataset
            existing = _verified_run(
                destination, baseline=baseline, dataset=dataset,
                expected_ids=expected_ids,
            ) if destination.is_dir() else None
            if existing is not None:
                print(f"SKIP complete {dataset}:{baseline}", flush=True)
                report["runs"].append({
                    "dataset": dataset, "baseline": baseline,
                    "skipped_existing_complete_run": True, **existing,
                })
                _atomic_json(output_manifest, report)
                continue
            if destination.exists():
                raise ExternalValidationMatrixError(
                    f"partial or unverifiable output exists; move it aside manually: {destination}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            argv = [
                str(python), "-m", "tooling.run_external_fusion",
                "--baseline", baseline, "--dataset", dataset,
                "--source-root", str(baseline_paths[baseline]["source_root"]),
                "--weights", str(baseline_paths[baseline]["weights"]),
                "--visible-dir", str(stages[dataset]["visible_dir"]),
                "--thermal-dir", str(stages[dataset]["thermal_dir"]),
                "--output-dir", str(destination),
                "--expected-revision", locked[baseline]["revision"],
                "--expected-weights-sha256", locked[baseline]["weights_sha256"],
                "--partition", "validation", "--device", device,
            ]
            for collision_path in baseline_paths[baseline]["allowed_case_collision_paths"]:
                argv.extend(
                    ["--allow-nonexecuted-case-collision", str(collision_path)]
                )
            stdout_path = logs / f"{dataset}-{baseline}.stdout.log"
            stderr_path = logs / f"{dataset}-{baseline}.stderr.log"
            if stdout_path.exists() or stderr_path.exists():
                raise ExternalValidationMatrixError(f"run log already exists: {dataset}:{baseline}")
            print(f"START {dataset}:{baseline}", flush=True)
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                completed = subprocess.run(
                    argv, cwd=repository, stdin=subprocess.DEVNULL,
                    stdout=stdout, stderr=stderr, shell=False, check=False,
                )
            if completed.returncode != 0:
                raise ExternalValidationMatrixError(
                    f"external validation run failed ({completed.returncode}): "
                    f"{dataset}:{baseline}; inspect {stderr_path}"
                )
            verified = _verified_run(
                destination, baseline=baseline, dataset=dataset,
                expected_ids=expected_ids,
            )
            if verified is None:
                raise ExternalValidationMatrixError(
                    f"external validation output failed verification: {dataset}:{baseline}"
                )
            report["runs"].append({
                "dataset": dataset, "baseline": baseline,
                "skipped_existing_complete_run": False,
                "stdout_sha256": _sha256(stdout_path),
                "stderr_sha256": _sha256(stderr_path),
                **verified,
            })
            _atomic_json(output_manifest, report)
            print(f"DONE {dataset}:{baseline}", flush=True)
    report["matrix_complete"] = len(report["runs"]) == len(DATASETS) * len(BASELINES)
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(output_manifest, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = run(args.spec, args.output_manifest)
    print(json.dumps({
        "partition": report["partition"],
        "matrix_complete": report["matrix_complete"],
        "run_count": len(report["runs"]),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
