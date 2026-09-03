"""Run the frozen, fusion-only PRISM-EGT latency and failure benchmark.

The primary timing table is deliberately a microbenchmark: decoded and
normalized 192 x 192 float32 tensors are resident on one device before timing,
batch size is one, gradients/AMP/compilation are disabled, and CUDA is
synchronized immediately before and after every measured call.  One fixed
hash-ranked pool of 16 samples from each protocol dataset is warmed once and
then traversed ten times.  This makes average, maximum, deterministic
OpenPRISM, and PRISM-EGT luminance kernels comparable without pretending that
image decoding, device transfer, display conversion, or disk output is free.

PRISM-EGT and deterministic OpenPRISM also have a separately labelled operator
endpoint.  Each endpoint computes fused luminance once and derives the RGB
operator rendering from that same result; there is no duplicate fusion/model
call inside an iteration.

Completed external-baseline manifests are audited but never mixed into the
primary timing table.  Their historical adapter timer used native image sizes,
included host-to-device transfer, output device-to-host transfer and PNG save,
excluded source-image decode, had no controlled warm-up, and measured each
sample once.  Re-labelling those observations as the common 192 x 192
fusion-only benchmark would be scientifically invalid.

The test partition is inaccessible unless this process was launched by the
one-shot final-suite controller: both its environment marker and exact
manifest-digest token are required before any stage manifest or checkpoint is
opened.  This module must itself be listed as a frozen final-suite step.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np
import PIL
from PIL import Image
import torch
from torch import Tensor


if __package__ in {None, ""}:  # Support ``python tooling/benchmark_fusion_latency.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openprism.learning.baselines import fuse_baseline  # noqa: E402
from openprism.learning.checkpoint import load_checkpoint  # noqa: E402


SCHEMA_VERSION = "openprism.fusion-latency-benchmark/1.0"
STAGE_SCHEMA_VERSION = "openprism.protocol-pair-stage/1.0"
FINAL_SUITE_TOKEN = "FROM_OPENPRISM_FINAL_SUITE_ENV"
IMAGE_SIZE = 192
BATCH_SIZE = 1
SAMPLES_PER_DATASET = 16
WARMUP_PASSES = 1
MEASURED_PASSES = 10
DATASETS = ("llvip", "msrs", "caltech")
EXTERNAL_METHODS = ("seafusion", "cddfuse", "paif", "c2rf")
SCIENTIFICALLY_COMPLETE_EXTERNAL_STATUSES = frozenset(
    {"complete", "completed_with_runtime_failures"}
)
EXPECTED_COUNTS = {
    "validation": {"llvip": 1_730, "msrs": 270, "caltech": 177},
    "test": {"llvip": 1_733, "msrs": 170, "caltech": 389},
}
_SHA256 = frozenset("0123456789abcdef")
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class BenchmarkError(RuntimeError):
    """The benchmark could not preserve its frozen scientific contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchmarkError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise BenchmarkError(f"non-finite JSON number is forbidden: {value}")


def _read_json(path: Path, *, maximum_bytes: int = 512 * 1024 * 1024) -> tuple[dict[str, Any], str]:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise BenchmarkError(f"cannot stat JSON artifact {path}: {error}") from error
    if path.is_symlink() or not path.is_file():
        raise BenchmarkError(f"JSON artifact must be a non-symlink regular file: {path}")
    if metadata.st_size > maximum_bytes:
        raise BenchmarkError(f"JSON artifact exceeds {maximum_bytes} bytes: {path}")
    raw = path.read_bytes()
    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except BenchmarkError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise BenchmarkError(f"invalid strict JSON artifact {path}: {error}") from error
    if type(decoded) is not dict:
        raise BenchmarkError(f"JSON artifact must contain one top-level object: {path}")
    return decoded, hashlib.sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and not (set(value) - _SHA256)
    )


def _is_portable_basename(value: object) -> bool:
    return (
        type(value) is str
        and value not in {"", ".", ".."}
        and PurePosixPath(value).name == value
        and PureWindowsPath(value).name == value
        and not value.endswith((".", " "))
    )


def _partition_authorization(
    partition: str,
    *,
    unlock_final_test: bool,
    final_suite_token: str | None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Authorize a partition without touching any data or artifact path."""

    if partition not in EXPECTED_COUNTS:
        raise BenchmarkError("partition must be validation or test")
    env = os.environ if environment is None else environment
    if partition == "validation":
        if unlock_final_test or final_suite_token is not None:
            raise BenchmarkError("validation benchmark must not carry final-test unlock material")
        return {
            "partition": "validation",
            "final_test_unlocked": False,
            "one_shot_controller_authorized": False,
            "controller_manifest_canonical_sha256": None,
            "test_data_accessed": False,
        }

    # These checks intentionally precede every Path operation in run_benchmark.
    if not unlock_final_test:
        raise BenchmarkError(
            "final-test latency benchmark is locked; --unlock-final-test is required"
        )
    if env.get("OPENPRISM_FINAL_SUITE") != "1":
        raise BenchmarkError("final test requires the one-shot controller environment")
    controller_digest = env.get("OPENPRISM_FINAL_SUITE_MANIFEST_SHA256")
    if not _is_sha256(controller_digest):
        raise BenchmarkError("one-shot controller manifest digest is absent or malformed")
    if final_suite_token != FINAL_SUITE_TOKEN:
        raise BenchmarkError(
            f"--final-suite-token must be the literal controller handoff token {FINAL_SUITE_TOKEN!r}"
        )
    return {
        "partition": "test",
        "final_test_unlocked": True,
        "one_shot_controller_authorized": True,
        "controller_manifest_canonical_sha256": controller_digest,
        "test_data_accessed": True,
    }


def _attest_one_shot_claim(token: str) -> dict[str, object]:
    """Verify that the final controller has an active, persistent claim.

    Environment variables alone are forgeable.  The controller publishes its
    claim in the repository's common Git directory before starting a child
    process, then marks exactly one declared step as running.  Requiring that
    ledger makes manual environment spoofing insufficient and binds this
    benchmark to the permanently consumed manifest identity.
    """

    root = Path(__file__).resolve().parents[1]
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--git-common-dir"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise BenchmarkError(f"cannot locate one-shot controller ledger: {error}") from error
    common_git = Path(completed.stdout.strip())
    if not common_git.is_absolute():
        common_git = root / common_git
    ledger_path = (
        common_git.resolve()
        / "openprism-final-suite"
        / "runs"
        / token
        / "ledger.json"
    )
    ledger, ledger_sha256 = _read_json(ledger_path, maximum_bytes=16 * 1024 * 1024)
    if ledger.get("schema_version") != "openprism.final-test-ledger/1.0":
        raise BenchmarkError("one-shot claim has an unsupported ledger schema")
    if ledger.get("manifest_canonical_sha256") != token:
        raise BenchmarkError("one-shot ledger is not bound to the supplied manifest token")
    if ledger.get("arming_confirmed") is not True:
        raise BenchmarkError("one-shot ledger is not armed")
    if ledger.get("status") != "running" or ledger.get("commands_started") is not True:
        raise BenchmarkError("one-shot ledger has no actively running command")
    repository = ledger.get("repository")
    if type(repository) is not dict or type(repository.get("root")) is not str:
        raise BenchmarkError("one-shot ledger lacks repository provenance")
    if Path(repository["root"]).resolve() != root:
        raise BenchmarkError("one-shot ledger belongs to a different repository")
    steps = ledger.get("steps")
    if type(steps) is not list:
        raise BenchmarkError("one-shot ledger has no declared steps")
    running = [step for step in steps if type(step) is dict and step.get("status") == "running"]
    if len(running) != 1:
        raise BenchmarkError("one-shot ledger must contain exactly one running step")
    step = running[0]
    argv = step.get("argv")
    if type(argv) is not list or not all(type(item) is str for item in argv):
        raise BenchmarkError("running one-shot step has invalid argv provenance")
    module_declared = any(
        item == "tooling.benchmark_fusion_latency"
        or Path(item).name == Path(__file__).name
        for item in argv
    )
    required_tokens = {
        "--partition",
        "test",
        "--unlock-final-test",
        "--final-suite-token",
        FINAL_SUITE_TOKEN,
    }
    if not module_declared or not required_tokens.issubset(set(argv)):
        raise BenchmarkError("running one-shot step is not this locked latency benchmark")
    return {
        "ledger_path": str(ledger_path),
        "ledger_sha256_at_benchmark_start": ledger_sha256,
        "running_step_id": step.get("id"),
        "repository_commit": repository.get("commit"),
        "repository_tag": repository.get("tag"),
        "manifest_file_sha256": ledger.get("manifest_file_sha256"),
        "persistent_one_shot_claim_verified": True,
    }


def _manifest_items(document: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    items = document.get("items")
    if type(items) is not list or not items:
        raise BenchmarkError(f"stage manifest has no complete item list: {path}")
    if not all(type(item) is dict for item in items):
        raise BenchmarkError(f"stage manifest contains a non-object item: {path}")
    return items


def _load_stage_manifest(path: Path, dataset: str, partition: str) -> dict[str, Any]:
    document, raw_sha256 = _read_json(path)
    if document.get("schema_version") != STAGE_SCHEMA_VERSION:
        raise BenchmarkError(f"unsupported stage-manifest schema: {path}")
    if document.get("dataset") != dataset or document.get("partition") != partition:
        raise BenchmarkError(f"stage manifest does not match {dataset}/{partition}: {path}")
    if partition == "test" and document.get("final_test_unlocked") is not True:
        raise BenchmarkError(f"test stage is not explicitly unlocked: {path}")
    if partition == "validation" and document.get("final_test_unlocked") is not False:
        raise BenchmarkError(f"validation stage carries an invalid final-test flag: {path}")
    expected_count = EXPECTED_COUNTS[partition][dataset]
    if document.get("protocol_count") != expected_count:
        raise BenchmarkError(
            f"stage count mismatch for {dataset}/{partition}: "
            f"expected {expected_count}, found {document.get('protocol_count')!r}"
        )
    items = _manifest_items(document, path)
    if len(items) != expected_count:
        raise BenchmarkError(f"stage item list is partial for {dataset}/{partition}: {path}")
    payload_digest = document.get("manifest_payload_sha256")
    payload = dict(document)
    payload.pop("manifest_payload_sha256", None)
    if not _is_sha256(payload_digest) or payload_digest != _canonical_sha256(payload):
        raise BenchmarkError(f"stage payload digest mismatch: {path}")
    declared_output = document.get("output_directory")
    if type(declared_output) is not str or Path(declared_output).resolve() != path.parent.resolve():
        raise BenchmarkError(f"stage manifest moved away from its declared output directory: {path}")

    sample_ids: set[str] = set()
    for index, item in enumerate(items):
        sample_id = item.get("sample_id")
        if type(sample_id) is not str or not sample_id or sample_id in sample_ids:
            raise BenchmarkError(f"invalid or duplicate sample ID at {path}:items[{index}]")
        sample_ids.add(sample_id)
        for modality in ("visible", "thermal"):
            artifact = item.get(modality)
            if type(artifact) is not dict:
                raise BenchmarkError(f"missing {modality} artifact for {sample_id}: {path}")
            staged = artifact.get("staged")
            if (
                not _is_portable_basename(staged)
                or Path(staged).suffix.lower() not in _IMAGE_SUFFIXES
                or not _is_sha256(artifact.get("sha256"))
            ):
                raise BenchmarkError(f"invalid staged {modality} artifact for {sample_id}: {path}")
    return {
        "path": path.resolve(),
        "document": document,
        "raw_sha256": raw_sha256,
        "sample_ids": frozenset(sample_ids),
        "items_by_id": {str(item["sample_id"]): item for item in items},
    }


def _ranked_ids(dataset: str, sample_ids: Iterable[str]) -> tuple[str, ...]:
    ranked = sorted(
        sample_ids,
        key=lambda sample_id: (
            hashlib.sha256(f"{dataset}:{sample_id}".encode("utf-8")).hexdigest(),
            sample_id,
        ),
    )
    if len(ranked) < SAMPLES_PER_DATASET:
        raise BenchmarkError(
            f"{dataset} has fewer than {SAMPLES_PER_DATASET} staged samples"
        )
    return tuple(ranked[:SAMPLES_PER_DATASET])


def _thermal_normalize(value: np.ndarray) -> np.ndarray:
    thermal = np.asarray(value, dtype=np.float32)
    if thermal.ndim == 3:
        thermal = thermal.mean(axis=2)
    finite = thermal[np.isfinite(thermal)]
    if not finite.size:
        return np.zeros(thermal.shape, dtype=np.float32)
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low + 1e-8:
        return np.zeros(thermal.shape, dtype=np.float32)
    return np.clip((thermal - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _resize_center_crop(
    visible: np.ndarray, thermal: np.ndarray, size: int = IMAGE_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    height, width = visible.shape[:2]
    if thermal.shape != (height, width):
        raise BenchmarkError(
            f"visible/thermal geometry mismatch: {visible.shape[:2]} versus {thermal.shape}"
        )
    scale = max(size / height, size / width)
    if scale > 1.0:
        new_size = (int(round(width * scale)), int(round(height * scale)))
        visible = np.asarray(
            Image.fromarray(visible, mode="RGB").resize(new_size, Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        thermal = np.asarray(
            Image.fromarray(thermal.astype(np.float32), mode="F").resize(
                new_size, Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        height, width = visible.shape[:2]
    top = max(0, (height - size) // 2)
    left = max(0, (width - size) // 2)
    visible = visible[top : top + size, left : left + size]
    thermal = thermal[top : top + size, left : left + size]
    if visible.shape != (size, size, 3) or thermal.shape != (size, size):
        raise BenchmarkError(
            f"fixed preprocessing failed to produce {size}x{size}: "
            f"{visible.shape}/{thermal.shape}"
        )
    return np.ascontiguousarray(visible), np.ascontiguousarray(thermal)


def _prepare_pair(
    stage: Mapping[str, Any], dataset: str, sample_id: str, device: torch.device
) -> dict[str, object]:
    item = stage["items_by_id"][sample_id]
    stage_root = Path(stage["path"]).parent
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for modality in ("visible", "thermal"):
        artifact = item[modality]
        candidate = stage_root / modality / artifact["staged"]
        if candidate.is_symlink() or not candidate.is_file():
            raise BenchmarkError(
                f"staged {modality} is not a non-symlink regular file for "
                f"{dataset}:{sample_id}"
            )
        observed = _sha256(candidate)
        if observed != artifact["sha256"]:
            raise BenchmarkError(
                f"staged {modality} hash mismatch for {dataset}:{sample_id}"
            )
        paths[modality] = candidate
        hashes[modality] = observed
    with Image.open(paths["visible"]) as image:
        visible = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(paths["thermal"]) as image:
        thermal = _thermal_normalize(np.asarray(image.convert("F"), dtype=np.float32))
    visible, thermal = _resize_center_crop(visible, thermal)
    rgb = torch.from_numpy(np.moveaxis(visible.astype(np.float32) / 255.0, -1, 0).copy())[
        None
    ].to(device)
    thermal_tensor = torch.from_numpy(thermal.copy())[None, None].to(device)
    evidence = torch.ones((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32, device=device)
    return {
        "key": f"{dataset}:{sample_id}",
        "dataset": dataset,
        "sample_id": sample_id,
        "rgb": rgb,
        "thermal": thermal_tensor,
        "evidence": evidence,
        "visible_sha256": hashes["visible"],
        "thermal_sha256": hashes["thermal"],
    }


def _replace_luminance_tensor(rgb: Tensor, target_y: Tensor, preserve: float = 0.88) -> Tensor:
    """Torch equivalent of the published OpenPRISM operator renderer."""

    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape Bx3xHxW")
    if target_y.shape != (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]):
        raise ValueError("target_y must have shape Bx1xHxW matching rgb")
    red, green, blue = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    source_y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = (blue - source_y) * (0.564 * preserve)
    cr = (red - source_y) * (0.713 * preserve)
    return torch.cat(
        (
            target_y + 1.403 * cr,
            target_y - 0.344 * cb - 0.714 * cr,
            target_y + 1.773 * cb,
        ),
        dim=1,
    ).clamp(0.0, 1.0)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_method_output(value: object, expected_outputs: Sequence[str]) -> None:
    tensors = value if isinstance(value, tuple) else (value,)
    if len(tensors) != len(expected_outputs) or not all(isinstance(item, Tensor) for item in tensors):
        raise BenchmarkError(
            f"method did not return declared outputs {tuple(expected_outputs)}"
        )
    expected_channels = {"luminance": 1, "operator_rgb_float": 3}
    for name, tensor in zip(expected_outputs, tensors):
        assert isinstance(tensor, Tensor)
        expected_shape = (BATCH_SIZE, expected_channels[name], IMAGE_SIZE, IMAGE_SIZE)
        if tuple(tensor.shape) != expected_shape:
            raise BenchmarkError(f"{name} output shape is {tuple(tensor.shape)}, expected {expected_shape}")
        if not bool(torch.isfinite(tensor).all().item()):
            raise BenchmarkError(f"{name} output contains a non-finite value")


def _failure_summary(failures: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    counts: Counter[str] = Counter(str(item["reason"]) for item in failures)
    return [
        {
            "reason": reason,
            "count": count,
            "example_sample_keys": [
                str(item["sample_key"])
                for item in failures
                if item["reason"] == reason
            ][:8],
        }
        for reason, count in sorted(counts.items())
    ]


def _latency_summary(milliseconds: Sequence[float]) -> dict[str, float | int | None]:
    values = np.asarray(milliseconds, dtype=np.float64)
    if not values.size:
        return {
            "recorded": 0,
            "mean_ms": None,
            "standard_deviation_ms": None,
            "minimum_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "maximum_ms": None,
            "throughput_images_per_second": None,
        }
    total_seconds = float(values.sum()) / 1_000.0
    return {
        "recorded": int(values.size),
        "mean_ms": float(values.mean()),
        "standard_deviation_ms": float(values.std()),
        "minimum_ms": float(values.min()),
        "p50_ms": float(np.quantile(values, 0.50)),
        "p90_ms": float(np.quantile(values, 0.90)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "maximum_ms": float(values.max()),
        "throughput_images_per_second": (
            float(values.size / total_seconds) if total_seconds > 0.0 else None
        ),
    }


def _benchmark_callable(
    name: str,
    function: Callable[[Mapping[str, object]], object],
    inputs: Sequence[Mapping[str, object]],
    device: torch.device,
    *,
    expected_outputs: Sequence[str],
) -> dict[str, object]:
    if not inputs:
        raise BenchmarkError("benchmark input pool is empty")
    warmup_failures: list[dict[str, object]] = []
    measured_failures: list[dict[str, object]] = []
    latency: list[float] = []
    start_allocated: int | None = None
    if device.type == "cuda":
        _synchronize(device)
        start_allocated = int(torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)

    def execute(
        sample: Mapping[str, object], phase: str, pass_index: int
    ) -> tuple[bool, float | None]:
        try:
            _synchronize(device)
            started_ns = time.perf_counter_ns()
            value = function(sample)
            _synchronize(device)
            elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
            _validate_method_output(value, expected_outputs)
            return True, elapsed_ms
        except Exception as error:  # Failure accounting is a benchmark outcome.
            reason = f"{type(error).__name__}: {error}"
            target = warmup_failures if phase == "warmup" else measured_failures
            target.append(
                {
                    "phase": phase,
                    "pass_index": pass_index,
                    "sample_key": str(sample["key"]),
                    "reason": reason,
                }
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            return False, None

    with torch.inference_mode():
        for pass_index in range(WARMUP_PASSES):
            for sample in inputs:
                execute(sample, "warmup", pass_index)
        for pass_index in range(MEASURED_PASSES):
            for sample in inputs:
                successful, elapsed = execute(sample, "measured", pass_index)
                if successful and elapsed is not None:
                    latency.append(elapsed)

    attempted = MEASURED_PASSES * len(inputs)
    failed = len(measured_failures)
    peak_process = None
    incremental_peak = None
    end_allocated = None
    if device.type == "cuda":
        _synchronize(device)
        peak_process = int(torch.cuda.max_memory_allocated(device))
        end_allocated = int(torch.cuda.memory_allocated(device))
        incremental_peak = max(0, peak_process - int(start_allocated or 0))
    return {
        "method": name,
        "declared_outputs": list(expected_outputs),
        "attempted": attempted,
        "successful": attempted - failed,
        "failed": failed,
        "failure_rate": failed / attempted,
        "failure_reasons": _failure_summary(measured_failures),
        "warmup": {
            "passes": WARMUP_PASSES,
            "attempted": WARMUP_PASSES * len(inputs),
            "failed": len(warmup_failures),
            "failure_reasons": _failure_summary(warmup_failures),
        },
        "measured_passes_per_input": MEASURED_PASSES,
        "latency": _latency_summary(latency),
        "cuda_memory": {
            "starting_process_allocated_bytes": start_allocated,
            "peak_process_allocated_bytes": peak_process,
            "incremental_peak_above_method_start_bytes": incremental_peak,
            "ending_process_allocated_bytes": end_allocated,
            "interpretation": (
                "incremental allocated memory above the method-start process allocation; "
                "allocator reserved memory and external-process VRAM are excluded"
                if device.type == "cuda"
                else "not applicable on a non-CUDA device"
            ),
        },
    }


def _checkpoint_size(report: Mapping[str, Any]) -> dict[str, object]:
    path_value = report.get("weights")
    declared = report.get("weight_files")
    declared_total = report.get("weights_total_bytes")
    declared_sizes = report.get("weight_file_bytes")
    if type(path_value) is not str or type(declared) is not dict:
        return {
            "bytes": declared_total if type(declared_total) is int else None,
            "available": type(declared_total) is int,
            "reason": "checkpoint path and byte inventory not both recorded by adapter",
        }
    root = Path(path_value)
    if root.is_file():
        paths = {root.name: root}
    elif root.is_dir():
        paths = {}
        for relative in declared:
            if type(relative) is not str:
                return {
                    "bytes": None,
                    "available": False,
                    "reason": "checkpoint member name is not a string",
                }
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
                return {
                    "bytes": None,
                    "available": False,
                    "reason": "checkpoint member path is unsafe",
                }
            paths[relative] = root / Path(*pure.parts)
    else:
        return {"bytes": None, "available": False, "reason": "checkpoint path unavailable"}
    try:
        total = sum(path.stat().st_size for path in paths.values())
    except OSError as error:
        return {"bytes": None, "available": False, "reason": str(error)}
    return {
        "bytes": int(total),
        "available": True,
        "reason": None,
        "byte_hashes_reverified": False,
        "declared_total_bytes": declared_total if type(declared_total) is int else None,
        "declared_file_bytes": declared_sizes if type(declared_sizes) is dict else None,
        "current_sizes_match_declared": (
            type(declared_total) is int
            and type(declared_sizes) is dict
            and total == declared_total
            and {
                relative: int(path.stat().st_size)
                for relative, path in paths.items()
            }
            == declared_sizes
        ),
        "declared_weights_sha256": report.get("weights_sha256"),
    }


def _external_latency_report(
    dataset: str,
    method: str,
    path: Path,
    stage: Mapping[str, Any],
) -> dict[str, object]:
    document, manifest_sha256 = _read_json(path)
    if document.get("schema_version") not in {
        "openprism.external-fusion-run/1.1",
        "openprism.external-fusion-run/1.2",
        "openprism.external-fusion-run/1.3",
    }:
        raise BenchmarkError(f"unsupported external run schema: {path}")
    if document.get("baseline") != method:
        raise BenchmarkError(f"external manifest baseline mismatch for {dataset}:{method}")
    inputs = document.get("inputs")
    outputs = document.get("outputs")
    if type(inputs) is not dict or type(outputs) is not list:
        raise BenchmarkError(f"external run has incomplete inputs/outputs: {path}")
    paired = inputs.get("paired_ids_sorted")
    expected_ids = set(stage["sample_ids"])
    if type(paired) is not list or set(paired) != expected_ids or len(paired) != len(expected_ids):
        raise BenchmarkError(f"external run is not complete for {dataset}: {path}")
    stage_root = Path(stage["path"]).parent.resolve()
    for modality in ("visible", "thermal"):
        declared_directory = inputs.get(f"{modality}_directory")
        if type(declared_directory) is not str or Path(declared_directory).resolve() != stage_root / modality:
            raise BenchmarkError(
                f"external run {dataset}:{method} is not bound to the frozen stage"
            )
    if document.get("input_count") != len(expected_ids):
        raise BenchmarkError(f"external input count mismatch for {dataset}:{method}")

    output_by_id: dict[str, Mapping[str, Any]] = {}
    invalid: list[dict[str, object]] = []
    timing_ms: list[float] = []
    for index, output in enumerate(outputs):
        if type(output) is not dict:
            invalid.append({"reason": "non-object output entry", "index": index})
            continue
        sample_id = output.get("sample_id")
        relative = output.get("path")
        elapsed = output.get("elapsed_seconds")
        if type(sample_id) is not str or sample_id in output_by_id:
            invalid.append({"reason": "invalid or duplicate sample_id", "index": index})
            continue
        output_by_id[sample_id] = output
        if (
            not _is_portable_basename(relative)
            or not _is_sha256(output.get("sha256"))
        ):
            invalid.append({"reason": "invalid output artifact metadata", "sample_id": sample_id})
            continue
        artifact = path.parent / relative
        if artifact.is_symlink() or not artifact.is_file():
            invalid.append({"reason": "missing output artifact", "sample_id": sample_id})
            continue
        if _sha256(artifact) != output["sha256"]:
            invalid.append({"reason": "output SHA-256 mismatch", "sample_id": sample_id})
            continue
        if isinstance(elapsed, bool) or not isinstance(elapsed, int | float):
            invalid.append({"reason": "missing elapsed_seconds", "sample_id": sample_id})
            continue
        elapsed_value = float(elapsed)
        if not math.isfinite(elapsed_value) or elapsed_value < 0.0:
            invalid.append({"reason": "invalid elapsed_seconds", "sample_id": sample_id})
            continue
        timing_ms.append(elapsed_value * 1_000.0)
    declared_accounting = document.get("failure_accounting")
    declared_runtime_failures: dict[str, Mapping[str, Any]] = {}
    accounting_valid = True
    if declared_accounting is not None:
        if type(declared_accounting) is not dict:
            accounting_valid = False
        else:
            declared_failure_list = declared_accounting.get("failures")
            if type(declared_failure_list) is not list:
                accounting_valid = False
            else:
                for failure in declared_failure_list:
                    if (
                        type(failure) is not dict
                        or type(failure.get("sample_id")) is not str
                        or failure["sample_id"] in declared_runtime_failures
                    ):
                        accounting_valid = False
                        continue
                    declared_runtime_failures[failure["sample_id"]] = failure
            expected_accounting = {
                "attempted": len(expected_ids),
                "successful": len(output_by_id),
                "failed": len(expected_ids) - len(output_by_id),
            }
            if any(
                declared_accounting.get(key) != value
                for key, value in expected_accounting.items()
            ):
                accounting_valid = False
    if not accounting_valid:
        invalid.append({"reason": "invalid external failure-accounting manifest"})

    missing_ids = sorted(expected_ids - set(output_by_id))
    unexpected_ids = sorted(set(output_by_id) - expected_ids)
    for sample_id in missing_ids:
        runtime_failure = declared_runtime_failures.get(sample_id)
        invalid.append(
            {
                "reason": (
                    "external runtime failure: " + str(runtime_failure.get("reason"))
                    if runtime_failure is not None
                    else "missing output entry"
                ),
                "sample_id": sample_id,
                "failure_class": (
                    "runtime" if runtime_failure is not None else "integrity_or_completeness"
                ),
            }
        )
    invalid.extend(
        {
            "reason": "unexpected output entry",
            "sample_id": sample_id,
            "failure_class": "integrity_or_completeness",
        }
        for sample_id in unexpected_ids
    )
    undeclared_failures = sorted(set(declared_runtime_failures) - set(missing_ids))
    invalid.extend(
        {
            "reason": "declared runtime failure also has an output",
            "sample_id": sample_id,
            "failure_class": "integrity_or_completeness",
        }
        for sample_id in undeclared_failures
    )
    attempted = len(expected_ids)
    failed_ids = {
        str(item.get("sample_id")) for item in invalid if item.get("sample_id") is not None
    }
    successful = max(0, attempted - len(failed_ids)) if not any(
        item.get("sample_id") is None for item in invalid
    ) else len(timing_ms)
    output_digest = _canonical_sha256(
        [
            [sample_id, output_by_id[sample_id].get("sha256")]
            for sample_id in sorted(output_by_id)
        ]
    )
    adapter = document.get("adapter") if type(document.get("adapter")) is dict else {}
    inventory = document.get("parameter_inventory")
    if type(inventory) is not dict:
        inventory = adapter.get("parameter_inventory")
    if type(inventory) is not dict:
        inventory = {}
    total_parameters = inventory.get("total_parameters")
    trainable_parameters = inventory.get("trainable_parameters")
    if isinstance(total_parameters, bool) or not isinstance(total_parameters, int):
        total_parameters = None
    if isinstance(trainable_parameters, bool) or not isinstance(trainable_parameters, int):
        trainable_parameters = None
    has_integrity_failure = any(
        item.get("failure_class") != "runtime" for item in invalid
    )
    archived_total_seconds = document.get("elapsed_seconds")
    if (
        isinstance(archived_total_seconds, bool)
        or not isinstance(archived_total_seconds, int | float)
        or not math.isfinite(float(archived_total_seconds))
        or float(archived_total_seconds) <= 0.0
    ):
        archived_total_seconds = None
    resources = document.get("runtime_resources")
    peak_memory = (
        resources.get("peak_cuda_allocated_bytes")
        if type(resources) is dict
        else None
    )
    return {
        "dataset": dataset,
        "method": method,
        "status": (
            "complete"
            if not invalid
            else "failed_integrity_or_completeness"
            if has_integrity_failure
            else "completed_with_runtime_failures"
        ),
        "attempted": attempted,
        "successful": successful,
        "failed": attempted - successful,
        "failure_rate": (attempted - successful) / attempted,
        "failure_reasons": _failure_summary(
            [
                {
                    "reason": str(item["reason"]),
                    "sample_key": f"{dataset}:{item.get('sample_id', 'manifest')}"
                }
                for item in invalid
            ]
        ),
        "historical_latency": _latency_summary(timing_ms),
        "archived_total_run": {
            "elapsed_seconds": (
                float(archived_total_seconds)
                if archived_total_seconds is not None
                else None
            ),
            "attempted_images_per_second": (
                attempted / float(archived_total_seconds)
                if archived_total_seconds is not None
                else None
            ),
            "scope": "all attempts plus input decode and manifest-loop overhead",
        },
        "cuda_memory": {
            "peak_allocated_bytes": peak_memory if type(peak_memory) is int else None,
            "available": type(peak_memory) is int,
            "reason": (
                None
                if type(peak_memory) is int
                else (
                    "the archived native-resolution adapter did not reset and sample "
                    "peak CUDA memory; no value is inferred after the fact"
                )
            ),
            "runtime_resources": resources if type(resources) is dict else None,
        },
        "timing_scope": {
            "comparison_class": "external_native_resolution_archived_pipeline_noncomparable",
            "input_shape": "native per-image resolution; not fixed at 192x192",
            "batch_size": 1,
            "warmup": "none recorded",
            "repetitions_per_input": 1,
            "includes": [
                "adapter preprocessing after decoded tensors",
                "host-to-device transfer",
                "model inference",
                "device-to-host transfer",
                "uint8 conversion",
                "PNG encoding and disk write",
            ],
            "excludes": ["source image decode", "downstream perception"],
            "cuda_synchronization": (
                "implicit at device-to-host transfer before PNG save; no explicit pre-call sync"
            ),
            "prohibition": (
                "must not be ranked numerically against the common fixed-shape fusion-only table"
            ),
        },
        "parameter_count": {
            "total": total_parameters,
            "trainable": trainable_parameters,
            "available": total_parameters is not None,
            "reason": None if total_parameters is not None else "not recorded by external adapter",
            "counting_policy": inventory.get("counting_policy"),
            "unique_parameter_tensors": inventory.get("unique_parameter_tensors"),
            "shared_parameter_references_deduplicated": inventory.get(
                "shared_parameter_references_deduplicated"
            ),
        },
        "checkpoint": _checkpoint_size(document),
        "provenance": {
            "run_manifest_path": str(path.resolve()),
            "run_manifest_sha256": manifest_sha256,
            "revision": document.get("revision"),
            "weights_sha256": document.get("weights_sha256"),
            "adapter_source_sha256": document.get("adapter_source_sha256"),
            "source_attestation": document.get("source_attestation"),
            "runtime": document.get("runtime"),
            "declared_output_set_sha256": output_digest,
            "output_bytes_reverified": True,
            "stage_manifest_sha256": stage["raw_sha256"],
        },
    }


def _runtime_environment(device: torch.device) -> dict[str, object]:
    driver_versions: list[str] | None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        driver_versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        driver_versions = None
    try:
        process_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        visible_compute_processes = [
            line.strip()
            for line in process_result.stdout.splitlines()
            if line.strip()
        ]
    except (OSError, subprocess.SubprocessError):
        visible_compute_processes = None
    cuda_device = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        cuda_device = {
            "index": int(index),
            "name": properties.name,
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
        }
    packages: dict[str, str | None] = {}
    for package in ("numpy", "Pillow", "torch", "torchvision"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_sha256": _sha256(Path(sys.executable)),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "packages": packages,
        "numpy_runtime": np.__version__,
        "pillow_runtime": PIL.__version__,
        "torch_runtime": torch.__version__,
        "device": str(device),
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_device,
        "nvidia_driver_versions": driver_versions,
        "nvidia_smi_compute_processes_at_report_time": visible_compute_processes,
        "current_process_id": os.getpid(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    }


@contextmanager
def _fixed_torch_policy() -> Iterator[None]:
    deterministic_before = torch.are_deterministic_algorithms_enabled()
    warn_before = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic_before = torch.backends.cudnn.deterministic
    cudnn_benchmark_before = torch.backends.cudnn.benchmark
    matmul_tf32_before = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32_before = torch.backends.cudnn.allow_tf32
    torch.manual_seed(20260903)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260903)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic_before, warn_only=warn_before)
        torch.backends.cudnn.deterministic = cudnn_deterministic_before
        torch.backends.cudnn.benchmark = cudnn_benchmark_before
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32_before
        torch.backends.cudnn.allow_tf32 = cudnn_tf32_before


def _git_provenance(root: Path) -> dict[str, object]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return completed.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        status = git("status", "--porcelain=v1", "--untracked-files=all")
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": None, "worktree_clean": None}
    return {
        "available": True,
        "commit": commit,
        "worktree_clean": not bool(status),
        "dirty_entry_count": len(status.splitlines()) if status else 0,
    }


def _parse_assignment(value: str, *, external: bool) -> tuple[tuple[str, str] | str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected KEY=PATH")
    key, raw_path = value.split("=", 1)
    path = Path(raw_path)
    if external:
        if ":" not in key:
            raise argparse.ArgumentTypeError("external key must be DATASET:METHOD")
        dataset, method = key.split(":", 1)
        if dataset not in DATASETS or method not in EXTERNAL_METHODS:
            raise argparse.ArgumentTypeError("unsupported external dataset or method")
        return (dataset, method), path
    if key not in DATASETS:
        raise argparse.ArgumentTypeError(f"stage key must be one of {DATASETS}")
    return key, path


def _assignments(
    values: Sequence[tuple[tuple[str, str] | str, Path]], label: str
) -> dict[tuple[str, str] | str, Path]:
    result: dict[tuple[str, str] | str, Path] = {}
    for key, path in values:
        if key in result:
            raise BenchmarkError(f"duplicate {label} assignment: {key}")
        result[key] = path
    return result


def run_benchmark(
    checkpoint: str | Path,
    stage_manifests: Mapping[str, str | Path],
    output: str | Path,
    *,
    partition: str,
    external_runs: Mapping[tuple[str, str], str | Path] | None = None,
    device_name: str = "auto",
    unlock_final_test: bool = False,
    final_suite_token: str | None = None,
    environment: Mapping[str, str] | None = None,
) -> dict[str, object]:
    authorization = _partition_authorization(
        partition,
        unlock_final_test=unlock_final_test,
        final_suite_token=final_suite_token,
        environment=environment,
    )
    if partition == "test":
        # Only controller metadata is touched here; stage/checkpoint paths are
        # still unopened until the persistent one-shot claim is verified.
        controller_digest = authorization["controller_manifest_canonical_sha256"]
        assert isinstance(controller_digest, str)
        authorization.update(_attest_one_shot_claim(controller_digest))
    if set(stage_manifests) != set(DATASETS):
        raise BenchmarkError(f"exactly one stage manifest is required for each of {DATASETS}")
    destination = Path(output).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"benchmark output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Data/checkpoint access starts only after the final-test authorization above.
    stages = {
        dataset: _load_stage_manifest(
            Path(stage_manifests[dataset]).resolve(), dataset, partition
        )
        for dataset in DATASETS
    }
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise BenchmarkError("CUDA was requested but is unavailable")
    checkpoint_path = Path(checkpoint).resolve()
    checkpoint_sha256 = _sha256(checkpoint_path)
    checkpoint_bytes = checkpoint_path.stat().st_size
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    preparation_failures: list[dict[str, object]] = []
    benchmark_inputs: list[dict[str, object]] = []
    selected_ids: dict[str, tuple[str, ...]] = {}
    for dataset in DATASETS:
        selected_ids[dataset] = _ranked_ids(dataset, stages[dataset]["sample_ids"])
        for sample_id in selected_ids[dataset]:
            try:
                benchmark_inputs.append(
                    _prepare_pair(stages[dataset], dataset, sample_id, device)
                )
            except Exception as error:
                preparation_failures.append(
                    {
                        "sample_key": f"{dataset}:{sample_id}",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
    if not benchmark_inputs:
        raise BenchmarkError(f"no benchmark input could be prepared: {preparation_failures}")

    def baseline(name: str) -> Callable[[Mapping[str, object]], Tensor]:
        return lambda sample: fuse_baseline(
            name,
            sample["rgb"],  # type: ignore[arg-type]
            sample["thermal"],  # type: ignore[arg-type]
            sample["evidence"],  # type: ignore[arg-type]
        )

    def learned_luminance(sample: Mapping[str, object]) -> Tensor:
        return model(
            sample["rgb"],  # type: ignore[arg-type]
            sample["thermal"],  # type: ignore[arg-type]
            sample["evidence"],  # type: ignore[arg-type]
            task_ids=None,
            pose_context=None,
        ).fused_luminance

    def deterministic_shared(sample: Mapping[str, object]) -> tuple[Tensor, Tensor]:
        luminance = baseline("deterministic_openprism")(sample)
        operator = _replace_luminance_tensor(sample["rgb"], luminance)  # type: ignore[arg-type]
        return luminance, operator

    def learned_shared(sample: Mapping[str, object]) -> tuple[Tensor, Tensor]:
        learned = model(
            sample["rgb"],  # type: ignore[arg-type]
            sample["thermal"],  # type: ignore[arg-type]
            sample["evidence"],  # type: ignore[arg-type]
            task_ids=None,
            pose_context=None,
        )
        operator = _replace_luminance_tensor(
            sample["rgb"], learned.fused_luminance  # type: ignore[arg-type]
        )
        return learned.fused_luminance, operator

    with _fixed_torch_policy():
        common = [
            _benchmark_callable(
                "average_luminance",
                baseline("average"),
                benchmark_inputs,
                device,
                expected_outputs=("luminance",),
            ),
            _benchmark_callable(
                "maximum_luminance",
                baseline("maximum"),
                benchmark_inputs,
                device,
                expected_outputs=("luminance",),
            ),
            _benchmark_callable(
                "deterministic_openprism_luminance_tensor",
                baseline("deterministic_openprism"),
                benchmark_inputs,
                device,
                expected_outputs=("luminance",),
            ),
            _benchmark_callable(
                "prism_egt_luminance_automatic_task",
                learned_luminance,
                benchmark_inputs,
                device,
                expected_outputs=("luminance",),
            ),
        ]
        operator_endpoints = [
            _benchmark_callable(
                "deterministic_openprism_shared_luminance_operator_tensor",
                deterministic_shared,
                benchmark_inputs,
                device,
                expected_outputs=("luminance", "operator_rgb_float"),
            ),
            _benchmark_callable(
                "prism_egt_shared_luminance_operator_automatic_task",
                learned_shared,
                benchmark_inputs,
                device,
                expected_outputs=("luminance", "operator_rgb_float"),
            ),
        ]
        runtime = _runtime_environment(device)

    external_map = dict(external_runs or {})
    unknown_external = set(external_map) - {
        (dataset, method) for dataset in DATASETS for method in EXTERNAL_METHODS
    }
    if unknown_external:
        raise BenchmarkError(f"unsupported external run assignments: {sorted(unknown_external)}")
    external: list[dict[str, object]] = []
    missing_external: list[str] = []
    for dataset in DATASETS:
        for method in EXTERNAL_METHODS:
            key = (dataset, method)
            if key not in external_map:
                missing_external.append(f"{dataset}:{method}")
                external.append(
                    {
                        "dataset": dataset,
                        "method": method,
                        "status": "missing_manifest_not_benchmarked",
                        "attempted": 0,
                        "successful": 0,
                        "failed": 0,
                        "failure_rate": None,
                        "failure_reasons": [
                            {
                                "reason": "external run manifest was not supplied",
                                "count": 1,
                                "example_sample_keys": [],
                            }
                        ],
                        "historical_latency": _latency_summary(()),
                        "archived_total_run": {
                            "elapsed_seconds": None,
                            "attempted_images_per_second": None,
                            "scope": "unavailable",
                        },
                        "cuda_memory": {
                            "peak_allocated_bytes": None,
                            "available": False,
                            "reason": "external run manifest was not supplied",
                        },
                        "timing_scope": {
                            "comparison_class": "external_native_resolution_archived_pipeline_noncomparable",
                            "prohibition": "must not be ranked against the common table",
                        },
                        "parameter_count": {
                            "total": None,
                            "trainable": None,
                            "available": False,
                            "reason": "external run manifest was not supplied",
                        },
                        "checkpoint": {
                            "bytes": None,
                            "available": False,
                            "reason": "external run manifest was not supplied",
                        },
                    }
                )
                continue
            external.append(
                _external_latency_report(
                    dataset, method, Path(external_map[key]).resolve(), stages[dataset]
                )
            )

    source_root = Path(__file__).resolve().parents[1]
    source_files = (
        Path(__file__).resolve(),
        source_root / "openprism" / "learning" / "baselines.py",
        source_root / "openprism" / "learning" / "model.py",
        source_root / "openprism" / "learning" / "checkpoint.py",
        source_root / "openprism" / "fusion.py",
    )
    method_parameters = {
        "average_luminance": {"total": 0, "trainable": 0, "checkpoint_bytes": 0},
        "maximum_luminance": {"total": 0, "trainable": 0, "checkpoint_bytes": 0},
        "deterministic_openprism_luminance_tensor": {
            "total": 0,
            "trainable": 0,
            "checkpoint_bytes": 0,
        },
        "prism_egt_luminance_automatic_task": {
            "total": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable": int(
                sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            ),
            "checkpoint_bytes": int(checkpoint_bytes),
        },
    }
    for method in common:
        method["model_artifact"] = method_parameters[str(method["method"])]
    operator_endpoints[0]["model_artifact"] = method_parameters[
        "deterministic_openprism_luminance_tensor"
    ]
    operator_endpoints[1]["model_artifact"] = method_parameters[
        "prism_egt_luminance_automatic_task"
    ]

    expected_prepared = len(DATASETS) * SAMPLES_PER_DATASET
    all_method_reports = [*common, *operator_endpoints]
    benchmark_complete = (
        len(benchmark_inputs) == expected_prepared
        and not preparation_failures
        and not missing_external
        and all(item["failed"] == 0 and item["warmup"]["failed"] == 0 for item in all_method_reports)
        and all(
            item["status"] in SCIENTIFICALLY_COMPLETE_EXTERNAL_STATUSES
            for item in external
        )
    )
    input_pool_manifest = [
        {
            "key": item["key"],
            "visible_sha256": item["visible_sha256"],
            "thermal_sha256": item["thermal_sha256"],
        }
        for item in benchmark_inputs
    ]
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "locked_final_test_result_requires_human_review"
            if partition == "test"
            else "validation_benchmark_not_final_test"
        ),
        "partition": partition,
        "final_test_unlocked": bool(authorization["final_test_unlocked"]),
        "test_lock": authorization,
        "benchmark_complete": benchmark_complete,
        "fixed_protocol": {
            "input_shape": [BATCH_SIZE, "channels", IMAGE_SIZE, IMAGE_SIZE],
            "batch_size": BATCH_SIZE,
            "dtype": "float32",
            "device": str(device),
            "sample_pool": (
                f"{SAMPLES_PER_DATASET} samples per dataset ranked by "
                "SHA-256(dataset + ':' + sample_id), independent of pixels and outputs"
            ),
            "prepared_sample_count_expected": expected_prepared,
            "warmup_passes_over_complete_pool": WARMUP_PASSES,
            "measured_passes_over_complete_pool": MEASURED_PASSES,
            "inference": (
                "torch.inference_mode; eval; batch=1; float32; no AMP; no compile; "
                "automatic task for PRISM-EGT; no pose context"
            ),
            "cuda_timing": "wall clock with torch.cuda.synchronize immediately before and after each call",
            "included_primary_scope": [
                "fusion/model tensor operations",
                "automatic task-head computation for PRISM-EGT",
            ],
            "excluded_primary_scope": [
                "file I/O and image decoding",
                "thermal percentile normalization",
                "resize/center crop",
                "host-to-device transfer",
                "device-to-host transfer",
                "uint8 quantization, display, and disk output",
                "registration, detector, segmenter, and application orchestration",
            ],
            "evidence_policy": (
                "all-one validity/registration/timing support for publisher-aligned staged pairs; "
                "evidence construction is excluded"
            ),
            "operator_endpoint_policy": (
                "fusion/model called exactly once per iteration; luminance and float RGB operator "
                "outputs share that result; operator endpoint is reported separately"
            ),
        },
        "input_preparation": {
            "attempted": expected_prepared,
            "successful": len(benchmark_inputs),
            "failed": len(preparation_failures),
            "failure_rate": len(preparation_failures) / expected_prepared,
            "failure_reasons": _failure_summary(preparation_failures),
            "selected_sample_ids": {
                dataset: list(selected_ids[dataset]) for dataset in DATASETS
            },
            "prepared_input_pool_sha256": _canonical_sha256(input_pool_manifest),
            "prepared_input_artifacts": input_pool_manifest,
        },
        "stage_manifests": {
            dataset: {
                "path": str(stages[dataset]["path"]),
                "sha256": stages[dataset]["raw_sha256"],
                "manifest_payload_sha256": stages[dataset]["document"][
                    "manifest_payload_sha256"
                ],
                "protocol_sample_manifest_sha256": stages[dataset]["document"][
                    "protocol_sample_manifest_sha256"
                ],
                "protocol_count": stages[dataset]["document"]["protocol_count"],
            }
            for dataset in DATASETS
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "bytes": int(checkpoint_bytes),
            "metadata_artifact_sha256": metadata.artifact_sha256,
            "model_id": metadata.model_id,
            "training_provenance": metadata.training_provenance,
            "validation_scope": metadata.validation_scope,
            "selected_epoch": metadata.epoch,
        },
        "source_provenance": {
            "git": _git_provenance(source_root),
            "files_sha256": {
                path.relative_to(source_root).as_posix(): _sha256(path)
                for path in source_files
            },
        },
        "runtime_environment": runtime,
        "common_fusion_only_luminance": common,
        "shared_operator_endpoints": operator_endpoints,
        "external_archived_native_resolution_runs": external,
        "external_completeness": {
            "expected": [
                f"{dataset}:{method}"
                for dataset in DATASETS
                for method in EXTERNAL_METHODS
            ],
            "missing": missing_external,
            "all_supplied": not missing_external,
            "comparison_warning": (
                "external historical latency is non-comparable and is excluded from the "
                "common fusion-only ranking"
            ),
        },
        "limitations": [
            "The common table is a warm-device microbenchmark, not end-to-end sensor-to-display latency.",
            "Inputs use declared all-one evidence and do not benchmark evidence estimation or registration.",
            "Only one batch size, precision, resolution, device, and automatic-task policy are measured.",
            "CUDA synchronization increases host-visible latency but prevents asynchronous under-reporting.",
            "External run timers have a different scope and must remain in a separate table.",
            "No calibrated Pixhawk flight, navigation, mapping, or operator-performance claim is evaluated.",
        ],
    }
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--stage-manifest",
        action="append",
        type=lambda value: _parse_assignment(value, external=False),
        required=True,
        metavar="DATASET=PATH",
    )
    parser.add_argument(
        "--external-run",
        action="append",
        type=lambda value: _parse_assignment(value, external=True),
        default=[],
        metavar="DATASET:METHOD=RUN_MANIFEST",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("validation", "test"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument(
        "--final-suite-token",
        help=(
            f"literal {FINAL_SUITE_TOKEN!r} handoff token; the controller injects its "
            "non-circular canonical manifest SHA-256 through the protected environment; "
            "required only for partition=test"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    stages = _assignments(args.stage_manifest, "stage manifest")
    external = _assignments(args.external_run, "external run")
    report = run_benchmark(
        args.checkpoint,
        stages,  # type: ignore[arg-type]
        args.output,
        partition=args.partition,
        external_runs=external,  # type: ignore[arg-type]
        device_name=args.device,
        unlock_final_test=args.unlock_final_test,
        final_suite_token=args.final_suite_token,
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "partition": report["partition"],
                "benchmark_complete": report["benchmark_complete"],
                "prepared_samples": report["input_preparation"]["successful"],  # type: ignore[index]
            },
            indent=2,
        )
    )
    if report["benchmark_complete"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


__all__ = [
    "BATCH_SIZE",
    "BenchmarkError",
    "DATASETS",
    "EXPECTED_COUNTS",
    "EXTERNAL_METHODS",
    "FINAL_SUITE_TOKEN",
    "IMAGE_SIZE",
    "MEASURED_PASSES",
    "SAMPLES_PER_DATASET",
    "SCHEMA_VERSION",
    "WARMUP_PASSES",
    "run_benchmark",
]
