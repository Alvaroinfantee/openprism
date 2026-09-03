"""Stage one frozen protocol partition as flat visible/thermal image pairs.

The external learned-fusion adapters require two flat directories with exactly
the same sample identifiers.  This command derives those identifiers from the
live OpenPRISM protocol, stages immutable hardlinks when possible (with a copy
fallback), and writes a content-addressed manifest.  Final-test staging is
locked behind both an explicit unlock and the previously frozen sample-manifest
digest, so a partial or changed final-test selection fails closed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Iterable, Mapping, Sequence


if __package__ in {None, ""}:  # Support ``python tooling/stage_protocol_pairs.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openprism.learning.data import (  # noqa: E402
    ProtocolItem,
    item_scene_group,
    protocol_items,
    protocol_manifest,
)


SCHEMA_VERSION = "openprism.protocol-pair-stage/1.0"
DATASETS = ("llvip", "msrs", "caltech")
PARTITIONS = ("train", "validation", "test")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_SAMPLE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _selection_token(item: ProtocolItem) -> str:
    return (
        f"{item.record.split}:{item.record.sample_id}:"
        f"{item_scene_group(item)}"
    )


def _selection_sha256(items: Sequence[ProtocolItem]) -> str:
    tokens = sorted(_selection_token(item) for item in items)
    return hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()


def _expected_protocol_values(
    manifest: Mapping[str, object], dataset: str, partition: str
) -> tuple[int, str]:
    try:
        counts = manifest["counts"]
        digests = manifest["sample_manifest_sha256"]
        expected_count = int(counts[partition][dataset])  # type: ignore[index]
        expected_digest = str(digests[partition][dataset])  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"protocol manifest has no complete {dataset}/{partition} selection"
        ) from error
    if expected_count <= 0 or not _SHA256.fullmatch(expected_digest):
        raise ValueError(
            f"protocol manifest contains invalid {dataset}/{partition} selection metadata"
        )
    return expected_count, expected_digest


def _validate_sample_ids(items: Sequence[ProtocolItem]) -> None:
    exact: set[str] = set()
    case_insensitive: dict[str, str] = {}
    for item in items:
        sample_id = item.record.sample_id
        if not _SAMPLE_ID.fullmatch(sample_id) or sample_id.endswith((".", " ")):
            raise ValueError(f"sample identifier is unsafe for flat staging: {sample_id!r}")
        if sample_id in exact:
            raise ValueError(f"duplicate protocol sample identifier: {sample_id}")
        exact.add(sample_id)
        folded = sample_id.casefold()
        if previous := case_insensitive.get(folded):
            raise ValueError(
                "protocol sample identifiers collide on a case-insensitive filesystem: "
                f"{previous!r} and {sample_id!r}"
            )
        case_insensitive[folded] = sample_id


def _select_protocol_items(
    data_root: Path,
    dataset: str,
    partition: str,
    *,
    unlock_final_test: bool,
    expected_sample_manifest_sha256: str | None,
) -> tuple[tuple[ProtocolItem, ...], dict[str, object], str]:
    if dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}")
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    frozen_digest = (
        expected_sample_manifest_sha256.lower().strip()
        if expected_sample_manifest_sha256 is not None
        else None
    )
    if frozen_digest is not None and not _SHA256.fullmatch(frozen_digest):
        raise ValueError("expected sample-manifest SHA-256 must be 64 lowercase hex characters")
    if partition == "test":
        if not unlock_final_test:
            raise ValueError(
                "final-test pair staging is locked; pass --unlock-final-test only after freezing artifacts"
            )
        if frozen_digest is None:
            raise ValueError(
                "final-test pair staging requires --expected-sample-manifest-sha256"
            )

    # Both calls intentionally use the live protocol implementation.  Their
    # independently recomputed count and token digest must agree before any
    # output directory is created.
    manifest = protocol_manifest(data_root)
    selected = tuple(
        item
        for item in protocol_items(data_root, partition)
        if item.record.dataset == dataset
    )
    if not selected:
        raise ValueError(f"protocol selected no {dataset}/{partition} samples")
    if any(item.partition != partition for item in selected):
        raise ValueError("protocol item partition does not match requested partition")
    _validate_sample_ids(selected)
    expected_count, manifest_digest = _expected_protocol_values(
        manifest, dataset, partition
    )
    actual_digest = _selection_sha256(selected)
    if len(selected) != expected_count or actual_digest != manifest_digest:
        raise ValueError(
            "protocol selection is partial or inconsistent with its manifest: "
            f"selected={len(selected)}/{expected_count}, "
            f"digest={actual_digest}/{manifest_digest}"
        )
    if frozen_digest is not None and actual_digest != frozen_digest:
        raise ValueError(
            "protocol selection differs from the explicitly frozen sample manifest: "
            f"expected {frozen_digest}, found {actual_digest}"
        )
    return selected, manifest, actual_digest


def _stage_file(source: Path, destination: Path) -> dict[str, object]:
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise ValueError(f"unsupported source image suffix: {source}")
    source_hash = _sha256(source)
    fallback: dict[str, object] | None = None
    try:
        os.link(source, destination)
        method = "hardlink"
    except OSError as error:
        shutil.copy2(source, destination)
        method = "copy_fallback"
        fallback = {
            "errno": error.errno,
            "winerror": getattr(error, "winerror", None),
            "reason": str(error),
        }
    staged_hash = _sha256(destination)
    if staged_hash != source_hash:
        raise IOError(f"staged content hash mismatch for {source}")
    result: dict[str, object] = {
        "source": str(source.resolve()),
        "staged": destination.name,
        "bytes": source.stat().st_size,
        "sha256": source_hash,
        "method": method,
    }
    if fallback is not None:
        result["hardlink_failure"] = fallback
    return result


def stage_protocol_pairs(
    data_root: str | Path,
    dataset: str,
    partition: str,
    output_dir: str | Path,
    *,
    unlock_final_test: bool = False,
    expected_sample_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Stage one complete dynamic protocol selection and return its manifest."""

    root = Path(data_root).resolve()
    destination = Path(output_dir).resolve()
    selected, protocol, selection_digest = _select_protocol_items(
        root,
        dataset,
        partition,
        unlock_final_test=unlock_final_test,
        expected_sample_manifest_sha256=expected_sample_manifest_sha256,
    )
    if destination.exists():
        raise FileExistsError(
            f"staging destination already exists; use a new immutable path: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-", dir=str(destination.parent)
        )
    )
    try:
        visible_dir = staging / "visible"
        thermal_dir = staging / "thermal"
        visible_dir.mkdir()
        thermal_dir.mkdir()
        entries: list[dict[str, object]] = []
        for item in sorted(selected, key=lambda value: value.record.sample_id):
            record = item.record
            visible_name = f"{record.sample_id}{record.visible_path.suffix.lower()}"
            thermal_name = f"{record.sample_id}{record.thermal_path.suffix.lower()}"
            visible = _stage_file(record.visible_path, visible_dir / visible_name)
            thermal = _stage_file(record.thermal_path, thermal_dir / thermal_name)
            entries.append(
                {
                    "sample_id": record.sample_id,
                    "source_split": record.split,
                    "scene_group": item_scene_group(item),
                    "visible": visible,
                    "thermal": thermal,
                }
            )

        visible_ids = {path.stem for path in visible_dir.iterdir() if path.is_file()}
        thermal_ids = {path.stem for path in thermal_dir.iterdir() if path.is_file()}
        expected_ids = {item.record.sample_id for item in selected}
        if visible_ids != expected_ids or thermal_ids != expected_ids:
            raise RuntimeError("staged visible/thermal ID sets are partial or inconsistent")
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "partition": partition,
            "final_test_unlocked": partition == "test" and unlock_final_test,
            "data_root": str(root),
            "output_directory": str(destination),
            "protocol_schema_version": protocol["schema_version"],
            "protocol_sample_manifest_sha256": selection_digest,
            "protocol_count": len(selected),
            "protocol_capture_groups": protocol["scene_groups"][partition][dataset],  # type: ignore[index]
            "staging_policy": "hardlink_with_copy2_fallback; exact source/staged SHA-256 equality",
            "items": entries,
            "total_source_bytes": sum(
                int(entry[modality]["bytes"])  # type: ignore[index]
                for entry in entries
                for modality in ("visible", "thermal")
            ),
        }
        manifest["manifest_payload_sha256"] = _canonical_sha256(manifest)
        (staging / "stage_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(staging, destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--partition", choices=PARTITIONS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument(
        "--expected-sample-manifest-sha256",
        help=(
            "frozen digest from protocol_manifest; mandatory for final-test staging "
            "and optional as a validation/train reproducibility assertion"
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = stage_protocol_pairs(
        args.data_root,
        args.dataset,
        args.partition,
        args.output_dir,
        unlock_final_test=args.unlock_final_test,
        expected_sample_manifest_sha256=args.expected_sample_manifest_sha256,
    )
    print(
        json.dumps(
            {
                "dataset": manifest["dataset"],
                "partition": manifest["partition"],
                "count": manifest["protocol_count"],
                "sample_manifest_sha256": manifest[
                    "protocol_sample_manifest_sha256"
                ],
                "output_directory": manifest["output_directory"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "DATASETS",
    "PARTITIONS",
    "SCHEMA_VERSION",
    "stage_protocol_pairs",
]
