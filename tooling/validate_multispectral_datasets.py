"""Validate the local RGB/thermal datasets staged under ``data/``.

This script uses only the Python standard library. It checks that paired
modalities and annotations have matching sample identifiers, prints a JSON
report, and optionally writes that report to disk.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def file_ids(
    directory: Path,
    *,
    suffixes: set[str] | None = None,
    normalize: Callable[[str], str] | None = None,
) -> set[str]:
    if not directory.is_dir():
        return set()
    allowed = suffixes or IMAGE_SUFFIXES
    normalizer = normalize or (lambda value: value)
    return {
        normalizer(path.stem)
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in allowed
    }


def compare_groups(groups: dict[str, set[str]]) -> dict[str, object]:
    nonempty = [values for values in groups.values() if values]
    common = set.intersection(*nonempty) if nonempty else set()
    union = set.union(*nonempty) if nonempty else set()
    return {
        "counts": {name: len(values) for name, values in groups.items()},
        "paired_count": len(common),
        "unmatched_counts": {
            name: len(union - values) for name, values in groups.items()
        },
        "valid": bool(groups) and len(nonempty) == len(groups) and common == union,
    }


def validate_llvip(root: Path) -> dict[str, object]:
    report: dict[str, object] = {"path": str(root), "exists": root.is_dir()}
    splits: dict[str, object] = {}
    all_images: set[str] = set()
    valid = root.is_dir()
    for split in ("train", "test"):
        groups = {
            "visible": file_ids(root / "visible" / split),
            "infrared": file_ids(root / "infrared" / split),
        }
        split_report = compare_groups(groups)
        splits[split] = split_report
        all_images.update(groups["visible"])
        valid = valid and bool(split_report["valid"])

    annotation_ids = file_ids(root / "Annotations", suffixes={".xml"})
    annotations = {
        "count": len(annotation_ids),
        "missing_for_images": len(all_images - annotation_ids),
        "without_images": len(annotation_ids - all_images),
    }
    valid = valid and annotation_ids == all_images
    report.update(
        {
            "splits": splits,
            "annotations": annotations,
            "total_pairs": len(all_images),
            "valid": valid,
        }
    )
    return report


def validate_msrs(root: Path) -> dict[str, object]:
    report: dict[str, object] = {"path": str(root), "exists": root.is_dir()}
    splits: dict[str, object] = {}
    valid = root.is_dir()
    for split in ("train", "test"):
        groups = {
            "visible": file_ids(root / split / "vi"),
            "infrared": file_ids(root / split / "ir"),
            "segmentation": file_ids(root / split / "Segmentation_labels"),
        }
        split_report = compare_groups(groups)
        splits[split] = split_report
        valid = valid and bool(split_report["valid"])

    detection_groups = {
        "visible": file_ids(root / "detection" / "vi"),
        "infrared": file_ids(root / "detection" / "ir"),
        "labels": {
            sample_id
            for sample_id in file_ids(
                root / "detection" / "labels", suffixes={".txt"}
            )
            if sample_id != "classes"
        },
    }
    detection = compare_groups(detection_groups)
    valid = valid and bool(detection["valid"])
    report.update({"splits": splits, "detection": detection, "valid": valid})
    return report


def normalize_caltech_id(stem: str) -> str:
    return re.sub(r"_(?:eo|thermal|mask)-", "_frame-", stem)


def validate_caltech(root: Path) -> dict[str, object]:
    groups = {
        "color": file_ids(root / "color", normalize=normalize_caltech_id),
        "thermal8": file_ids(root / "thermal8", normalize=normalize_caltech_id),
        "thermal16": file_ids(root / "thermal16", normalize=normalize_caltech_id),
        "annotations": file_ids(
            root / "annotations", normalize=normalize_caltech_id
        ),
        "thermal_annotation_overlay": file_ids(
            root / "thermal_ann_overlay", normalize=normalize_caltech_id
        ),
    }
    comparison = compare_groups(groups)
    return {
        "path": str(root),
        "exists": root.is_dir(),
        **comparison,
        "valid": root.is_dir() and bool(comparison["valid"]),
    }


def build_report() -> dict[str, object]:
    datasets = {
        "llvip": validate_llvip(DATA_ROOT / "LLVIP"),
        "msrs": validate_msrs(DATA_ROOT / "MSRS"),
        "caltech_aerial_rgbt": validate_caltech(
            DATA_ROOT / "Caltech_Aerial_RGBT"
        ),
    }
    return {
        "data_root": str(DATA_ROOT),
        "datasets": datasets,
        "valid": all(bool(item["valid"]) for item in datasets.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        type=Path,
        help="Also write the JSON report to this path.",
    )
    args = parser.parse_args()
    report = build_report()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.write:
        output = args.write if args.write.is_absolute() else REPO_ROOT / args.write
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
