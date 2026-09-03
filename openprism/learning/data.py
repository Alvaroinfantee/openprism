"""Leakage-aware dataset protocol and synthetic sensor corruptions."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..datasets import DatasetCatalog, SampleRecord
from .model import TASK_NAMES


PROTOCOL_VERSION = "openprism.egtcf-protocol/1.1"
CORRUPTION_MODES = ("declared_shift", "hidden_shift", "dropout", "noise")


@dataclass(frozen=True, slots=True)
class ProtocolItem:
    record: SampleRecord
    task: str
    partition: str


def _bucket(value: str, modulus: int = 10) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % modulus


def _caltech_partition(scene_group: str | None) -> str:
    bucket = _bucket(scene_group or "ungrouped")
    if bucket <= 6:
        return "train"
    if bucket == 7:
        return "validation"
    return "test"


def _llvip_test_partition(sample_id: str) -> str:
    # LLVIP file prefixes identify capture sequences.  Keep entire sequences
    # together rather than randomly distributing adjacent video frames.
    sequence = sample_id[:2]
    return "validation" if sequence in {"19", "21", "23"} else "test"


def msrs_scene_group(sample_id: str, block_size: int = 100) -> str:
    """Return the preregistered contiguous-block unit for MSRS statistics.

    MSRS exposes ordered frame identifiers and day/night suffixes, but no
    publisher scene identity.  We therefore use conservative, non-overlapping
    numeric blocks rather than treating adjacent frames as independent.
    """

    if block_size <= 0:
        raise ValueError("block_size must be positive")
    stem = str(sample_id).strip()
    if len(stem) < 2 or not stem[:-1].isdigit() or stem[-1] not in {"D", "N"}:
        raise ValueError(f"unsupported MSRS sample identifier: {sample_id}")
    return f"{stem[-1].lower()}-block-{int(stem[:-1]) // block_size:03d}"


def _msrs_train_partition(sample_id: str) -> str:
    """Assign a complete MSRS capture block to one custom partition.

    The publisher train/test files interleave adjacent frames from the same
    sequences, so preserving that split while carving validation from train
    leaks capture blocks.  We instead partition the union of publisher train
    and test semantic frames by the stable 100-frame day/night group.
    """

    bucket = _bucket(msrs_scene_group(sample_id), modulus=10)
    if bucket <= 6:
        return "train"
    if bucket == 7:
        return "validation"
    return "test"


def item_scene_group(item: ProtocolItem) -> str:
    """Return the frozen resampling unit used by every protocol evaluator."""

    if item.record.scene_group:
        return item.record.scene_group
    if item.record.dataset == "llvip":
        return item.record.sample_id[:2]
    if item.record.dataset == "msrs":
        if item.record.split == "detection":
            return f"detection-block-{int(item.record.sample_id) // 100:03d}"
        return msrs_scene_group(item.record.sample_id)
    raise ValueError(
        f"no preregistered scene-group rule for dataset {item.record.dataset}"
    )


def protocol_items(
    data_root: str | Path,
    partition: str,
    *,
    include_detection_subset: bool = True,
) -> list[ProtocolItem]:
    """Return the frozen cross-dataset partition without copying source data."""

    if partition not in {"train", "validation", "test"}:
        raise ValueError("partition must be train, validation, or test")
    catalog = DatasetCatalog(data_root)
    items: list[ProtocolItem] = []

    for split in ("train", "test"):
        for index in range(catalog.count("llvip", split)):
            record = catalog.record("llvip", split, index)
            assigned = "train" if split == "train" else _llvip_test_partition(record.sample_id)
            if assigned == partition:
                items.append(ProtocolItem(record, "search", assigned))

    for split in ("train", "test"):
        for index in range(catalog.count("msrs", split)):
            record = catalog.record("msrs", split, index)
            assigned = _msrs_train_partition(record.sample_id)
            if assigned == partition:
                items.append(ProtocolItem(record, "navigate", assigned))
    # The separate MSRS detection directory uses numeric identifiers without
    # day/night suffixes, so its capture-block relation to the semantic frames
    # cannot be established from the released filenames.  It is excluded from
    # every learned-fusion partition to avoid an unverifiable overlap.
    del include_detection_subset

    for index in range(catalog.count("caltech", "all")):
        record = catalog.record("caltech", "all", index)
        assigned = _caltech_partition(record.scene_group)
        if assigned == partition:
            items.append(ProtocolItem(record, "terrain", assigned))
    return items


def protocol_manifest(data_root: str | Path) -> dict[str, object]:
    partitions = {
        name: protocol_items(data_root, name) for name in ("train", "validation", "test")
    }
    counts: dict[str, dict[str, int]] = {}
    scene_groups: dict[str, dict[str, list[str]]] = {}
    sample_tokens: dict[str, dict[str, list[str]]] = {}
    for partition, items in partitions.items():
        counts[partition] = {}
        scene_groups[partition] = {}
        sample_tokens[partition] = {}
        for item in items:
            dataset = item.record.dataset
            counts[partition][dataset] = counts[partition].get(dataset, 0) + 1
            group = item_scene_group(item)
            scene_groups[partition].setdefault(dataset, []).append(group)
            sample_tokens[partition].setdefault(dataset, []).append(
                f"{item.record.split}:{item.record.sample_id}:{group}"
            )
        for dataset in scene_groups[partition]:
            scene_groups[partition][dataset] = sorted(
                set(scene_groups[partition][dataset])
            )
            sample_tokens[partition][dataset].sort()
    for dataset in {name for values in scene_groups.values() for name in values}:
        assigned = {
            partition: set(scene_groups[partition].get(dataset, []))
            for partition in partitions
        }
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
            if overlap := assigned[first] & assigned[second]:
                raise ValueError(
                    f"protocol leaks {dataset} capture groups across {first}/{second}: "
                    f"{sorted(overlap)}"
                )
    sample_manifest_sha256 = {
        partition: {
            dataset: hashlib.sha256("\n".join(tokens).encode("utf-8")).hexdigest()
            for dataset, tokens in values.items()
        }
        for partition, values in sample_tokens.items()
    }
    return {
        "schema_version": PROTOCOL_VERSION,
        "split_policy": {
            "llvip": "publisher train; publisher test separated by capture-sequence prefix",
            "msrs": (
                "publisher train and test semantic frames combined, then assigned by "
                "stable day/night contiguous 100-frame blocks (SHA-256 modulo 10; "
                "buckets 0-6 train, 7 validation, 8-9 test); numeric detection subset excluded"
            ),
            "caltech": "SHA-256 bucket of scene_group; 0-6 train, 7 validation, 8-9 test",
        },
        "counts": counts,
        "scene_groups": scene_groups,
        "sample_manifest_sha256": sample_manifest_sha256,
        "capture_groups_disjoint_across_partitions": True,
    }


def _load_pair(record: SampleRecord) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(record.visible_path) as image:
        visible = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(record.thermal_path) as image:
        thermal = np.asarray(image.convert("F"), dtype=np.float32)
    finite = thermal[np.isfinite(thermal)]
    if finite.size:
        low, high = np.percentile(finite, (1.0, 99.0))
        thermal = (
            np.clip((thermal - low) / max(float(high - low), 1e-6), 0.0, 1.0)
            if high > low
            else np.zeros_like(thermal)
        )
    else:
        thermal = np.zeros_like(thermal)
    return visible, thermal


def _resize_and_crop(
    visible: np.ndarray,
    thermal: np.ndarray,
    size: int,
    rng: random.Random,
    training: bool,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = visible.shape[:2]
    scale = max(size / height, size / width)
    if scale > 1.0:
        new_size = (int(round(width * scale)), int(round(height * scale)))
        visible = np.asarray(
            Image.fromarray(np.rint(visible * 255.0).astype(np.uint8)).resize(
                new_size, Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        thermal = np.asarray(
            Image.fromarray(thermal.astype(np.float32), mode="F").resize(
                new_size, Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        height, width = visible.shape[:2]
    if training:
        top = rng.randint(0, max(0, height - size))
        left = rng.randint(0, max(0, width - size))
    else:
        top = max(0, (height - size) // 2)
        left = max(0, (width - size) // 2)
    visible = visible[top : top + size, left : left + size]
    thermal = thermal[top : top + size, left : left + size]
    if training and rng.random() < 0.5:
        visible = visible[:, ::-1]
        thermal = thermal[:, ::-1]
    return np.ascontiguousarray(visible), np.ascontiguousarray(thermal)


def _translate(value: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = value.shape
    shifted = np.zeros_like(value)
    support = np.zeros_like(value, dtype=np.float32)
    source_y0 = max(0, -dy)
    source_y1 = min(height, height - dy)
    source_x0 = max(0, -dx)
    source_x1 = min(width, width - dx)
    target_y0 = max(0, dy)
    target_x0 = max(0, dx)
    target_y1 = target_y0 + max(0, source_y1 - source_y0)
    target_x1 = target_x0 + max(0, source_x1 - source_x0)
    if target_y1 > target_y0 and target_x1 > target_x0:
        shifted[target_y0:target_y1, target_x0:target_x1] = value[
            source_y0:source_y1, source_x0:source_x1
        ]
        support[target_y0:target_y1, target_x0:target_x1] = 1.0
    return shifted, support


class FusionPatchDataset(Dataset[dict[str, Tensor | str]]):
    """Aligned patches with declared and hidden sensor failures for abstention."""

    def __init__(
        self,
        data_root: str | Path,
        partition: str,
        *,
        patch_size: int = 192,
        seed: int = 20260902,
        max_samples: int | None = None,
        corruption_probability: float = 0.65,
        apply_corruptions: bool | None = None,
        corruption_modes: tuple[str, ...] = CORRUPTION_MODES,
    ) -> None:
        if patch_size < 32:
            raise ValueError("patch_size must be at least 32")
        self.partition = partition
        self.training = partition == "train"
        self.patch_size = patch_size
        self.seed = seed
        self.epoch = 0
        self.corruption_probability = corruption_probability
        self.apply_corruptions = self.training if apply_corruptions is None else apply_corruptions
        unknown_modes = set(corruption_modes) - set(CORRUPTION_MODES)
        if unknown_modes or not corruption_modes:
            raise ValueError(
                f"corruption_modes must be a non-empty subset of {CORRUPTION_MODES}"
            )
        self.corruption_modes = tuple(corruption_modes)
        items = protocol_items(data_root, partition)
        if max_samples is not None and len(items) > max_samples:
            chooser = random.Random(seed + {"train": 0, "validation": 1, "test": 2}[partition])
            selected = sorted(chooser.sample(range(len(items)), max_samples))
            items = [items[index] for index in selected]
        self.items = tuple(items)

    def set_epoch(self, epoch: int) -> None:
        """Select a deterministic, epoch-specific augmentation realization."""

        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        item = self.items[index]
        epoch_offset = self.epoch * 15_485_863 if self.training else 0
        rng = random.Random(self.seed + epoch_offset + index * 1_000_003)
        visible, thermal = _load_pair(item.record)
        visible, thermal = _resize_and_crop(
            visible, thermal, self.patch_size, rng, self.training
        )
        validity = np.ones_like(thermal, dtype=np.float32)
        registration = np.ones_like(thermal, dtype=np.float32)
        timing = np.ones_like(thermal, dtype=np.float32)
        corruption = np.zeros_like(thermal, dtype=np.float32)
        corruption_mode = "clean"

        if self.apply_corruptions and rng.random() < self.corruption_probability:
            mode = rng.choice(self.corruption_modes)
            corruption_mode = mode
            if mode in {"declared_shift", "hidden_shift"}:
                maximum = max(2, self.patch_size // 20)
                dx = rng.choice(tuple(i for i in range(-maximum, maximum + 1) if i))
                dy = rng.choice(tuple(i for i in range(-maximum, maximum + 1) if i))
                thermal, geometric = _translate(thermal, dy, dx)
                severity = min(1.0, float(np.hypot(dx, dy)) / maximum)
                corruption[:] = severity
                corruption[geometric == 0.0] = 1.0
                if mode == "declared_shift":
                    registration[:] = 1.0 - severity
                    registration *= geometric
            elif mode == "dropout":
                height, width = thermal.shape
                box_h = rng.randint(height // 5, height // 2)
                box_w = rng.randint(width // 5, width // 2)
                top = rng.randint(0, height - box_h)
                left = rng.randint(0, width - box_w)
                thermal[top : top + box_h, left : left + box_w] = 0.0
                validity[top : top + box_h, left : left + box_w] = 0.0
                corruption[top : top + box_h, left : left + box_w] = 1.0
            else:
                sigma = rng.uniform(0.04, 0.18)
                noise_rng = np.random.default_rng(self.seed + epoch_offset + index)
                thermal = np.clip(
                    thermal + noise_rng.normal(0.0, sigma, thermal.shape), 0.0, 1.0
                ).astype(np.float32)
                corruption[:] = min(1.0, sigma / 0.18)

        rgb_tensor = torch.from_numpy(np.moveaxis(visible, -1, 0).copy()).float()
        thermal_tensor = torch.from_numpy(thermal[None].copy()).float()
        evidence = torch.from_numpy(
            np.stack((validity, registration, timing), axis=0)
        ).float()
        return {
            "rgb": rgb_tensor,
            "thermal": thermal_tensor,
            "evidence": evidence,
            "corruption_target": torch.from_numpy(corruption[None]).float(),
            "task_id": torch.tensor(TASK_NAMES.index(item.task), dtype=torch.long),
            "dataset": item.record.dataset,
            "sample_id": item.record.sample_id,
            "scene_group": item_scene_group(item),
            "partition": item.partition,
            "corruption_mode": corruption_mode,
        }


def dataset_counts(items: Iterable[ProtocolItem]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in items:
        result[item.record.dataset] = result.get(item.record.dataset, 0) + 1
    return result
