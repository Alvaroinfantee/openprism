"""Train a visible-only Caltech terrain probe and apply it unchanged to RGB-T views."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..contracts import SynchronizationStatus
from ..datasets import DatasetCatalog, SampleRecord
from ..fusion import EvidenceFusionEngine
from .caltech_segmentation_evaluation import (
    evaluate_caltech_semantics,
    paired_caltech_scene_group_comparison,
)
from .data import protocol_items, protocol_manifest
from .engine import LearnedFusionEngine
from .segmentation_evaluation import FrozenModelProvenance, SegmentationCase
from .segmentation_probe import (
    DEFAULT_VIEWS,
    _ConvBlock,
    _load_mask,
    _load_rgb,
    _seed_everything,
    _sha256,
    _string_manifest_sha256,
    _view_rgb,
)


CHECKPOINT_SCHEMA = "openprism.caltech-visible-terrain-probe/1.0"
REPORT_SCHEMA = "openprism.caltech-multiview-terrain-probe/1.0"
IGNORE_INDEX = 255


@dataclass(frozen=True, slots=True)
class CaltechProbeConfig:
    base_channels: int = 12
    image_size: int = 256
    classes: int = 11

    def __post_init__(self) -> None:
        for name in ("base_channels", "image_size", "classes"):
            raw = getattr(self, name)
            value = int(raw)
            if value <= 0 or value != raw:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16")
        if self.classes != 11:
            raise ValueError("the frozen Caltech taxonomy requires exactly 11 classes")

    def as_dict(self) -> dict[str, int]:
        return {
            "base_channels": self.base_channels,
            "image_size": self.image_size,
            "classes": self.classes,
        }


class CaltechVisibleTerrainProbe(nn.Module):
    """Compact two-level U-Net-like model for classes 1--11."""

    def __init__(self, config: CaltechProbeConfig = CaltechProbeConfig()) -> None:
        super().__init__()
        self.config = config
        base = config.base_channels
        self.encoder_one = _ConvBlock(3, base)
        self.encoder_two = _ConvBlock(base, base * 2)
        self.bottleneck = _ConvBlock(base * 2, base * 4)
        self.decoder_two = _ConvBlock(base * 6, base * 2)
        self.decoder_one = _ConvBlock(base * 3, base)
        self.classifier = nn.Conv2d(base, config.classes, 1)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError("probe input must have shape N x 3 x H x W")
        first = self.encoder_one(value)
        second = self.encoder_two(F.max_pool2d(first, 2))
        latent = self.bottleneck(F.max_pool2d(second, 2))
        up_two = F.interpolate(latent, size=second.shape[-2:], mode="bilinear", align_corners=False)
        decoded_two = self.decoder_two(torch.cat((up_two, second), dim=1))
        up_one = F.interpolate(decoded_two, size=first.shape[-2:], mode="bilinear", align_corners=False)
        return self.classifier(self.decoder_one(torch.cat((up_one, first), dim=1)))


def _target_indices(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask)
    supplied = set(int(item) for item in np.unique(value))
    unknown = sorted(supplied - set(range(12)) - {255})
    if unknown:
        raise ValueError(f"Caltech mask contains labels outside 0..11/255: {unknown}")
    result = np.full(value.shape, IGNORE_INDEX, dtype=np.int64)
    labeled = (value >= 1) & (value <= 11)
    result[labeled] = value[labeled].astype(np.int64) - 1
    return result


def _resize_rgb(rgb: np.ndarray, size: int) -> Tensor:
    value = np.asarray(rgb)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError("probe view must be HxWx3 or HxW")
    normalized = value.astype(np.float32)
    if normalized.max(initial=0.0) > 1.0:
        normalized /= 255.0
    if not np.all(np.isfinite(normalized)):
        raise ValueError("probe view contains non-finite values")
    tensor = torch.from_numpy(np.moveaxis(np.clip(normalized, 0.0, 1.0), -1, 0).copy())[None]
    return F.interpolate(tensor, (size, size), mode="bilinear", align_corners=False)[0]


def _resize_target(mask: np.ndarray, size: int) -> Tensor:
    value = torch.from_numpy(_target_indices(mask))[None, None].float()
    return F.interpolate(value, (size, size), mode="nearest")[0, 0].long()


class _TrainingDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, records: Sequence[SampleRecord], size: int) -> None:
        self.records = tuple(records)
        self.size = size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records[index]
        if record.annotation_path is None:
            raise ValueError(f"missing semantic mask for {record.sample_id}")
        return (
            _resize_rgb(_load_rgb(record.visible_path), self.size),
            _resize_target(_load_mask(record.annotation_path), self.size),
        )


def _semantic_records(data_root: Path, partition: str) -> tuple[SampleRecord, ...]:
    return tuple(
        item.record
        for item in protocol_items(data_root, partition, include_detection_subset=False)
        if item.record.dataset == "caltech"
        and item.record.annotation_kind == "semantic_mask"
        and item.record.annotation_path is not None
        and item.record.scene_group is not None
    )


def class_weights_from_training_records(
    records: Sequence[SampleRecord],
) -> tuple[np.ndarray, np.ndarray]:
    counts = np.zeros(11, dtype=np.int64)
    for record in records:
        if record.annotation_path is None:
            continue
        target = _target_indices(_load_mask(record.annotation_path))
        for class_index in range(11):
            counts[class_index] += int(np.count_nonzero(target == class_index))
    present = counts > 0
    if not np.any(present):
        raise ValueError("training records contain no evaluated Caltech labels")
    weights = np.zeros(11, dtype=np.float64)
    weights[present] = np.sqrt(np.sum(counts[present]) / counts[present])
    weights[present] /= np.mean(weights[present])
    return weights.astype(np.float32), counts


def _predict_mask(
    model: CaltechVisibleTerrainProbe,
    rgb: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    height, width = np.asarray(rgb).shape[:2]
    value = _resize_rgb(rgb, model.config.image_size)[None].to(device)
    with torch.inference_mode():
        logits = model(value)
        logits = F.interpolate(logits, (height, width), mode="bilinear", align_corners=False)
        return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int16) + 1


@dataclass(frozen=True, slots=True)
class CaltechCheckpointMetadata:
    path: Path
    artifact_sha256: str
    model_id: str
    training_run_id: str
    best_epoch: int
    validation_mean_iou: float
    development_subset: bool
    config: CaltechProbeConfig
    class_weights: tuple[float, ...]
    training_sample_manifest_sha256: str
    validation_sample_manifest_sha256: str


def _metadata(path: Path, payload: Mapping[str, object]) -> CaltechCheckpointMetadata:
    return CaltechCheckpointMetadata(
        path=path.resolve(),
        artifact_sha256=_sha256(path),
        model_id=str(payload["model_id"]),
        training_run_id=str(payload["training_run_id"]),
        best_epoch=int(payload["best_epoch"]),
        validation_mean_iou=float(payload["validation_mean_iou"]),
        development_subset=bool(payload["development_subset"]),
        config=CaltechProbeConfig(**dict(payload["config"])),
        class_weights=tuple(float(item) for item in payload["class_weights"]),
        training_sample_manifest_sha256=str(payload["training_sample_manifest_sha256"]),
        validation_sample_manifest_sha256=str(payload["validation_sample_manifest_sha256"]),
    )


def _save_checkpoint(
    path: Path,
    model: CaltechVisibleTerrainProbe,
    *,
    model_id: str,
    training_run_id: str,
    epoch: int,
    validation_mean_iou: float,
    development_subset: bool,
    class_weights: np.ndarray,
    training_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> CaltechCheckpointMetadata:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "model_id": model_id,
        "training_run_id": training_run_id,
        "input_domain": "visible_rgb_only",
        "training_partition": "protocol_train_scene_groups",
        "selection_partition": "protocol_validation_scene_groups",
        "best_epoch": int(epoch),
        "validation_mean_iou": float(validation_mean_iou),
        "development_subset": bool(development_subset),
        "config": model.config.as_dict(),
        "class_weights": [float(item) for item in class_weights],
        "training_sample_manifest_sha256": _string_manifest_sha256(tuple(training_ids)),
        "validation_sample_manifest_sha256": _string_manifest_sha256(tuple(validation_ids)),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, path)
    return _metadata(path, payload)


def load_caltech_probe_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CaltechVisibleTerrainProbe, CaltechCheckpointMetadata]:
    source = Path(path)
    payload = torch.load(source, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported Caltech probe checkpoint schema")
    required = {
        "model_id", "training_run_id", "input_domain", "training_partition",
        "selection_partition", "best_epoch", "validation_mean_iou",
        "development_subset", "config", "class_weights",
        "training_sample_manifest_sha256", "validation_sample_manifest_sha256", "state_dict",
    }
    if missing := required - payload.keys():
        raise ValueError(f"Caltech checkpoint is missing fields: {sorted(missing)}")
    if payload["input_domain"] != "visible_rgb_only":
        raise ValueError("Caltech probe was not trained on visible RGB only")
    if payload["training_partition"] != "protocol_train_scene_groups" or payload["selection_partition"] != "protocol_validation_scene_groups":
        raise ValueError("Caltech checkpoint does not satisfy grouped protocol partitions")
    weights = np.asarray(payload["class_weights"], dtype=np.float64)
    if weights.shape != (11,) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Caltech checkpoint class weights are malformed")
    if int(payload["best_epoch"]) <= 0 or not 0.0 <= float(payload["validation_mean_iou"]) <= 1.0:
        raise ValueError("Caltech checkpoint selection metadata is malformed")
    config = CaltechProbeConfig(**dict(payload["config"]))
    model = CaltechVisibleTerrainProbe(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, _metadata(source, payload)


def _validation_score(
    model: CaltechVisibleTerrainProbe,
    records: Sequence[SampleRecord],
    device: torch.device,
) -> float:
    cases = []
    for record in records:
        assert record.annotation_path is not None and record.scene_group is not None
        truth = _load_mask(record.annotation_path)
        cases.append(
            SegmentationCase(
                record.sample_id,
                record.scene_group,
                truth,
                _predict_mask(model, _load_rgb(record.visible_path), device),
            )
        )
    provenance = FrozenModelProvenance(
        "in-training-caltech-visible-probe",
        None,
        "OpenPRISM in-process training",
        CHECKPOINT_SCHEMA,
        f"square-bilinear-{model.config.image_size}",
        False,
        False,
        "Caltech protocol train visible RGB only",
    )
    report = evaluate_caltech_semantics(
        cases,
        provenance,
        partition="validation",
        bootstrap_replicates=1,
    )
    score = report["metrics"]["mean_iou"]
    if score is None:
        raise ValueError("Caltech validation contains no evaluated labels")
    return float(score)


def train_caltech_visible_probe(
    data_root: str | Path,
    checkpoint_path: str | Path,
    *,
    config: CaltechProbeConfig = CaltechProbeConfig(),
    epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    device_name: str = "auto",
    seed: int = 20260903,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
) -> dict[str, object]:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("training hyperparameters are invalid")
    root = Path(data_root)
    training = list(_semantic_records(root, "train"))
    validation = list(_semantic_records(root, "validation"))
    if max_train_samples is not None:
        training = training[:max_train_samples]
    if max_validation_samples is not None:
        validation = validation[:max_validation_samples]
    if not training or not validation:
        raise ValueError("Caltech protocol train and validation both require labeled frames")
    train_groups = {record.scene_group for record in training}
    validation_groups = {record.scene_group for record in validation}
    if train_groups & validation_groups:
        raise ValueError("Caltech training and validation scene groups overlap")

    _seed_everything(seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    weights, counts = class_weights_from_training_records(training)
    model = CaltechVisibleTerrainProbe(config).to(device)
    objective = nn.CrossEntropyLoss(
        weight=torch.from_numpy(weights).to(device), ignore_index=IGNORE_INDEX
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    loader = DataLoader(
        _TrainingDataset(training, config.image_size),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )
    run_id = hashlib.sha256(
        f"{seed}:{config.as_dict()}:{[r.sample_id for r in training]}".encode("utf-8")
    ).hexdigest()[:16]
    model_id = f"openprism-caltech-visible-terrain-probe-{run_id}"
    destination = Path(checkpoint_path)
    best_score = -1.0
    best: CaltechCheckpointMetadata | None = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if not bool(torch.any(targets != IGNORE_INDEX)):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(images), targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Caltech semantic loss became non-finite")
            loss.backward()
            optimizer.step()
            batch_examples = int(images.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_examples
            examples += batch_examples
        if examples == 0:
            raise ValueError("training epoch contains no evaluated semantic pixels")
        model.eval()
        score = _validation_score(model, validation, device)
        history.append({
            "epoch": epoch,
            "training_class_weighted_cross_entropy": total_loss / examples,
            "validation_mean_iou": score,
        })
        if score > best_score:
            best_score = score
            best = _save_checkpoint(
                destination,
                model,
                model_id=model_id,
                training_run_id=run_id,
                epoch=epoch,
                validation_mean_iou=score,
                development_subset=(
                    max_train_samples is not None or max_validation_samples is not None
                ),
                class_weights=weights,
                training_ids=[record.sample_id for record in training],
                validation_ids=[record.sample_id for record in validation],
            )
    assert best is not None
    report = {
        "schema_version": "openprism.caltech-visible-terrain-probe-training/1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(best.path),
            "artifact_sha256": best.artifact_sha256,
            "model_id": best.model_id,
            "best_epoch": best.best_epoch,
            "validation_mean_iou": best.validation_mean_iou,
        },
        "config": config.as_dict(),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "batch_size": batch_size,
            "seed": seed,
        },
        "data_protocol": protocol_manifest(root),
        "data_use": {
            "input_domain": "visible_rgb_only",
            "training_partition": "protocol_train_scene_groups",
            "checkpoint_selection_partition": "protocol_validation_scene_groups",
            "final_test_accessed": False,
            "training_samples": len(training),
            "validation_samples": len(validation),
            "training_scene_groups": len(train_groups),
            "validation_scene_groups": len(validation_groups),
            "development_subset": best.development_subset,
        },
        "class_weighting": {
            "method": "inverse_sqrt_frequency_normalized_over_present_classes",
            "counts_train_only": counts.tolist(),
            "weights": weights.tolist(),
            "ignore_labels": [0, 255],
        },
        "history": history,
        "runtime": {
            "device": str(device),
            "seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "limitations": [
            "The compact probe is a controlled evaluator, not a state-of-the-art segmenter.",
            "It is trained only on visible RGB and is biased toward that domain.",
            "Square resizing changes aspect ratio and limits absolute boundary quality.",
        ],
    }
    destination.with_suffix(destination.suffix + ".training.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _declared_aligned(frame):
    synchronization = SynchronizationStatus.declared_replay_aligned(
        tuple(frame.observations),
        clock_domain=frame.timestamp.effective_clock_domain,
        declaration="Caltech publisher-provided rectified RGB/LWIR replay pair",
    )
    provenance = dict(frame.provenance)
    provenance["evaluation_alignment_assumption"] = (
        "publisher-provided rectification; replay timing is declared, not measured"
    )
    return replace(frame, synchronization=synchronization, provenance=provenance)


def evaluate_caltech_frozen_probe(
    checkpoint_path: str | Path,
    data_root: str | Path,
    output_path: str | Path,
    *,
    partition: str = "validation",
    unlock_final_test: bool = False,
    views: Sequence[str] = DEFAULT_VIEWS,
    prism_egt_checkpoint: str | Path | None = None,
    external_fused: Mapping[str, str | Path] | None = None,
    device_name: str = "auto",
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 20260903,
    max_samples: int | None = None,
) -> dict[str, object]:
    if partition not in {"validation", "test"}:
        raise ValueError("Caltech probe partition must be validation or test")
    if partition == "test" and not unlock_final_test:
        raise ValueError("Caltech final test is locked; freeze all artifacts before explicit unlock")
    if partition == "test" and max_samples is not None:
        raise ValueError("Caltech final-test evaluation must include the complete partition")
    requested = tuple(views)
    external = {str(name): Path(path) for name, path in (external_fused or {}).items()}
    requested += tuple(
        f"external:{name}" for name in external if f"external:{name}" not in requested
    )
    if len(set(requested)) != len(requested):
        raise ValueError("evaluation views must be unique")
    if not requested:
        raise ValueError("evaluation requires at least one view")
    unknown = set(requested) - set(DEFAULT_VIEWS) - {
        f"external:{name}" for name in external
    }
    if unknown:
        raise ValueError(f"unknown views: {sorted(unknown)}")
    if any(name.startswith("prism_egt_") for name in requested) and prism_egt_checkpoint is None:
        raise ValueError("prism_egt_checkpoint is required for PRISM-EGT views")

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    model, metadata = load_caltech_probe_checkpoint(checkpoint_path, device=device)
    if partition == "test" and metadata.development_subset:
        raise ValueError("Caltech final test cannot use a development-subset checkpoint")
    provenance = FrozenModelProvenance(
        metadata.model_id,
        metadata.artifact_sha256,
        "OpenPRISM compact Caltech visible-domain terrain probe",
        metadata.training_run_id,
        f"square-bilinear-{metadata.config.image_size}",
        False,
        True,
        "Caltech protocol train visible RGB only",
        ("best checkpoint selected on grouped protocol validation mean IoU",),
    )
    learned_engine = (
        LearnedFusionEngine.from_checkpoint(str(prism_egt_checkpoint), device=device)
        if prism_egt_checkpoint is not None else None
    )
    deterministic_engine = EvidenceFusionEngine()
    root = Path(data_root)
    items = [
        item for item in protocol_items(root, partition, include_detection_subset=False)
        if item.record.dataset == "caltech"
        and item.record.annotation_path is not None
        and item.record.scene_group is not None
    ]
    if max_samples is not None:
        items = items[:max_samples]
    if not items:
        raise ValueError("selected partition contains no labeled Caltech samples")
    catalog = DatasetCatalog(root)
    record_indexes = {
        catalog.record("caltech", "all", index).sample_id: index
        for index in range(catalog.count("caltech", "all"))
    }
    cases = {name: [] for name in requested}
    for item in items:
        record = item.record
        frame = _declared_aligned(
            catalog.load("caltech", "all", record_indexes[record.sample_id])
        )
        truth = np.asarray(frame.semantic_mask)
        view_cache: dict[str, object] = {}
        for name in requested:
            started = time.perf_counter()
            prediction = None
            failure_reason = None
            try:
                view = _view_rgb(
                    name,
                    frame,
                    deterministic_engine=deterministic_engine,
                    learned_engine=learned_engine,
                    external=external,
                    learned_task="terrain",
                    cache=view_cache,
                )
                prediction = _predict_mask(model, view, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except Exception as error:
                failure_reason = f"{type(error).__name__}:{error}"
            cases[name].append(
                SegmentationCase(
                    record.sample_id,
                    record.scene_group,
                    truth,
                    prediction,
                    latency_ms=(time.perf_counter() - started) * 1_000.0,
                    failure_reason=failure_reason,
                )
            )
    view_reports = {
        name: evaluate_caltech_semantics(
            value,
            provenance,
            partition=partition,
            unlock_final_test=unlock_final_test,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        for name, value in cases.items()
    }
    comparisons = {}
    if "visible_rgb" in cases:
        for name in requested:
            if name == "visible_rgb":
                continue
            try:
                comparisons[name] = paired_caltech_scene_group_comparison(
                    cases["visible_rgb"],
                    cases[name],
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed,
                )
            except ValueError as error:
                comparisons[name] = {"available": False, "reason": str(error)}
    external_hashes = {}
    for name, directory in external.items():
        files = (
            sorted(path for path in directory.iterdir() if path.is_file())
            if directory.is_dir() else []
        )
        external_hashes[name] = {
            "directory": str(directory.resolve()),
            "directory_available": directory.is_dir(),
            "file_count": len(files),
            "manifest_sha256": _string_manifest_sha256(
                [f"{path.name}:{_sha256(path)}" for path in files]
            ),
        }
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development_subset_not_for_paper" if max_samples is not None
            else "caltech_final_test_unlocked_requires_human_review"
            if partition == "test" else "protocol_validation"
        ),
        "partition": partition,
        "final_test_unlocked": partition == "test" and unlock_final_test,
        "visible_probe_checkpoint": {
            "path": str(metadata.path),
            "artifact_sha256": metadata.artifact_sha256,
            "model_id": metadata.model_id,
            "config": metadata.config.as_dict(),
            "best_epoch": metadata.best_epoch,
            "validation_mean_iou": metadata.validation_mean_iou,
            "input_domain": "visible_rgb_only",
            "selection_partition": "protocol_validation_scene_groups",
        },
        "prism_egt_checkpoint": (
            {
                "path": str(learned_engine.checkpoint.path),
                "artifact_sha256": learned_engine.checkpoint.artifact_sha256,
                "model_id": learned_engine.checkpoint.model_id,
            }
            if learned_engine is not None and learned_engine.checkpoint is not None
            else None
        ),
        "sample_manifest": {
            "count": len(items),
            "scene_groups": len({item.record.scene_group for item in items}),
            "sha256": _string_manifest_sha256([item.record.sample_id for item in items]),
            "max_samples": max_samples,
        },
        "views": list(requested),
        "task_mapping": {
            "dataset_proxy": "terrain",
            "prism_egt_operator": "terrain",
            "prism_egt_luminance": "terrain",
            "prism_egt_operator_automatic": "automatic",
            "prism_egt_luminance_automatic": "automatic",
        },
        "view_reports": view_reports,
        "paired_against_visible": comparisons,
        "external_fused_inputs": external_hashes,
        "runtime": {
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "The unchanged probe is trained only on visible RGB and is biased toward that domain.",
            "Cross-view differences measure accessibility to this probe, not universal terrain quality.",
            "Caltech replay registration is publisher-provided and is not a live calibration result.",
            "Development max_samples runs are not final results.",
        ],
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _external_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("external view must be NAME=DIRECTORY")
    return name, Path(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    train = subcommands.add_parser("train")
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--base-channels", type=int, default=12)
    train.add_argument("--image-size", type=int, default=256)
    train.add_argument("--learning-rate", type=float, default=2e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=20260903)
    train.add_argument("--max-train-samples", type=int)
    train.add_argument("--max-validation-samples", type=int)
    evaluate = subcommands.add_parser("evaluate")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--partition", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--unlock-final-test", action="store_true")
    evaluate.add_argument("--prism-egt-checkpoint", type=Path)
    evaluate.add_argument("--external-fused", action="append", type=_external_argument, default=[])
    evaluate.add_argument("--view", action="append", choices=DEFAULT_VIEWS)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--bootstrap-replicates", type=int, default=2_000)
    evaluate.add_argument("--bootstrap-seed", type=int, default=20260903)
    evaluate.add_argument("--max-samples", type=int)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "train":
        report = train_caltech_visible_probe(
            args.data_root,
            args.checkpoint,
            config=CaltechProbeConfig(args.base_channels, args.image_size),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device_name=args.device,
            seed=args.seed,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
        )
        print(json.dumps(report["checkpoint"], indent=2))
        return
    report = evaluate_caltech_frozen_probe(
        args.checkpoint,
        args.data_root,
        args.output,
        partition=args.partition,
        unlock_final_test=args.unlock_final_test,
        views=tuple(args.view or DEFAULT_VIEWS),
        prism_egt_checkpoint=args.prism_egt_checkpoint,
        external_fused=dict(args.external_fused),
        device_name=args.device,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        max_samples=args.max_samples,
    )
    print(json.dumps({
        "partition": report["partition"],
        "final_test_unlocked": report["final_test_unlocked"],
        "views": report["views"],
    }, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "CaltechCheckpointMetadata",
    "CaltechProbeConfig",
    "CaltechVisibleTerrainProbe",
    "evaluate_caltech_frozen_probe",
    "load_caltech_probe_checkpoint",
    "train_caltech_visible_probe",
]
