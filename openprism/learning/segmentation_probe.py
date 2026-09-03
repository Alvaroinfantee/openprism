"""Train and run a frozen visible-domain semantic probe on MSRS views.

The compact probe is deliberately trained on visible RGB only.  At evaluation
time one unchanged checkpoint is applied to identical samples rendered through
each comparison view.  This measures accessibility to a visible-domain model;
it is not an assertion that the probe is unbiased for thermal or fused imagery.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time
from typing import Iterable, Mapping, MutableMapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..contracts import SynchronizationStatus
from ..datasets import DatasetCatalog, SampleRecord
from ..fusion import EvidenceFusionEngine
from .data import msrs_scene_group, protocol_items, protocol_manifest
from .engine import LearnedFusionEngine
from .segmentation_evaluation import (
    FrozenModelProvenance,
    SegmentationCase,
    evaluate_msrs_semantics,
    paired_scene_group_comparison,
)


CHECKPOINT_SCHEMA = "openprism.msrs-visible-segmentation-probe/1.0"
REPORT_SCHEMA = "openprism.msrs-multiview-semantic-probe/1.0"
DEFAULT_VIEWS = (
    "visible_rgb",
    "thermal_grayscale",
    "average",
    "maximum",
    "deterministic_openprism_operator",
    "prism_egt_operator",
    "prism_egt_luminance",
    "prism_egt_operator_automatic",
    "prism_egt_luminance_automatic",
)
IGNORE_INDEX = 255


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _string_manifest_sha256(values: Sequence[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    base_channels: int = 12
    image_size: int = 256
    classes: int = 8

    def __post_init__(self) -> None:
        for name in ("base_channels", "image_size", "classes"):
            value = int(getattr(self, name))
            if value <= 0 or value != getattr(self, name):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, value)
        if self.image_size < 16:
            raise ValueError("image_size must be at least 16")
        if self.classes != 8:
            raise ValueError("the frozen MSRS taxonomy requires exactly 8 classes")

    def as_dict(self) -> dict[str, int]:
        return {
            "base_channels": self.base_channels,
            "image_size": self.image_size,
            "classes": self.classes,
        }


class _ConvBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(1, outputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1, bias=False),
            nn.GroupNorm(1, outputs),
            nn.ReLU(inplace=True),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class MSRSVisibleSegmentationProbe(nn.Module):
    """Small deterministic two-level U-Net-like 8-class segmenter."""

    def __init__(self, config: ProbeConfig = ProbeConfig()) -> None:
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


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"semantic mask must be a 2-D integer image: {path}")
    return value


def _target_indices(mask: np.ndarray) -> np.ndarray:
    value = np.asarray(mask)
    supplied = set(int(item) for item in np.unique(value))
    unknown = sorted(supplied - set(range(9)) - {255})
    if unknown:
        raise ValueError(f"MSRS mask contains labels outside 0..8/255: {unknown}")
    result = np.full(value.shape, IGNORE_INDEX, dtype=np.int64)
    labeled = (value >= 1) & (value <= 8)
    result[labeled] = value[labeled].astype(np.int64) - 1
    return result


def _resize_rgb_tensor(rgb: np.ndarray, size: int) -> Tensor:
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


def _resize_target_tensor(mask: np.ndarray, size: int) -> Tensor:
    target = torch.from_numpy(_target_indices(mask))[None, None].float()
    return F.interpolate(target, (size, size), mode="nearest")[0, 0].long()


class _TrainingDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, records: Sequence[SampleRecord], image_size: int) -> None:
        self.records = tuple(records)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        record = self.records[index]
        if record.annotation_path is None:
            raise ValueError(f"missing semantic annotation for {record.sample_id}")
        return (
            _resize_rgb_tensor(_load_rgb(record.visible_path), self.image_size),
            _resize_target_tensor(_load_mask(record.annotation_path), self.image_size),
        )


def _semantic_records(data_root: Path, partition: str) -> tuple[SampleRecord, ...]:
    return tuple(
        item.record
        for item in protocol_items(data_root, partition, include_detection_subset=False)
        if item.record.dataset == "msrs"
        and item.record.annotation_kind == "semantic_mask"
        and item.record.annotation_path is not None
    )


def class_weights_from_training_records(records: Sequence[SampleRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-square-root class weights computed from training labels only."""

    counts = np.zeros(8, dtype=np.int64)
    for record in records:
        if record.annotation_path is None:
            continue
        values = _target_indices(_load_mask(record.annotation_path))
        for class_index in range(8):
            counts[class_index] += int(np.count_nonzero(values == class_index))
    present = counts > 0
    if not np.any(present):
        raise ValueError("training records contain no evaluated MSRS labels")
    weights = np.zeros(8, dtype=np.float64)
    weights[present] = np.sqrt(np.sum(counts[present]) / counts[present])
    weights[present] /= np.mean(weights[present])
    return weights.astype(np.float32), counts


def _predict_mask(
    model: MSRSVisibleSegmentationProbe,
    rgb: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    height, width = np.asarray(rgb).shape[:2]
    input_tensor = _resize_rgb_tensor(rgb, model.config.image_size)[None].to(device)
    with torch.inference_mode():
        logits = model(input_tensor)
        logits = F.interpolate(logits, (height, width), mode="bilinear", align_corners=False)
        prediction = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int16) + 1
    return prediction


@dataclass(frozen=True, slots=True)
class ProbeCheckpointMetadata:
    path: Path
    artifact_sha256: str
    model_id: str
    training_run_id: str
    best_epoch: int
    validation_mean_iou: float
    development_subset: bool
    config: ProbeConfig
    class_weights: tuple[float, ...]
    training_sample_manifest_sha256: str
    validation_sample_manifest_sha256: str


def _save_checkpoint(
    path: Path,
    model: MSRSVisibleSegmentationProbe,
    *,
    model_id: str,
    training_run_id: str,
    epoch: int,
    validation_mean_iou: float,
    development_subset: bool,
    class_weights: np.ndarray,
    training_ids: Sequence[str],
    validation_ids: Sequence[str],
) -> ProbeCheckpointMetadata:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "model_id": model_id,
        "training_run_id": training_run_id,
        "input_domain": "visible_rgb_only",
        "training_partition": "protocol_train",
        "selection_partition": "protocol_validation",
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
    return _checkpoint_metadata(path, payload)


def _checkpoint_metadata(path: Path, payload: Mapping[str, object]) -> ProbeCheckpointMetadata:
    return ProbeCheckpointMetadata(
        path=path.resolve(),
        artifact_sha256=_sha256(path),
        model_id=str(payload["model_id"]),
        training_run_id=str(payload["training_run_id"]),
        best_epoch=int(payload["best_epoch"]),
        validation_mean_iou=float(payload["validation_mean_iou"]),
        development_subset=bool(payload["development_subset"]),
        config=ProbeConfig(**dict(payload["config"])),
        class_weights=tuple(float(item) for item in payload["class_weights"]),
        training_sample_manifest_sha256=str(payload["training_sample_manifest_sha256"]),
        validation_sample_manifest_sha256=str(payload["validation_sample_manifest_sha256"]),
    )


def load_probe_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MSRSVisibleSegmentationProbe, ProbeCheckpointMetadata]:
    source = Path(path)
    payload = torch.load(source, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported MSRS probe checkpoint schema")
    required = {
        "model_id", "training_run_id", "input_domain", "training_partition",
        "selection_partition", "best_epoch", "validation_mean_iou",
        "development_subset", "config", "class_weights",
        "training_sample_manifest_sha256", "validation_sample_manifest_sha256", "state_dict",
    }
    if missing := required - payload.keys():
        raise ValueError(f"probe checkpoint is missing fields: {sorted(missing)}")
    if payload["input_domain"] != "visible_rgb_only":
        raise ValueError("probe checkpoint was not trained on the frozen visible-only domain")
    if payload["training_partition"] != "protocol_train" or payload["selection_partition"] != "protocol_validation":
        raise ValueError("probe checkpoint does not satisfy the frozen train/validation protocol")
    weights = np.asarray(payload["class_weights"], dtype=np.float64)
    if weights.shape != (8,) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("probe checkpoint class weights are malformed")
    if int(payload["best_epoch"]) <= 0 or not 0.0 <= float(payload["validation_mean_iou"]) <= 1.0:
        raise ValueError("probe checkpoint selection metadata is malformed")
    config = ProbeConfig(**dict(payload["config"]))
    model = MSRSVisibleSegmentationProbe(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, _checkpoint_metadata(source, payload)


def _validation_score(
    model: MSRSVisibleSegmentationProbe,
    records: Sequence[SampleRecord],
    device: torch.device,
) -> float:
    cases = []
    for record in records:
        assert record.annotation_path is not None
        rgb = _load_rgb(record.visible_path)
        truth = _load_mask(record.annotation_path)
        cases.append(
            SegmentationCase(
                record.sample_id,
                msrs_scene_group(record.sample_id),
                truth,
                _predict_mask(model, rgb, device),
            )
        )
    development_provenance = FrozenModelProvenance(
        model_id="in-training-visible-probe",
        artifact_sha256=None,
        source="OpenPRISM in-process training",
        source_revision=CHECKPOINT_SCHEMA,
        preprocessing_id=f"square-bilinear-{model.config.image_size}",
        pretrained=False,
        frozen=False,
        training_data="MSRS protocol train only",
    )
    report = evaluate_msrs_semantics(
        cases,
        development_provenance,
        partition="validation",
        bootstrap_replicates=1,
    )
    score = report["metrics"]["mean_iou"]
    if score is None:
        raise ValueError("validation split contains no evaluated labels")
    return float(score)


def train_visible_probe(
    data_root: str | Path,
    checkpoint_path: str | Path,
    *,
    config: ProbeConfig = ProbeConfig(),
    epochs: int = 12,
    batch_size: int = 8,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    device_name: str = "auto",
    seed: int = 20260903,
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
) -> dict[str, object]:
    """Train on protocol-train RGB and select one checkpoint on validation mIoU."""

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("training hyperparameters are invalid")
    root = Path(data_root)
    train_records = list(_semantic_records(root, "train"))
    validation_records = list(_semantic_records(root, "validation"))
    if max_train_samples is not None:
        train_records = train_records[:max_train_samples]
    if max_validation_samples is not None:
        validation_records = validation_records[:max_validation_samples]
    if not train_records or not validation_records:
        raise ValueError("protocol train and validation must both contain MSRS semantic masks")
    if {record.sample_id for record in train_records} & {
        record.sample_id for record in validation_records
    }:
        raise ValueError("training and validation sample identifiers overlap")
    training_groups = {msrs_scene_group(record.sample_id) for record in train_records}
    validation_groups = {
        msrs_scene_group(record.sample_id) for record in validation_records
    }
    if training_groups & validation_groups:
        raise ValueError("training and validation scene groups overlap")

    _seed_everything(seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    class_weights, class_counts = class_weights_from_training_records(train_records)
    model = MSRSVisibleSegmentationProbe(config).to(device)
    objective = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device),
        ignore_index=IGNORE_INDEX,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        _TrainingDataset(train_records, config.image_size),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    training_run_id = hashlib.sha256(
        f"{seed}:{config.as_dict()}:{[r.sample_id for r in train_records]}".encode("utf-8")
    ).hexdigest()[:16]
    model_id = f"openprism-msrs-visible-probe-{training_run_id}"
    best_score = -1.0
    best_metadata: ProbeCheckpointMetadata | None = None
    history: list[dict[str, float | int]] = []
    destination = Path(checkpoint_path)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        examples = 0
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if not bool(torch.any(targets != IGNORE_INDEX)):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(images), targets)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("segmentation probe loss became non-finite")
            loss.backward()
            optimizer.step()
            batch_examples = int(images.shape[0])
            loss_total += float(loss.detach().cpu()) * batch_examples
            examples += batch_examples
        if examples == 0:
            raise ValueError("training epoch contains no evaluated semantic pixels")
        model.eval()
        validation_mean_iou = _validation_score(model, validation_records, device)
        history.append(
            {
                "epoch": epoch,
                "training_class_weighted_cross_entropy": loss_total / examples,
                "validation_mean_iou": validation_mean_iou,
            }
        )
        if validation_mean_iou > best_score:
            best_score = validation_mean_iou
            best_metadata = _save_checkpoint(
                destination,
                model,
                model_id=model_id,
                training_run_id=training_run_id,
                epoch=epoch,
                validation_mean_iou=validation_mean_iou,
                development_subset=(
                    max_train_samples is not None or max_validation_samples is not None
                ),
                class_weights=class_weights,
                training_ids=[record.sample_id for record in train_records],
                validation_ids=[record.sample_id for record in validation_records],
            )
    assert best_metadata is not None
    report = {
        "schema_version": "openprism.msrs-visible-probe-training/1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "path": str(best_metadata.path),
            "artifact_sha256": best_metadata.artifact_sha256,
            "model_id": best_metadata.model_id,
            "best_epoch": best_metadata.best_epoch,
            "validation_mean_iou": best_metadata.validation_mean_iou,
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
            "training_partition": "protocol_train",
            "checkpoint_selection_partition": "protocol_validation",
            "publisher_test_accessed": False,
            "training_samples": len(train_records),
            "validation_samples": len(validation_records),
            "development_subset": best_metadata.development_subset,
        },
        "class_weighting": {
            "method": "inverse_sqrt_frequency_normalized_over_present_classes",
            "counts_train_only": class_counts.tolist(),
            "weights": class_weights.tolist(),
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
            "This compact probe is a controlled evaluator, not a state-of-the-art segmenter.",
            "It is trained only on visible RGB and is therefore biased toward its visible input domain.",
            "Square resizing changes image aspect ratio and limits absolute segmentation quality.",
        ],
    }
    report_path = destination.with_suffix(destination.suffix + ".training.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _normalize_thermal(value: np.ndarray) -> np.ndarray:
    thermal = np.asarray(value, dtype=np.float32)
    if thermal.ndim == 3:
        thermal = thermal.mean(axis=2)
    finite = thermal[np.isfinite(thermal)]
    if not finite.size:
        return np.zeros(thermal.shape, dtype=np.float32)
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        return np.zeros(thermal.shape, dtype=np.float32)
    return np.clip((thermal - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    value = np.asarray(rgb, dtype=np.float32)
    if value.max(initial=0.0) > 1.0:
        value /= 255.0
    return 0.299 * value[..., 0] + 0.587 * value[..., 1] + 0.114 * value[..., 2]


def _external_image(directory: Path, sample_id: str, shape: tuple[int, int]) -> np.ndarray:
    candidates = [
        directory / f"{sample_id}{suffix}"
        for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        if (directory / f"{sample_id}{suffix}").is_file()
    ]
    if len(candidates) != 1:
        raise FileNotFoundError(f"expected one external image for {sample_id}; found {len(candidates)}")
    image = _load_rgb(candidates[0])
    if image.shape[:2] != shape:
        raise ValueError(f"external image geometry {image.shape[:2]} does not match {shape}")
    return image


def _view_rgb(
    name: str,
    frame,
    *,
    deterministic_engine: EvidenceFusionEngine,
    learned_engine: LearnedFusionEngine | None,
    external: Mapping[str, Path],
    learned_task: str = "navigate",
    cache: MutableMapping[str, object] | None = None,
) -> np.ndarray:
    visible = np.asarray(frame.observations["visible"].data)
    thermal = _normalize_thermal(frame.observations["thermal"].data)
    thermal_rgb = np.repeat(thermal[..., None], 3, axis=2)
    visible_float = visible.astype(np.float32) / 255.0
    if name == "visible_rgb":
        return visible
    if name == "thermal_grayscale":
        return thermal_rgb
    visible_luminance = _luminance(visible)
    if name == "average":
        value = 0.5 * visible_luminance + 0.5 * thermal
        return np.repeat(value[..., None], 3, axis=2)
    if name == "maximum":
        value = np.maximum(visible_luminance, thermal)
        return np.repeat(value[..., None], 3, axis=2)
    if name == "deterministic_openprism_operator":
        cache_key = "deterministic_openprism"
        if cache is not None and cache_key in cache:
            deterministic_output = cache[cache_key]
        else:
            deterministic_output = deterministic_engine.fuse(frame)
            if cache is not None:
                cache[cache_key] = deterministic_output
        return deterministic_output.operator_rgb
    if name in {
        "prism_egt_operator",
        "prism_egt_luminance",
        "prism_egt_operator_automatic",
        "prism_egt_luminance_automatic",
    }:
        if learned_engine is None:
            raise ValueError("a frozen PRISM-EGT checkpoint is required for learned views")
        selected_task = "automatic" if name.endswith("_automatic") else learned_task
        cache_key = f"prism_egt:{selected_task}"
        if cache is not None and cache_key in cache:
            output = cache[cache_key]
        else:
            output = learned_engine.fuse(frame, task=selected_task)
            if cache is not None:
                cache[cache_key] = output
        if not output.provenance.get("learned_fusion_applied", False):
            raise RuntimeError("PRISM-EGT did not pass the evidence gate")
        if name in {"prism_egt_operator", "prism_egt_operator_automatic"}:
            return output.operator_rgb
        channel = output.machine_tensor[
            output.channel_names.index("learned_fused_luminance")
        ]
        return np.repeat(channel[..., None], 3, axis=2)
    if name.startswith("external:"):
        key = name.split(":", 1)[1]
        if key not in external:
            raise ValueError(f"unknown external view: {key}")
        return _external_image(external[key], frame.provenance["sample_id"], visible.shape[:2])
    raise ValueError(f"unknown segmentation view: {name}")


def _declared_aligned_replay(frame):
    synchronization = SynchronizationStatus.declared_replay_aligned(
        tuple(frame.observations),
        clock_domain=frame.timestamp.effective_clock_domain,
        declaration="MSRS publisher-provided aligned visible/infrared replay pair",
    )
    provenance = dict(frame.provenance)
    provenance["evaluation_alignment_assumption"] = (
        "publisher-provided spatial alignment; no measured capture timing"
    )
    return replace(frame, synchronization=synchronization, provenance=provenance)


def evaluate_frozen_probe(
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
    """Apply one visible-trained checkpoint unchanged to all requested views."""

    if partition not in {"validation", "test"}:
        raise ValueError("probe evaluation partition must be validation or test")
    if partition == "test" and not unlock_final_test:
        raise ValueError("publisher test is locked; freeze all artifacts before explicit unlock")
    if partition == "test" and max_samples is not None:
        raise ValueError("publisher final-test evaluation must include the complete partition")
    requested = tuple(views)
    external = {str(name): Path(path) for name, path in (external_fused or {}).items()}
    requested += tuple(f"external:{name}" for name in external if f"external:{name}" not in requested)
    if len(set(requested)) != len(requested):
        raise ValueError("evaluation views must be unique")
    unknown = set(requested) - set(DEFAULT_VIEWS) - {f"external:{name}" for name in external}
    if unknown:
        raise ValueError(f"unknown views: {sorted(unknown)}")
    if any(name.startswith("prism_egt_") for name in requested) and prism_egt_checkpoint is None:
        raise ValueError("prism_egt_checkpoint is required for PRISM-EGT views")

    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    model, metadata = load_probe_checkpoint(checkpoint_path, device=device)
    if partition == "test" and metadata.development_subset:
        raise ValueError("publisher test cannot be unlocked with a development-subset checkpoint")
    provenance = FrozenModelProvenance(
        model_id=metadata.model_id,
        artifact_sha256=metadata.artifact_sha256,
        source="OpenPRISM compact MSRS visible-domain probe",
        source_revision=metadata.training_run_id,
        preprocessing_id=f"square-bilinear-{metadata.config.image_size}",
        pretrained=False,
        frozen=True,
        training_data="MSRS protocol train visible RGB only",
        notes=("best checkpoint selected on protocol validation mean IoU",),
    )
    learned_engine = (
        LearnedFusionEngine.from_checkpoint(str(prism_egt_checkpoint), device=device)
        if prism_egt_checkpoint is not None
        else None
    )
    deterministic_engine = EvidenceFusionEngine()
    root = Path(data_root)
    items = [
        item for item in protocol_items(root, partition, include_detection_subset=False)
        if item.record.dataset == "msrs" and item.record.annotation_path is not None
    ]
    if max_samples is not None:
        items = items[:max_samples]
    if not items:
        raise ValueError("selected protocol partition contains no MSRS semantic samples")

    catalog = DatasetCatalog(root)
    record_indexes = {
        (split, catalog.record("msrs", split, index).sample_id): index
        for split in ("train", "test")
        for index in range(catalog.count("msrs", split))
    }
    cases = {name: [] for name in requested}
    for item in items:
        record = item.record
        frame = _declared_aligned_replay(
            catalog.load("msrs", record.split, record_indexes[(record.split, record.sample_id)])
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
                    learned_task="navigate",
                    cache=view_cache,
                )
                prediction = _predict_mask(model, view, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            except Exception as error:
                failure_reason = f"{type(error).__name__}:{error}"
            latency_ms = (time.perf_counter() - started) * 1_000.0
            cases[name].append(
                SegmentationCase(
                    record.sample_id,
                    msrs_scene_group(record.sample_id),
                    truth,
                    prediction,
                    latency_ms=latency_ms,
                    failure_reason=failure_reason,
                )
            )

    view_reports = {
        name: evaluate_msrs_semantics(
            value,
            provenance,
            partition=partition,
            unlock_final_test=unlock_final_test,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        for name, value in cases.items()
    }
    comparisons: dict[str, object] = {}
    for name in requested:
        if name == "visible_rgb":
            continue
        try:
            comparisons[name] = paired_scene_group_comparison(
                cases["visible_rgb"],
                cases[name],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
        except ValueError as error:
            comparisons[name] = {"available": False, "reason": str(error)}

    external_hashes = {}
    for name, directory in external.items():
        files = sorted(path for path in directory.iterdir() if path.is_file())
        external_hashes[name] = {
            "directory": str(directory.resolve()),
            "file_count": len(files),
            "manifest_sha256": _string_manifest_sha256(
                [f"{path.name}:{_sha256(path)}" for path in files]
            ),
        }
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development_subset_not_for_paper"
            if max_samples is not None
            else
            "publisher_test_unlocked_requires_human_review"
            if partition == "test"
            else "protocol_validation"
        ),
        "partition": partition,
        "publisher_test_unlocked": partition == "test" and unlock_final_test,
        "visible_probe_checkpoint": {
            "path": str(metadata.path),
            "artifact_sha256": metadata.artifact_sha256,
            "model_id": metadata.model_id,
            "config": metadata.config.as_dict(),
            "best_epoch": metadata.best_epoch,
            "validation_mean_iou": metadata.validation_mean_iou,
            "input_domain": "visible_rgb_only",
            "checkpoint_selection_partition": "protocol_validation",
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
            "sha256": _string_manifest_sha256([item.record.sample_id for item in items]),
            "max_samples": max_samples,
        },
        "views": list(requested),
        "task_mapping": {
            "dataset_proxy": "navigate",
            "prism_egt_operator": "navigate",
            "prism_egt_luminance": "navigate",
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
            "The unchanged probe was trained only on visible RGB and is biased toward that domain.",
            "Cross-view differences measure accessibility to this probe, not universal semantic quality.",
            "MSRS replay alignment is publisher-declared; capture timing was not measured.",
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
    train = subcommands.add_parser("train", help="train visible-only semantic probe")
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

    evaluate = subcommands.add_parser("evaluate", help="run frozen multiview probe")
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
        report = train_visible_probe(
            args.data_root,
            args.checkpoint,
            config=ProbeConfig(args.base_channels, args.image_size),
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
    report = evaluate_frozen_probe(
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
        "publisher_test_unlocked": report["publisher_test_unlocked"],
        "views": report["views"],
    }, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_VIEWS",
    "MSRSVisibleSegmentationProbe",
    "ProbeCheckpointMetadata",
    "ProbeConfig",
    "class_weights_from_training_records",
    "evaluate_frozen_probe",
    "load_probe_checkpoint",
    "train_visible_probe",
]
