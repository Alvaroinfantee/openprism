"""Frozen LLVIP person-detection probe for fused operator views.

This evaluator deliberately uses the same detector, thresholds, and matching
code for every view.  It is a probe of information accessibility, not a claim
that a COCO-visible detector is an unbiased thermal benchmark.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from ..datasets import DatasetCatalog, SampleRecord
from ..fusion import EvidenceFusionEngine
from .baselines import fuse_baseline
from .data import ProtocolItem, protocol_items
from .engine import LearnedFusionEngine
from .training import seed_everything


DETECTOR_ID = "torchvision/fasterrcnn_resnet50_fpn_v2-coco-v1"
BUILTIN_VIEWS = (
    "visible_rgb",
    "thermal_grayscale",
    "average",
    "maximum",
    "deterministic_openprism_operator",
    "prism_egt_operator",
    "prism_egt_luminance",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if not len(boxes):
        return np.empty((0,), dtype=np.float64)
    intersection_min = np.maximum(box[:2], boxes[:, :2])
    intersection_max = np.minimum(box[2:], boxes[:, 2:])
    intersection = np.maximum(0.0, intersection_max - intersection_min)
    intersection_area = intersection[:, 0] * intersection[:, 1]
    area = np.prod(np.maximum(0.0, box[2:] - box[:2]))
    areas = np.prod(np.maximum(0.0, boxes[:, 2:] - boxes[:, :2]), axis=1)
    return intersection_area / np.maximum(area + areas - intersection_area, 1e-12)


def _match_detections(
    ground_truth: Mapping[str, Sequence[Sequence[float]]],
    predictions: Sequence[tuple[str, float, Sequence[float]]],
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered = sorted(predictions, key=lambda item: (-item[1], item[0]))
    truth = {
        image_id: np.asarray(boxes, dtype=np.float64).reshape((-1, 4))
        for image_id, boxes in ground_truth.items()
    }
    matched = {
        image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in truth.items()
    }
    true_positive = np.zeros(len(ordered), dtype=np.float64)
    false_positive = np.zeros(len(ordered), dtype=np.float64)
    scores = np.zeros(len(ordered), dtype=np.float64)
    for index, (image_id, score, raw_box) in enumerate(ordered):
        scores[index] = score
        boxes = truth.get(image_id, np.empty((0, 4), dtype=np.float64))
        available = ~matched.get(image_id, np.empty((0,), dtype=bool))
        overlaps = _box_iou(np.asarray(raw_box, dtype=np.float64), boxes)
        overlaps[~available] = -1.0
        if overlaps.size and float(overlaps.max()) >= iou_threshold:
            selected = int(overlaps.argmax())
            matched[image_id][selected] = True
            true_positive[index] = 1.0
        else:
            false_positive[index] = 1.0
    return scores, true_positive, false_positive, np.asarray(
        [sum(len(boxes) for boxes in truth.values())], dtype=np.float64
    )


def _average_precision(
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    ground_truth_count: int,
) -> float:
    if ground_truth_count <= 0 or not len(true_positive):
        return 0.0
    recall = np.cumsum(true_positive) / ground_truth_count
    precision = np.cumsum(true_positive) / np.maximum(
        np.cumsum(true_positive + false_positive), 1e-12
    )
    recall_envelope = np.concatenate(([0.0], recall, [1.0]))
    precision_envelope = np.concatenate(([0.0], precision, [0.0]))
    precision_envelope = np.maximum.accumulate(precision_envelope[::-1])[::-1]
    changes = np.where(recall_envelope[1:] != recall_envelope[:-1])[0]
    return float(
        np.sum(
            (recall_envelope[changes + 1] - recall_envelope[changes])
            * precision_envelope[changes + 1]
        )
    )


def _log_average_miss_rate(
    true_positive: np.ndarray,
    false_positive: np.ndarray,
    ground_truth_count: int,
    image_count: int,
) -> float:
    if ground_truth_count <= 0 or image_count <= 0:
        return 1.0
    fppi = np.cumsum(false_positive) / image_count
    miss = 1.0 - np.cumsum(true_positive) / ground_truth_count
    reference_fppi = np.logspace(-2.0, 0.0, 9)
    sampled = []
    for reference in reference_fppi:
        eligible = miss[fppi <= reference]
        sampled.append(float(eligible.min()) if eligible.size else 1.0)
    return float(np.exp(np.mean(np.log(np.maximum(sampled, 1e-10)))))


def detection_metrics(
    ground_truth: Mapping[str, Sequence[Sequence[float]]],
    predictions: Sequence[tuple[str, float, Sequence[float]]],
) -> dict[str, float | int]:
    """Compute deterministic AP and log-average miss rate without pycocotools."""

    thresholds = np.arange(0.50, 0.951, 0.05)
    average_precisions = []
    ap50_tp = ap50_fp = None
    truth_count = sum(len(boxes) for boxes in ground_truth.values())
    for threshold in thresholds:
        _, true_positive, false_positive, _ = _match_detections(
            ground_truth, predictions, float(threshold)
        )
        average_precisions.append(
            _average_precision(true_positive, false_positive, truth_count)
        )
        if abs(float(threshold) - 0.5) < 1e-6:
            ap50_tp, ap50_fp = true_positive, false_positive
    assert ap50_tp is not None and ap50_fp is not None
    return {
        "ap_50_95": float(np.mean(average_precisions)),
        "ap50": float(average_precisions[0]),
        "log_average_miss_rate": _log_average_miss_rate(
            ap50_tp, ap50_fp, truth_count, len(ground_truth)
        ),
        "ground_truth_boxes": truth_count,
        "predicted_boxes": len(predictions),
        "images": len(ground_truth),
    }


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


def _tensor_rgb(value: np.ndarray) -> Tensor:
    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    result = array.astype(np.float32)
    if result.max(initial=0.0) > 1.0:
        result /= 255.0
    return torch.from_numpy(np.moveaxis(result, -1, 0).copy()).clamp(0.0, 1.0)


def _ground_truth_boxes(frame) -> list[list[float]]:
    height, width = frame.observations["visible"].data.shape[:2]
    return [
        [
            detection.x * width,
            detection.y * height,
            (detection.x + detection.width) * width,
            (detection.y + detection.height) * height,
        ]
        for detection in frame.detections
        if detection.label.lower() == "person"
    ]


def _external_image(directory: Path, sample_id: str, expected: tuple[int, int]) -> Tensor:
    matches = [
        candidate
        for suffix in (".png", ".jpg", ".jpeg", ".bmp")
        if (candidate := directory / f"{sample_id}{suffix}").is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one external fused image for {sample_id} in {directory}; found {matches}"
        )
    with Image.open(matches[0]) as source:
        image = np.asarray(source.convert("RGB"))
    if image.shape[:2] != expected:
        raise ValueError(
            f"external fused image {matches[0]} has geometry {image.shape[:2]}, expected {expected}"
        )
    return _tensor_rgb(image)


def _view_tensor(
    name: str,
    frame,
    *,
    learned_engine: LearnedFusionEngine | None,
    external_fused: Mapping[str, Path],
    cache: dict[str, object],
) -> Tensor:
    visible = np.asarray(frame.observations["visible"].data)
    thermal = _normalize_thermal(frame.observations["thermal"].data)
    if name == "visible_rgb":
        return _tensor_rgb(visible)
    if name == "thermal_grayscale":
        return _tensor_rgb(thermal)
    if name in {"average", "maximum"}:
        rgb_tensor = _tensor_rgb(visible)[None]
        thermal_tensor = torch.from_numpy(thermal[None, None])
        evidence = torch.ones((1, 3, *thermal.shape), dtype=torch.float32)
        luminance = fuse_baseline(name, rgb_tensor, thermal_tensor, evidence)[0, 0]
        return luminance.repeat(3, 1, 1)
    if name == "deterministic_openprism_operator":
        if "deterministic" not in cache:
            cache["deterministic"] = EvidenceFusionEngine().fuse(frame)
        return _tensor_rgb(cache["deterministic"].operator_rgb)
    if name in {"prism_egt_operator", "prism_egt_luminance"}:
        if learned_engine is None:
            raise ValueError("PRISM-EGT views require a learned checkpoint")
        if "prism_egt" not in cache:
            cache["prism_egt"] = learned_engine.fuse(frame, task="search")
        output = cache["prism_egt"]
        if name == "prism_egt_operator":
            return _tensor_rgb(output.operator_rgb)
        luminance = output.machine_tensor[
            output.channel_names.index("learned_fused_luminance")
        ]
        return _tensor_rgb(luminance)
    if name.startswith("external:"):
        identifier = name.removeprefix("external:")
        if identifier not in external_fused:
            raise ValueError(f"no directory supplied for external view {identifier}")
        return _external_image(
            external_fused[identifier],
            str(frame.provenance["sample_id"]),
            visible.shape[:2],
        )
    raise ValueError(f"unknown detection view: {name}")


def _load_detector(device: torch.device):
    try:
        import torchvision
        from torchvision.models.detection import (
            FasterRCNN_ResNet50_FPN_V2_Weights,
            fasterrcnn_resnet50_fpn_v2,
        )
    except ImportError as error:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "the detection probe requires the 'evaluation' optional dependency"
        ) from error
    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()
    checkpoint_name = Path(weights.url).name
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_name
    return model, {
        "id": DETECTOR_ID,
        "torchvision_version": torchvision.__version__,
        "weights_url": weights.url,
        "weights_sha256": _sha256(checkpoint) if checkpoint.is_file() else None,
    }


def _select_llvip_items(
    data_root: Path, partition: str, max_samples: int | None, seed: int
) -> list[ProtocolItem]:
    items = [
        item
        for item in protocol_items(data_root, partition)
        if item.record.dataset == "llvip"
    ]
    if max_samples is not None and len(items) > max_samples:
        chooser = random.Random(seed)
        items = [items[index] for index in sorted(chooser.sample(range(len(items)), max_samples))]
    return items


def evaluate_llvip_detection(
    data_root: Path,
    output_path: Path,
    *,
    checkpoint_path: Path | None = None,
    partition: str = "validation",
    max_samples: int | None = None,
    score_floor: float = 0.001,
    max_detections: int = 100,
    device_name: str = "auto",
    seed: int = 20260902,
    unlock_final_test: bool = False,
    views: Sequence[str] = BUILTIN_VIEWS,
    external_fused: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Run one frozen detector over identical LLVIP samples and image views."""

    if partition not in {"validation", "test"}:
        raise ValueError("LLVIP detection partition must be validation or test")
    if partition == "test" and not unlock_final_test:
        raise ValueError("final test is locked; freeze all artifacts before unlocking")
    if not 0.0 <= score_floor < 1.0:
        raise ValueError("score_floor must be in [0, 1)")
    if max_detections <= 0:
        raise ValueError("max_detections must be positive")
    external = dict(external_fused or {})
    unknown = set(views) - set(BUILTIN_VIEWS) - {
        f"external:{name}" for name in external
    }
    if unknown:
        raise ValueError(f"unknown views: {sorted(unknown)}")
    if any(name.startswith("prism_egt_") for name in views) and checkpoint_path is None:
        raise ValueError("--checkpoint is required when evaluating PRISM-EGT")

    seed_everything(seed)
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    detector, detector_metadata = _load_detector(device)
    learned_engine = (
        LearnedFusionEngine.from_checkpoint(str(checkpoint_path), device=device)
        if checkpoint_path is not None
        else None
    )
    catalog = DatasetCatalog(data_root)
    items = _select_llvip_items(data_root, partition, max_samples, seed)
    if not items:
        raise ValueError("the selected LLVIP partition contains no samples")

    ground_truth: dict[str, list[list[float]]] = {}
    predictions: dict[str, list[tuple[str, float, Sequence[float]]]] = {
        name: [] for name in views
    }
    sample_ids = []
    record_indices = {
        (split, catalog.record("llvip", split, index).sample_id): index
        for split in ("train", "test")
        for index in range(catalog.count("llvip", split))
    }
    for item in items:
        record: SampleRecord = item.record
        frame = catalog.load(
            record.dataset,
            record.split,
            record_indices[(record.split, record.sample_id)],
        )
        sample_ids.append(record.sample_id)
        ground_truth[record.sample_id] = _ground_truth_boxes(frame)
        view_cache: dict[str, object] = {}
        for name in views:
            image = _view_tensor(
                name,
                frame,
                learned_engine=learned_engine,
                external_fused=external,
                cache=view_cache,
            ).to(device)
            with torch.inference_mode():
                prediction = detector([image])[0]
            selected = (
                (prediction["labels"] == 1)
                & (prediction["scores"] >= score_floor)
            )
            boxes = prediction["boxes"][selected][:max_detections].detach().cpu().numpy()
            scores = prediction["scores"][selected][:max_detections].detach().cpu().numpy()
            predictions[name].extend(
                (record.sample_id, float(score), box.tolist())
                for score, box in zip(scores, boxes)
            )

    result = {
        name: detection_metrics(ground_truth, values)
        for name, values in predictions.items()
    }
    report: dict[str, object] = {
        "schema_version": "openprism.llvip-detection-probe/1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development_subset_not_for_paper"
            if max_samples is not None
            else "validation_protocol_result_requires_human_review"
        ),
        "partition": partition,
        "final_test_unlocked": partition == "test" and unlock_final_test,
        "seed": seed,
        "sample_ids": sample_ids,
        "score_floor": score_floor,
        "max_detections_per_image": max_detections,
        "detector": detector_metadata,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device),
        },
        "views": list(views),
        "external_directories": {
            name: str(path.resolve()) for name, path in external.items()
        },
        "metrics": result,
        "limitations": [
            "The frozen COCO detector is biased toward visible RGB imagery.",
            "This probe measures accessibility to one unchanged detector, not universal task utility.",
            "LLVIP annotations contain people; terrain and semantic claims require separate evaluators.",
            "A development subset is never a paper result.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _external_argument(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("external view must be NAME=DIRECTORY")
    return name, Path(raw_path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/prism-egt/llvip-detection.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--view", action="append", choices=BUILTIN_VIEWS)
    parser.add_argument(
        "--external-fused",
        action="append",
        type=_external_argument,
        default=[],
        metavar="NAME=DIRECTORY",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    external = dict(args.external_fused)
    views = tuple(args.view or BUILTIN_VIEWS) + tuple(
        f"external:{name}" for name in external
    )
    report = evaluate_llvip_detection(
        args.data_root,
        args.output,
        checkpoint_path=args.checkpoint,
        partition=args.partition,
        max_samples=args.max_samples,
        score_floor=args.score_floor,
        max_detections=args.max_detections,
        device_name=args.device,
        seed=args.seed,
        unlock_final_test=args.unlock_final_test,
        views=views,
        external_fused=external,
    )
    print(json.dumps({
        "scientific_status": report["scientific_status"],
        "partition": report["partition"],
        "metrics": report["metrics"],
    }, indent=2))


if __name__ == "__main__":
    main()
