"""Frozen Caltech Aerial RGB-T semantic evaluation with scene-group statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Mapping, Sequence

import numpy as np

from ..datasets import CALTECH_CLASSES
from .segmentation_evaluation import (
    FrozenModelProvenance,
    SegmentationCase,
    summarize_latency_and_failures,
)


EVALUATOR_SCHEMA = "openprism.caltech-aerial-rgbt-segmentation-evaluation/1.0"
EVALUATOR_ID = "openprism-caltech-terrain-iou-scene-bootstrap-v1"
CALTECH_EVALUATED_CLASS_IDS = tuple(
    sorted(class_id for class_id in CALTECH_CLASSES if class_id)
)
IGNORE_LABELS = (0, 255)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _Counts:
    true_positive: np.ndarray
    truth_pixels: np.ndarray
    predicted_pixels: np.ndarray
    correct_pixels: int
    evaluated_pixels: int

    @classmethod
    def zeros(cls) -> "_Counts":
        zeros = np.zeros(len(CALTECH_EVALUATED_CLASS_IDS), dtype=np.int64)
        return cls(zeros.copy(), zeros.copy(), zeros.copy(), 0, 0)

    def __add__(self, other: "_Counts") -> "_Counts":
        return _Counts(
            self.true_positive + other.true_positive,
            self.truth_pixels + other.truth_pixels,
            self.predicted_pixels + other.predicted_pixels,
            self.correct_pixels + other.correct_pixels,
            self.evaluated_pixels + other.evaluated_pixels,
        )


def _validate_labels(value: np.ndarray, name: str) -> None:
    allowed = set(CALTECH_EVALUATED_CLASS_IDS) | set(IGNORE_LABELS)
    unknown = sorted(set(int(item) for item in np.unique(value)) - allowed)
    if unknown:
        raise ValueError(f"{name} contains labels outside the frozen Caltech taxonomy: {unknown}")


def _case_counts(case: SegmentationCase) -> _Counts:
    if case.prediction is None:
        return _Counts.zeros()
    truth = np.asarray(case.ground_truth)
    prediction = np.asarray(case.prediction)
    _validate_labels(truth, "ground_truth")
    _validate_labels(prediction, "prediction")
    evaluated = ~np.isin(truth, IGNORE_LABELS)
    tp = np.asarray(
        [np.count_nonzero(evaluated & (truth == item) & (prediction == item))
         for item in CALTECH_EVALUATED_CLASS_IDS],
        dtype=np.int64,
    )
    truth_pixels = np.asarray(
        [np.count_nonzero(evaluated & (truth == item))
         for item in CALTECH_EVALUATED_CLASS_IDS],
        dtype=np.int64,
    )
    predicted_pixels = np.asarray(
        [np.count_nonzero(evaluated & (prediction == item))
         for item in CALTECH_EVALUATED_CLASS_IDS],
        dtype=np.int64,
    )
    return _Counts(
        tp,
        truth_pixels,
        predicted_pixels,
        int(np.count_nonzero(evaluated & (truth == prediction))),
        int(np.count_nonzero(evaluated)),
    )


def _sum_counts(values: Sequence[_Counts]) -> _Counts:
    total = _Counts.zeros()
    for value in values:
        total = total + value
    return total


def _metrics(counts: _Counts) -> dict[str, object]:
    union = counts.truth_pixels + counts.predicted_pixels - counts.true_positive
    present = union > 0
    iou = np.full(len(CALTECH_EVALUATED_CLASS_IDS), np.nan, dtype=np.float64)
    iou[present] = counts.true_positive[present] / union[present]
    return {
        "mean_iou": float(np.mean(iou[present])) if np.any(present) else None,
        "per_class": {
            str(class_id): {
                "name": CALTECH_CLASSES[class_id],
                "iou": float(iou[index]) if present[index] else None,
                "intersection_pixels": int(counts.true_positive[index]),
                "union_pixels": int(union[index]),
                "ground_truth_pixels": int(counts.truth_pixels[index]),
                "predicted_pixels": int(counts.predicted_pixels[index]),
            }
            for index, class_id in enumerate(CALTECH_EVALUATED_CLASS_IDS)
        },
        "evaluated_class_count": int(np.count_nonzero(present)),
        "pixel_accuracy": (
            counts.correct_pixels / counts.evaluated_pixels
            if counts.evaluated_pixels else None
        ),
        "evaluated_pixels": counts.evaluated_pixels,
    }


def _interval(values: Sequence[float], confidence: float = 0.95) -> dict[str, float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    alpha = (1.0 - confidence) / 2.0
    return {
        "lower": float(np.quantile(finite, alpha)),
        "upper": float(np.quantile(finite, 1.0 - alpha)),
    }


def caltech_scene_group_bootstrap(
    group_counts: Mapping[str, _Counts],
    *,
    replicates: int = 2_000,
    seed: int = 20260903,
    confidence: float = 0.95,
) -> dict[str, object]:
    if not group_counts:
        raise ValueError("at least one Caltech scene_group is required")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap settings are invalid")
    groups = tuple(sorted(group_counts))
    rng = np.random.default_rng(seed)
    means: list[float] = []
    classes: dict[str, list[float]] = {
        str(item): [] for item in CALTECH_EVALUATED_CLASS_IDS
    }
    for _ in range(replicates):
        selected = rng.integers(0, len(groups), size=len(groups))
        counts = _sum_counts([group_counts[groups[int(index)]] for index in selected])
        metric = _metrics(counts)
        if metric["mean_iou"] is not None:
            means.append(float(metric["mean_iou"]))
        for class_id in CALTECH_EVALUATED_CLASS_IDS:
            value = metric["per_class"][str(class_id)]["iou"]
            if value is not None:
                classes[str(class_id)].append(float(value))
    return {
        "method": "percentile_complete_scene_group_bootstrap",
        "confidence_level": confidence,
        "requested_replicates": replicates,
        "effective_mean_iou_replicates": len(means),
        "seed": int(seed),
        "scene_group_count": len(groups),
        "mean_iou": _interval(means, confidence),
        "per_class_iou": {
            key: _interval(values, confidence) for key, values in classes.items()
        },
    }


def _validated_counts(
    cases: Sequence[SegmentationCase],
) -> tuple[dict[str, _Counts], dict[str, _Counts]]:
    per_sample: dict[str, _Counts] = {}
    per_group: dict[str, _Counts] = {}
    for case in cases:
        if case.sample_id in per_sample:
            raise ValueError(f"duplicate sample_id: {case.sample_id}")
        if not case.scene_group:
            raise ValueError("Caltech evaluation requires a non-empty scene_group")
        counts = _case_counts(case)
        per_sample[case.sample_id] = counts
        per_group[case.scene_group] = per_group.get(case.scene_group, _Counts.zeros()) + counts
    return per_sample, per_group


def _check_test_lock(
    partition: str,
    unlock_final_test: bool,
    provenance: FrozenModelProvenance,
) -> None:
    if partition not in {"validation", "test"}:
        raise ValueError("partition must be validation or test")
    if partition == "test" and not unlock_final_test:
        raise ValueError("Caltech final test is locked; freeze all artifacts before explicit unlock")
    if partition == "test" and (
        not provenance.frozen or provenance.artifact_sha256 is None
    ):
        raise ValueError("Caltech final test requires frozen hashed model provenance")


def evaluate_caltech_semantics(
    cases: Sequence[SegmentationCase],
    provenance: FrozenModelProvenance,
    *,
    partition: str = "validation",
    unlock_final_test: bool = False,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 20260903,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate classes 1--11, including terrain, vehicle, and person."""

    _check_test_lock(partition, unlock_final_test, provenance)
    selected = tuple(cases)
    if not selected:
        raise ValueError("evaluation requires at least one prediction attempt")
    per_sample, per_group = _validated_counts(selected)
    metrics = _metrics(_sum_counts(tuple(per_sample.values())))
    intervals = caltech_scene_group_bootstrap(
        per_group,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    report: dict[str, object] = {
        "schema_version": EVALUATOR_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "caltech_final_test_unlocked_requires_human_review"
            if partition == "test" else "protocol_validation"
        ),
        "partition": partition,
        "final_test_unlocked": partition == "test" and unlock_final_test,
        "evaluator": {
            "id": EVALUATOR_ID,
            "implementation_sha256": _sha256(Path(__file__)),
            "dataset": "Caltech Aerial RGB-T",
            "taxonomy": {str(key): value for key, value in CALTECH_CLASSES.items()},
            "evaluated_class_ids": list(CALTECH_EVALUATED_CLASS_IDS),
            "ignore_ground_truth_labels": list(IGNORE_LABELS),
            "absent_class_policy": "exclude class when aggregate union is zero",
            "failed_prediction_policy": "exclude from IoU and report separately",
        },
        "model_provenance": provenance.as_dict(),
        "metrics": metrics,
        "confidence_intervals_95": intervals,
        "scene_groups": {
            group: {
                "attempted_samples": sum(case.scene_group == group for case in selected),
                "successful_samples": sum(
                    case.scene_group == group and case.prediction is not None
                    for case in selected
                ),
                "metrics": _metrics(counts),
            }
            for group, counts in sorted(per_group.items())
        },
        "runtime_and_failures": summarize_latency_and_failures(selected),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "limitations": [
            "Failed predictions are excluded from IoU and must be read with failure rate.",
            "Confidence intervals treat complete flight/scene groups, not frames, as resampling units.",
            "Frozen visible-domain probe results do not establish thermal-domain fairness.",
        ],
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def paired_caltech_scene_group_comparison(
    baseline_cases: Sequence[SegmentationCase],
    candidate_cases: Sequence[SegmentationCase],
    *,
    replicates: int = 2_000,
    seed: int = 20260903,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Paired candidate-minus-baseline comparison on identical Caltech frames."""

    baseline = {case.sample_id: case for case in baseline_cases}
    candidate = {case.sample_id: case for case in candidate_cases}
    if len(baseline) != len(baseline_cases) or len(candidate) != len(candidate_cases):
        raise ValueError("paired inputs contain duplicate sample identifiers")
    if set(baseline) != set(candidate):
        raise ValueError("paired comparison requires identical sample identifiers")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap settings are invalid")
    complete: list[tuple[SegmentationCase, SegmentationCase]] = []
    for sample_id in sorted(baseline):
        left, right = baseline[sample_id], candidate[sample_id]
        if left.scene_group != right.scene_group:
            raise ValueError(f"scene_group mismatch for {sample_id}")
        if not np.array_equal(left.ground_truth, right.ground_truth):
            raise ValueError(f"ground_truth mismatch for {sample_id}")
        if left.prediction is not None and right.prediction is not None:
            complete.append((left, right))
    if not complete:
        raise ValueError("paired comparison has no samples completed by both methods")
    grouped: dict[str, tuple[_Counts, _Counts]] = {}
    for left, right in complete:
        old = grouped.get(left.scene_group, (_Counts.zeros(), _Counts.zeros()))
        grouped[left.scene_group] = (
            old[0] + _case_counts(left), old[1] + _case_counts(right)
        )
    baseline_metric = _metrics(_sum_counts([item[0] for item in grouped.values()]))
    candidate_metric = _metrics(_sum_counts([item[1] for item in grouped.values()]))
    if baseline_metric["mean_iou"] is None or candidate_metric["mean_iou"] is None:
        raise ValueError("paired comparison has no evaluated semantic classes")
    names = tuple(sorted(grouped))
    rng = np.random.default_rng(seed)
    mean_deltas: list[float] = []
    class_deltas: dict[str, list[float]] = {
        str(item): [] for item in CALTECH_EVALUATED_CLASS_IDS
    }
    for _ in range(replicates):
        selected = rng.integers(0, len(names), size=len(names))
        left_metric = _metrics(_sum_counts([grouped[names[int(i)]][0] for i in selected]))
        right_metric = _metrics(_sum_counts([grouped[names[int(i)]][1] for i in selected]))
        if left_metric["mean_iou"] is not None and right_metric["mean_iou"] is not None:
            mean_deltas.append(float(right_metric["mean_iou"] - left_metric["mean_iou"]))
        for class_id in CALTECH_EVALUATED_CLASS_IDS:
            left_iou = left_metric["per_class"][str(class_id)]["iou"]
            right_iou = right_metric["per_class"][str(class_id)]["iou"]
            if left_iou is not None and right_iou is not None:
                class_deltas[str(class_id)].append(float(right_iou - left_iou))
    observed_class = {}
    for class_id in CALTECH_EVALUATED_CLASS_IDS:
        left = baseline_metric["per_class"][str(class_id)]["iou"]
        right = candidate_metric["per_class"][str(class_id)]["iou"]
        observed_class[str(class_id)] = (
            float(right - left) if left is not None and right is not None else None
        )
    delta = float(candidate_metric["mean_iou"] - baseline_metric["mean_iou"])
    return {
        "method": "paired_complete_scene_group_percentile_bootstrap",
        "direction": "candidate_minus_baseline",
        "attempted_pairs": len(baseline),
        "complete_pairs": len(complete),
        "incomplete_pairs": len(baseline) - len(complete),
        "complete_pair_scene_groups": len(grouped),
        "bootstrap_replicates": replicates,
        "seed": int(seed),
        "confidence_level": confidence,
        "mean_iou": {
            "baseline": baseline_metric["mean_iou"],
            "candidate": candidate_metric["mean_iou"],
            "delta": delta,
            "delta_interval": _interval(mean_deltas, confidence),
            "bootstrap_probability_candidate_better": float(
                np.mean(np.asarray(mean_deltas) > 0.0)
            ) if mean_deltas else None,
        },
        "per_class_iou_delta": {
            str(class_id): {
                "delta": observed_class[str(class_id)],
                "delta_interval": _interval(class_deltas[str(class_id)], confidence),
            }
            for class_id in CALTECH_EVALUATED_CLASS_IDS
        },
        "pairing_policy": "only frames completed by both methods enter paired IoU; attrition is explicit",
    }


__all__ = [
    "CALTECH_EVALUATED_CLASS_IDS",
    "EVALUATOR_ID",
    "IGNORE_LABELS",
    "caltech_scene_group_bootstrap",
    "evaluate_caltech_semantics",
    "paired_caltech_scene_group_comparison",
]
