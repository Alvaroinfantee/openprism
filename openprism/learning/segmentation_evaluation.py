"""Frozen MSRS semantic-segmentation evaluation and grouped statistics.

The evaluator consumes label masks rather than importing a particular model.
That keeps the metric implementation identical for RGB, thermal, OpenPRISM,
and external fusion methods while still requiring traceable model provenance.
Frames from the same scene group are resampled together; adjacent frames are
never treated as independent bootstrap observations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from ..datasets import MSRS_CLASSES


EVALUATOR_SCHEMA = "openprism.msrs-segmentation-evaluation/1.0"
EVALUATOR_ID = "openprism-msrs-iou-scene-bootstrap-v1"
MSRS_EVALUATED_CLASS_IDS = tuple(sorted(class_id for class_id in MSRS_CLASSES if class_id))
DEFAULT_IGNORE_LABELS = (0, 255)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class FrozenModelProvenance:
    """Identity of the unchanged model/evaluator input being measured."""

    model_id: str
    artifact_sha256: str | None
    source: str
    source_revision: str
    preprocessing_id: str
    pretrained: bool
    frozen: bool
    training_data: str = "not_declared"
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("model_id", "source", "source_revision", "preprocessing_id"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} is required")
            object.__setattr__(self, name, value)
        if self.artifact_sha256 is not None:
            checksum = str(self.artifact_sha256).lower().strip()
            if not _SHA256.fullmatch(checksum):
                raise ValueError("artifact_sha256 must be 64 lowercase hexadecimal characters")
            object.__setattr__(self, "artifact_sha256", checksum)
        object.__setattr__(self, "pretrained", bool(self.pretrained))
        object.__setattr__(self, "frozen", bool(self.frozen))
        object.__setattr__(self, "training_data", str(self.training_data).strip())
        object.__setattr__(self, "notes", tuple(str(item) for item in self.notes))

    @classmethod
    def from_artifact(
        cls,
        artifact_path: str | Path,
        *,
        model_id: str,
        source: str,
        source_revision: str,
        preprocessing_id: str,
        pretrained: bool,
        frozen: bool = True,
        training_data: str = "not_declared",
        notes: Sequence[str] = (),
    ) -> "FrozenModelProvenance":
        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(f"model artifact is not a file: {path}")
        return cls(
            model_id=model_id,
            artifact_sha256=_sha256(path),
            source=source,
            source_revision=source_revision,
            preprocessing_id=preprocessing_id,
            pretrained=pretrained,
            frozen=frozen,
            training_data=training_data,
            notes=tuple(notes),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "artifact_sha256": self.artifact_sha256,
            "source": self.source,
            "source_revision": self.source_revision,
            "preprocessing_id": self.preprocessing_id,
            "pretrained": self.pretrained,
            "frozen": self.frozen,
            "training_data": self.training_data,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class SegmentationCase:
    """One MSRS prediction attempt with explicit grouping and failure state."""

    sample_id: str
    scene_group: str
    ground_truth: np.ndarray
    prediction: np.ndarray | None
    latency_ms: float | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        sample_id = str(self.sample_id).strip()
        scene_group = str(self.scene_group).strip()
        if not sample_id or not scene_group:
            raise ValueError("sample_id and scene_group are required")
        truth = np.asarray(self.ground_truth)
        if truth.ndim != 2 or not np.issubdtype(truth.dtype, np.integer):
            raise ValueError("ground_truth must be a two-dimensional integer mask")
        prediction = self.prediction
        reason = None if self.failure_reason is None else str(self.failure_reason).strip()
        if prediction is None:
            if not reason:
                raise ValueError("a missing prediction requires failure_reason")
        else:
            predicted = np.asarray(prediction)
            if predicted.ndim != 2 or not np.issubdtype(predicted.dtype, np.integer):
                raise ValueError("prediction must be a two-dimensional integer mask")
            if predicted.shape != truth.shape:
                raise ValueError("prediction and ground_truth geometry must match")
            if reason:
                raise ValueError("a completed prediction cannot also have failure_reason")
            predicted = np.array(predicted, dtype=np.int64, copy=True)
            predicted.setflags(write=False)
            object.__setattr__(self, "prediction", predicted)
        latency = self.latency_ms
        if latency is not None:
            latency = float(latency)
            if not np.isfinite(latency) or latency < 0.0:
                raise ValueError("latency_ms must be finite and non-negative")
            object.__setattr__(self, "latency_ms", latency)
        truth = np.array(truth, dtype=np.int64, copy=True)
        truth.setflags(write=False)
        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "scene_group", scene_group)
        object.__setattr__(self, "ground_truth", truth)
        object.__setattr__(self, "failure_reason", reason)


@dataclass(frozen=True, slots=True)
class _Counts:
    true_positive: np.ndarray
    truth_pixels: np.ndarray
    predicted_pixels: np.ndarray
    correct_pixels: int
    evaluated_pixels: int

    @classmethod
    def zeros(cls, class_count: int) -> "_Counts":
        return cls(
            np.zeros(class_count, dtype=np.int64),
            np.zeros(class_count, dtype=np.int64),
            np.zeros(class_count, dtype=np.int64),
            0,
            0,
        )

    def __add__(self, other: "_Counts") -> "_Counts":
        return _Counts(
            self.true_positive + other.true_positive,
            self.truth_pixels + other.truth_pixels,
            self.predicted_pixels + other.predicted_pixels,
            self.correct_pixels + other.correct_pixels,
            self.evaluated_pixels + other.evaluated_pixels,
        )


def _validate_labels(
    values: np.ndarray,
    *,
    name: str,
    class_ids: tuple[int, ...],
    ignore_labels: tuple[int, ...],
) -> None:
    allowed = set(class_ids) | set(ignore_labels)
    supplied = set(int(item) for item in np.unique(values))
    unknown = sorted(supplied - allowed)
    if unknown:
        raise ValueError(f"{name} contains labels outside the frozen taxonomy: {unknown}")


def _case_counts(
    case: SegmentationCase,
    class_ids: tuple[int, ...],
    ignore_labels: tuple[int, ...],
) -> _Counts:
    if case.prediction is None:
        return _Counts.zeros(len(class_ids))
    truth = np.asarray(case.ground_truth)
    prediction = np.asarray(case.prediction)
    _validate_labels(truth, name="ground_truth", class_ids=class_ids, ignore_labels=ignore_labels)
    _validate_labels(prediction, name="prediction", class_ids=class_ids, ignore_labels=ignore_labels)
    evaluated = ~np.isin(truth, ignore_labels)
    true_positive = np.asarray(
        [np.count_nonzero(evaluated & (truth == item) & (prediction == item)) for item in class_ids],
        dtype=np.int64,
    )
    truth_pixels = np.asarray(
        [np.count_nonzero(evaluated & (truth == item)) for item in class_ids],
        dtype=np.int64,
    )
    predicted_pixels = np.asarray(
        [np.count_nonzero(evaluated & (prediction == item)) for item in class_ids],
        dtype=np.int64,
    )
    return _Counts(
        true_positive,
        truth_pixels,
        predicted_pixels,
        int(np.count_nonzero(evaluated & (truth == prediction))),
        int(np.count_nonzero(evaluated)),
    )


def _metrics(counts: _Counts, class_ids: tuple[int, ...]) -> dict[str, object]:
    union = counts.truth_pixels + counts.predicted_pixels - counts.true_positive
    iou = np.full(len(class_ids), np.nan, dtype=np.float64)
    present = union > 0
    iou[present] = counts.true_positive[present] / union[present]
    per_class = {
        str(class_id): {
            "name": MSRS_CLASSES.get(class_id, f"class_{class_id}"),
            "iou": float(iou[index]) if present[index] else None,
            "intersection_pixels": int(counts.true_positive[index]),
            "union_pixels": int(union[index]),
            "ground_truth_pixels": int(counts.truth_pixels[index]),
            "predicted_pixels": int(counts.predicted_pixels[index]),
        }
        for index, class_id in enumerate(class_ids)
    }
    return {
        "mean_iou": float(np.mean(iou[present])) if np.any(present) else None,
        "per_class": per_class,
        "evaluated_class_count": int(np.count_nonzero(present)),
        "pixel_accuracy": (
            counts.correct_pixels / counts.evaluated_pixels
            if counts.evaluated_pixels
            else None
        ),
        "evaluated_pixels": counts.evaluated_pixels,
    }


def _sum_counts(values: Sequence[_Counts], class_count: int) -> _Counts:
    result = _Counts.zeros(class_count)
    for value in values:
        result = result + value
    return result


def _percentile_interval(values: Sequence[float], confidence: float) -> dict[str, float] | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None
    alpha = (1.0 - confidence) / 2.0
    return {
        "lower": float(np.quantile(finite, alpha)),
        "upper": float(np.quantile(finite, 1.0 - alpha)),
    }


def scene_group_bootstrap(
    group_counts: Mapping[str, _Counts],
    *,
    class_ids: Sequence[int] = MSRS_EVALUATED_CLASS_IDS,
    replicates: int = 2_000,
    seed: int = 20260902,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Percentile bootstrap with complete scene groups as sampling units."""

    classes = tuple(int(item) for item in class_ids)
    if not group_counts:
        raise ValueError("at least one scene group is required for bootstrap")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    groups = tuple(sorted(group_counts))
    values = tuple(group_counts[group] for group in groups)
    rng = np.random.default_rng(seed)
    mean_iou: list[float] = []
    class_iou: list[list[float]] = [[] for _ in classes]
    for _ in range(replicates):
        indexes = rng.integers(0, len(groups), size=len(groups))
        counts = _sum_counts([values[int(index)] for index in indexes], len(classes))
        metric = _metrics(counts, classes)
        if metric["mean_iou"] is not None:
            mean_iou.append(float(metric["mean_iou"]))
        for index, class_id in enumerate(classes):
            value = metric["per_class"][str(class_id)]["iou"]
            if value is not None:
                class_iou[index].append(float(value))
    return {
        "method": "percentile_scene_group_bootstrap",
        "confidence_level": confidence,
        "requested_replicates": replicates,
        "effective_mean_iou_replicates": len(mean_iou),
        "seed": int(seed),
        "scene_group_count": len(groups),
        "mean_iou": _percentile_interval(mean_iou, confidence),
        "per_class_iou": {
            str(class_id): _percentile_interval(class_iou[index], confidence)
            for index, class_id in enumerate(classes)
        },
    }


def summarize_latency_and_failures(cases: Sequence[SegmentationCase]) -> dict[str, object]:
    attempted = len(cases)
    failures = [case for case in cases if case.prediction is None]
    successful = [case for case in cases if case.prediction is not None]
    failure_counts: dict[str, int] = {}
    for case in failures:
        assert case.failure_reason is not None
        failure_counts[case.failure_reason] = failure_counts.get(case.failure_reason, 0) + 1

    def latency_summary(selected: Sequence[SegmentationCase]) -> dict[str, object]:
        latency = np.asarray(
            [case.latency_ms for case in selected if case.latency_ms is not None],
            dtype=np.float64,
        )
        summary: dict[str, object] = {
            "recorded": int(latency.size),
            "missing": len(selected) - int(latency.size),
        }
        if latency.size:
            summary.update(
                {
                    "mean": float(np.mean(latency)),
                    "standard_deviation": float(np.std(latency)),
                    "minimum": float(np.min(latency)),
                    "p50": float(np.quantile(latency, 0.50)),
                    "p90": float(np.quantile(latency, 0.90)),
                    "p95": float(np.quantile(latency, 0.95)),
                    "p99": float(np.quantile(latency, 0.99)),
                    "maximum": float(np.max(latency)),
                }
            )
        return summary

    return {
        "attempted": attempted,
        "successful": len(successful),
        "failed": len(failures),
        "failure_rate": len(failures) / attempted if attempted else None,
        "failures_by_reason": dict(sorted(failure_counts.items())),
        "latency_ms_all_attempts": latency_summary(cases),
        "latency_ms_successful": latency_summary(successful),
    }


def _validated_cases(
    cases: Sequence[SegmentationCase],
    class_ids: tuple[int, ...],
    ignore_labels: tuple[int, ...],
) -> tuple[dict[str, _Counts], dict[str, _Counts]]:
    identifiers: set[str] = set()
    per_sample: dict[str, _Counts] = {}
    per_group: dict[str, _Counts] = {}
    for case in cases:
        if case.sample_id in identifiers:
            raise ValueError(f"duplicate sample_id: {case.sample_id}")
        identifiers.add(case.sample_id)
        counts = _case_counts(case, class_ids, ignore_labels)
        per_sample[case.sample_id] = counts
        per_group[case.scene_group] = per_group.get(
            case.scene_group, _Counts.zeros(len(class_ids))
        ) + counts
    return per_sample, per_group


def _check_test_lock(
    partition: str,
    unlock_final_test: bool,
    provenance: FrozenModelProvenance,
) -> None:
    if partition not in {"train", "validation", "test"}:
        raise ValueError("partition must be train, validation, or test")
    if partition == "test" and not unlock_final_test:
        raise ValueError(
            "final test is locked; freeze model, preprocessing, evaluator, and baselines before unlocking"
        )
    if partition == "test" and (not provenance.frozen or provenance.artifact_sha256 is None):
        raise ValueError(
            "final test requires frozen model provenance with an artifact SHA-256"
        )


def evaluate_msrs_semantics(
    cases: Sequence[SegmentationCase],
    provenance: FrozenModelProvenance,
    *,
    partition: str = "validation",
    unlock_final_test: bool = False,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 20260902,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate MSRS masks under the frozen taxonomy and test-lock policy."""

    _check_test_lock(partition, unlock_final_test, provenance)
    selected = tuple(cases)
    if not selected:
        raise ValueError("evaluation requires at least one prediction attempt")
    class_ids = MSRS_EVALUATED_CLASS_IDS
    ignore_labels = DEFAULT_IGNORE_LABELS
    per_sample, per_group = _validated_cases(selected, class_ids, ignore_labels)
    aggregate = _sum_counts(list(per_sample.values()), len(class_ids))
    metrics = _metrics(aggregate, class_ids)
    intervals = scene_group_bootstrap(
        per_group,
        class_ids=class_ids,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    group_metrics = {
        group: {
            "attempted_samples": sum(case.scene_group == group for case in selected),
            "successful_samples": sum(
                case.scene_group == group and case.prediction is not None for case in selected
            ),
            "metrics": _metrics(counts, class_ids),
        }
        for group, counts in sorted(per_group.items())
    }
    report: dict[str, object] = {
        "schema_version": EVALUATOR_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": {
            "id": EVALUATOR_ID,
            "implementation_sha256": _sha256(Path(__file__)),
            "dataset": "MSRS",
            "taxonomy": {str(key): value for key, value in MSRS_CLASSES.items()},
            "evaluated_class_ids": list(class_ids),
            "ignore_ground_truth_labels": list(ignore_labels),
            "absent_class_policy": "exclude class when aggregate union is zero",
            "failed_prediction_policy": "exclude from IoU; report separately without silent substitution",
        },
        "partition": partition,
        "final_test_unlocked": partition == "test" and unlock_final_test,
        "scientific_status": (
            "locked_final_test_result_requires_human_review"
            if partition == "test"
            else "development_or_validation_result_not_final_test"
        ),
        "model_provenance": provenance.as_dict(),
        "metrics": metrics,
        "confidence_intervals_95": intervals,
        "scene_groups": group_metrics,
        "runtime_and_failures": summarize_latency_and_failures(selected),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "limitations": [
            "IoU is computed only where MSRS ground truth is not ignored.",
            "Failed predictions are excluded from IoU and must be interpreted with the reported failure rate.",
            "Bootstrap validity depends on scene_group identifying genuinely independent capture groups.",
            "Recorded provenance does not establish that a pretrained model is unbiased for RGB-thermal imagery.",
        ],
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def paired_scene_group_comparison(
    baseline_cases: Sequence[SegmentationCase],
    candidate_cases: Sequence[SegmentationCase],
    *,
    replicates: int = 2_000,
    seed: int = 20260902,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Paired candidate-minus-baseline IoU comparison on identical samples."""

    baseline = {case.sample_id: case for case in baseline_cases}
    candidate = {case.sample_id: case for case in candidate_cases}
    if len(baseline) != len(baseline_cases) or len(candidate) != len(candidate_cases):
        raise ValueError("paired inputs contain duplicate sample identifiers")
    if set(baseline) != set(candidate):
        raise ValueError("paired comparison requires identical sample identifiers")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap settings are invalid")

    class_ids = MSRS_EVALUATED_CLASS_IDS
    ignore_labels = DEFAULT_IGNORE_LABELS
    complete: list[tuple[SegmentationCase, SegmentationCase]] = []
    for sample_id in sorted(baseline):
        left = baseline[sample_id]
        right = candidate[sample_id]
        if left.scene_group != right.scene_group:
            raise ValueError(f"scene_group mismatch for paired sample {sample_id}")
        if not np.array_equal(left.ground_truth, right.ground_truth):
            raise ValueError(f"ground_truth mismatch for paired sample {sample_id}")
        if left.prediction is not None and right.prediction is not None:
            complete.append((left, right))
    if not complete:
        raise ValueError("paired comparison has no samples completed by both methods")

    grouped: dict[str, tuple[_Counts, _Counts]] = {}
    for left, right in complete:
        left_counts = _case_counts(left, class_ids, ignore_labels)
        right_counts = _case_counts(right, class_ids, ignore_labels)
        old_left, old_right = grouped.get(
            left.scene_group,
            (_Counts.zeros(len(class_ids)), _Counts.zeros(len(class_ids))),
        )
        grouped[left.scene_group] = (old_left + left_counts, old_right + right_counts)

    all_left = _sum_counts([item[0] for item in grouped.values()], len(class_ids))
    all_right = _sum_counts([item[1] for item in grouped.values()], len(class_ids))
    baseline_metric = _metrics(all_left, class_ids)
    candidate_metric = _metrics(all_right, class_ids)
    observed_mean_delta = float(candidate_metric["mean_iou"] - baseline_metric["mean_iou"])
    observed_class_delta: dict[str, float | None] = {}
    for class_id in class_ids:
        left = baseline_metric["per_class"][str(class_id)]["iou"]
        right = candidate_metric["per_class"][str(class_id)]["iou"]
        observed_class_delta[str(class_id)] = (
            float(right - left) if left is not None and right is not None else None
        )

    names = tuple(sorted(grouped))
    rng = np.random.default_rng(seed)
    mean_deltas: list[float] = []
    class_deltas: dict[str, list[float]] = {str(item): [] for item in class_ids}
    for _ in range(replicates):
        chosen = rng.integers(0, len(names), size=len(names))
        left_counts = _sum_counts([grouped[names[int(i)]][0] for i in chosen], len(class_ids))
        right_counts = _sum_counts([grouped[names[int(i)]][1] for i in chosen], len(class_ids))
        left_metric = _metrics(left_counts, class_ids)
        right_metric = _metrics(right_counts, class_ids)
        if left_metric["mean_iou"] is not None and right_metric["mean_iou"] is not None:
            mean_deltas.append(float(right_metric["mean_iou"] - left_metric["mean_iou"]))
        for class_id in class_ids:
            left_iou = left_metric["per_class"][str(class_id)]["iou"]
            right_iou = right_metric["per_class"][str(class_id)]["iou"]
            if left_iou is not None and right_iou is not None:
                class_deltas[str(class_id)].append(float(right_iou - left_iou))

    return {
        "method": "paired_scene_group_percentile_bootstrap",
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
            "delta": observed_mean_delta,
            "delta_interval": _percentile_interval(mean_deltas, confidence),
            "bootstrap_probability_candidate_better": (
                float(np.mean(np.asarray(mean_deltas) > 0.0)) if mean_deltas else None
            ),
        },
        "per_class_iou_delta": {
            str(class_id): {
                "delta": observed_class_delta[str(class_id)],
                "delta_interval": _percentile_interval(class_deltas[str(class_id)], confidence),
            }
            for class_id in class_ids
        },
        "pairing_policy": "only samples completed by both methods enter paired IoU; attrition is explicit",
    }


def load_prediction_cases(
    ground_truth_dir: str | Path,
    prediction_dir: str | Path,
    scene_groups: Mapping[str, str],
    *,
    latency_ms: Mapping[str, float] | None = None,
    failures: Mapping[str, str] | None = None,
) -> tuple[SegmentationCase, ...]:
    """Load precomputed masks; missing predictions become explicit failures."""

    truth_root = Path(ground_truth_dir)
    prediction_root = Path(prediction_dir)
    latency = dict(latency_ms or {})
    declared_failures = dict(failures or {})
    image_suffixes = {".png", ".bmp", ".tif", ".tiff"}
    truth_paths = sorted(
        path
        for path in truth_root.iterdir()
        if path.is_file() and path.suffix.lower() in image_suffixes
    )
    if not truth_paths:
        raise ValueError(f"no ground-truth masks found in {truth_root}")
    prediction_paths = (
        sorted(
            path
            for path in prediction_root.iterdir()
            if path.is_file() and path.suffix.lower() in image_suffixes
        )
        if prediction_root.is_dir()
        else []
    )
    if len({path.stem for path in prediction_paths}) != len(prediction_paths):
        raise ValueError("prediction directory contains ambiguous duplicate stems")
    predictions = {path.stem: path for path in prediction_paths}
    cases: list[SegmentationCase] = []
    for truth_path in truth_paths:
        sample_id = truth_path.stem
        if sample_id not in scene_groups:
            raise ValueError(f"scene-group manifest has no entry for {sample_id}")
        with Image.open(truth_path) as image:
            truth = np.asarray(image)
        prediction_path = predictions.get(sample_id)
        failure_reason = declared_failures.get(sample_id)
        prediction = None
        if failure_reason is None and prediction_path is not None:
            try:
                with Image.open(prediction_path) as image:
                    prediction = np.asarray(image)
            except Exception as error:  # Pillow format/decode errors are recorded, not hidden.
                failure_reason = f"prediction_decode_error:{type(error).__name__}"
        elif failure_reason is None:
            failure_reason = "missing_prediction"
        cases.append(
            SegmentationCase(
                sample_id,
                scene_groups[sample_id],
                truth,
                prediction,
                latency_ms=latency.get(sample_id),
                failure_reason=failure_reason,
            )
        )
    return tuple(cases)


def _load_json_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--scene-groups", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    provenance = FrozenModelProvenance(**_load_json_mapping(args.provenance))
    scene_groups = {
        str(key): str(value) for key, value in _load_json_mapping(args.scene_groups).items()
    }
    cases = load_prediction_cases(
        args.ground_truth_dir, args.prediction_dir, scene_groups
    )
    report = evaluate_msrs_semantics(
        cases,
        provenance,
        partition=args.partition,
        unlock_final_test=args.unlock_final_test,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        output_path=args.output,
    )
    print(json.dumps({
        "partition": report["partition"],
        "mean_iou": report["metrics"]["mean_iou"],
        "attempted": report["runtime_and_failures"]["attempted"],
        "failure_rate": report["runtime_and_failures"]["failure_rate"],
    }, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_IGNORE_LABELS",
    "EVALUATOR_ID",
    "FrozenModelProvenance",
    "MSRS_EVALUATED_CLASS_IDS",
    "SegmentationCase",
    "evaluate_msrs_semantics",
    "load_prediction_cases",
    "paired_scene_group_comparison",
    "scene_group_bootstrap",
    "summarize_latency_and_failures",
]
