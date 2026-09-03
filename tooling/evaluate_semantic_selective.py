"""Locked semantic mIoU and semantic-error selective evaluation for PRISM."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Iterable, Sequence

import numpy as np
import torch

from openprism.datasets import DatasetCatalog
from openprism.learning.caltech_segmentation_evaluation import evaluate_caltech_semantics
from openprism.learning.caltech_segmentation_probe import (
    _declared_aligned as caltech_aligned,
    load_caltech_probe_checkpoint,
)
from openprism.learning.data import item_scene_group, protocol_items, protocol_manifest
from openprism.learning.engine import LearnedFusionEngine
from openprism.learning.evaluation import (
    grouped_selective_bootstrap,
    selective_diagnostic_curves,
    selective_metrics,
)
from openprism.learning.segmentation_evaluation import (
    FrozenModelProvenance,
    SegmentationCase,
    evaluate_msrs_semantics,
)
from openprism.learning.segmentation_probe import (
    _declared_aligned_replay as msrs_aligned,
    _luminance,
    _normalize_thermal,
    _predict_mask,
    load_probe_checkpoint,
)


SCHEMA = "openprism.semantic-selective-evaluation/1.0"
PROVIDED_VIEWS = ("prism_egt_operator", "prism_egt_luminance")
AUTOMATIC_VIEWS = (
    "prism_egt_operator_automatic", "prism_egt_luminance_automatic"
)
VIEWS = PROVIDED_VIEWS + AUTOMATIC_VIEWS
SPATIAL_STRIDE = 8
TIE_POLICY = (
    "threshold-tied: every equal-score pixel enters together; AURC uses "
    "right-continuous block-end risk weighted by the block coverage increment"
)
RANKINGS = (
    "prism_predictive_uncertainty",
    "learned_abstention",
    "evidence_insufficiency",
    "rgb_thermal_luminance_disagreement",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _string_manifest_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def semantic_score_metrics(
    score: np.ndarray, truth: np.ndarray, prediction: np.ndarray, *, ignore=(0, 255)
) -> dict[str, float | None]:
    """Pair the PRISM score with the frozen probe's binary semantic error."""

    score = np.asarray(score, dtype=np.float32)
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    if score.shape != truth.shape or prediction.shape != truth.shape:
        raise ValueError("score, truth, and prediction geometry must match")
    valid = ~np.isin(truth, ignore)
    if not np.any(valid):
        raise ValueError("semantic mask contains no evaluated pixels")
    error = (prediction != truth).astype(np.float32)
    return selective_metrics(
        torch.from_numpy(score[valid]), torch.from_numpy(error[valid])
    )


def evaluate(
    dataset: str,
    probe_checkpoint: Path,
    prism_checkpoint: Path,
    data_root: Path,
    output: Path,
    *,
    device_name: str = "auto",
    views: Sequence[str] = VIEWS,
    bootstrap_seed: int = 20260903,
    bootstrap_replicates: int = 2_000,
    partition: str = "validation",
    unlock_final_test: bool = False,
    spatial_stride: int = SPATIAL_STRIDE,
) -> dict[str, object]:
    evaluation_started = time.perf_counter()
    if dataset not in {"msrs", "caltech"}:
        raise ValueError("dataset must be msrs or caltech")
    requested = tuple(views)
    allowed = (set(PROVIDED_VIEWS), set(VIEWS))
    if set(requested) not in allowed or len(requested) != len(set(requested)):
        raise ValueError(
            "semantic evaluation requires both provided-task PRISM views and, "
            "outside reduced ablations, both automatic-task PRISM views"
        )
    if partition not in {"validation", "test"}:
        raise ValueError("semantic selective partition must be validation or test")
    if partition == "test" and not unlock_final_test:
        raise ValueError("final test is locked; the one-shot controller must explicitly unlock it")
    if spatial_stride != SPATIAL_STRIDE:
        raise ValueError(f"semantic selective spatial_stride is frozen at {SPATIAL_STRIDE}")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else "cpu" if device_name == "auto" else device_name
    )
    if dataset == "caltech":
        probe, metadata = load_caltech_probe_checkpoint(probe_checkpoint, device=device)
        align = caltech_aligned
        ignore = (0, 255)
    else:
        probe, metadata = load_probe_checkpoint(probe_checkpoint, device=device)
        align = msrs_aligned
        ignore = (0, 255)
    if metadata.development_subset:
        raise ValueError("ablation evaluation refuses a development probe checkpoint")
    engine = LearnedFusionEngine.from_checkpoint(str(prism_checkpoint), device=device)
    if engine.checkpoint is None:
        raise ValueError("PRISM-EGT checkpoint provenance is required")
    catalog = DatasetCatalog(data_root)
    items = [
        item for item in protocol_items(data_root, partition, include_detection_subset=False)
        if item.record.dataset == dataset and item.record.annotation_path is not None
    ]
    if not items:
        raise ValueError(f"{partition} partition has no labeled semantic samples")
    indexes = {
        (split, catalog.record(dataset, split, index).sample_id): index
        for split in (("all",) if dataset == "caltech" else ("train", "test"))
        for index in range(catalog.count(dataset, split))
    }
    cases = {view: [] for view in requested}
    scores = {
        view: {ranking: {} for ranking in RANKINGS} for view in requested
    }
    errors = {view: {} for view in requested}
    task = "terrain" if dataset == "caltech" else "navigate"
    for item in items:
        record = item.record
        frame = align(catalog.load(dataset, record.split, indexes[(record.split, record.sample_id)]))
        truth = np.asarray(frame.semantic_mask)
        visible_luminance = _luminance(frame.observations["visible"].data)
        thermal_luminance = _normalize_thermal(frame.observations["thermal"].data)
        modality_disagreement = np.abs(
            visible_luminance - thermal_luminance
        ).astype(np.float32)
        group = item_scene_group(item)
        started = time.perf_counter()
        try:
            fused_by_mode = {
                "provided": engine.fuse(frame, task=task),
            }
            if any(view.endswith("_automatic") for view in requested):
                fused_by_mode["automatic"] = engine.fuse(frame, task="automatic")
            images = {}
            ranking_scores = {}
            for mode, fused in fused_by_mode.items():
                if not fused.provenance.get("learned_fusion_applied", False):
                    raise RuntimeError(f"PRISM-EGT {mode} task did not pass the evidence gate")
                uncertainty = np.asarray(fused.machine_tensor[
                    fused.channel_names.index("learned_predictive_uncertainty")
                ], dtype=np.float32)
                evidence_support = np.asarray(fused.machine_tensor[
                    fused.channel_names.index("evidence_support")
                ], dtype=np.float32)
                learned_abstention = np.asarray(fused.machine_tensor[
                    fused.channel_names.index("learned_abstention_probability")
                ], dtype=np.float32)
                luminance = np.asarray(fused.machine_tensor[
                    fused.channel_names.index("learned_fused_luminance")
                ], dtype=np.float32)
                suffix = "_automatic" if mode == "automatic" else ""
                images[f"prism_egt_operator{suffix}"] = fused.operator_rgb
                images[f"prism_egt_luminance{suffix}"] = np.repeat(
                    luminance[..., None], 3, axis=2
                )
                ranking_scores[mode] = {
                    "prism_predictive_uncertainty": uncertainty,
                    "learned_abstention": learned_abstention,
                    "evidence_insufficiency": 1.0 - evidence_support,
                    "rgb_thermal_luminance_disagreement": modality_disagreement,
                }
        except Exception as error:
            elapsed = (time.perf_counter() - started) * 1_000.0
            for view in requested:
                cases[view].append(SegmentationCase(
                    record.sample_id, group, truth, None, latency_ms=elapsed,
                    failure_reason=f"{type(error).__name__}:{error}",
                ))
            continue
        fusion_seconds = time.perf_counter() - started
        valid = ~np.isin(truth, ignore)
        for view in requested:
            view_started = time.perf_counter()
            try:
                predicted = _predict_mask(probe, images[view], device)
                mode = "automatic" if view.endswith("_automatic") else "provided"
                sampled_valid = valid[::spatial_stride, ::spatial_stride]
                sampled_error = (predicted != truth)[::spatial_stride, ::spatial_stride]
                if not np.any(sampled_valid):
                    raise ValueError("semantic mask has no evaluated pixels at the frozen stride")
                for ranking in RANKINGS:
                    sampled_score = ranking_scores[mode][ranking][
                        ::spatial_stride, ::spatial_stride
                    ]
                    scores[view][ranking].setdefault(group, []).append(
                        sampled_score[sampled_valid]
                    )
                errors[view].setdefault(group, []).append(
                    sampled_error[sampled_valid].astype(np.float32)
                )
                cases[view].append(SegmentationCase(
                    record.sample_id, group, truth, predicted,
                    latency_ms=(fusion_seconds + time.perf_counter() - view_started) * 1_000.0,
                ))
            except Exception as error:
                cases[view].append(SegmentationCase(
                    record.sample_id, group, truth, None,
                    latency_ms=(fusion_seconds + time.perf_counter() - view_started) * 1_000.0,
                    failure_reason=f"{type(error).__name__}:{error}",
                ))
    provenance = FrozenModelProvenance(
        metadata.model_id, metadata.artifact_sha256,
        f"OpenPRISM frozen {dataset} visible-only probe", metadata.training_run_id,
        f"square-bilinear-{metadata.config.image_size}", False, True,
        f"{dataset} protocol train visible RGB only",
    )
    results = {}
    for view in requested:
        semantic = (
            evaluate_caltech_semantics if dataset == "caltech" else evaluate_msrs_semantics
        )(
            cases[view], provenance, partition=partition,
            unlock_final_test=unlock_final_test,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        if semantic["runtime_and_failures"]["failed"]:
            raise RuntimeError(f"{dataset} {view} contains failed {partition} samples")
        rankings = {}
        for ranking in RANKINGS:
            group_metrics = {
                group: selective_metrics(
                    torch.from_numpy(np.concatenate(scores[view][ranking][group])),
                    torch.from_numpy(np.concatenate(errors[view][group])),
                )
                for group in sorted(scores[view][ranking])
            }
            all_scores = np.concatenate([
                values for group in sorted(scores[view][ranking])
                for values in scores[view][ranking][group]
            ])
            all_errors = np.concatenate([
                values for group in sorted(errors[view])
                for values in errors[view][group]
            ])
            rankings[ranking] = {
                "semantic_selective_metrics_by_capture_group": group_metrics,
                "semantic_selective_confidence_intervals_95": grouped_selective_bootstrap(
                    group_metrics, replicates=bootstrap_replicates, seed=bootstrap_seed
                ),
                "semantic_selective_diagnostics": selective_diagnostic_curves(
                    torch.from_numpy(all_scores), torch.from_numpy(all_errors)
                ),
            }
        results[view] = {
            "semantic_evaluation": semantic,
            "semantic_selective_rankings": rankings,
            "semantic_selective_runtime_and_failures": {
                "attempted": len(cases[view]),
                "successful": sum(case.prediction is not None for case in cases[view]),
                "failed": sum(case.prediction is None for case in cases[view]),
                "evaluated_score_pixels": sum(
                    array.size
                    for grouped_arrays in scores[view][RANKINGS[0]].values()
                    for array in grouped_arrays
                ),
            },
        }
    protocol = protocol_manifest(data_root)
    sample_ids = [
        f"{item.record.dataset}:{item.record.split}:{item.record.sample_id}"
        for item in items
    ]
    report = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "final_test_unlocked_requires_human_review"
            if partition == "test" else "protocol_validation"
        ),
        "partition": partition,
        "final_test_unlocked": partition == "test" and unlock_final_test,
        "test_lock": {
            "state": "explicitly_unlocked" if partition == "test" else "not_accessed",
            "complete_partition_required": True,
            "partial_evaluation_supported": False,
        },
        "dataset": dataset,
        "task": task,
        "view_task_modes": {
            view: ("automatic" if view.endswith("_automatic") else "provided")
            for view in requested
        },
        "score_target": "binary semantic error from the frozen visible-only probe",
        "ranking_scores": {
            "prism_predictive_uncertainty": "learned_predictive_uncertainty channel",
            "learned_abstention": "learned_abstention_probability channel",
            "evidence_insufficiency": "1 - evidence_support channel",
            "rgb_thermal_luminance_disagreement": "absolute normalized visible-thermal luminance difference",
        },
        "tie_policy": TIE_POLICY,
        "score_sampling": {
            "spatial_stride": spatial_stride,
            "origin": [0, 0],
            "policy": "deterministic_regular_grid_then_ignore_masking",
        },
        "views": list(requested),
        "sample_count": len(items),
        "sample_manifest": {
            "count": len(sample_ids),
            "sha256": _string_manifest_sha256(sample_ids),
            "max_samples": None,
        },
        "source_hashes": {
            "data_protocol_sha256": _json_sha256(protocol),
            "evaluator_implementation_sha256": _sha256(Path(__file__)),
            "probe_checkpoint_sha256": _sha256(probe_checkpoint),
            "prism_egt_checkpoint_sha256": _sha256(prism_checkpoint),
        },
        "probe_checkpoint": {
            "path": str(probe_checkpoint.resolve()),
            "artifact_sha256": metadata.artifact_sha256,
        },
        "prism_egt_checkpoint": {
            "path": str(Path(engine.checkpoint.path).resolve()),
            "artifact_sha256": engine.checkpoint.artifact_sha256,
        },
        "bootstrap": {
            "seed": bootstrap_seed, "replicates": bootstrap_replicates,
            "unit": "complete capture group",
        },
        "evaluation_config": {
            "dataset": dataset,
            "task": task,
            "partition": partition,
            "views": list(requested),
            "spatial_stride": spatial_stride,
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "partial_evaluation_supported": False,
        },
        "runtime": {
            "elapsed_seconds": time.perf_counter() - evaluation_started,
            "device": str(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("msrs", "caltech"), required=True)
    parser.add_argument("--probe-checkpoint", type=Path, required=True)
    parser.add_argument("--prism-egt-checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bootstrap-seed", type=int, default=20260903)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--spatial-stride", type=int, default=SPATIAL_STRIDE)
    parser.add_argument("--view", action="append", choices=VIEWS)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = evaluate(
        args.dataset, args.probe_checkpoint, args.prism_egt_checkpoint,
        args.data_root, args.output, device_name=args.device,
        views=tuple(args.view or VIEWS), bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates, partition=args.partition,
        unlock_final_test=args.unlock_final_test, spatial_stride=args.spatial_stride,
    )
    print(json.dumps({"dataset": report["dataset"], "sample_count": report["sample_count"]}))


if __name__ == "__main__":
    main()


__all__ = [
    "AUTOMATIC_VIEWS", "PROVIDED_VIEWS", "SPATIAL_STRIDE", "TIE_POLICY",
    "RANKINGS", "VIEWS", "evaluate", "semantic_score_metrics",
]
