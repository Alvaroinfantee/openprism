"""Evaluate PRISM-EGT invariants and robustness without leaking the test set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .baselines import BASELINE_NAMES, fuse_baseline
from .data import CORRUPTION_MODES, FusionPatchDataset, protocol_manifest
from .objective import EGTCFLoss
from .objective import fusion_proxy_metrics, proxy_targets
from .training import seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _threshold_aurc(score: Tensor, risk: Tensor) -> float:
    """Area for threshold selection without splitting equal-score pixels.

    At each distinct score, the complete tie block enters together. The area
    is the right-continuous conditional risk weighted by that block's coverage
    increment. With unique scores this reduces to the conventional mean of
    cumulative risks over ranks.
    """

    order = torch.argsort(score, descending=False, stable=True)
    ordered_score = score[order]
    ordered_risk = risk[order]
    _, counts = torch.unique_consecutive(ordered_score, return_counts=True)
    endpoints = torch.cumsum(counts, dim=0) - 1
    cumulative_risk = torch.cumsum(ordered_risk, dim=0)
    retained = endpoints.to(torch.float32) + 1.0
    conditional_risk = cumulative_risk[endpoints] / retained
    coverage_increment = counts.to(torch.float32) / score.numel()
    return float(torch.sum(coverage_increment * conditional_risk))


def _threshold_augrc(score: Tensor, risk: Tensor) -> float:
    """Trapezoidal area under the generalized-risk/coverage curve.

    Generalized risk is the retained error mass divided by the original sample
    count, rather than conditional error among retained samples.  Complete
    equal-score blocks define the attainable threshold points and the origin is
    included, matching the estimator introduced by Traub et al. (NeurIPS 2024).
    """

    order = torch.argsort(score, descending=False, stable=True)
    ordered_score = score[order]
    ordered_risk = risk[order]
    _, counts = torch.unique_consecutive(ordered_score, return_counts=True)
    endpoints = torch.cumsum(counts, dim=0) - 1
    coverage = (endpoints.to(torch.float64) + 1.0) / score.numel()
    generalized_risk = torch.cumsum(
        ordered_risk.to(torch.float64), dim=0
    )[endpoints] / score.numel()
    coverage = torch.cat((torch.zeros(1, dtype=torch.float64), coverage))
    generalized_risk = torch.cat(
        (torch.zeros(1, dtype=torch.float64), generalized_risk)
    )
    return float(torch.trapezoid(generalized_risk, coverage))


def _threshold_working_point(
    score: Tensor, risk: Tensor, *, requested_coverage: float
) -> dict[str, float]:
    """Return the first complete score block meeting a requested coverage."""

    if not 0.0 < requested_coverage <= 1.0:
        raise ValueError("requested_coverage must be within (0, 1]")
    order = torch.argsort(score, descending=False, stable=True)
    ordered_score = score[order]
    ordered_risk = risk[order]
    unique_scores, counts = torch.unique_consecutive(
        ordered_score, return_counts=True
    )
    endpoints = torch.cumsum(counts, dim=0) - 1
    retained = endpoints.to(torch.float64) + 1.0
    coverage = retained / score.numel()
    index = min(
        int(
            torch.searchsorted(
                coverage,
                torch.tensor(requested_coverage, dtype=torch.float64),
                right=False,
            )
        ),
        coverage.numel() - 1,
    )
    cumulative_risk = torch.cumsum(ordered_risk.to(torch.float64), dim=0)
    return {
        "requested_coverage": requested_coverage,
        "realized_coverage": float(coverage[index]),
        "selective_risk": float(cumulative_risk[endpoints[index]] / retained[index]),
        "score_threshold": float(unique_scores[index]),
    }


def selective_metrics(
    uncertainty: Tensor,
    risk_target: Tensor,
    *,
    bins: int = 15,
) -> dict[str, float | None]:
    """Compute calibration and risk--coverage metrics without scikit-learn."""

    score = uncertainty.detach().float().flatten().cpu().clamp(0.0, 1.0)
    risk = risk_target.detach().float().flatten().cpu().clamp(0.0, 1.0)
    if score.numel() != risk.numel() or score.numel() == 0:
        raise ValueError("uncertainty and risk_target must have equal non-zero size")
    brier = float(torch.mean((score - risk).square()))
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (score >= lower) & (score < upper if index < bins - 1 else score <= upper)
        if bool(selected.any()):
            weight = float(selected.float().mean())
            ece += weight * abs(float(score[selected].mean() - risk[selected].mean()))

    aurc = _threshold_aurc(score, risk)
    oracle_aurc = _threshold_aurc(risk, risk)
    random_aurc = float(risk.mean())
    augrc = _threshold_augrc(score, risk)
    oracle_augrc = _threshold_augrc(risk, risk)
    random_augrc = 0.5 * float(risk.mean())
    risk_at_80 = _threshold_working_point(
        score, risk, requested_coverage=0.8
    )

    labels = risk >= 0.5
    positives = int(labels.sum())
    negatives = labels.numel() - positives
    auroc: float | None = None
    if positives and negatives:
        # Mann--Whitney rank statistic with average ranks for ties.
        sorted_score, sorted_indices = torch.sort(score)
        unique, counts = torch.unique_consecutive(sorted_score, return_counts=True)
        del unique
        starts = torch.cumsum(counts, dim=0) - counts
        average_ranks = starts.float() + (counts.float() + 1.0) / 2.0
        sorted_ranks = torch.repeat_interleave(average_ranks, counts)
        ranks = torch.empty_like(sorted_ranks)
        ranks[sorted_indices] = sorted_ranks
        positive_rank_sum = float(ranks[labels].sum())
        auroc = (
            positive_rank_sum - positives * (positives + 1) / 2.0
        ) / float(positives * negatives)
    return {
        "brier": brier,
        "expected_calibration_error": ece,
        "risk_coverage_area": aurc,
        "oracle_risk_coverage_area": oracle_aurc,
        "random_order_expected_risk_coverage_area": random_aurc,
        "excess_risk_coverage_area": aurc - oracle_aurc,
        "generalized_risk_coverage_area": augrc,
        "oracle_generalized_risk_coverage_area": oracle_augrc,
        "random_order_expected_generalized_risk_coverage_area": random_augrc,
        "excess_generalized_risk_coverage_area": augrc - oracle_augrc,
        "selective_risk_at_80_percent_requested_coverage": risk_at_80[
            "selective_risk"
        ],
        "realized_coverage_at_80_percent_requested_coverage": risk_at_80[
            "realized_coverage"
        ],
        "uncertainty_auroc": auroc,
        "samples": float(score.numel()),
        "mean_target_risk": float(risk.mean()),
        "mean_uncertainty": float(score.mean()),
    }


def selective_diagnostic_curves(
    uncertainty: Tensor,
    risk_target: Tensor,
    *,
    bins: int = 15,
    coverage_points: int = 20,
) -> dict[str, object]:
    """Return compact, plot-ready calibration and risk--coverage diagnostics.

    The output summarizes the exact sampled pixels used by
    :func:`selective_metrics`. Fixed grids keep reports small and prevent a
    later plotting step from silently changing the analysis.
    """

    if bins <= 0:
        raise ValueError("bins must be positive")
    if coverage_points <= 0:
        raise ValueError("coverage_points must be positive")
    score = uncertainty.detach().float().flatten().cpu().clamp(0.0, 1.0)
    risk = risk_target.detach().float().flatten().cpu().clamp(0.0, 1.0)
    if score.numel() != risk.numel() or score.numel() == 0:
        raise ValueError("uncertainty and risk_target must have equal non-zero size")

    reliability = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = (score >= lower) & (
            score < upper if index < bins - 1 else score <= upper
        )
        count = int(selected.sum())
        mean_score = float(score[selected].mean()) if count else None
        mean_risk = float(risk[selected].mean()) if count else None
        reliability.append(
            {
                "bin_index": index,
                "lower_inclusive": lower,
                "upper": upper,
                "upper_inclusive": index == bins - 1,
                "count": count,
                "fraction": count / score.numel(),
                "mean_score": mean_score,
                "mean_target_risk": mean_risk,
                "absolute_gap": (
                    abs(mean_score - mean_risk)
                    if mean_score is not None and mean_risk is not None
                    else None
                ),
            }
        )

    def threshold_points(
        ranking_score: Tensor, observed_risk: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        order = torch.argsort(ranking_score, descending=False, stable=True)
        ordered_score = ranking_score[order]
        ordered_risk = observed_risk[order]
        unique_scores, counts = torch.unique_consecutive(
            ordered_score, return_counts=True
        )
        endpoints = torch.cumsum(counts, dim=0) - 1
        retained = endpoints.to(torch.float32) + 1.0
        coverage = retained / ranking_score.numel()
        conditional = torch.cumsum(ordered_risk, dim=0)[endpoints] / retained
        return unique_scores, coverage, conditional

    thresholds, coverages, conditional_risks = threshold_points(score, risk)
    oracle_thresholds, oracle_coverages, oracle_risks = threshold_points(risk, risk)
    overall = float(risk.mean())
    curve = []
    for step in range(1, coverage_points + 1):
        requested = step / coverage_points
        point = min(
            int(torch.searchsorted(coverages, torch.tensor(requested), right=False)),
            coverages.numel() - 1,
        )
        oracle_point = min(
            int(torch.searchsorted(
                oracle_coverages, torch.tensor(requested), right=False
            )),
            oracle_coverages.numel() - 1,
        )
        realized = float(coverages[point])
        curve.append(
            {
                "requested_coverage": requested,
                "realized_coverage": realized,
                "retained_pixels": int(round(realized * score.numel())),
                "score_threshold": float(thresholds[point]),
                "conditional_risk": float(conditional_risks[point]),
                "generalized_risk": float(
                    coverages[point] * conditional_risks[point]
                ),
                "oracle_realized_coverage": float(oracle_coverages[oracle_point]),
                "oracle_risk_threshold": float(oracle_thresholds[oracle_point]),
                "oracle_conditional_risk": float(oracle_risks[oracle_point]),
                "oracle_generalized_risk": float(
                    oracle_coverages[oracle_point] * oracle_risks[oracle_point]
                ),
                "random_order_expected_risk": overall,
            }
        )
    return {
        "schema_version": "openprism.selective-diagnostics/1.0",
        "sampled_pixels": int(score.numel()),
        "reliability_bin_count": bins,
        "reliability_bins": reliability,
        "risk_coverage_grid_points": coverage_points,
        "risk_coverage_curve": curve,
        "risk_coverage_tie_policy": (
            "threshold-tied: every equal-score pixel enters together; AURC uses "
            "right-continuous block-end risk weighted by the block coverage increment"
        ),
        "notes": [
            "Scores and targets are clipped to [0,1].",
            "Reliability bins compare mean score with mean target risk and do not imply probabilistic calibration.",
            "AUGRC is trapezoid-integrated over generalized risk at complete equal-score blocks, including the origin.",
            "Random-order references are mean target risk for AURC and half that mean for AUGRC.",
        ],
    }


def collect_selective_metrics(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    maximum_samples: int = 20_000,
) -> dict[str, float | None]:
    uncertainties: list[Tensor] = []
    targets: list[Tensor] = []
    collected = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            evidence = batch["evidence"].to(device)
            task_ids = batch["task_id"].to(device)
            output = model(rgb, thermal, evidence, task_ids=task_ids)
            # Regular spatial subsampling keeps the metric bounded and
            # deterministic while covering every evaluated image.
            score = output.predictive_uncertainty[..., ::8, ::8].flatten().cpu()
            _, intensity_target = proxy_targets(
                rgb, thermal, output.evidence_support, task_ids
            )
            proxy_error = (output.fused_luminance - intensity_target).abs()
            target = torch.maximum(
                proxy_error, batch["corruption_target"].to(device)
            )[..., ::8, ::8].flatten().cpu()
            take = min(score.numel(), maximum_samples - collected)
            if take <= 0:
                break
            uncertainties.append(score[:take])
            targets.append(target[:take])
            collected += take
    if not uncertainties:
        raise ValueError("evaluation loader produced no examples")
    return selective_metrics(torch.cat(uncertainties), torch.cat(targets))


def _latency_summary(values: list[float]) -> dict[str, float | int | None]:
    latency = np.asarray(values, dtype=np.float64)
    if not latency.size:
        return {
            "recorded_batches": 0,
            "mean_ms_per_image": None,
            "p50_ms_per_image": None,
            "p95_ms_per_image": None,
            "p99_ms_per_image": None,
        }
    return {
        "recorded_batches": int(latency.size),
        "mean_ms_per_image": float(np.mean(latency)),
        "standard_deviation_ms_per_image": float(np.std(latency)),
        "minimum_ms_per_image": float(np.min(latency)),
        "p50_ms_per_image": float(np.quantile(latency, 0.50)),
        "p90_ms_per_image": float(np.quantile(latency, 0.90)),
        "p95_ms_per_image": float(np.quantile(latency, 0.95)),
        "p99_ms_per_image": float(np.quantile(latency, 0.99)),
        "maximum_ms_per_image": float(np.max(latency)),
    }


def grouped_selective_bootstrap(
    metrics_by_group: dict[str, dict[str, float | None]],
    *,
    replicates: int = 2_000,
    seed: int = 20260902,
    confidence: float = 0.95,
) -> dict[str, object]:
    """Bootstrap equal-weight capture-group selective metrics.

    The estimand is the mean metric for an independent capture group, not a
    pixel-weighted frame mean.  This prevents long video sequences from being
    treated as thousands of independent observations.
    """

    if not metrics_by_group:
        raise ValueError("at least one capture group is required")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be within (0, 1)")
    groups = tuple(sorted(metrics_by_group))
    names = (
        "brier",
        "expected_calibration_error",
        "risk_coverage_area",
        "oracle_risk_coverage_area",
        "random_order_expected_risk_coverage_area",
        "excess_risk_coverage_area",
        "generalized_risk_coverage_area",
        "oracle_generalized_risk_coverage_area",
        "random_order_expected_generalized_risk_coverage_area",
        "excess_generalized_risk_coverage_area",
        "selective_risk_at_80_percent_requested_coverage",
        "realized_coverage_at_80_percent_requested_coverage",
        "uncertainty_auroc",
        "mean_target_risk",
        "mean_uncertainty",
    )
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(replicates):
        selected = rng.integers(0, len(groups), size=len(groups))
        for name in names:
            values = [
                metrics_by_group[groups[int(index)]].get(name)
                for index in selected
            ]
            finite = np.asarray(
                [value for value in values if value is not None], dtype=np.float64
            )
            if finite.size:
                draws[name].append(float(np.mean(finite)))
    alpha = (1.0 - confidence) / 2.0
    intervals: dict[str, dict[str, float] | None] = {}
    point_estimates: dict[str, float | None] = {}
    for name in names:
        values = np.asarray(
            [
                metrics[name]
                for metrics in metrics_by_group.values()
                if metrics.get(name) is not None
            ],
            dtype=np.float64,
        )
        point_estimates[name] = float(np.mean(values)) if values.size else None
        samples = np.asarray(draws[name], dtype=np.float64)
        intervals[name] = (
            {
                "lower": float(np.quantile(samples, alpha)),
                "upper": float(np.quantile(samples, 1.0 - alpha)),
            }
            if samples.size
            else None
        )
    return {
        "method": "percentile_capture_group_bootstrap",
        "estimand": "equal-weight mean across pre-specified capture groups",
        "confidence_level": confidence,
        "requested_replicates": replicates,
        "seed": seed,
        "capture_group_count": len(groups),
        "point_estimates": point_estimates,
        "intervals": intervals,
    }


def evaluate_model_pass(
    model,
    loader: DataLoader,
    objective: EGTCFLoss,
    device: torch.device,
    *,
    task_mode: str = "provided",
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 20260902,
    spatial_stride: int = 8,
) -> dict[str, object]:
    """Evaluate losses, dense selective risk, latency, and failures once."""

    if task_mode not in {"provided", "automatic"}:
        raise ValueError("task_mode must be provided or automatic")
    if spatial_stride <= 0:
        raise ValueError("spatial_stride must be positive")
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    maximum_violation = 0.0
    latency_ms_per_image: list[float] = []
    failures: list[dict[str, object]] = []
    scores: list[Tensor] = []
    targets: list[Tensor] = []
    grouped_scores: dict[str, list[Tensor]] = {}
    grouped_targets: dict[str, list[Tensor]] = {}
    comparator_names = (
        "evidence_insufficiency",
        "visible_thermal_disagreement",
        "learned_abstention",
    )
    comparator_scores: dict[str, list[Tensor]] = {
        name: [] for name in comparator_names
    }
    grouped_comparator_scores: dict[str, dict[str, list[Tensor]]] = {
        name: {} for name in comparator_names
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for batch_index, raw_batch in enumerate(loader):
            rgb = raw_batch["rgb"].to(device, non_blocking=True)
            thermal = raw_batch["thermal"].to(device, non_blocking=True)
            evidence = raw_batch["evidence"].to(device, non_blocking=True)
            task_ids = raw_batch["task_id"].to(device, non_blocking=True)
            corruption_target = raw_batch["corruption_target"].to(
                device, non_blocking=True
            )
            batch_size = int(rgb.shape[0])
            try:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                output = model(
                    rgb,
                    thermal,
                    evidence,
                    task_ids=(task_ids if task_mode == "provided" else None),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = (time.perf_counter() - started) * 1_000.0 / batch_size
                latency_ms_per_image.append(elapsed)
                _, components = objective(
                    output,
                    rgb,
                    thermal,
                    task_ids,
                    corruption_target=corruption_target,
                )
                _, intensity_target = proxy_targets(
                    rgb, thermal, output.evidence_support, task_ids
                )
                target = torch.maximum(
                    (output.fused_luminance - intensity_target).abs(),
                    corruption_target,
                )
            except Exception as error:
                failures.append(
                    {
                        "batch_index": batch_index,
                        "sample_ids": [str(value) for value in raw_batch["sample_id"]],
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue

            examples += batch_size
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.cpu()) * batch_size
            maximum_violation = max(
                maximum_violation,
                float(
                    torch.relu(
                        output.thermal_contribution - output.evidence_support
                    ).max().cpu()
                ),
            )
            predicted = output.task_logits.argmax(dim=1)
            totals["task_accuracy"] = totals.get("task_accuracy", 0.0) + float(
                (predicted == task_ids).float().sum().cpu()
            )
            sampled_score = output.predictive_uncertainty[
                ..., ::spatial_stride, ::spatial_stride
            ].detach().cpu()
            sampled_target = target[
                ..., ::spatial_stride, ::spatial_stride
            ].detach().cpu()
            visible_luminance = (
                0.299 * rgb[:, 0:1]
                + 0.587 * rgb[:, 1:2]
                + 0.114 * rgb[:, 2:3]
            )
            sampled_comparators = {
                "evidence_insufficiency": (1.0 - output.evidence_support)[
                    ..., ::spatial_stride, ::spatial_stride
                ].detach().cpu(),
                "visible_thermal_disagreement": (
                    visible_luminance - thermal
                ).abs()[..., ::spatial_stride, ::spatial_stride].detach().cpu(),
                "learned_abstention": output.abstention[
                    ..., ::spatial_stride, ::spatial_stride
                ].detach().cpu(),
            }
            for index in range(batch_size):
                score = sampled_score[index].flatten()
                risk = sampled_target[index].flatten()
                dataset = str(raw_batch["dataset"][index])
                scene_group = str(raw_batch["scene_group"][index])
                group = f"{dataset}:{scene_group}"
                scores.append(score)
                targets.append(risk)
                grouped_scores.setdefault(group, []).append(score)
                grouped_targets.setdefault(group, []).append(risk)
                for comparator_name in comparator_names:
                    comparator = sampled_comparators[comparator_name][index].flatten()
                    comparator_scores[comparator_name].append(comparator)
                    grouped_comparator_scores[comparator_name].setdefault(
                        group, []
                    ).append(comparator)

    if not examples or not scores:
        raise RuntimeError(f"evaluation produced no successful examples: {failures}")
    aggregate_metrics = {
        key: value / examples for key, value in totals.items()
    }
    aggregate_metrics["maximum_support_violation"] = maximum_violation
    aggregate_metrics["examples"] = float(examples)
    group_metrics = {
        group: selective_metrics(
            torch.cat(grouped_scores[group]), torch.cat(grouped_targets[group])
        )
        for group in sorted(grouped_scores)
    }
    attempted = examples + sum(
        len(failure["sample_ids"]) for failure in failures
    )
    all_scores = torch.cat(scores)
    all_targets = torch.cat(targets)
    comparator_reports: dict[str, object] = {}
    for comparator_name in comparator_names:
        all_comparator_scores = torch.cat(comparator_scores[comparator_name])
        per_group = {
            group: selective_metrics(
                torch.cat(grouped_comparator_scores[comparator_name][group]),
                torch.cat(grouped_targets[group]),
            )
            for group in sorted(grouped_targets)
        }
        comparator_reports[comparator_name] = {
            "metrics_pixel_weighted": selective_metrics(
                all_comparator_scores, all_targets
            ),
            "metrics_by_capture_group": per_group,
            "capture_group_confidence_intervals_95": grouped_selective_bootstrap(
                per_group,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            ),
            "diagnostic_curves_pixel_weighted": selective_diagnostic_curves(
                all_comparator_scores, all_targets
            ),
        }
    return {
        "task_input_mode": task_mode,
        "objective_and_proxy_metrics": aggregate_metrics,
        "failure_score_metrics_pixel_weighted": selective_metrics(
            all_scores, all_targets
        ),
        "failure_score_diagnostic_curves_pixel_weighted": selective_diagnostic_curves(
            all_scores, all_targets
        ),
        "failure_score_comparators": comparator_reports,
        "failure_score_comparator_definitions": {
            "evidence_insufficiency": "one minus the externally supplied evidence support",
            "visible_thermal_disagreement": "absolute normalized visible-luminance/thermal difference",
            "learned_abstention": "the model's learned abstention output",
        },
        "failure_score_metrics_by_capture_group": group_metrics,
        "capture_group_confidence_intervals_95": grouped_selective_bootstrap(
            group_metrics,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "runtime_and_failures": {
            "attempted_examples": attempted,
            "successful_examples": examples,
            "failed_examples": attempted - examples,
            "failure_rate": (attempted - examples) / attempted,
            "failures": failures,
            "latency": _latency_summary(latency_ms_per_image),
            "latency_scope": "model forward pass only; host preprocessing excluded",
            "cold_start_policy": "first measured batch retained",
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else None
            ),
        },
    }


def evaluate_baselines(
    loader: DataLoader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    totals = {
        name: {"gradient_l1": 0.0, "task_intensity_smooth_l1": 0.0}
        for name in BASELINE_NAMES
    }
    examples = 0
    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            evidence = batch["evidence"].to(device)
            task_ids = batch["task_id"].to(device)
            support = torch.prod(evidence, dim=1, keepdim=True)
            batch_size = int(rgb.shape[0])
            examples += batch_size
            for name in BASELINE_NAMES:
                fused = fuse_baseline(name, rgb, thermal, evidence)
                metrics = fusion_proxy_metrics(
                    fused, rgb, thermal, support, task_ids
                )
                for metric, value in metrics.items():
                    totals[name][metric] += float(value.cpu()) * batch_size
    for metrics in totals.values():
        for metric in metrics:
            metrics[metric] /= max(examples, 1)
        metrics["examples"] = float(examples)
    return totals


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_root: Path,
    output_path: Path,
    *,
    partition: str = "validation",
    patch_size: int = 192,
    batch_size: int = 12,
    max_samples: int | None = None,
    device_name: str = "auto",
    seed: int = 20260902,
    unlock_final_test: bool = False,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 20260902,
    spatial_stride: int = 8,
) -> dict[str, object]:
    if partition == "test" and not unlock_final_test:
        raise ValueError(
            "final test evaluation is locked; freeze the model and pass --unlock-final-test"
        )
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available() else
        "cpu" if device_name == "auto" else device_name
    )
    seed_everything(seed)
    model, metadata = load_checkpoint(checkpoint_path, device=device)
    objective = EGTCFLoss()

    def loader(
        corrupted: bool,
        *,
        corruption_modes: tuple[str, ...] = CORRUPTION_MODES,
    ) -> DataLoader:
        dataset = FusionPatchDataset(
            data_root,
            partition,
            patch_size=patch_size,
            seed=seed,
            max_samples=max_samples,
            corruption_probability=1.0 if corrupted else 0.0,
            apply_corruptions=corrupted,
            corruption_modes=corruption_modes,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    def model_pass(
        corrupted: bool,
        *,
        task_mode: str,
        corruption_modes: tuple[str, ...] = CORRUPTION_MODES,
    ) -> dict[str, object]:
        return evaluate_model_pass(
            model,
            loader(corrupted, corruption_modes=corruption_modes),
            objective,
            device,
            task_mode=task_mode,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
            spatial_stride=spatial_stride,
        )

    clean = {
        "provided_task": model_pass(False, task_mode="provided"),
        "automatic_task": model_pass(False, task_mode="automatic"),
    }
    stress = {
        "provided_task": model_pass(True, task_mode="provided"),
        "automatic_task": model_pass(True, task_mode="automatic"),
    }
    stress_strata = {
        mode: {
            "provided_task": model_pass(
                True,
                task_mode="provided",
                corruption_modes=(mode,),
            ),
            "automatic_task": model_pass(
                True,
                task_mode="automatic",
                corruption_modes=(mode,),
            ),
        }
        for mode in CORRUPTION_MODES
    }
    clean_baselines = evaluate_baselines(loader(False), device)
    stress_baselines = evaluate_baselines(loader(True), device)
    report: dict[str, object] = {
        "schema_version": "openprism.egtcf-evaluation-report/2.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_status": (
            "development_subset_not_for_paper"
            if max_samples is not None
            else "protocol_evaluation_requires_human_review"
        ),
        "checkpoint": {
            "path": str(metadata.path),
            "artifact_sha256": metadata.artifact_sha256,
            "model_id": metadata.model_id,
            "training_provenance": metadata.training_provenance,
            "validation_scope": metadata.validation_scope,
            "epoch": metadata.epoch,
            "bytes": metadata.path.stat().st_size,
        },
        "partition": partition,
        "final_test_unlocked": partition == "test" and unlock_final_test,
        "max_samples": max_samples,
        "evaluation_configuration": {
            "patch_size": patch_size,
            "batch_size": batch_size,
            "seed": seed,
            "spatial_sampling_stride": spatial_stride,
            "task_modes": ["provided", "automatic"],
            "corruption_modes": list(CORRUPTION_MODES),
            "reliability_bins": 15,
            "risk_coverage_grid_points": 20,
            "risk_coverage_tie_policy": (
                "threshold-tied: every equal-score pixel enters together; AURC uses "
                "right-continuous block-end risk weighted by the block coverage increment"
            ),
        },
        "clean": clean,
        "synthetic_corruption_stress": stress,
        "synthetic_corruption_strata": stress_strata,
        "proxy_baselines_clean": clean_baselines,
        "proxy_baselines_synthetic_corruption": stress_baselines,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "sampling_unit": "pre-specified complete capture group",
            "same_resampling_seed_for_every_condition": True,
        },
        "source_sha256": {
            "evaluation.py": _sha256(Path(__file__)),
            "data.py": _sha256(Path(__file__).with_name("data.py")),
            "model.py": _sha256(Path(__file__).with_name("model.py")),
            "objective.py": _sha256(Path(__file__).with_name("objective.py")),
            "checkpoint.py": _sha256(Path(__file__).with_name("checkpoint.py")),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "cuda_device": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
            "cuda_device_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda" else None
            ),
        },
        "protocol": protocol_manifest(data_root),
        "limitations": [
            "Proxy fusion losses are not fused-image ground truth.",
            "This report does not include detection or segmentation task metrics.",
            "Synthetic corruptions do not replace calibrated flight validation.",
            "The reported failure score is not a calibrated posterior probability or a safety guarantee.",
            "Automatic task labels are dataset-derived proxies, not operator intent labels.",
            "No acceptance or state-of-the-art claim follows from this report alone.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/prism-egt/evaluation.json"))
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--unlock-final-test", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260902)
    parser.add_argument("--spatial-stride", type=int, default=8)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = evaluate_checkpoint(
        args.checkpoint,
        args.data_root,
        args.output,
        partition=args.partition,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        device_name=args.device,
        seed=args.seed,
        unlock_final_test=args.unlock_final_test,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        spatial_stride=args.spatial_stride,
    )
    print(json.dumps({
        "scientific_status": report["scientific_status"],
        "partition": report["partition"],
        "clean_loss": report["clean"]["provided_task"]
        ["objective_and_proxy_metrics"]["loss"],
        "stress_loss": report["synthetic_corruption_stress"]["provided_task"]
        ["objective_and_proxy_metrics"]["loss"],
    }, indent=2))


if __name__ == "__main__":
    main()
