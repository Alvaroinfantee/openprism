"""Evaluate PRISM-EGT invariants and robustness without leaking the test set."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .baselines import BASELINE_NAMES, fuse_baseline
from .data import FusionPatchDataset, protocol_manifest
from .objective import EGTCFLoss
from .objective import fusion_proxy_metrics
from .training import run_epoch, seed_everything


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

    order = torch.argsort(score, descending=False)
    ordered_risk = risk[order]
    cumulative_risk = torch.cumsum(ordered_risk, dim=0) / torch.arange(
        1, ordered_risk.numel() + 1, dtype=torch.float32
    )
    aurc = float(cumulative_risk.mean())

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
        "uncertainty_auroc": auroc,
        "samples": float(score.numel()),
        "mean_target_risk": float(risk.mean()),
        "mean_uncertainty": float(score.mean()),
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
            target = batch["corruption_target"][..., ::8, ::8].flatten().cpu()
            take = min(score.numel(), maximum_samples - collected)
            if take <= 0:
                break
            uncertainties.append(score[:take])
            targets.append(target[:take])
            collected += take
    if not uncertainties:
        raise ValueError("evaluation loader produced no examples")
    return selective_metrics(torch.cat(uncertainties), torch.cat(targets))


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

    def loader(corrupted: bool) -> DataLoader:
        dataset = FusionPatchDataset(
            data_root,
            partition,
            patch_size=patch_size,
            seed=seed,
            max_samples=max_samples,
            corruption_probability=1.0 if corrupted else 0.0,
            apply_corruptions=corrupted,
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    clean_loader = loader(False)
    stress_loader = loader(True)
    clean = run_epoch(model, clean_loader, objective, device)
    stress = run_epoch(model, stress_loader, objective, device)
    stress_selective = collect_selective_metrics(model, loader(True), device)
    clean_baselines = evaluate_baselines(loader(False), device)
    stress_baselines = evaluate_baselines(loader(True), device)
    report: dict[str, object] = {
        "schema_version": "openprism.egtcf-evaluation-report/1.0",
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
        },
        "partition": partition,
        "final_test_unlocked": partition == "test" and unlock_final_test,
        "max_samples": max_samples,
        "clean": clean,
        "synthetic_corruption_stress": stress,
        "synthetic_corruption_selective_metrics": stress_selective,
        "proxy_baselines_clean": clean_baselines,
        "proxy_baselines_synthetic_corruption": stress_baselines,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "protocol": protocol_manifest(data_root),
        "limitations": [
            "Proxy fusion losses are not fused-image ground truth.",
            "This report does not include detection or segmentation task metrics.",
            "Synthetic corruptions do not replace calibrated flight validation.",
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
    )
    print(json.dumps({
        "scientific_status": report["scientific_status"],
        "partition": report["partition"],
        "clean_loss": report["clean"]["loss"],
        "stress_loss": report["synthetic_corruption_stress"]["loss"],
    }, indent=2))


if __name__ == "__main__":
    main()
