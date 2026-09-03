"""Reproducible training and development evaluation for PRISM-EGT."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .data import CORRUPTION_MODES, FusionPatchDataset, protocol_manifest
from .model import EGTCF, EGTCFConfig
from .objective import EGTCFLoss, EGTCFLossConfig


EXPERIMENT_VARIANTS = (
    "full",
    "no_task_conditioning",
    "no_learned_abstention",
    "no_calibration_loss",
    "no_hidden_corruption",
    "soft_evidence_envelope",
)


def _source_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    names = ("training.py", "data.py", "model.py", "objective.py", "checkpoint.py")
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in names
    }


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    data_root: Path
    output_dir: Path
    epochs: int = 20
    batch_size: int = 12
    patch_size: int = 192
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    seed: int = 20260902
    workers: int = 0
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    base_channels: int = 24
    pose_features: int = 0
    variant: str = "full"
    device: str = "auto"


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _average(sums: dict[str, float], count: int) -> dict[str, float]:
    return {key: value / max(count, 1) for key, value in sums.items()}


def run_epoch(
    model: EGTCF,
    loader: DataLoader[dict[str, Any]],
    objective: EGTCFLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    task_mode: str = "provided",
) -> dict[str, float]:
    if task_mode not in {"provided", "automatic"}:
        raise ValueError("task_mode must be provided or automatic")
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    examples = 0
    for raw_batch in loader:
        batch = _batch_to_device(raw_batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(
                batch["rgb"],
                batch["thermal"],
                batch["evidence"],
                task_ids=(batch["task_id"] if task_mode == "provided" else None),
            )
            loss, components = objective(
                output,
                batch["rgb"],
                batch["thermal"],
                batch["task_id"],
                corruption_target=batch["corruption_target"],
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        batch_size = int(batch["rgb"].shape[0])
        examples += batch_size
        for name, value in components.items():
            totals[name] = totals.get(name, 0.0) + float(value.cpu()) * batch_size
        violation = torch.relu(output.thermal_contribution - output.evidence_support)
        totals["maximum_support_violation"] = max(
            totals.get("maximum_support_violation", 0.0),
            float(violation.detach().max().cpu()),
        )
        predicted = output.task_logits.argmax(dim=1)
        totals["task_accuracy"] = totals.get("task_accuracy", 0.0) + float(
            (predicted == batch["task_id"]).float().sum().cpu()
        )
    averaged = _average(
        {key: value for key, value in totals.items() if key != "maximum_support_violation"},
        examples,
    )
    averaged["maximum_support_violation"] = totals.get("maximum_support_violation", 0.0)
    averaged["examples"] = float(examples)
    return averaged


def _loader(
    dataset: FusionPatchDataset,
    config: TrainingConfig,
    *,
    shuffle: bool,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(config.seed + (0 if shuffle else 1))
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        # Workers are recreated each epoch so the epoch-specific deterministic
        # crop/corruption state is propagated on spawn-based platforms too.
        persistent_workers=False,
    )


def train(config: TrainingConfig) -> dict[str, object]:
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.variant not in EXPERIMENT_VARIANTS:
        raise ValueError(f"unknown experiment variant: {config.variant}")
    seed_everything(config.seed)
    device = _device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    corruption_modes = (
        ("declared_shift", "dropout")
        if config.variant == "no_hidden_corruption"
        else CORRUPTION_MODES
    )
    training_data = FusionPatchDataset(
        config.data_root,
        "train",
        patch_size=config.patch_size,
        seed=config.seed,
        max_samples=config.max_train_samples,
        corruption_modes=corruption_modes,
    )
    validation_data = FusionPatchDataset(
        config.data_root,
        "validation",
        patch_size=config.patch_size,
        seed=config.seed,
        max_samples=config.max_validation_samples,
        corruption_probability=0.0,
    )
    if not training_data or not validation_data:
        raise ValueError("training and validation partitions must be non-empty")
    training_loader = _loader(training_data, config, shuffle=True)
    validation_loader = _loader(validation_data, config, shuffle=False)

    model = EGTCF(
        EGTCFConfig(
            base_channels=config.base_channels,
            pose_features=config.pose_features,
            use_task_conditioning=config.variant != "no_task_conditioning",
            use_learned_abstention=config.variant != "no_learned_abstention",
            hard_evidence_envelope=config.variant != "soft_evidence_envelope",
        )
    ).to(device)
    objective = EGTCFLoss(
        EGTCFLossConfig(
            abstention_weight=(
                0.0 if config.variant == "no_learned_abstention" else 0.8
            ),
            calibration_weight=(
                0.0 if config.variant == "no_calibration_loss" else 0.5
            ),
        )
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.epochs)
    )

    history: list[dict[str, object]] = []
    best_loss = float("inf")
    best_checkpoint: str | None = None
    limited = config.max_train_samples is not None or config.max_validation_samples is not None
    run_kind = "development_smoke_test" if limited else "full_protocol_training"
    started = datetime.now(timezone.utc).isoformat()
    for epoch in range(1, config.epochs + 1):
        training_data.set_epoch(epoch - 1)
        train_metrics = run_epoch(
            model, training_loader, objective, device, optimizer=optimizer
        )
        validation_metrics = run_epoch(
            model, validation_loader, objective, device, optimizer=None
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            checkpoint_path = config.output_dir / "best.pt"
            metadata = save_checkpoint(
                checkpoint_path,
                model,
                model_id=f"prism-egt-{config.variant}-{run_kind}",
                training_provenance=(
                    f"{run_kind}; variant={config.variant}; seed={config.seed}; "
                    f"train_examples={len(training_data)}; validation_examples={len(validation_data)}"
                ),
                validation_scope=(
                    "development-only; not a paper result"
                    if limited
                    else "frozen validation partition; final test partition untouched"
                ),
                epoch=epoch,
                metrics=validation_metrics,
            )
            best_checkpoint = str(metadata.path)

    report: dict[str, object] = {
        "schema_version": "openprism.egtcf-training-report/1.0",
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "run_kind": run_kind,
        "experiment_variant": config.variant,
        "paper_result": False,
        "paper_result_reason": (
            "development subset used" if limited else "test partition not evaluated"
        ),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "configuration": {
            **asdict(config),
            "data_root": str(config.data_root.resolve()),
            "output_dir": str(config.output_dir.resolve()),
        },
        "model_configuration": model.config.as_dict(),
        "objective_configuration": asdict(objective.config),
        "training_corruption_modes": list(corruption_modes),
        "source_sha256": _source_hashes(),
        "protocol": protocol_manifest(config.data_root),
        "history": history,
        "best_validation_loss": best_loss,
        "best_checkpoint": best_checkpoint,
    }
    (config.output_dir / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/prism-egt"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--pose-features", type=int, default=0)
    parser.add_argument("--variant", choices=EXPERIMENT_VARIANTS, default="full")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = train(
        TrainingConfig(
            data_root=args.data_root,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patch_size=args.patch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            workers=args.workers,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
            base_channels=args.base_channels,
            pose_features=args.pose_features,
            variant=args.variant,
            device=args.device,
        )
    )
    summary = {
        "run_kind": report["run_kind"],
        "paper_result": report["paper_result"],
        "best_validation_loss": report["best_validation_loss"],
        "best_checkpoint": report["best_checkpoint"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
