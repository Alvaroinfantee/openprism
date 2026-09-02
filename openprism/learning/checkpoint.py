"""Safe, explicit checkpoint serialization for PRISM-EGT."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from .model import EGTCF, EGTCFConfig


CHECKPOINT_SCHEMA_VERSION = "openprism.egtcf-checkpoint/1.0"


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    path: Path
    artifact_sha256: str
    model_id: str
    training_provenance: str
    validation_scope: str
    epoch: int
    metrics: Mapping[str, float]


def save_checkpoint(
    path: str | Path,
    model: EGTCF,
    *,
    model_id: str,
    training_provenance: str,
    validation_scope: str,
    epoch: int,
    metrics: Mapping[str, float],
) -> CheckpointMetadata:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not model_id.strip() or not training_provenance.strip() or not validation_scope.strip():
        raise ValueError("checkpoint provenance fields cannot be empty")
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_id": model_id,
        "training_provenance": training_provenance,
        "validation_scope": validation_scope,
        "epoch": int(epoch),
        "metrics": {str(key): float(value) for key, value in metrics.items()},
        "config": model.config.as_dict(),
        "state_dict": model.state_dict(),
    }
    torch.save(payload, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return CheckpointMetadata(
        path=destination.resolve(),
        artifact_sha256=digest,
        model_id=model_id,
        training_provenance=training_provenance,
        validation_scope=validation_scope,
        epoch=int(epoch),
        metrics=payload["metrics"],
    )


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[EGTCF, CheckpointMetadata]:
    source = Path(path)
    # weights_only rejects arbitrary pickle globals and still supports the
    # primitive metadata plus tensors used by this artifact.
    payload = torch.load(source, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported PRISM-EGT checkpoint schema")
    required = {
        "model_id",
        "training_provenance",
        "validation_scope",
        "epoch",
        "metrics",
        "config",
        "state_dict",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"checkpoint is missing fields: {sorted(missing)}")
    config = EGTCFConfig.from_dict(payload["config"])
    model = EGTCF(config).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    metrics = {str(key): float(value) for key, value in payload["metrics"].items()}
    metadata = CheckpointMetadata(
        path=source.resolve(),
        artifact_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        model_id=str(payload["model_id"]),
        training_provenance=str(payload["training_provenance"]),
        validation_scope=str(payload["validation_scope"]),
        epoch=int(payload["epoch"]),
        metrics=metrics,
    )
    return model, metadata
