"""Learned, evidence-bounded RGB--thermal fusion.

The learning stack is optional so the deterministic OpenPRISM runtime remains
usable without PyTorch.  Install it with ``pip install openprism-fusion[learned]``.
"""

from .engine import LearnedFusionEngine
from .checkpoint import CheckpointMetadata, load_checkpoint, save_checkpoint
from .model import EGTCF, EGTCFConfig, EGTCFOutput, TASK_NAMES
from .objective import EGTCFLoss, EGTCFLossConfig
from .baselines import BASELINE_NAMES, fuse_baseline

__all__ = [
    "EGTCF",
    "EGTCFConfig",
    "EGTCFLoss",
    "EGTCFLossConfig",
    "EGTCFOutput",
    "CheckpointMetadata",
    "LearnedFusionEngine",
    "TASK_NAMES",
    "BASELINE_NAMES",
    "fuse_baseline",
    "load_checkpoint",
    "save_checkpoint",
]
