"""Explicit, optional Rust kernel selection; NumPy remains the reference default."""
from __future__ import annotations

from functools import lru_cache
import importlib
import os
from types import ModuleType


@lru_cache(maxsize=1)
def _load_native() -> ModuleType:
    try:
        native = importlib.import_module("openprism_native")
    except ImportError as error:
        raise RuntimeError(
            "Rust backend requested but openprism-native is unavailable. "
            "Install it with `python -m pip install ./native` from the repository, "
            "or set OPENPRISM_BACKEND=numpy."
        ) from error
    if getattr(native, "API_VERSION", None) != 1:
        raise RuntimeError("Unsupported openprism-native API version; rebuild ./native")
    return native


def native_kernels() -> ModuleType | None:
    backend = os.environ.get("OPENPRISM_BACKEND", "numpy").lower()
    if backend == "numpy":
        return None
    if backend == "rust":
        return _load_native()
    raise ValueError("OPENPRISM_BACKEND must be 'numpy' or 'rust'")


def backend_info() -> dict[str, str | int]:
    """Describe the selected implementation for provenance and benchmark records."""
    native = native_kernels()
    if native is None:
        return {"name": "numpy", "kernel_api": 1}
    return {"name": "rust", "kernel_api": native.API_VERSION, "version": native.__version__}
