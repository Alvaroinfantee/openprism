"""Reproducible synthetic CPU benchmark, separate from frozen model evaluation.

Each backend/workload/size uses a fresh process. Timings include the public
Python API, conversions, allocation and output construction. Input creation,
imports, file I/O and correctness comparisons are outside timing. Memory is
OS process peak resident memory, including both Python and Rust allocations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openprism.acceleration import backend_info, native_kernels
from openprism.contracts import PrismFrame, SensorObservation, SynchronizationStatus, Timestamp
from openprism.fusion import EvidenceFusionEngine, box_blur, thermal_colormap


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                    "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile",
                )
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.peak_working_set)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def workload(name: str, height: int, width: int):
    rng = np.random.default_rng(20261021)
    value = rng.random((height, width), dtype=np.float32)
    if name == "blur":
        return lambda: box_blur(value, radius=3)
    if name == "palette":
        return lambda: thermal_colormap(value)
    timestamp = Timestamp(1000, clock_id="synthetic-benchmark")
    observations = {}
    for key, modality, data, encoding in (
        ("visible", "visible_rgb", rng.integers(0, 256, (height, width, 3), dtype=np.uint8), "rgb8"),
        ("thermal", "thermal_lwir", rng.integers(0, 65536, (height, width), dtype=np.uint16), "mono16"),
    ):
        observations[key] = SensorObservation(
            sensor_id=key, modality=modality, frame_id=key,
            timestamp=timestamp, data=data, encoding=encoding,
        )
    frame = PrismFrame(
        frame_id="synthetic-benchmark", timestamp=timestamp, reference_frame="visible",
        observations=observations,
        synchronization=SynchronizationStatus.declared_replay_aligned(
            tuple(observations), clock_domain="synthetic-benchmark", declaration="seeded synthetic benchmark"
        ),
    )
    engine = EvidenceFusionEngine()
    return lambda: engine.fuse(frame)


def worker(args):
    os.environ["OPENPRISM_BACKEND"] = args.backend
    info = backend_info()  # Load the extension outside timing.
    native = native_kernels()
    binary_sha256 = None
    if native is not None:
        extension = sys.modules[native.palette.__module__]
        binary_sha256 = hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest()
    run = workload(args.workload, args.height, args.width)
    for _ in range(args.warmup):
        result = run()
        del result
    gc.collect()
    elapsed = []
    for _ in range(args.repeats):
        start = time.perf_counter_ns()
        result = run()
        elapsed.append((time.perf_counter_ns() - start) / 1e6)
        del result
    peak = peak_rss_bytes()  # Correctness comparison must not inflate this peak.
    actual = run()
    os.environ["OPENPRISM_BACKEND"] = "numpy"
    expected = run()
    if args.workload == "fusion":
        tensor_error = float(np.max(np.abs(actual.machine_tensor - expected.machine_tensor)))
        rgb_error = int(np.max(np.abs(actual.operator_rgb.astype(int) - expected.operator_rgb.astype(int))))
        np.testing.assert_allclose(actual.machine_tensor, expected.machine_tensor, atol=2e-6, rtol=0)
        np.testing.assert_array_equal(actual.thermal_view, expected.thermal_view)
        if rgb_error > 1:
            raise AssertionError(f"operator RGB error exceeds one level: {rgb_error}")
        error = {"machine_max_abs": tensor_error, "operator_max_levels": rgb_error}
    elif args.workload == "palette":
        np.testing.assert_array_equal(actual, expected)
        error = {"max_abs": 0}
    else:
        np.testing.assert_allclose(actual, expected, atol=8e-7, rtol=0)
        error = {"max_abs": float(np.max(np.abs(actual - expected)))}
    return {
        "backend": info, "workload": args.workload,
        "native_binary_sha256": binary_sha256,
        "height": args.height, "width": args.width, "warmup": args.warmup,
        "repeats": args.repeats, "samples_ms": elapsed,
        "median_ms": statistics.median(elapsed), "peak_rss_bytes": peak,
        "error_vs_numpy": error,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/native-benchmark.json"))
    parser.add_argument("--sizes", nargs="+", default=["480x640", "720x1280", "1080x1920"])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=("numpy", "rust"))
    parser.add_argument("--workload", choices=("blur", "palette", "fusion"))
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 0:
        parser.error("repeats must be positive and warmup non-negative")
    if args.worker:
        print(json.dumps(worker(args)))
        return
    rows = []
    for size in args.sizes:
        height, width = map(int, size.lower().split("x"))
        if min(height, width) < 1:
            parser.error("image dimensions must be positive")
        for name in ("blur", "palette", "fusion"):
            for backend in ("numpy", "rust"):
                command = [
                    sys.executable, str(Path(__file__).resolve()), "--worker",
                    "--backend", backend, "--workload", name,
                    "--height", str(height), "--width", str(width),
                    "--repeats", str(args.repeats), "--warmup", str(args.warmup),
                ]
                row = json.loads(subprocess.check_output(command, text=True))
                rows.append(row)
                print(f"{size} {name:7s} {backend:5s}: {row['median_ms']:.3f} ms, {row['peak_rss_bytes'] / 2**20:.1f} MiB peak RSS", flush=True)
    files = ["native/src/lib.rs", "native/Cargo.lock", "openprism/acceleration.py", "openprism/fusion.py", "tooling/benchmark_native_kernels.py"]
    report = {
        "schema": "openprism.synthetic-native-benchmark/1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "cpu": platform.processor(),
        "python": platform.python_version(), "numpy": np.__version__,
        "data": "seeded synthetic inputs; no publisher data or model quality evaluation",
        "memory_metric": "OS process peak resident memory including interpreter, inputs, outputs and native allocations; correctness comparison excluded",
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in files},
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
