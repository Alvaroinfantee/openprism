# Optional Rust acceleration

OpenPRISM can run its thermal local-mean and palette kernels in Rust while
keeping the Python API and the deterministic evidence gates. This backend is
opt-in. The default NumPy implementation remains the numerical reference.
This addition accelerates deterministic CPU fusion; it does not accelerate
PRISM-EGT neural inference, registration FFTs, terrain ray casting or I/O.

## Installation and selection

Install [Rust](https://rust-lang.org/tools/install/) and its platform linker
(on Windows, the Visual Studio C++ build tools). From the repository root,
using the same Python environment as OpenPRISM:

```sh
python -m pip install -e ".[dev]"
python -m pip install ./native
```

The second command builds a release-mode extension through
[Maturin](https://www.maturin.rs/tutorial) and PyO3. The native component is a
separate package so ordinary OpenPRISM installations do not require a compiler.
Its stable-ABI wheels support standard CPython 3.10 or newer on the platform
for which they were built. The source uses the checked-in Cargo lockfile;
no GPU, Go runtime, subprocess service or Rust compiler is needed at runtime.
No native package has been published to PyPI by this change.

PowerShell:

```powershell
$env:OPENPRISM_BACKEND = "rust"
openprism
```

Linux/macOS:

```sh
OPENPRISM_BACKEND=rust openprism
```

Set `OPENPRISM_BACKEND=numpy` to select the reference implementation. Select
the backend before starting a process; changing the environment during
concurrent fusion calls is not supported. Requesting Rust without a compatible
extension raises an actionable error instead of silently changing the recorded
backend. Invalid backend names are rejected.

```python
from openprism.acceleration import backend_info
print(backend_info())
# {'name': 'rust', 'kernel_api': 1, 'version': '0.1.0'}
```

Every deterministic `FusionOutput.provenance` includes `compute_backend`.
Install/upgrade the extension and restart the process after changing Rust code.
The environment variable describes the selected backend; general-domain blur
inputs outside the supported native domain still use the reference kernel.

## Algorithm and numerical contract

The reference radius-three blur adds 49 shifted image planes. The Rust kernel
maintains vertical column sums and slides one horizontal sum over each row.
For height H, width W and radius r, work is O(HW + r(H + W)), compared with
O(HW(2r+1)^2) in the reference. Its scratch storage is W float64 values,
in addition to the float32 output; it needs no padded image or full-size
intermediate plane. Reflected borders match NumPy's `mode="reflect"`, including
singleton dimensions and radii larger than the image dimensions.

Accumulation uses float64 and the result is cast once to float32. This produces
a more accurate local mean than repeated float32 summation, but changes the
rounding order. The backend therefore does **not** promise bitwise-equivalent
fusion tensors. Tests use these contracts:

- Radius-three local means differ from the NumPy reference by at most 8e-7
  absolute on the tested normalized inputs. Independently computed float64
  neighborhood means round to exactly the same float32 values as Rust.
- Complete fusion tensors and support maps differ by at most 2e-6 absolute
  on the synthetic regression fixtures and benchmark frames. Operator and
  semantic RGB can differ by one 8-bit level at rounding boundaries.
- The original synchronization, registration, sensor-validity and uncertainty
  logic is unchanged. Tests require exactly zero thermal contribution wherever
  the reference gates it out, including zero gain and no-fusion-zone cases.
- Native blur handles finite normalized [0, 1] images at radii 1â€“64. Other
  ranges, NaN/Inf and larger radii follow the original NumPy path, retaining
  its local non-finite propagation. Radius zero retains the original behavior.

The Rust palette renderer writes each uint8 RGB pixel directly. It preserves
float32 interpolation, later-interval precedence at shared stops, clipping and
ties-to-even rounding. Dense sweeps, all palette stops and their neighboring
float32 values, out-of-range values, NaN/Inf, scalar and strided inputs are
tested for byte equality with both built-in palettes. No lookup-table
quantization is introduced.

Aligned contiguous float32 inputs are borrowed without an input copy. Other
layouts/dtypes are converted at the Python boundary. Native entry points also
validate layout, alignment and domain. Inputs are not modified. The kernels
retain the GIL, use one CPU thread, and introduce no application-level unsafe
Rust or fast-math. As with NumPy, callers must not mutate shared inputs while
an operation reads them.

These are implementation parity tests, not a new validation of sensor
calibration, downstream detection accuracy or operational flight performance.
Use NumPy when reproducing historical results that require the original
floating-point ordering. The frozen model-evaluation tools and final-test
locks are unchanged.

## Reproduce the measurements

```sh
python tooling/benchmark_native_kernels.py --output output/native-benchmark.json
```

The benchmark uses seed 20261021 and synthetic float32 thermal fields or
RGB/16-bit thermal pairs at 480x640, 720x1280 and 1080x1920. Each backend,
workload and size runs in its own fresh process, with two warm-ups and seven
measured calls. Timings include the public Python call, conversions,
allocations and output construction; imports, input creation, decode, I/O and
correctness checks are excluded. Outputs are released between calls.

The memory metric is the OS process peak resident memory (Windows peak working
set; `getrusage` on Linux/macOS), including interpreter, input/output buffers
and native allocations. It is measured before a separate comparison against
NumPy. Python-only allocation tracers would omit Rust allocations, so they
are not used. Each JSON record contains individual timings and errors, along
with environment details, raw local source-byte hashes (including checkout
line endings) and the loaded extension binary hash.
Run benchmarks without other CPU-heavy jobs; results are hardware-dependent
and are not CI performance thresholds.

The checked-in [Windows measurement](../../benchmarks/native-windows-x86_64.json)
was produced on an AMD Ryzen 7 8700F, Windows x86-64, Python 3.12 and NumPy
2.5.4, with a release build using Rust 1.99.1.

| Image | Workload | NumPy ms | Rust ms | Speedup | NumPy / Rust peak MiB |
|---|---|---:|---:|---:|---:|
| 480x640 | blur | 2.77 | 1.39 | 2.00x | 41.1 / 39.1 |
| 480x640 | palette | 42.85 | 3.87 | 11.07x | 54.3 / 38.5 |
| 480x640 | fusion | 98.06 | 58.53 | 1.68x | 105.2 / 105.4 |
| 720x1280 | blur | 8.04 | 4.16 | 1.93x | 50.2 / 44.3 |
| 720x1280 | palette | 128.60 | 11.55 | 11.13x | 90.7 / 42.6 |
| 720x1280 | fusion | 286.94 | 167.28 | 1.72x | 237.8 / 239.1 |
| 1080x1920 | blur | 32.19 | 9.69 | 3.32x | 67.8 / 52.3 |
| 1080x1920 | palette | 288.07 | 25.88 | 11.13x | 157.1 / 50.2 |
| 1080x1920 | fusion | 695.31 | 380.04 | 1.83x | 488.7 / 488.8 |

The memory reduction is in the accelerated kernels. Complete fusion peak RSS
is essentially unchanged because its other full-frame evidence buffers still
dominate the peak. All 11 machine channels and immutable source observations
are retained. These measurements are synthetic CPU implementation benchmarks,
not trained-model quality or publisher-dataset results.

## Validation and packaging

```sh
cargo test --manifest-path native/Cargo.toml --locked
cargo clippy --manifest-path native/Cargo.toml --locked --all-targets -- -D warnings
python -m pytest tests/test_openprism_acceleration.py -q
python tooling/run_core_tests.py
python -m build native --outdir dist/native
```

Run the core suite with each backend selected. Install `.[dev,learned]` and
run `python -m pytest -q` for the full suite. Native parity tests skip explicitly
when the optional extension is absent. CI additionally builds the native wheel
from its source distribution, installs that wheel, and runs the core suite on
Linux, Windows and macOS, including Python 3.10 and 3.13 on Linux.
