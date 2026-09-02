# OpenPRISM Validation and Benchmark Plan

The objective is not to win one image-fusion metric. It is to prove that the
system helps models and operators find people, vehicles, and terrain while
remaining calibrated, inspectable, fast, and safe under sensor failure.

## 1. Scorecards

### Registration

- reprojection/target registration error in pixels and angular units;
- robust inlier coverage and spatial distribution;
- per-pixel uncertainty calibration and risk–coverage curve;
- occlusion/FOV/invalid-region precision and recall;
- temporal warp stability, double-edge/ghost rate, and track jitter;
- residual under parallax, rolling shutter, platform motion, thermal inversion,
  low texture, saturation, blur, and changing focus/zoom; and
- correct abstention rate when support is insufficient.

### Machine perception

- people/vehicle AP, AP50, AP75, small-object recall, and miss rate;
- terrain mIoU, per-class IoU, boundary F-score, and geographic-domain transfer;
- tracking HOTA/IDF1, ID switches, track latency, and coast error;
- ECE, NLL/Brier score, OOD detection, selective risk, and abstention coverage;
- localization/geoposition error with propagated uncertainty; and
- visible-only, thermal-only, operator-rendered, reversible machine-tensor, and
  feature/token-fusion baselines.

### Human operator

- target acquisition time and search-path efficiency;
- miss/false-alarm rate and d-prime;
- terrain identification accuracy and boundary errors;
- raw-source/evidence-lens usage and disagreement resolution time;
- trust calibration after nominal and injected-fault trials;
- change blindness when an automatic mode or palette changes;
- NASA-TLX workload and structured situation-awareness probes; and
- color-vision, contrast, keyboard, touch, and reduced-motion accessibility.

### System

- sensor-to-evidence, evidence-to-display, and end-to-end latency distributions;
- jitter, deadline/liveliness misses, frame drops, and recovery time;
- CPU/GPU/RAM/VRAM, network bandwidth, storage, power, and thermal load;
- deterministic replay and bit/numerical reproducibility;
- sustained-rate performance at target resolutions and sensor counts; and
- raw-view/health availability under fusion/model process failure.

Image entropy, SSIM, mutual information, visual information fidelity, and
similar fusion metrics MAY be reported as secondary diagnostics. They do not
substitute for task, calibration, human, or system results.

## 2. Dataset protocol

### LLVIP

- Preserve supplied train/test split.
- Report day/night and target-size strata where metadata can be recovered.
- Evaluate people detection and low-light robustness.
- Do not treat as a commercial training corpus; its terms are non-commercial.

### MSRS

- Preserve supplied train/test semantic split and separate detection subset.
- Report daytime/nighttime and class-specific segmentation.
- Keep dataset identity when combining experiments; avoid implicit pooled
  normalization that leaks sensor/scenario information.
- Treat redistribution/commercial rights as unspecified until clarified.

### Caltech Aerial RGB-T

- Split by complete flight/scene group, never random neighboring frames.
- Report random, temporal, geographic, and terrain-domain transfer.
- Preserve uint16 thermal counts before any display/model normalization.
- Report bare ground, rocky terrain, developed structures, road, shrubs, trees,
  sky, water, vehicles, and people separately.
- Treat as non-commercial under the authoritative CaltechDATA record.

### Cross-dataset

- Train on one or two dataset families and test on the held-out family.
- Hold out complete sensor models, locations, flights, times, and weather where
  possible.
- Report per-dataset as well as aggregate metrics; do not allow a large set to
  hide collapse on a small one.
- Never infer temperature from uncalibrated thermal code values.

## 3. Mandatory ablations

1. visible only;
2. thermal only;
3. deterministic operator rendering only;
4. reversible OpenPRISM machine tensor;
5. early, intermediate, late, and gated multi-depth feature fusion;
6. with/without explicit validity and source-qualified registration support;
7. with/without timestamp age/skew channels;
8. calibrated versus fixed fusion weights;
9. sensor-dropout and missing-modality training;
10. ground-truth alignment versus online residual registration; and
11. source-contribution ablation for every detection/track.

The operator rendering and machine representation MUST be benchmarked
separately. A model may not train only on the aesthetically optimized bitmap.

## 4. Fault-injection matrix

| Fault | Injection | Expected behavior |
| --- | --- | --- |
| Clock offset/drift | Static/ramped timestamp error | Pairing state changes; pixel fusion gate trips; late fusion remains |
| Stale/frozen sensor | Repeat old payload with increasing age | STALE state, last-good age, no false live appearance |
| Packet loss/reorder | Drop/burst/reorder frames and metadata | QoS counters, bounded latency, deterministic recovery |
| Miscalibration | Perturb intrinsics/extrinsics/zoom | Residual rises; uncertainty and No-Fusion Zones expand |
| Parallax/occlusion | Depth discontinuities and camera baseline | Invalid/occluded pixels not overlaid |
| Low texture | Flat or repeated patterns | Registrar abstains instead of asserting a transform |
| RGB degradation | Darkness, glare, blur, rain, compression | Reliability shifts; thermal remains attributable |
| Thermal degradation | NUC drift, dead pixels, bloom, saturation | Validity/health flags; no temperature claim |
| Sensor loss | Remove RGB/thermal/lidar/radar | Explicit single/partial-sensor mode |
| Invalid geolocation | Remove/corrupt GNSS/pose | Map output suppressed, image evidence retained |
| Model OOD | New sensor/weather/object family | Tentative/abstaining evidence and OOD state |
| Process crash | Kill fusion/model service | Raw view and health path remain available |

Recovery MUST be tested as carefully as failure entry. State hysteresis should
prevent rapid oscillation near thresholds.

## 5. Tests in the reference implementation

`tests/test_openprism.py` currently verifies:

- sensor arrays own immutable storage and nested provenance is frozen;
- semantic masks resolve against the explicit reference frame;
- known-shift registration direction and exact alignment;
- low-texture registration abstention;
- delayed, missing, and stale synchronization gating;
- one-dimensional auxiliary observations for IMU-like sensors;
- exact dataset counts and pair adapters;
- content-sniffed Caltech images and preserved uint16 thermal data;
- explicit ground-truth attribution;
- deterministic, finite, bounded dual-rail fusion across all three datasets;
- API image/metadata contract; and
- candidate schema structure.

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_openprism.py" -v
```

## 6. Next automated gates

- subpixel/projective synthetic registration and calibrated uncertainty;
- corrupt/truncated payload handling;
- grouped Caltech split-leakage assertions;
- golden synthetic operator render without restricted dataset redistribution;
- a versioned runtime-to-Frame-Bundle serializer and independent reader;
- Draft 2020-12 validation of complete serialized golden bundles, including
  exact 3x3/4x4 transform cardinality and registered/extension modalities;
- round-trip tests proving payload hashes, units, validity, uncertainty,
  calibration, quality, contributor IDs, and provenance survive serialization;
- unknown-time fixtures proving `tai_ns=None` is emitted as unknown/null time
  with an explicit quality flag, never as timestamp evidence;
- API concurrency, cancellation, cache, and malformed-query tests;
- browser interaction checks for all presets, lens, toggles, slider, keyboard,
  narrow layout, missing channels, and server errors;
- CPU/GPU latency budgets and 8 GB VRAM limit; and
- fault-injection replay fixtures with expected state transitions.

## 7. Acceptance sequence

1. **Contract:** two independent readers reproduce a Frame Bundle.
2. **Geometry:** registration error/uncertainty meet mission thresholds.
3. **Perception:** fusion beats the strongest single modality without hiding
   regressions, with calibrated abstention.
4. **Operator:** blinded studies improve acquisition/terrain decisions without
   increasing false confidence or workload.
5. **System:** end-to-end latency, recovery, and resource limits pass.
6. **Interoperability:** two vendors exchange live/replayed bundles and evidence.
7. **Safety/security:** fault, threat, privacy, and domain safety cases close.

Only after all seven should the profile be described as operationally mature.
