# OpenPRISM

[![CI](https://github.com/Alvaroinfantee/openprism/actions/workflows/ci.yml/badge.svg)](https://github.com/Alvaroinfantee/openprism/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Pre-JOSS research release. Complete author metadata and add an archival DOI
> before a JOSS submission. See [JOSS readiness](JOSS_READINESS.md).

**Provenance-Rich Integrated Sensor Model** — one synchronized scene state,
two faithful projections.

OpenPRISM treats fusion as a reversible view function, not as a flattened image
file. A `PrismFrame` preserves source measurements, timing, geometry, validity,
uncertainty, and provenance. From that same evidence it produces:

1. a named, multi-channel machine tensor for detection, segmentation, tracking,
   mapping, or multimodal tokenization; and
2. a deterministic operator canvas with visible color, thermal evidence,
   semantic context, confidence, and instant raw-source inspection.

The human rendering is never used as the only model input or source of truth.

## Run the reference console

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m openprism
```

Then open <http://127.0.0.1:8765>. Use `--no-browser` when running headless.

The console reads separately staged LLVIP, MSRS, and Caltech Aerial RGB-T
datasets from `data/`. It does not copy, rename, or rewrite publisher imagery,
and the repository does not redistribute those archives.

## Install

```bash
python -m venv .venv
python -m pip install -e .
openprism
```

Pass `--data-root /path/to/data` when the archives are not under the current
working directory. Install `.[pixhawk]` only when MAVLink log/live support is
needed.

## Automatic fusion and AI-ready evidence

The **Constrained fusion agent** extracts ten bounded scene features from a
neutral fusion probe, recommends Navigate, Search, Terrain, or Integrity, and
selects a thermal gain. Its small versioned linear policy is inspectable and
explicitly marked `expert_initialized_not_fitted`: no benchmark-optimal or
trained-model claim is made.

Hard synchronization, registration, validity, and uncertainty gates always
take precedence. When evidence is unsupported, the controller selects
Integrity and zero thermal contribution. Operators can disable automatic
control at any time.

Every frame also has an image-free JSON digest for downstream agents:

```text
GET /api/ai/context?dataset=llvip&split=train&index=0
```

The digest includes named tensor channels and statistics, scene evidence,
recommendation confidence and rationale, the exact policy artifact hash, and a
machine-readable safety contract. See [Automatic Fusion](openprism/docs/AUTOMATIC_FUSION.md)
and the [digest schema](openprism/spec/ai-scene-digest.schema.json).

## Build the PRISM Atlas demonstration

PRISM Atlas turns calibrated camera captures plus Pixhawk pose records into a
north-up tactical map and a separate geolocated people/vehicle track layer:

```powershell
.\.venv\Scripts\python.exe tooling\build_openprism_atlas_demo.py --overwrite
.\.venv\Scripts\python.exe -m openprism
```

Select **Atlas** in the same operator window. The bundled demonstration is
analytically generated and watermarked `SYNTHETIC`; it contains no Caltech
imagery, real GPS, or real terrain. It exercises the complete API and UI path
without inventing coordinates for the downloaded paired-image datasets.

The live algorithm is deliberately evidence-preserving:

1. match each exposure to a bounded Pixhawk pose on an explicitly anchored
   clock;
2. apply the declared camera-to-vehicle-to-ENU frame graph and MSL-to-ellipsoid
   conversion;
3. intersect rectified RGB pixels with a declared ground plane or 2.5-D height
   field;
4. probe pose, timing, registration, and terrain uncertainty and abstain when a
   ray is unsupported;
5. accumulate RGB, normalized thermal, terrain semantics, support, uncertainty,
   and source provenance in north-up cells; and
6. publish one immutable, hash-verified generation while keeping dynamic
   people/vehicles in a time-bounded track layer rather than terrain texture.

This live product is an **orthodrape**, not terrain reconstruction. A real
surface model comes from a range sensor or the post-flight SfM/MVS path. See
[Terrain Mapping Architecture](docs/TERRAIN_MAPPING.md) for the two-speed live
and survey design, coordinate conventions, flight requirements, and validation
gates.

### Real flight prerequisites

Install the optional log/live MAVLink adapter into this workspace with:

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r openprism\requirements-pixhawk.txt
```

- overlapping full-resolution RGB captures and registered thermal evidence;
- traceable RGB/thermal intrinsics, distortion, cross-sensor registration, and
  camera-to-vehicle rotation plus lever arm;
- exposure or trigger timestamps tied to the Pixhawk clock with measured delay
  and uncertainty;
- selected, non-ambiguous MAVLink navigation and GPS streams with covariance;
- declared acceleration and angular-acceleration bounds whenever a capture pose
  must be interpolated between telemetry samples;
- an explicitly named MSL vertical datum/geoid model and its uncertainty;
- RTK/PPK and surveyed check points when metric accuracy matters; and
- immutable source images and complete autopilot/companion logs for replay.

GPS alone cannot reconstruct terrain geometry. Photogrammetry needs parallax,
texture, overlap, calibrated optics, and an optimized camera trajectory; thermal
may be projected onto that geometry only after temporal and spatial alignment.

## Prepare a true post-flight reconstruction

The survey handoff stages original RGB captures and `CameraPoseRecord` JSON as
a content-addressed OpenDroneMap project. It checks exact image/pose matching,
declared datum and geoid identity, position quality, minimum image count, and a
coarse overlap/extent prerequisite before writing a position-only `geo.txt`, a
hash manifest, and the exact Docker command:

```powershell
.\.venv\Scripts\python.exe tooling\prepare_openprism_survey.py `
  --image-dir <RGB_CAPTURE_DIRECTORY> `
  --poses <CAMERA_POSE_RECORDS_JSON> `
  --output-root <ODM_DATASETS_DIRECTORY> `
  --project-name <MISSION_NAME> `
  --vertical-datum <EXPLICIT_DATUM_ID> `
  --geoid-model <EXPLICIT_GEOID_ID> `
  --nominal-agl-m <METERS> `
  --horizontal-fov-deg <DEGREES> `
  --vertical-fov-deg <DEGREES> `
  --planned-forward-overlap 0.80 `
  --planned-side-overlap 0.70
```

Preparation does not run Docker. Add `--run` only when you intend to launch
ODM, and pin `--odm-image` to a tested digest for a reproducible operational
run. The generated `survey_plan.json` states the required reconstruction and
checkpoint checks. RGB supplies the initial geometry; calibrated thermal and
semantic evidence are projected onto the optimized surface in a later stage.

## Use the machine projection

```python
from openprism.datasets import DatasetCatalog
from openprism.fusion import EvidenceFusionEngine

catalog = DatasetCatalog("data")
frame = catalog.load("llvip", "train", 0)
product = EvidenceFusionEngine().fuse(frame)

model_input = product.machine_tensor  # float32 [channels, height, width]
channel_index = {name: i for i, name in enumerate(product.channel_names)}
operator_view = product.operator_rgb  # uint8 [height, width, 3]
```

Models should select channels by name, not by a hard-coded position. This keeps
the interface evolvable when depth, radar, lidar, event, pose, or other sensor
adapters are added.

## What is implemented

- immutable, caller-isolated sensor observations and deep-frozen provenance;
- arbitrary auxiliary sensor tensors for future IMU, GNSS, radar, lidar, event,
  SAR, depth, or hyperspectral adapters;
- explicit hardware-style TAI timestamps and timestamp uncertainty;
- a watermark synchronizer that marks observations `exact`, `interpolated`,
  `late`, or `missing` and gates pixel fusion;
- a conservative No-Fusion Zone when clocks, timing uncertainty, registration,
  or a measurement-uncertainty model cannot support pixel blending;
- content-sniffing adapters for LLVIP, MSRS, and Caltech;
- preservation of Caltech 16-bit thermal counts before display normalization;
- declared publisher alignment plus an abstaining edge-phase residual registrar;
- deterministic confidence-aware visible/thermal fusion;
- an explainable, versioned automatic fusion controller whose outputs remain
  subordinate to evidence safety gates;
- an image-free, schema-backed AI scene digest with channel statistics,
  provenance, and policy artifact hashing;
- an 11-channel machine projection, including source-qualified registration
  support, sensor validity, per-pixel thermal contribution, and fusion support;
- ground-truth box and semantic-mask provenance without implying inference;
- one browser canvas with Navigate, Search, Terrain, and Integrity presets;
- a fifth Atlas preset for north-up RGB/thermal mapping, uncertainty support,
  transient object tracks, and explicit fresh/stale/snapshot state;
- strict Pixhawk/MAVLink capture matching with selectable vehicle, component,
  camera, navigation, and GPS streams plus acceleration-bounded interpolation;
- dependency-light flat-ground and supplied-height-field orthodraping with
  terrain-aware visibility probes and bounded memory/ray work;
- atomic hash-manifest Atlas generations and position-prior export for the
  post-flight photogrammetry path;
- content-addressed ODM survey staging with exact RGB/pose matching, coarse
  overlap gates, immutable inputs, explicit datums, and opt-in execution;
- a coordinate-locked evidence lens that reveals a raw source in place;
- a candidate JSON Schema for a portable OpenPRISM Frame Bundle;
- a dependency-light local HTTP API and browser UI.

## Validate

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test*.py" -v
.\.venv\Scripts\python.exe tooling\validate_multispectral_datasets.py
```

Unit tests run without third-party datasets. Dataset integration tests activate
when all three archives are staged under `data/`.

## Package map

```text
openprism/
├── contracts.py          immutable evidence and projection contracts
├── autonomy.py           constrained policy + AI scene digest
├── datasets.py           LLVIP, MSRS, and Caltech adapters
├── synchronization.py    asynchronous sensor watermarking and fusion gates
├── registration.py       identity/declaration and residual registration
├── fusion.py             machine tensor + deterministic operator renderer
├── pixhawk.py            strict MAVLink capture/pose bridge and ODM priors
├── mapping.py            geodesy, ray casting, uncertainty, orthomosaic grid
├── atlas.py              mission gates, dynamic objects, atomic map bundles
├── survey.py             deterministic, fail-closed ODM survey handoff
├── server.py             local API and static console server
├── web/                  single-window operator interface
├── spec/                 candidate interoperable JSON Schema
└── docs/                 architecture, standards, and validation strategy
```

## Important limitations

- This is a tested **reference baseline**, not a certified operational system.
- The three archives contain paired RGB/thermal imagery; they do not validate a
  live hardware clock, camera calibration, rolling-shutter compensation, lidar,
  radar, GNSS, or IMU integration.
- Publisher rectification is labeled *declared, not measured*. A live system
  must estimate residual alignment and its uncertainty continuously.
- Thermal pixels are labeled `relative thermal intensity` or `raw sensor count`.
  They are not temperatures without a traceable radiometric calibration.
- The browser currently displays dataset ground truth. It runs no detector and
  labels that distinction explicitly.
- Atlas does not yet perform online visual-inertial bundle adjustment or infer a
  new elevation surface. Its live height layer records intersections with the
  supplied plane/DEM. The included survey tool prepares an SfM/MVS job, but it
  does not itself validate or certify the resulting reconstruction.
- LLVIP and Caltech are non-commercial datasets; MSRS has no explicit license
  in its repository. They can seed research and conformance tests but cannot be
  assumed suitable for a commercial training corpus.

See [Reference Architecture](openprism/docs/REFERENCE_ARCHITECTURE.md),
[Standards Profile](openprism/docs/STANDARDS_PROFILE.md),
[Terrain Mapping](openprism/docs/TERRAIN_MAPPING.md), and
[Validation Plan](openprism/docs/VALIDATION_PLAN.md).
