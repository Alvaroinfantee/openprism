# Changelog

All notable changes are documented here. The project follows Semantic
Versioning after its first public release.

## [Unreleased]

- Prepare independent packaging, community files, CI, JOSS manuscript, and
  contribution guidance.
- Add exact publisher URLs, source revisions, archive digests, and a sanitized
  inventory for reproducible acquisition of the three third-party datasets.
- Add PRISM-EGT evidence-gated task-conditioned learned fusion, safe checkpoint
  loading, training and locked-test evaluation CLIs, operator integration, and
  a leakage-aware LLVIP/MSRS/Caltech experiment protocol.
- Add overlap-tiled full-resolution inference and executable RGB-only,
  thermal-only, average, maximum, and deterministic OpenPRISM proxy baselines.
- Add a test-locked, checksum-recording LLVIP person-detection probe with
  identical frozen inference across source, built-in fusion, PRISM-EGT, and
  externally generated learned-baseline views.
- Add a revision-verifying, weights-only external runner for reviewed
  SeaFusion, CDDFuse, PAIF, and C2RF checkouts without vendoring third-party
  code.
- Add an official-format anonymous TMLR manuscript draft whose result cells are
  deliberately blocked until full baselines and final evaluation are complete.

## [0.3.0] - 2026-09-02

### Added

- Explainable automatic selection of Navigate, Search, Terrain, or Integrity.
- Bounded thermal-gain recommendation with immutable policy artifact hashing.
- Schema-backed, image-free AI scene context endpoint.
- Operator auto/advice switch, rationale, evidence metrics, and safety status.

### Changed

- Zero thermal gain now means exactly zero thermal contribution.
- Installed applications resolve default data and Atlas paths from the working
  directory rather than from inside the Python package.

### Safety

- Automatic control cannot override synchronization, registration, validity, or
  measurement-uncertainty gates.

## [0.2.0] - 2026-09-02

- Added Pixhawk capture/pose matching, live orthodraping, uncertainty support,
  immutable Atlas generations, dynamic object tracks, and ODM survey staging.

## [0.1.0] - 2026-09-01

- Initial dual-rail RGB/thermal fusion engine, data adapters, schemas, HTTP API,
  and operator console.
