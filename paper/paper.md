---
title: "OpenPRISM: evidence-preserving multisensor fusion for machines, operators, and aerial maps"
tags:
  - Python
  - sensor fusion
  - thermal imaging
  - computer vision
  - unmanned aerial vehicles
  - terrain mapping
authors:
  - name: Alvaro Infante Flores
    orcid: 0009-0002-0680-9612
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 02 September 2026
bibliography: paper.bib
---

# Summary

OpenPRISM (Provenance-Rich Integrated Sensor Model) is a Python framework for
combining visible and thermal imagery while preserving the evidence needed by
both computational models and human operators. Rather than treating fusion as
the production of a single authoritative picture, OpenPRISM represents a scene
as immutable source observations, timing and registration state, validity,
uncertainty, annotations, and provenance. The same state is projected into (1)
a named multi-channel tensor for detection, segmentation, tracking, mapping, or
multimodal encoding and (2) a deterministic single-window operator view with
raw-source inspection.

The framework includes dataset adapters, synchronization and registration
gates, an explainable automatic fusion controller, a machine-readable scene
digest, a browser-based operator console, a Pixhawk/MAVLink pose bridge, live
2.5-D orthodraping, dynamic-object tracks, and a content-addressed handoff to
OpenDroneMap for post-flight structure-from-motion and multi-view stereo. It is
designed as research infrastructure, not as certified flight or life-safety
software.

# Statement of need

Visible cameras provide texture and color but may lose targets in darkness,
smoke, haze, or low contrast. Thermal cameras often retain target contrast while
discarding visible appearance. Existing experimental pipelines commonly solve
one downstream task, flatten two sensors into an image without recording how
alignment and confidence were established, or build a model input that human
operators cannot audit. The resulting image can look convincing even when
clocks, geometric registration, valid overlap, or radiometry do not support
pixel-level fusion.

Researchers need a reusable boundary between sensor evidence, fusion policy,
machine input, operator rendering, and geospatial products. OpenPRISM supplies
that boundary and fails closed: unknown or incompatible clocks, excessive skew,
unmodeled measurement uncertainty, weak registration, or invalid coverage enter
a No-Fusion Zone. A model recommendation cannot make an ineligible measurement
eligible. This is especially important for aerial mapping, where GPS alone does
not reconstruct terrain and where camera pose, vertical datum, parallax,
calibration, overlap, and check points determine what can be claimed.

# State of the field

LLVIP provides aligned low-light visible/infrared imagery and pedestrian labels
[@jia2021llvip]. MSRS derives aligned day/night road scenes from the MFNet
multispectral benchmark [@ha2017mfnet; @tang2022seafusion]. The Caltech Aerial
RGB-T dataset extends paired sensing to natural aerial environments with
position, inertial data, and terrain semantics [@lee2024cart]. These resources
support algorithm comparison, but they use different structures, radiometric
representations, annotations, and capture assumptions.

Robotics middleware such as ROS 2 transports heterogeneous observations and
supports distributed systems [@macenski2022ros2], while OpenCV supplies widely
used image operations [@bradski2000opencv]. OpenDroneMap reconstructs geospatial
products from overlapping aerial photographs [@opendronemap2026]. These systems
are complementary to OpenPRISM. OpenPRISM focuses on the missing evidence
contract between paired RGB/thermal data, adaptive human rendering, named model
channels, and honest live-versus-survey map products. Its dependency-light core
also lets archive experiments and contract tests run without a robotics runtime
or GPU.

# Software design

`PrismFrame` owns immutable `SensorObservation` arrays and explicit timestamps,
clock identities, uncertainty, coordinate frames, validity masks, provenance,
semantic masks, and detections. A watermark synchronizer classifies evidence as
exact, bounded-skew, late, future, missing, or incompatible. Registration is
either publisher-declared or measured by an abstaining residual estimator; the
representation never silently upgrades a declared prior into a measured
confidence.

`EvidenceFusionEngine` produces an eleven-channel `float32` tensor containing
visible sRGB, normalized thermal evidence, modality-specific detail and
saliency, validity, registration support, actual thermal contribution, and
fusion support. Consumers select channels by name. The operator canvas uses the
same output but remains a view rather than model truth. Navigate, Search,
Terrain, and Integrity presets change interpretation without destroying the
sources, and a coordinate-locked evidence lens exposes registered raw evidence
in place.

The automatic controller runs a neutral fusion probe, extracts ten bounded
features, and evaluates a small versioned linear policy. It recommends a preset
and bounded thermal gain, emits probabilities and rationale, and identifies the
policy artifact by SHA-256. The bundled coefficients are explicitly
`expert_initialized_not_fitted`; they are a reproducible reference policy, not
a claim of learned or universal optimality. Hard synchronization, registration,
validity, and uncertainty gates remain authoritative. An image-free JSON digest
exposes tensor statistics, scene evidence, controls, provenance, and safety
semantics to downstream AI systems.

For UAV work, Pixhawk messages are filtered by selected vehicle, component,
camera, navigation, and GPS streams before exposure-to-pose matching. Declared
camera-to-vehicle geometry and vertical datums feed ray intersections with a
ground plane or supplied height field. Live Atlas output is therefore labeled a
2.5-D orthodrape and keeps transient people/vehicles outside the static terrain
texture. Post-flight tooling stages exact RGB/pose pairs, hashes all inputs, and
creates an opt-in OpenDroneMap command; it does not claim that preparation alone
performed or validated reconstruction.

# Research impact statement

OpenPRISM turns several usually implicit experimental choices into inspectable
software contracts: whether time alignment is measured, whether registration is
a prior, which pixels contributed thermal evidence, why automatic control chose
a view, and whether a map depicts a supplied surface or reconstructed geometry.
This supports reproducible ablations across LLVIP, MSRS, and Caltech Aerial
RGB-T while providing one extension point for additional cameras, lidar, radar,
IMU, GNSS, depth, event, SAR, or hyperspectral observations.

At this pre-JOSS release stage, automated tests exercise deterministic fusion,
safety abstention, schema behavior, pose matching, uncertainty propagation,
atomic Atlas publication, dynamic/static separation, and survey staging. The
repository does **not yet claim** independent adoption, certified field
performance, or operator-performance improvement. Before JOSS submission, the
authors will add documented public research use, held-out task results, a real
calibrated flight with checkpoint errors, and—if human-performance claims are
made—a preregistered operator study. This limitation is recorded to keep impact
claims falsifiable rather than aspirational.

# AI usage disclosure

OpenAI Codex using GPT-5 (accessed September 2026) assisted
with initial code, tests, documentation, repository scaffolding, and manuscript
prose. It did not supply experimental observations or independently validate
scientific claims. Before submission, every named human author must review,
edit, execute, and validate the software and manuscript, accept responsibility
for all content, and record the exact tool/model versions and scope used. Core
design and publication decisions must remain human decisions. AI systems will
not communicate with JOSS editors or reviewers except for translation if
explicitly permitted by JOSS policy.

# Acknowledgements

The authors thank the maintainers and participants of LLVIP, MFNet/MSRS,
Caltech Aerial RGB-T, ROS, OpenCV, MAVLink/Pixhawk, and OpenDroneMap. Funding and
institutional acknowledgements must be added before submission. The authors
must also disclose any financial or non-financial conflicts of interest; none
are asserted by this draft.

# References
