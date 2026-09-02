# OpenPRISM Multisensor Evidence and Fusion Profile

Status: candidate profile 0.1. This document composes existing standards; it is
not itself an approved standard.

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** indicate proposed
conformance requirements.

## 1. Scope

This profile defines a portable evidence bundle, transform/calibration
registry, uncertainty vocabulary, degraded-state model, operator contract, and
bindings for multisensor machine perception. It applies to visible, thermal,
depth, lidar, radar, event, IMU, GNSS, SAR, hyperspectral, and derived scene
evidence.

It does not define a universal fusion neural network or color palette. It
defines the evidence and safety contract within which those implementations can
be compared and replaced.

## 2. Normative invariants

1. Source measurements MUST remain immutable and addressable.
2. A human-rendered composite MUST NOT be the only machine representation.
3. Every observation MUST declare modality, encoding, physical units or an
   explicit uncalibrated unit, sensor/frame ID, capture interval, clock domain,
   timestamp uncertainty, validity, health, and calibration reference.
4. Every derived result MUST identify its source observations, transforms,
   processing configuration, contribution/validity, and uncertainty.
5. Physical uncertainty and ML confidence MUST remain distinct.
6. Frame pairing MUST be explicit: `exact`, `interpolated`, `late`, `missing`,
   or `unknown`.
7. Pixel fusion MUST be disabled where temporal/spatial support fails policy.
8. Raw-source inspection and health indication MUST remain available when the
   fusion/model pipeline is degraded.
9. Synthetic or generative imagery MUST be labeled and MUST NOT masquerade as
   a physical thermal measurement.
10. Ground truth, model output, prediction/coasting, and operator confirmation
    MUST retain distinct provenance.

## 3. Standards composition

| Concern | Preferred standards/bindings | Profile use |
| --- | --- | --- |
| Network time | [IEEE 1588-2019 PTP](https://standards.ieee.org/ieee/1588/6825/), [IEEE 802.1AS-2025](https://standards.ieee.org/ieee/802.1AS/11968/) | Hardware time, domain, uncertainty, holdover |
| Facility media time | [SMPTE ST 2059-2:2021](https://pub.smpte.org/latest/st2059-2/st2059-2-2021.pdf), [ST 2110-10:2022](https://pub.smpte.org/pub/st2110-10/st2110-10-2022.pdf) | Synchronized professional video deployments |
| Sensor description | [OGC SensorML 3.0](https://docs.ogc.org/is/23-000/23-000.pdf) | Sensor, process, calibration, and deployment metadata |
| Observations | [OGC OMS 3.0](https://www.ogc.org/standards/om/), [SensorThings API 1.1](https://www.ogc.org/standards/sensorthings/) | Observation semantics and external APIs |
| Pose/frames | [OGC GeoPose 1.0](https://www.ogc.org/standards/geopose/), [ROS REP 103](https://www.ros.org/reps/rep-0103.html), [REP 105](https://www.ros.org/reps/rep-0105.html) | Axis conventions and transform tree |
| Coordinate reference | [ISO 19111:2019](https://www.iso.org/standard/74039.html), [ISO 19130-1:2018](https://www.iso.org/standard/66847.html) | CRS, moving-platform imagery, propagated error |
| Sensor calibration | [ISO/TS 19159-1:2014](https://www.iso.org/standard/60080.html), [ISO/TS 19124-1:2023](https://www.iso.org/standard/79352.html) | Optical/radiometric calibration and validation |
| Measurement uncertainty | [JCGM 100:2008](https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf) | Physical uncertainty vocabulary/propagation |
| Processing lineage | [W3C PROV-O](https://www.w3.org/TR/prov-o/) | Observation → alignment → fusion → inference → action |
| Signed exports | [C2PA 2.4](https://spec.c2pa.org/specifications/specifications/2.4/index.html) | Tamper-evident exported images/reports |
| Runtime pub/sub | [OMG DDS 1.4](https://www.omg.org/spec/DDS/1.4/PDF), [DDSI-RTPS](https://www.omg.org/spec/DDSI-RTPS), [DDS-XTypes 1.3](https://www.omg.org/spec/DDS-XTypes/1.3) | Typed telemetry, calibration, tracks, and control |
| Managed video LAN | [SMPTE ST 2110](https://www.smpte.org/standards/st2110), [ST 2110-41:2024](https://pub.smpte.org/latest/st2110-41/st2110-41-2024.pdf), [ST 2022-7:2019](https://pub.smpte.org/latest/st2022-7/st2022-7-2019.pdf) | Essence, fast metadata, redundant paths |
| Remote/browser video | [RTP RFC 3550](https://www.rfc-editor.org/info/rfc3550), [SRTP RFC 3711](https://www.rfc-editor.org/info/rfc3711), [WebRTC](https://www.w3.org/TR/webrtc/), [RFC 8834](https://www.rfc-editor.org/rfc/rfc8834.html) | Low-latency operator transport |
| Camera edge | [ONVIF Profile T](https://www.onvif.org/profiles/profile-t/), [Profile M](https://www.onvif.org/profiles/profile-m/) | Video, events, and analytics metadata |
| Aerial motion imagery | [NGA MISB registry](https://nsgreg.nga.mil/misb.jsp) | UAS metadata, VMTI/tracks, quality, identifiers |
| Geospatial output | [OGC API Features](https://www.ogc.org/standards/ogcapi-features/), [GeoTIFF](https://www.ogc.org/standards/geotiff/), [3D Tiles](https://www.ogc.org/standards/3DTiles/) | Tracks, footprints, terrain, raster/mesh products |
| Human-centered design | [ISO 9241-210:2019](https://www.iso.org/standard/77520.html), [ISO 9241-110:2020](https://www.iso.org/standard/75258.html), [FAA HF-STD-001B](https://hf.tc.faa.gov/publications/2016-12-human-factors-design-standard/full_text.pdf) | User research, interaction and workload |
| Accessible presentation | [WCAG 2.2](https://www.w3.org/TR/WCAG22/) | Non-color-only state, contrast, keyboard operation |
| AI risk | [NIST AI RMF 1.0](https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-ai-rmf-10), [ISO/IEC 23894:2023](https://www.iso.org/standard/77304.html) | Governance, risk, validation, monitoring |

Several ISO, IEEE, IEC, and SAE standards require purchase or free-access
registration. Implementers MUST verify the licensed normative text and pin exact
versions in procurement and conformance records.

## 4. Time profile

Systems MUST use integer TAI nanoseconds internally. UTC is a presentation
format and MUST identify the leap-second table used.

An image observation MUST record:

- capture start, midpoint, and end;
- exposure/integration and rolling-shutter line timing;
- hardware clock/domain and timestamp source;
- timestamp uncertainty;
- ingress, processing, and presentation time; and
- offset, drift, holdover, path-asymmetry, and source-switch health where
  supplied by the time network.

RTP timestamps alone are insufficient as absolute capture time. Every remote
overlay MUST reference an exact `frame_bundle_id`.

Proposed engineering policy: disable pixel fusion when timestamp uncertainty
projected through measured/estimated scene motion exceeds about 0.5 pixel.
This threshold is not specified by IEEE/SMPTE and MUST be mission-calibrated.

## 5. Coordinate and calibration profile

The default transform chain is:

```text
Earth/ECEF → map → odom → platform/base_link → gimbal → sensor → optical
```

Every transform MUST state parent/child direction, axis/handedness convention,
units, quaternion ordering, covariance, source, calibration ID, and validity
interval. Geospatial frames MUST identify CRS, axis order, vertical datum, and
coordinate epoch; `WGS84` alone is insufficient.

Factory calibration is immutable. Field and online residual updates create new
calibration epochs linked through provenance.

Thermal calibration SHOULD include NUC state, sensor temperature, bad-pixel
mask, spectral response/band, integration time, radiometric response, and any
emissivity/environment assumptions.

Proposed engineering policy: above about 1-pixel reprojection uncertainty,
disable pixel overlay and retain labeled late/object fusion and raw comparison.

## 6. Frame Bundle profile

The candidate JSON schema is
[`prism-frame.schema.json`](../spec/prism-frame.schema.json). A conforming bundle
MUST include:

- schema version, mission/session/bundle ID, and sequence;
- capture interval and pairing state;
- reference frame and coordinate convention;
- one or more source observations with hashed payload and validity assets;
- versioned transforms and covariance;
- quality/degraded state;
- provenance activity, software/configuration hash, source IDs, and optional
  model/calibration IDs; and
- contributor IDs for every derived machine/operator projection.

Per-pixel contribution/uncertainty assets MAY use tiled rasters, tensor stores,
or shared-memory/zero-copy handles. JSON MUST reference rather than inline
high-rate imagery.

### 6.1 Portable schema versus reference runtime

The JSON Schema defines the candidate **portable Frame Bundle**. It is not the
wire representation of the current lightweight Python `Timestamp`,
`SensorObservation`, `PrismFrame`, or `FusionOutput` classes, nor of the local
operator API. Those in-memory contracts intentionally omit several portable
fields and do not yet serialize referenced payloads, validity/uncertainty
assets, hashes, complete capture intervals, health/calibration records, quality
state, or processing-activity provenance. Passing the reference unit tests does
not establish Frame Bundle conformance or any conformance level in Section 10.

The required next step is a versioned serializer plus an independent reader.
The serializer MUST materialize and hash every referenced asset, map runtime
modalities and units without loss, construct required capture/geometry/quality
and provenance records, validate the result with a Draft 2020-12 validator, and
pass a round-trip golden-vector test. Until that exists, the Python objects and
operator responses MUST be labeled `reference runtime`, not `portable bundle`.

Unknown archive capture time MUST remain unknown. The reference runtime uses
`tai_ns=None`; it MUST NOT be promoted to timestamp evidence. When authoritative
time is unavailable, portable JSON uses `null` for the unavailable TAI and timing-
uncertainty fields, sets `pairing_state` to `unknown`, and carries an explicit
quality/provenance flag. A numeric zero is permitted only when supported by a
traceable clock source.

## 7. Runtime QoS profile

- High-rate preview imagery SHOULD use best-effort delivery with finite lifespan.
- Calibration, alarms, acknowledgements, and operator commands MUST use reliable,
  ordered delivery and suitable durability.
- Implementations MUST monitor deadline, liveliness, latency, and stale-frame
  events.
- Transient visual overlays MAY use an unreliable low-latency channel.
- Alerts, operator decisions, and control MUST use a reliable ordered channel.
- Structured metadata MUST remain synchronized but separate from the display
  bitmap; burning all metadata into video is non-conforming.

For MISB deployments, procurement MUST pin registry versions and assess at
least UAS metadata, VMTI/track metadata, interpretability/quality, platform and
sensor identifiers, and KLV transport/container requirements.

## 8. Operator profile

The stable central scene MUST expose:

- current mode and any automatic mode change;
- live/replay/test state;
- source health, age, latency, synchronization, and registration state;
- relative/physical thermal units and palette scale;
- geolocation validity;
- model/version/OOD state for derived evidence;
- reversible RGB, thermal, semantic, and Integrity modes; and
- a coordinate-locked raw evidence reveal.

Class confidence, spatial uncertainty, age, contributing sensors, and quality
flags SHOULD be presented separately. A single opaque confidence percentage is
insufficient.

Observed, predicted/coasting, and operator-confirmed evidence MUST use shape,
line, and text distinctions in addition to color. Operator corrections MUST be
quarantined from training/ground truth until reviewed.

Operator validation SHOULD measure detection/acquisition time, miss and false
alarm rate, d-prime, trust calibration, and workload with
[NASA-TLX](https://www.nasa.gov/human-systems-integration-division/nasa-task-load-index-tlx/),
not visual preference alone.

## 9. Degraded-state profile

Conforming implementations MUST expose these states:

```text
NOMINAL
DEGRADED
UNSYNCHRONIZED
UNREGISTERED
STALE
LOST
TEST_REPLAY
```

Minimum fallbacks:

| Condition | Required behavior |
| --- | --- |
| Excess time/registration error | Pixel fusion off; labeled late fusion/raw inspection retained |
| RGB loss | Explicit thermal-only mode; color-semantic caveat |
| Thermal loss | Explicit RGB-only mode; low-light caveat |
| Invalid geolocation | Image evidence retained; map coordinates suppressed |
| OOD/uncalibrated model | Tentative evidence only |
| Frozen/stale frame | Hatch/blank, last-good time and age; never appear live |
| Prediction on newer video | Label predicted, show age, grow uncertainty |
| Recovery | Apply hysteresis and audit the transition |

Deployments SHOULD select the governing safety process from domain standards,
for example IEC 61508, ISO 21448 for road vehicles, or SAE ARP4754B/ARP4761A
for civil-aircraft systems. OpenPRISM does not replace a domain safety case.

## 10. Conformance levels

### Level 0 — Recorded evidence

- Immutable observations, hashes, units, IDs, timestamps, and provenance.
- Replay reproduces machine/operator projections bit-for-bit or within declared
  numerical tolerance.

### Level 1 — Registered multimodal frame

- Level 0 plus calibration/transform graph, validity, residual registration,
  uncertainty, contribution maps, and No-Fusion Zones.

### Level 2 — Live synchronized system

- Level 1 plus hardware time, pairing states, latency/staleness monitoring,
  transport bindings, fault injection, and degraded-state fallbacks.

### Level 3 — Operational perception

- Level 2 plus calibrated AI confidence/OOD behavior, operator evidence states,
  human-factors validation, cybersecurity, audit/export integrity, and the
  deployment-domain safety case.

## 11. Standardization path

An industry profile requires more than a repository. The project SHOULD publish:

1. open JSON Schema and DDS/IDL bindings with compatibility rules;
2. calibration registry examples and transform test vectors;
3. legal, synthetic golden recordings and packet captures;
4. injected clock, packet, calibration, frozen-frame, OOD, and sensor-loss faults;
5. conformance tooling and deterministic expected results;
6. at least two independent implementations;
7. cross-vendor plugfests; and
8. a version matrix for every referenced standard.
