# Frozen PRISM-EGT experiment protocol

Status: implementation-complete protocol draft; final model selection and test
evaluation have not been performed.

## Question

Does evidence-bounded selective fusion improve downstream perception and
selective risk under sensor degradation, compared with single modalities,
deterministic OpenPRISM, and representative learned fusion baselines?

## Partitions

- LLVIP: publisher training split for training. Publisher test sequences `19`,
  `21`, and `23` are validation; `20`, `22`, `24`, and `26` are final test.
  Sequence prefixes are never split across partitions.
- MSRS: publisher training split for training and publisher test split only for
  final testing. The detection subset may be used for training.
- Caltech Aerial RGB-T: complete `scene_group` values are assigned by the first
  32 bits of SHA-256 modulo 10: buckets 0--6 train, 7 validation, and 8--9 test.
  Frames from one flight/scene group never cross partitions.

The executable manifest is produced by
`openprism.learning.data.protocol_manifest`. Current staged counts are:

| Partition | LLVIP | MSRS | Caltech |
|---|---:|---:|---:|
| Train | 12,025 | 1,163 | 1,716 |
| Validation | 1,730 | 0 | 177 |
| Final test | 1,733 | 361 | 389 |

## Baselines

1. RGB only.
2. Thermal only.
3. Pixel average and maximum-luminance fusion.
4. Deterministic OpenPRISM reference fusion.
5. At least three learned methods representing reconstruction, perception-aware
   fusion, and registration-aware fusion. Candidate source revisions and their
   repository-license status are frozen in `baselines.lock.json`; model-weight
   checksums and development executions are frozen; full-validation artifacts
   and environments still need to be frozen.

Items 1--4 are executable in `openprism.learning.baselines`. Item 5 remains a
submission blocker; proxy-loss comparisons are not substitutes for those
external learned baselines. Only SeaFusion had an explicit repository license
at the audited revision. CDDFuse, PAIF, and C2RF must remain external and must
not be copied into the project unless their copyright holders provide a
license.

Reviewed SeaFusion, CDDFuse, PAIF, and C2RF source checkouts can be invoked by
`tooling/run_external_fusion.py`. The adapter verifies the exact Git revision,
loads only the reviewed model-definition path, uses safe weights-only
deserialization, and writes per-output hashes plus a run manifest. The
external repository's test script is not executed. This adapter does not grant
or imply a license to third-party code or weights. The PAIF adapter records
its fail-closed shims for two unused dependency paths that are unavailable in
the published checkout/runtime. The C2RF adapter executes the complete
registration-plus-fusion inference path from its four-file RoadScene checkpoint
and records its CUDA-only constraint.

## Ablations

- no task conditioning;
- no learned abstention;
- uncertainty head without calibration loss;
- no hidden-corruption training;
- evidence envelope replaced by a learned soft gate;
- no pose/time context for sequence experiments; and
- automatic task selection versus operator/mission task input.

## Outcomes

Primary outcomes:

- LLVIP person-detection AP, AP50, and miss rate using a frozen evaluator;
- MSRS semantic mIoU and per-class IoU using a frozen evaluator;
- Caltech terrain mIoU and person/vehicle metrics where labels support them;
- risk--coverage area, Brier score, expected calibration error, and AUROC for
  abstention under held-out degradations; and
- maximum observed violation of `thermal_contribution <= evidence_support`
  (required value: zero within numerical tolerance).

Secondary outcomes include SSIM, mutual information, entropy, gradient
retention, LPIPS where license-compatible, parameters, memory, and throughput.
No-reference fusion metrics are never the sole evidence for superiority.

The LLVIP detection probe is executable as
`openprism-evaluate-llvip-detection`. It uses the same frozen TorchVision
Faster R-CNN ResNet-50-FPN-v2 COCO checkpoint for every view, records the
weight checksum, computes AP@[.50:.95], AP50, and log-average miss rate, and
accepts externally generated fused-image directories without vendoring their
implementations. Its visible-domain bias is a declared limitation; it is one
downstream probe, not a universal perception evaluator. PRISM-EGT's
human-facing color rendering and machine-facing fused luminance are reported
separately rather than selecting whichever looks better after evaluation.

## Stress conditions

- spatial translation, local warp, calibration drift, and dropped regions;
- thermal noise, clipping, nonuniformity, and dead pixels;
- visible low light, blur, glare, and compression;
- paired timestamp offsets and uncertainty-budget sweeps; and
- combinations absent from training corruptions.

## Statistical plan

Report per-scene results and scene-group bootstrap 95% confidence intervals.
Use paired comparisons on identical frames/groups. Report all predefined
primary outcomes, seed-level results, failures, and negative results. Do not
select baselines or metrics after opening the final test partition.

## Test lock

The evaluation command refuses `--partition test` unless
`--unlock-final-test` is passed. That flag should be used once, after the model,
hyperparameters, baselines, evaluator revisions, and analysis script are
committed and tagged.
