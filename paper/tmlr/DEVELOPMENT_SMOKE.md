# Development smoke run

Date: 2026-09-02. Scientific status: **not a paper result**.

The local GPU smoke run used 64 sampled training patches, 32 validation
patches, one epoch, 64-pixel crops, and an eight-channel base model. It exists
only to verify data loading, optimization, checkpoint round-tripping, runtime
integration, and metric serialization.

- checkpoint SHA-256:
  `8b175fa028c94db4a22a1a042a24cb5796c6fc6a221443a5d1426d040c7654bd`
- maximum observed evidence-bound violation: `0.0`
- training examples: `64`
- validation examples: `32`
- development validation loss: `0.7228109688`

The run's automatic task accuracy was not useful and must not be reported as a
positive result. Its checkpoint remains under the git-ignored `runs/`
directory and is visibly labeled `development-only; not a paper result` in the
operator API. Full training, learned baselines, downstream evaluators,
multi-seed analysis, and the locked final test remain outstanding.

## Larger development audit

A second non-paper run used 512 training patches, 128 validation patches,
three epochs, 96-pixel crops, and 16 base channels (63,288 parameters).
Validation loss decreased monotonically from `0.53` to `0.34`, sampled task
proxy accuracy reached `0.90`, synthetic-corruption uncertainty AUROC was
`0.819`, and the maximum evidence-bound violation remained `0.0`.

The task value may reflect dataset-domain recognition rather than mission
understanding. Moreover, deterministic OpenPRISM still had a lower
task-intensity proxy loss (`0.00104`) than PRISM-EGT (`0.00173`) on the sampled
clean validation set. This is a useful negative result: the learned model has
not yet demonstrated superiority, and downstream tasks—not optimization proxy
scores—must decide whether it earns its added complexity.

External baseline revisions have now been recorded in
`baselines.lock.json`. Their pretrained-weight checksums and executable
environments have not been frozen, so none is recorded as an executed result.

## Frozen-detector development probe

An eight-image LLVIP validation smoke test exercised the downstream evaluator
using one unchanged TorchVision Faster R-CNN ResNet-50-FPN-v2 COCO checkpoint
(`dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf`).
The report checksum is
`409214940f330283eafd2d2142a4f035f9e8767184daea58aae27344aa1bd5e2`.

| View | AP@[.50:.95] | AP50 | log-average miss rate |
|---|---:|---:|---:|
| Visible RGB | 0.555 | 0.894 | 0.276 |
| Thermal grayscale | 0.573 | 0.842 | 0.287 |
| Average | **0.623** | 0.873 | 0.223 |
| Maximum | 0.535 | 0.805 | 0.382 |
| Deterministic OpenPRISM operator view | 0.539 | 0.788 | 0.336 |
| CDDFuse (frozen external adapter) | 0.542 | 0.800 | 0.348 |
| PAIF (frozen external adapter, without AAT) | 0.558 | 0.800 | 0.200 |
| SeaFusion (frozen external adapter) | 0.608 | 0.858 | 0.212 |
| C2RF (frozen registration + fusion adapter) | **0.614** | **0.900** | **0.100** |
| PRISM-EGT operator view | 0.500 | 0.834 | 0.194 |
| PRISM-EGT machine luminance | 0.537 | 0.791 | 0.325 |

This is explicitly not a paper result: there are only eight images and 20
ground-truth boxes. Average fusion had the highest AP, while PRISM-EGT's
operator view had the lowest miss rate among OpenPRISM variants; C2RF was the
strongest external method on all three detector metrics. PAIF nearly matched
PRISM-EGT's miss rate but did not exceed visible RGB in AP. The machine
luminance did not improve the probe. The mixed outcome rules out a blanket
superiority claim and supports the predefined multi-metric, full-validation
experiment.

SeaFusion, CDDFuse, PAIF, and C2RF were run from the exact revisions in
`baselines.lock.json` through the reviewed external adapter. The adapter
manifests record weight and output hashes. CDDFuse required a documented
checkpoint-key compatibility path because its bundled weight dictionary uses
`DIDF_Encoder`/`DIDF_Decoder`, while the current upstream test script requests
`CDDF_Encoder`/`CDDF_Decoder`. PAIF required fail-closed stubs for unused
imports because the audited repository omits `antialias` and eagerly imports
its unrelated segmentation stack.
C2RF used the complete four-file RoadScene checkpoint and both its registration
and fusion paths; its audited code is CUDA-only and emits framework-version
compatibility warnings documented in the baseline lock.

The combined external-baseline detector report has SHA-256
`088aa375b3906c116d0e1283f20b6426369300ad0709e17b25dd3d35d31498e2`.
