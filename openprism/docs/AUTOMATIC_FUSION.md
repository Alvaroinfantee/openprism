# Automatic fusion and AI scene context

## Design objective

OpenPRISM makes one scene consumable in two forms without turning either form
into hidden truth:

1. a named `float32 [C,H,W]` evidence tensor for models; and
2. a deterministic fused canvas for human operators.

The automatic controller configures the human projection. It does not replace
source imagery, synthesize pixels, or certify perception accuracy.

## Decision path

```text
PrismFrame
  -> neutral fusion probe (gain 1.0)
  -> 10 bounded scene features
  -> hard evidence gate -----------------> Integrity + gain 0 (if unsafe)
  -> versioned linear policy
  -> preset probabilities + bounded gain
  -> operator canvas + machine-readable digest
```

The features summarize visible luminance/detail, thermal detail/saliency,
registration and fusion support, joint validity, day/night evidence, and the
presence of supplied detections or semantics. The policy is a small matrix and
bias vector stored in `policy/fusion_policy_v1.json`; its SHA-256 is emitted
with every decision.

The v1 policy is `expert_initialized_not_fitted`. It exists to make the control
surface reproducible and testable, not to claim learned optimality. A future
fitted policy must use a new model identifier and publish its training data,
loss, splits, calibration, subgroup results, and failure analysis.

## Safety precedence

Before the model can influence a view, the controller checks whether pixel
fusion was applied by the core engine and whether mean registration support and
valid coverage clear artifact-defined minima. The fusion engine independently
abstains when clocks, skew, measurement uncertainty, registration, or source
validity cannot support blending. An automatic decision can reduce evidence but
cannot turn an ineligible frame into an eligible one.

The operator can turn automatic control off. The policy then remains visible as
advice while the gain slider is editable. Integrity is never selected through
generative repair: unsupported thermal contribution is exactly zero.

## AI scene digest

`GET /api/ai/context` returns an `openprism.ai-context-envelope/1.0` object with
an `openprism.ai-scene-digest/1.0` payload. It intentionally omits base64 images
and includes:

- frame and scene summary;
- applied and recommended controls;
- all feature values and recommendation rationale;
- machine tensor shape, dtype, ordered channel names, and p05/p50/p95/mean;
- detection, semantics, synchronization, and pixel-fusion status;
- policy identity, training provenance, validation scope, and artifact hash;
- explicit statements that hard gates cannot be overridden, no pixels were
  generated, operator override exists, and the decision is not a certificate.

Consumers must validate the JSON against
`spec/ai-scene-digest.schema.json`, select tensor channels by name, and treat
unknown/missing values as unknown rather than fabricating defaults.

## Replacing the policy

Construct `FusionPolicyModel.from_json(path)` and inject it into
`AdaptiveFusionController`. The loader rejects non-finite values, duplicate
features, missing presets, invalid thresholds, and out-of-range gain bounds.
The controller rejects feature names it cannot compute. Artifact changes must
add tests and a changelog entry; modifying v1 in place after release is not
allowed.

## Evaluation protocol

Control quality needs task- and operator-specific ground truth. At minimum,
evaluate:

1. safety invariants under missing, late, incompatible-clock, invalid, and
   misregistered inputs;
2. deterministic replay and artifact-hash agreement;
3. downstream task deltas for detection/segmentation/tracking, reported by
   dataset and condition rather than only as an aggregate;
4. operator search time, errors, confidence calibration, and workload in a
   preregistered human-subject protocol; and
5. robustness to unseen sensors, radiometric ranges, weather, altitude, and
   geographic domain shift.

Until those experiments exist, describe v1 as an explainable reference policy,
not as a universally optimal controller.
