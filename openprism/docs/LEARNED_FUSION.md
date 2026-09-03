# PRISM-EGT learned fusion

PRISM-EGT is OpenPRISM's optional Evidence-Gated, Task-Conditioned learned
fusion method. It is an experimental research implementation, not a certified
flight component and not yet a state-of-the-art claim.

## Research hypothesis

Image fusion should be selective rather than compulsory. A learned system
should be able to use complementary thermal evidence, abstain when the pair is
untrustworthy, expose calibrated uncertainty, and preserve the source evidence
needed by downstream models and operators.

Task conditioning, controllable fusion, and uncertainty-aware registration are
already represented in the 2026 literature. The candidate contribution tested
here is their integration into an explicit **selective-risk contract**:

1. the deterministic OpenPRISM clock, registration, validity, and uncertainty
   gates decide whether learned pixel fusion is eligible;
2. a dual encoder represents registered RGB and thermal evidence;
3. an operator task or automatic task distribution conditions the selector;
4. relative Pixhawk/camera motion may condition the selector without exposing
   raw absolute GPS as an image shortcut;
5. reliability, abstention, and predictive-uncertainty heads are retained as
   named machine channels; and
6. the thermal contribution is a convex, evidence-bounded coefficient.

For pixel `p`, the implemented bound is

```text
e(p) = validity(p) * registration(p) * timing(p)
0 <= alpha_T(p) <= e(p)
Y_fused(p) = (1 - alpha_T(p)) * Y_RGB(p) + alpha_T(p) * T(p)
```

The network cannot violate the inequality for any parameter values. When
`e(p)=0`, abstention and predictive uncertainty are forced to one and thermal
contribution is forced to zero. This architectural invariant complements—but
does not replace—calibration and robustness evaluation.

## Install and train

```bash
python -m pip install -e ".[learned]"
openprism-train-egtcf \
  --data-root /path/to/data \
  --output-dir runs/prism-egt-full \
  --epochs 20 \
  --batch-size 12 \
  --patch-size 192
```

Development subsets must be labeled as such:

```bash
openprism-train-egtcf \
  --data-root /path/to/data \
  --output-dir runs/prism-egt-smoke \
  --epochs 1 \
  --max-train-samples 64 \
  --max-validation-samples 32
```

Checkpoints contain their architecture, training provenance, validation scope,
epoch, metrics, and an external SHA-256 identity. Loading uses PyTorch's
`weights_only=True` mode.

## Operator console

Enable a trained checkpoint explicitly:

```bash
openprism \
  --data-root /path/to/data \
  --learned-checkpoint runs/prism-egt-full/best.pt
```

Automatic control then uses PRISM-EGT and shows an `EGT` badge. Manual thermal
control retains the deterministic reference engine. Development checkpoints
remain visibly marked `development-only; not a paper result` in the API.

The learned output adds these named machine channels:

- `learned_visible_reliability`
- `learned_thermal_reliability`
- `learned_abstention_probability`
- `learned_predictive_uncertainty`
- `learned_fused_luminance`

The original visible, thermal, validity, registration, and support evidence is
not replaced.

Full-resolution inference is overlap-tiled by default. Automatic task choice
is made once from a bounded context image and then held fixed across tiles.
Hann-weighted overlap prevents tile seams; the evidence bound is re-applied to
the assembled output. This limits activation memory without silently resizing
the operator or machine product.

## Evaluation discipline

The final test split is locked in the evaluation CLI. It can be opened only
after model selection is frozen:

```bash
openprism-evaluate-egtcf \
  --checkpoint runs/prism-egt-full/best.pt \
  --data-root /path/to/data \
  --partition test \
  --unlock-final-test \
  --output runs/prism-egt-full/final-test.json
```

Proxy fusion losses are useful for optimization but are not scientific ground
truth. A complete evaluation should report downstream person/vehicle
detection, semantic segmentation, calibration/selective risk, registration and
timing corruption stress tests, and latency. A real Pixhawk flight is required
for any claim about aerial map accuracy or operational performance.

The frozen LLVIP person-detection probe applies one unchanged COCO detector to
every view and records its weights checksum:

```bash
python -m pip install -e ".[evaluation]"
openprism-evaluate-llvip-detection \
  --data-root /path/to/data \
  --checkpoint runs/prism-egt-full/best.pt \
  --partition validation \
  --output runs/prism-egt-full/llvip-detection-validation.json
```

Third-party fusion code does not need to be copied into OpenPRISM. Generate
its fused images independently and pass an exact directory keyed by LLVIP
sample ID:

```bash
openprism-evaluate-llvip-detection \
  --data-root /path/to/data \
  --checkpoint runs/prism-egt-full/best.pt \
  --external-fused seafusion=/path/to/seafusion/images
```

Any `--max-samples` run is serialized as development-only. The detector's
visible-domain bias is explicit in every report, and the final test requires
the same one-way unlock as the selective-risk evaluator.

Reviewed external checkouts are executed without vendoring their code:

```bash
python tooling/run_external_fusion.py \
  --baseline seafusion \
  --source-root /path/to/frozen/SeAFusion \
  --weights /path/to/frozen/SeAFusion/model/Fusion/fusionmodel_final.pth \
  --visible-dir /path/to/paired/vi \
  --thermal-dir /path/to/paired/ir \
  --output-dir runs/external/seafusion \
  --expected-revision f217e1cd4b1cda8ca505d70cfb98481fac552408 \
  --expected-weights-sha256 8d65a7f9a50f36349803368cf9920f40420776995e7e8cc8a56d8d30e499431d
```

The runner refuses revision drift, imports only the reviewed model-definition
path, loads checkpoints with `weights_only=True`, preserves source geometry,
writes lossless PNGs, and records the adapter, input IDs, checkpoint, and
per-output hashes. It currently supports
SeaFusion, CDDFuse, PAIF, and C2RF. PAIF's published checkout lacks an `antialias`
helper and imports unrelated segmentation dependencies at module load; the
adapter replaces only those unused paths with fail-closed stubs and records
that fact in its manifest. None of these adapters grants a license to external
code or weights.

C2RF uses its official four-file RoadScene checkpoint and executes both its
registration and fusion subnetworks. Its audited revision constructs CUDA
tensors directly, so this adapter fails clearly on a CPU-only runtime. The
adapter executes the exact three fusion class definitions needed for inference
without instantiating the upstream VGG training-only default object. Multi-file
checkpoint directories receive a deterministic aggregate checksum plus a hash
for every member.

## Current limitations

- The automatic task head uses dataset/task proxies during bootstrap training;
  mission-context labels or downstream differentiable objectives are required
  before claiming reliable automatic task selection.
- The bundled repository does not contain a trained or validated checkpoint.
- The current objective is reference-free. It must be supplemented with frozen
  downstream evaluators and operator studies where human-performance claims are
  made.
- Pose conditioning is implemented as a relative context interface; temporal
  sequence training and real clock characterization remain future experiments.
- Novelty remains a hypothesis until the broader literature audit and all
  experiments are completed and reviewed by the human author.
