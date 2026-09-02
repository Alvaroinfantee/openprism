# PRISM-EGT novelty audit

Audit date: 2026-09-02. Status: targeted first pass, not a systematic review.

The original idea—task-conditioned, controllable, uncertainty-aware
RGB/thermal fusion—is not independently novel in 2026. The following close
works materially narrow the claim:

- [Customized Fusion](https://arxiv.org/abs/2604.08924) studies adaptive
  multi-task-aware infrared/visible fusion.
- [ConFusion](https://arxiv.org/abs/2607.23600) studies fine-grained controllable
  fusion.
- [SFRF](https://arxiv.org/abs/2605.13049) estimates registration uncertainty in
  a joint registration/fusion pipeline.
- [LER-YOLO](https://arxiv.org/abs/2605.20667) uses reliability-aware routing for
  misaligned RGB/infrared UAV detection.
- [PAIF](https://arxiv.org/abs/2308.03979) connects perception-aware fusion and
  attack-tolerant segmentation.
- [SeaFusion](https://doi.org/10.1016/j.inffus.2021.12.004) places fusion inside
  a high-level semantic task loop.

The candidate PRISM-EGT contribution is therefore narrower:

1. a parameter-independent per-pixel upper bound from externally auditable
   validity, registration, and timing evidence;
2. selective-risk evaluation of dense fusion abstention rather than using an
   uncertainty map only as a registration aid;
3. simultaneous delivery of a reversible machine evidence tensor and a
   human-visible convex rendering; and
4. a flight-oriented interface that carries relative pose/time context while
   keeping navigation state estimation and hard eligibility outside the
   network.

These are hypotheses, not priority claims. Before submission, search IEEE
Xplore, ACM Digital Library, Scopus/Web of Science where available, arXiv,
Semantic Scholar, and citation graphs using at least:

- `infrared visible fusion abstention selective risk`
- `multimodal fusion calibrated uncertainty reliability gate`
- `RGB thermal safety constrained fusion`
- `task conditioned controllable infrared visible fusion`
- `UAV RGB thermal fusion pose temporal consistency`
- `human operator uncertainty image fusion`

For every close method, record publication status, code/data license, exact
revision, training data, evaluation splits, tasks, uncertainty definition,
ability to abstain, enforcement versus soft regularization, and compatibility
with paired aerial data. The paper must avoid “first” and “state of the art”
unless this audit and the frozen results actually support those phrases.
