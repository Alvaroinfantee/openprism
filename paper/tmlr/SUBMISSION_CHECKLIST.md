# TMLR submission checklist

This repository is not ready to submit merely because the model executes. The
following items are blocking scientific requirements.

- [x] Optional learned model, checkpoint schema, runtime adapter, and hard-gate
  invariant implemented.
- [x] Leakage-aware split protocol and a locked final-test command implemented.
- [x] Development smoke training and evaluation completed without claiming a
  paper result.
- [x] Current TMLR template vendored from the official style repository under
  its Apache-2.0 license.
- [x] Structural manuscript validator added; submission mode fails while
  placeholders, unchecked items, or the rendered PDF remain.
- [ ] Human author independently reviews, understands, and approves the method,
  scientific claims, code, and AI-use disclosure.
- [ ] Complete systematic literature audit; freeze at least three executable
  learned baselines and exact revisions.
- [ ] Add frozen downstream detection and segmentation evaluators.
- [ ] Run multi-seed full-protocol training and validation.
- [ ] Freeze the model and analysis plan in a tagged commit.
- [ ] Unlock the final test exactly once and populate every table.
- [ ] Add scene-group confidence intervals, ablations, latency, failure cases,
  and model/data cards.
- [ ] Run a calibrated Pixhawk flight before making mapping/flight claims.
- [ ] Obtain ethics review before any human-operator experiment, if required by
  the applicable institution/jurisdiction.
- [ ] Keep the submitted PDF anonymous and all OpenReview profiles complete.
- [ ] Confirm that the main body length is justified; staying at or below 12
  pages supports TMLR's normal review target but does not guarantee timing.
- [ ] Export an anonymized code supplement without third-party datasets,
  personal paths, identities, or hidden repository metadata.
- [ ] Build the PDF with the unmodified official `tmlr.sty` and inspect every
  page visually.

Submission must not occur while result cells contain `TBD` or while the
manuscript calls development-subset results scientific evidence.
