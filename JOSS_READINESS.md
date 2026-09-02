# JOSS readiness dossier

Status on 2026-09-02: **public pre-JOSS release launched; not yet eligible for
JOSS submission**.

Current guidance: <https://joss.readthedocs.io/en/latest/submitting.html> and
<https://joss.readthedocs.io/en/latest/paper.html>.

JOSS can evaluate research software, but it does not guarantee acceptance. Its
current pre-submission rules require a public repository older than six months,
active iterative public development over that period, and demonstrated research
use or impact. A newly published repository cannot truthfully satisfy those
criteria on day one.

## Implemented readiness signals

- OSI-approved MIT license draft.
- Installable Python package and command-line entry point.
- Data-independent automated tests plus opt-in archive integration tests.
- CI across supported Python versions and a paper-format check.
- User, architecture, standards, mapping, policy, and validation documentation.
- Contribution, conduct, governance, support, security, and changelog files.
- Versioned schemas and policy artifact with provenance and SHA-256 identity.
- JOSS-format `paper/paper.md` and `paper/paper.bib`, including AI disclosure.
- Honest limitations and dataset licensing boundaries.
- Public, cloneable GitHub repository with Issues enabled.

## Next repository setup actions

- Confirm the final author list and complete human manuscript review before
  submission.
- Have every named author review the complete repository, policy claims, and
  paper; record approval.
- Enable Discussions and private vulnerability reporting if those support
  channels will be offered.
- Archive the tagged release with Zenodo or another long-term repository when
  appropriate.

## Evidence to accumulate before JOSS submission

1. Maintain the repository publicly for more than six months with real,
   reviewable commits, issues, releases, and responses to external users.
2. Obtain documented use in at least one genuine research workflow; publish a
   reproducible example, report, preprint, thesis, or independent citation.
3. Run task-level evaluation on held-out scenes and report per-condition results.
4. Run a preregistered human-operator study if usability or performance claims
   are retained.
5. Validate a real calibrated flight with traceable camera/Pixhawk timing,
   ground-control or checkpoints, and horizontal/vertical error metrics.
6. Add external contributors or explain the governance/support model with
   evidence that it works.
7. Freeze a reviewed release, archive it, add the DOI to `CITATION.cff`, and
   submit only after rechecking the then-current JOSS rules.

## Claim discipline

The software may be innovative without asserting that it is proven optimal,
certified, or an industry standard. The bundled automatic policy is expert
initialized and not fitted. The Atlas is a live evidence orthodrape; terrain
geometry reconstruction belongs to the SfM/MVS survey path. These distinctions
must remain in the public paper and documentation until evidence supports
stronger claims.
