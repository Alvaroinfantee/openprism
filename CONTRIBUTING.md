# Contributing to OpenPRISM

OpenPRISM welcomes reproducible bug reports, sensor adapters, validation data,
documentation, and implementation improvements. By participating, you agree to
follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before opening a change

1. Search existing issues and open a focused issue for substantial design work.
2. Never commit proprietary captures, personal location traces, dataset
   archives, credentials, or export-controlled information.
3. State whether data are measured, publisher-declared, simulated, inferred, or
   generated. Do not silently promote a prior to a measurement.
4. Preserve the fail-closed rule: automatic policy code must not bypass timing,
   registration, validity, uncertainty, or provenance gates.

## Development setup

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -p "test*.py" -v
```

The normal unit suite is data-independent. Run dataset integration tests only
with publisher archives staged under `data/`; they are skipped otherwise.

## Pull-request checklist

- Add or update tests for observable behavior.
- Update schemas and documentation when a public contract changes.
- Record user-visible changes in `CHANGELOG.md`.
- Keep policy artifacts versioned, immutable, and accompanied by honest
  training and validation provenance.
- Explain scientific assumptions and expected failure modes in the pull request.
- Confirm that the full data-independent suite passes on a supported Python.

Small fixes may be reviewed by one maintainer. Changes to coordinate frames,
safety gates, schemas, policy artifacts, or research claims require two-person
review when a second maintainer is available.
