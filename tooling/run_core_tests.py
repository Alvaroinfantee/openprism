"""Run the dependency-light OpenPRISM test suite."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CORE_TEST_MODULES = (
    "tests.test_external_validation_matrix",
    "tests.test_openprism",
    "tests.test_openprism_atlas",
    "tests.test_openprism_atlas_demo",
    "tests.test_openprism_atlas_ui",
    "tests.test_openprism_autonomy",
    "tests.test_openprism_mapping",
    "tests.test_openprism_pixhawk",
    "tests.test_openprism_survey",
)


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromNames(CORE_TEST_MODULES)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
