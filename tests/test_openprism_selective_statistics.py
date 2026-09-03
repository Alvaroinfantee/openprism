from __future__ import annotations

import pytest

from openprism.learning.evaluation import grouped_selective_bootstrap


def _group(value: float) -> dict[str, float | None]:
    return {
        "brier": value,
        "expected_calibration_error": value / 2.0,
        "risk_coverage_area": value * 2.0,
        "uncertainty_auroc": None if value == 0.0 else 1.0 - value,
        "mean_target_risk": value,
        "mean_uncertainty": value,
    }


def test_grouped_selective_bootstrap_is_deterministic_and_group_weighted() -> None:
    groups = {"long-sequence": _group(0.2), "short-sequence": _group(0.8)}
    first = grouped_selective_bootstrap(groups, replicates=128, seed=17)
    second = grouped_selective_bootstrap(groups, replicates=128, seed=17)
    assert first == second
    assert first["capture_group_count"] == 2
    assert first["point_estimates"]["brier"] == pytest.approx(0.5)
    interval = first["intervals"]["brier"]
    assert interval is not None
    assert interval["lower"] <= 0.5 <= interval["upper"]


def test_grouped_selective_bootstrap_validates_inputs() -> None:
    with pytest.raises(ValueError, match="at least one"):
        grouped_selective_bootstrap({})
    with pytest.raises(ValueError, match="positive"):
        grouped_selective_bootstrap({"a": _group(0.2)}, replicates=0)
    with pytest.raises(ValueError, match="confidence"):
        grouped_selective_bootstrap({"a": _group(0.2)}, confidence=1.0)
