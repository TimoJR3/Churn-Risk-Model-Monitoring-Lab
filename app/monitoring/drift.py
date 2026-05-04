from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np


DriftStatus = Literal["stable", "warning", "drift"]
EPSILON = 1e-6


def _clean_numeric_values(values: Sequence[float | int | None]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array[np.isfinite(array)]


def _build_bins(expected: np.ndarray, actual: np.ndarray, buckets: int) -> np.ndarray:
    combined = np.concatenate([expected, actual])
    if combined.size == 0:
        return np.linspace(0.0, 1.0, buckets + 1)

    minimum = float(np.min(combined))
    maximum = float(np.max(combined))
    if minimum == maximum:
        return np.linspace(minimum - 0.5, maximum + 0.5, buckets + 1)

    return np.linspace(minimum, maximum, buckets + 1)


def calculate_psi(
    expected: Sequence[float | int | None],
    actual: Sequence[float | int | None],
    buckets: int = 10,
) -> dict[str, object]:
    """Calculate Population Stability Index for numeric feature values."""
    if buckets < 1:
        raise ValueError("buckets must be greater than or equal to 1")

    expected_values = _clean_numeric_values(expected)
    actual_values = _clean_numeric_values(actual)
    bins = _build_bins(expected_values, actual_values, buckets)

    expected_counts, _ = np.histogram(expected_values, bins=bins)
    actual_counts, _ = np.histogram(actual_values, bins=bins)

    expected_total = max(int(expected_counts.sum()), 1)
    actual_total = max(int(actual_counts.sum()), 1)

    expected_share = np.maximum(expected_counts / expected_total, EPSILON)
    actual_share = np.maximum(actual_counts / actual_total, EPSILON)
    psi_values = (actual_share - expected_share) * np.log(
        actual_share / expected_share
    )

    return {
        "psi": float(np.sum(psi_values)),
        "buckets": buckets,
        "expected_count": int(expected_values.size),
        "actual_count": int(actual_values.size),
    }


def drift_status(psi: float) -> DriftStatus:
    """Map a PSI value to a monitoring status."""
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "warning"
    return "drift"
