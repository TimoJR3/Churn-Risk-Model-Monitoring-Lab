from __future__ import annotations

from collections.abc import Sequence

import numpy as np

EPSILON = 1e-6


def _clean_numeric(values: Sequence[float | int | None]) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    return data[np.isfinite(data)]


def _build_bucket_edges(
    expected: np.ndarray,
    actual: np.ndarray,
    buckets: int,
) -> np.ndarray:
    quantiles = np.linspace(0, 1, buckets + 1)
    edges = np.unique(np.quantile(expected, quantiles))

    if len(edges) < 2:
        combined = np.concatenate([expected, actual])
        minimum = float(np.min(combined))
        maximum = float(np.max(combined))
        if minimum == maximum:
            edges = np.array([minimum - 0.5, maximum + 0.5])
        else:
            edges = np.linspace(minimum, maximum, buckets + 1)

    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def drift_status(psi: float) -> str:
    if psi < 0.1:
        return "stable"
    if psi < 0.25:
        return "warning"
    return "drift"


def calculate_psi(
    expected: Sequence[float | int | None],
    actual: Sequence[float | int | None],
    buckets: int = 10,
) -> dict[str, object]:
    if buckets < 1:
        raise ValueError("buckets must be greater than or equal to 1.")

    expected_clean = _clean_numeric(expected)
    actual_clean = _clean_numeric(actual)

    if len(expected_clean) == 0 or len(actual_clean) == 0:
        return {
            "psi": 0.0,
            "status": "stable",
            "bucket_count": 0,
            "expected_count": int(len(expected_clean)),
            "actual_count": int(len(actual_clean)),
        }

    edges = _build_bucket_edges(expected_clean, actual_clean, buckets)
    expected_counts, _ = np.histogram(expected_clean, bins=edges)
    actual_counts, _ = np.histogram(actual_clean, bins=edges)

    expected_share = expected_counts / max(len(expected_clean), 1)
    actual_share = actual_counts / max(len(actual_clean), 1)
    expected_share = np.maximum(expected_share, EPSILON)
    actual_share = np.maximum(actual_share, EPSILON)

    psi_values = (actual_share - expected_share) * np.log(
        actual_share / expected_share
    )
    psi = max(float(np.sum(psi_values)), 0.0)

    return {
        "psi": psi,
        "status": drift_status(psi),
        "bucket_count": int(len(edges) - 1),
        "expected_count": int(len(expected_clean)),
        "actual_count": int(len(actual_clean)),
    }
