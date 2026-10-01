from __future__ import annotations

import pandas as pd

from app.ml.preprocessing import NUMERIC_FEATURES
from app.monitoring.split_report import build_split_psi_report


def test_split_report_has_one_row_per_numeric_feature() -> None:
    frame = pd.DataFrame({feature: range(100) for feature in NUMERIC_FEATURES})

    report = build_split_psi_report(frame, frame)

    assert set(report["feature"]) == set(NUMERIC_FEATURES)
    assert (report["psi"] == 0).all()
    assert (report["status"] == "stable").all()


def test_split_report_flags_shifted_feature() -> None:
    train = pd.DataFrame({feature: range(100) for feature in NUMERIC_FEATURES})
    shifted = train.copy()
    shifted["monthly_fee"] = shifted["monthly_fee"] + 1000

    report = build_split_psi_report(train, shifted)

    top = report.iloc[0]
    assert top["feature"] == "monthly_fee"
    assert top["status"] == "drift"
