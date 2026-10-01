"""PSI between the train and validation splits for every numeric model feature.

Baseline check of the split: both parts come from one random split of the same
data, so every feature is expected to be "stable". Run:

    python -m app.monitoring.split_report
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.ml.preprocessing import ARTIFACTS_DIR, NUMERIC_FEATURES, PROCESSED_DATA_DIR
from app.monitoring.drift import calculate_psi

REPORT_PATH = ARTIFACTS_DIR / "psi_train_vs_validation.csv"


def build_split_psi_report(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    buckets: int = 10,
) -> pd.DataFrame:
    rows = []
    for feature in NUMERIC_FEATURES:
        result = calculate_psi(train[feature], validation[feature], buckets=buckets)
        rows.append(
            {
                "feature": feature,
                "psi": round(result["psi"], 4),
                "status": result["status"],
            }
        )
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)


def main(output_path: Path = REPORT_PATH) -> pd.DataFrame:
    train = pd.read_csv(PROCESSED_DATA_DIR / "train_processed.csv")
    validation = pd.read_csv(PROCESSED_DATA_DIR / "validation_processed.csv")
    report = build_split_psi_report(train, validation)
    report.to_csv(output_path, index=False)
    print(report.to_string(index=False))
    print(f"Saved PSI report: {output_path}")
    return report


if __name__ == "__main__":
    main()
