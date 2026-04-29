from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.ml.data_loader import load_churn_data
from app.ml.features import add_feature_engineering


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TARGET_COLUMN = "churn"
DROP_COLUMNS = ["user_id", "signup_date"]

NUMERIC_FEATURES = [
    "monthly_fee",
    "days_active_last_30",
    "sessions_last_30",
    "support_tickets_last_30",
    "payments_failed_last_90",
    "avg_session_duration",
    "feature_usage_score",
    "last_login_days_ago",
    "activity_score",
    "payment_risk_score",
    "days_since_signup",
    "usage_per_session",
    "support_intensity",
]

CATEGORICAL_FEATURES = [
    "country",
    "plan_type",
    "engagement_level",
]


def build_preprocessing_pipeline() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_train_validation_split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN].astype(int)

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )


def transform_to_dataframe(
    transformer: ColumnTransformer,
    data: pd.DataFrame,
) -> pd.DataFrame:
    transformed = transformer.transform(data)
    columns = transformer.get_feature_names_out()
    return pd.DataFrame(transformed, columns=columns, index=data.index)


def prepare_processed_datasets(
    source: str = "csv",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Path]:
    data = load_churn_data(source=source)
    data = add_feature_engineering(data)

    train_x, valid_x, train_y, valid_y = make_train_validation_split(
        data,
        test_size=test_size,
        random_state=random_state,
    )

    train_x = train_x.drop(columns=DROP_COLUMNS)
    valid_x = valid_x.drop(columns=DROP_COLUMNS)

    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(train_x)

    train_processed = transform_to_dataframe(preprocessor, train_x)
    valid_processed = transform_to_dataframe(preprocessor, valid_x)
    train_processed[TARGET_COLUMN] = train_y.to_numpy()
    valid_processed[TARGET_COLUMN] = valid_y.to_numpy()

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": PROCESSED_DATA_DIR / "train_processed.csv",
        "validation": PROCESSED_DATA_DIR / "validation_processed.csv",
        "preprocessor": ARTIFACTS_DIR / "preprocessor.joblib",
    }

    train_processed.to_csv(paths["train"], index=False)
    valid_processed.to_csv(paths["validation"], index=False)
    joblib.dump(preprocessor, paths["preprocessor"])

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare processed churn datasets."
    )
    parser.add_argument("--source", choices=["csv", "postgres"], default="csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = prepare_processed_datasets(
        source=args.source,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Train dataset: {paths['train']}")
    print(f"Validation dataset: {paths['validation']}")
    print(f"Preprocessor artifact: {paths['preprocessor']}")


if __name__ == "__main__":
    main()
