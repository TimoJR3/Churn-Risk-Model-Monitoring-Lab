# Model Card: Churn Risk Baseline

## Model Purpose

This model estimates churn probability for a synthetic subscription-product
user. It is used in this repository as a reproducible churn modeling and model
monitoring demo for DS portfolio review.

The score should be interpreted as a risk signal for analysis and retention
prioritization, not as an automated business decision.

## Target

- Target column: `churn`
- Type: binary classification
- Meaning: `1` means the user churned, `0` means the user did not churn

## Input Features

Raw input fields:

- `signup_date`
- `country`
- `plan_type`
- `monthly_fee`
- `days_active_last_30`
- `sessions_last_30`
- `support_tickets_last_30`
- `payments_failed_last_90`
- `avg_session_duration`
- `feature_usage_score`
- `last_login_days_ago`

`user_id` may be accepted by the API for demo logging, but it is not used as a
model feature.

Engineered features:

- `activity_score`
- `payment_risk_score`
- `engagement_level`
- `days_since_signup`
- `usage_per_session`
- `support_intensity`

## Modeling Approach

Candidate models:

- Logistic Regression with balanced class weights
- Random Forest with balanced class weights
- HistGradientBoostingClassifier

Preprocessing:

- numeric median imputation;
- numeric standard scaling;
- categorical most-frequent imputation;
- one-hot encoding for categorical features;
- `user_id` and `signup_date` are dropped before model fitting.

## Validation Method

The project uses:

- stratified train/validation split, default `test_size=0.2`;
- `random_state=42` for reproducibility;
- `StratifiedKFold` cross-validation on the training split;
- ROC-AUC as the primary model-selection metric;
- F1 as a secondary tie-breaker.

## Current Validation Metrics

Current saved artifact metrics:

- Best model: `random_forest`
- ROC-AUC: `0.8408`
- F1: `0.5072`
- Precision: `0.3846`
- Recall: `0.7447`
- Confusion matrix: `[[297, 56], [12, 35]]`

These numbers come from synthetic data and should not be treated as business
benchmarks.

## Inference

The trained model is saved to `artifacts/trained_model.pkl`; the preprocessing
pipeline is saved to `artifacts/preprocessor.pkl`. FastAPI endpoints load these
artifacts with caching and do not retrain during requests.

The default classification threshold is `0.5`, configurable through
`PREDICTION_THRESHOLD`.

Risk bands:

- `low`: probability `< 0.35`
- `medium`: `0.35 <= probability < 0.65`
- `high`: probability `>= 0.65`

## Monitoring Signals

Implemented monitoring signals:

- prediction count;
- average churn probability;
- high-risk share;
- risk-band distribution;
- PSI drift check for a numeric feature;
- optional quality metrics from labels and scores.

PSI thresholds used in the demo:

- `stable`: PSI `< 0.1`
- `warning`: `0.1 <= PSI < 0.25`
- `drift`: PSI `>= 0.25`

## Known Limitations

- Synthetic data only; no real customer behavior is modeled.
- No temporal split; the dataset is a generated snapshot.
- Threshold is not optimized against business costs.
- No probability calibration.
- Monitoring is request-driven and simplified.
- No model registry, scheduled retraining, or alert routing.
- Dashboard is a local demo, not an authenticated operational console.

## Ethical and Product Risks

- Churn scores can be misused if treated as a final decision instead of a risk
  signal.
- Retention actions should be reviewed for fairness and user experience.
- Synthetic data does not capture real demographic, behavioral, or market
  biases.
- If adapted to real data, raw personal data should not be logged, and feature
  selection should be reviewed for privacy and fairness concerns.
