# Monitoring

The monitoring layer is intentionally small and testable. It exposes pure
functions in `app/monitoring` and API endpoints in `app/api/routers`.

## Prediction Summary

`GET /monitoring/summary` reads recent prediction logs and returns:

- `total_predictions`
- `average_probability`
- `high_risk_share`
- `risk_band_counts`

This is useful for a quick operational view of how many users are being scored
and whether the score distribution is shifting toward high risk.

## Population Stability Index

`POST /monitoring/drift` calculates Population Stability Index (PSI) for one
numeric feature.

PSI compares the distribution of expected values with actual values:

- very small PSI means the distributions are similar;
- larger PSI suggests feature distribution drift;
- PSI is univariate and should be interpreted with context.

Status thresholds:

| PSI | Status |
| --- | --- |
| `< 0.1` | `stable` |
| `0.1 <= PSI < 0.25` | `warning` |
| `>= 0.25` | `drift` |

The implementation handles NaN values, constant arrays, and zero bucket counts
with a small epsilon.

## Quality Metrics

`POST /monitoring/quality` accepts labels and prediction scores and returns:

- ROC-AUC;
- precision;
- recall;
- F1;
- sample count;
- positive count.

If labels are missing or `y_true` contains a single class, the endpoint returns
`roc_auc: null` instead of failing.

## Limitations

- Monitoring is request-driven; there is no scheduler yet.
- PSI is calculated per feature and does not detect multivariate drift.
- There is no alert routing.
- No production-grade model registry is included.
- The dataset is synthetic, so thresholds are illustrative.
