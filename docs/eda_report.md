# EDA Report

## Dataset Overview

- Rows: `2000`
- Columns: `13`
- Target: `churn`
- Churn rate: `0.117`

## Missing Values

| index | missing_count |
| --- | --- |
| user_id | 0 |
| signup_date | 0 |
| country | 0 |
| plan_type | 0 |
| monthly_fee | 0 |
| days_active_last_30 | 0 |
| sessions_last_30 | 0 |
| support_tickets_last_30 | 86 |
| payments_failed_last_90 | 0 |
| avg_session_duration | 85 |
| feature_usage_score | 76 |
| last_login_days_ago | 0 |
| churn | 0 |

Missing values appear in behavioral features. This is realistic for
event data: some session, support, or usage events can be absent.
The preprocessing pipeline handles them with median or most frequent
imputation.

## Outliers

| index | outlier_count |
| --- | --- |
| monthly_fee | 296 |
| sessions_last_30 | 40 |
| payments_failed_last_90 | 25 |
| avg_session_duration | 24 |
| last_login_days_ago | 14 |
| support_tickets_last_30 | 3 |
| feature_usage_score | 1 |
| user_id | 0 |
| days_active_last_30 | 0 |

Outliers are expected mostly in `sessions_last_30` and
`avg_session_duration`. They represent power users, bots, or noisy
event tracking. They are not removed at this stage, so the future
baseline model can see realistic edge cases.

## Numeric Distributions

| index | mean | std | min | 25% | 50% | 75% | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| user_id | 1000.5 | 577.49 | 1.0 | 500.75 | 1000.5 | 1500.25 | 2000.0 |
| monthly_fee | 18.13 | 10.43 | 4.99 | 10.04 | 16.8 | 20.84 | 44.76 |
| days_active_last_30 | 13.03 | 6.51 | 0.0 | 8.0 | 13.0 | 18.0 | 30.0 |
| sessions_last_30 | 30.11 | 22.86 | 0.0 | 16.0 | 26.0 | 40.0 | 252.0 |
| support_tickets_last_30 | 1.04 | 1.06 | 0.0 | 0.0 | 1.0 | 2.0 | 6.0 |
| payments_failed_last_90 | 0.43 | 0.67 | 0.0 | 0.0 | 0.0 | 1.0 | 6.0 |
| avg_session_duration | 35.69 | 28.67 | 1.0 | 25.99 | 33.29 | 40.85 | 416.29 |
| feature_usage_score | 41.02 | 20.44 | 0.0 | 25.68 | 40.92 | 55.27 | 100.0 |
| last_login_days_ago | 12.19 | 4.63 | 2.0 | 9.0 | 12.0 | 15.0 | 27.0 |
| churn | 0.12 | 0.32 | 0.0 | 0.0 | 0.0 | 0.0 | 1.0 |

## Categorical Distributions

| index | count | unique | top | freq |
| --- | --- | --- | --- | --- |
| country | 2000 | 8 | US | 484 |
| plan_type | 2000 | 3 | basic | 988 |

## Feature Relationship With Churn

| index | 0 | 1 |
| --- | --- | --- |
| days_active_last_30 | 13.92 | 6.23 |
| sessions_last_30 | 32.22 | 14.12 |
| support_tickets_last_30 | 0.98 | 1.44 |
| payments_failed_last_90 | 0.39 | 0.72 |
| avg_session_duration | 36.62 | 28.67 |
| feature_usage_score | 43.86 | 19.17 |
| last_login_days_ago | 11.75 | 15.55 |
| activity_score | 41.8 | 18.76 |
| payment_risk_score | 13.0 | 23.61 |
| days_since_signup | 460.81 | 491.55 |
| usage_per_session | 1.66 | 1.91 |
| support_intensity | 0.06 | 0.26 |

Churned users are less active on average, use the product less,
log in less recently, and have more failed payments. This matches
subscription-product logic: low engagement and payment friction often
come before churn.

## Churn By Plan

| plan_type | count | mean |
| --- | --- | --- |
| basic | 988 | 0.134 |
| standard | 709 | 0.1 |
| premium | 303 | 0.099 |

## Churn By Country

| country | count | mean |
| --- | --- | --- |
| FR | 260 | 0.142 |
| DE | 267 | 0.139 |
| NL | 153 | 0.124 |
| BR | 202 | 0.114 |
| US | 484 | 0.114 |
| IN | 301 | 0.113 |
| ES | 178 | 0.084 |
| PL | 155 | 0.084 |

## Leakage Check

Potential leakage features:

- `churn` must not be used as a feature because it is the target.
- `user_id` has no business signal and can let a model memorize rows.
- Features created after the churn date must not be added to training.
- `model_predictions`, `model_metrics`, and `drift_metrics` must not
  be used for training because they are created after model inference.

The preprocessing step drops `user_id` and `signup_date`. Instead of
the raw signup date, it uses the engineered `days_since_signup` feature.
