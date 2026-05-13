# Interview Notes

## 60-Second Pitch

This is a reproducible churn prediction and model monitoring demo. I generated
a synthetic subscription dataset, built feature engineering and preprocessing
outside notebooks, compared baseline scikit-learn models with stratified
cross-validation, saved model artifacts, and exposed online and batch scoring
through FastAPI. Predictions can be logged to PostgreSQL without storing raw
`user_id`, and the monitoring layer includes prediction summaries, quality
metrics, and PSI drift checks. A Streamlit dashboard shows the model, scoring
results, and monitoring outputs.

## How to Explain Churn

Churn is a binary classification problem where the target is whether a user
leaves the product. The business goal is to identify users with elevated risk
early enough to prioritize retention actions.

Important points:

- Churn is often imbalanced, so accuracy can be misleading.
- Recall matters if missing an at-risk user is costly.
- Precision matters if retention actions are expensive or intrusive.
- The threshold should eventually be optimized against business costs.
- A churn probability is a risk estimate, not a decision by itself.

## How to Explain PSI

Population Stability Index compares an expected numeric distribution with a
current distribution. It bins values, compares the share of observations per
bin, and sums the difference weighted by the log ratio.

In this project:

- PSI `< 0.1` means stable;
- `0.1 <= PSI < 0.25` means warning;
- PSI `>= 0.25` means drift.

PSI is easy to explain and useful for quick monitoring, but it is univariate.
It does not prove model degradation and should be combined with quality metrics
when labels become available.

## Questions an Interviewer Might Ask

1. Why did you use ROC-AUC instead of accuracy?
   Churn can be imbalanced. ROC-AUC evaluates ranking quality across thresholds,
   while accuracy can hide poor recall on churn users.

2. Why use stratified split and StratifiedKFold?
   To preserve the churn/no-churn class ratio in train, validation, and CV
   folds.

3. How would you choose the threshold?
   I would estimate the business cost of false positives and false negatives,
   then optimize the threshold using validation data and possibly calibration.

4. What happens when artifacts are missing?
   The API returns a controlled 503 with the training command instead of
   retraining during the request.

5. How is privacy handled in logs?
   Raw `user_id` is removed from stored input features and only a SHA-256 hash
   is saved.

6. How would you detect model degradation?
   I would monitor score distributions, PSI on important features, input data
   quality, and delayed label-based metrics such as ROC-AUC, precision, recall,
   and F1.

## Limitations to Admit Honestly

- The dataset is synthetic.
- There is no real production traffic.
- There is no temporal split or out-of-time validation.
- The threshold is fixed at `0.5` and not cost-optimized.
- The model is not calibrated.
- PSI is a simple per-feature drift demo.
- There is no scheduled monitoring, alerting, or model registry.

## How to Develop It Further

- Add probability calibration.
- Add threshold optimization using a cost matrix.
- Add time-based validation once time-series data exists.
- Track feature-level PSI for top features from feature importance.
- Add scheduled monitoring reports.
- Add model registry metadata and experiment tracking.
- Compare with gradient boosting libraries after the baseline is stable.
- Add a small label-feedback loop for post-deployment quality metrics.
