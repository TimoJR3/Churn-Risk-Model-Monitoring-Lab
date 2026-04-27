INSERT INTO model_metrics (
    model_version,
    metric_date,
    roc_auc,
    f1_score,
    precision_score,
    recall_score
)
VALUES
    ('baseline_pending', CURRENT_DATE, NULL, NULL, NULL, NULL)
ON CONFLICT (model_version, metric_date) DO NOTHING;

INSERT INTO drift_metrics (
    model_version,
    metric_date,
    feature_name,
    drift_score,
    drift_detected
)
VALUES
    ('baseline_pending', CURRENT_DATE, 'days_active_last_30', 0.00000, false),
    ('baseline_pending', CURRENT_DATE, 'payments_failed_last_90', 0.00000, false),
    ('baseline_pending', CURRENT_DATE, 'feature_usage_score', 0.00000, false)
ON CONFLICT (model_version, metric_date, feature_name) DO NOTHING;
