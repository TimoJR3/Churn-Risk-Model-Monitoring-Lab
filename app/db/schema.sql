CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    signup_date DATE NOT NULL,
    country VARCHAR(64) NOT NULL,
    plan_type VARCHAR(32) NOT NULL,
    monthly_fee NUMERIC(8, 2) NOT NULL,
    churn BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_features (
    feature_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    snapshot_date DATE NOT NULL,
    days_active_last_30 INTEGER,
    sessions_last_30 INTEGER,
    support_tickets_last_30 INTEGER,
    payments_failed_last_90 INTEGER,
    avg_session_duration NUMERIC(10, 2),
    feature_usage_score NUMERIC(5, 2),
    last_login_days_ago INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, snapshot_date)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    prediction_id BIGSERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id),
    model_version VARCHAR(64) NOT NULL,
    prediction_date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    churn_probability NUMERIC(6, 5) NOT NULL,
    predicted_churn BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction_logs (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL,
    user_id_hash TEXT,
    churn_probability NUMERIC NOT NULL,
    churn_prediction INTEGER NOT NULL,
    risk_band TEXT NOT NULL,
    threshold NUMERIC NOT NULL,
    model_version TEXT,
    input_features JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(64) NOT NULL,
    metric_date DATE NOT NULL,
    roc_auc NUMERIC(6, 5),
    f1_score NUMERIC(6, 5),
    precision_score NUMERIC(6, 5),
    recall_score NUMERIC(6, 5),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_version, metric_date)
);

CREATE TABLE IF NOT EXISTS drift_metrics (
    drift_id BIGSERIAL PRIMARY KEY,
    model_version VARCHAR(64) NOT NULL,
    metric_date DATE NOT NULL,
    feature_name VARCHAR(128) NOT NULL,
    drift_score NUMERIC(10, 5) NOT NULL,
    drift_detected BOOLEAN NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (model_version, metric_date, feature_name)
);
