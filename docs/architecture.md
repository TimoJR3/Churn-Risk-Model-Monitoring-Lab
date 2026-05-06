# Architecture

This project is a compact production-style churn prediction system. It keeps
the ML workflow, API, database logging, monitoring, and dashboard in separate
layers so each part can be tested independently.

## Data Flow

```text
data/raw synthetic CSV
    -> app.ml.preprocessing
    -> processed train/validation CSVs
    -> app.ml.training
    -> artifacts/trained_model.pkl + preprocessor.pkl + metrics.json
    -> FastAPI /predict and /predict/batch
    -> PostgreSQL prediction_logs
    -> monitoring endpoints
    -> Streamlit dashboard
```

## Runtime Components

```text
Client / Dashboard
       |
       v
FastAPI app
  |-- prediction router
  |-- monitoring router
  |-- Pydantic schemas
       |
       +--> ML inference layer
       |      |-- loads cached artifacts
       |      |-- applies feature engineering + preprocessor
       |      +-- returns churn probability and risk band
       |
       +--> DB layer
       |      |-- SQLAlchemy Core engine
       |      |-- prediction_logs insert/read helpers
       |
       +--> Monitoring layer
              |-- PSI drift checks
              |-- prediction summary
              +-- quality metrics
```

## Key Directories

- `app/api`: FastAPI app and routers.
- `app/schemas`: Pydantic request/response models.
- `app/ml`: synthetic data, preprocessing, training, inference.
- `app/db`: SQL schema and database helpers.
- `app/monitoring`: pure monitoring functions.
- `dashboard`: Streamlit UI and lightweight API client.
- `data/sample`: demo request payloads.
- `artifacts`: trained model, preprocessor, metrics, feature importance.
- `tests`: unit and API tests.

## Artifact Contract

Prediction uses these files:

- `artifacts/trained_model.pkl`
- `artifacts/preprocessor.pkl`
- `artifacts/metrics.json`

If they are missing, `/predict` returns HTTP `503` with the command needed to
train the model.

## Privacy Boundary

API requests may include `user_id` for demo purposes. The logging layer removes
raw `user_id` from saved input features and stores only `user_id_hash`.
