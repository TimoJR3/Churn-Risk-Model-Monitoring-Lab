# Архитектура

Проект устроен как компактный демонстрационный ML-сервис для churn prediction и
model monitoring. ML pipeline, API, database logging, monitoring logic и
dashboard разделены по слоям, чтобы каждую часть можно было тестировать
отдельно.

## Поток данных

```text
data/raw/synthetic_churn_dataset.csv
    -> app.ml.preprocessing
    -> data/processed train/validation CSV
    -> app.ml.training
    -> artifacts/trained_model.pkl + preprocessor.pkl + metrics.json
    -> FastAPI /predict и /predict/batch
    -> PostgreSQL prediction_logs
    -> monitoring endpoints
    -> Streamlit dashboard
```

## Runtime-компоненты

```text
Dashboard / API client
       |
       v
FastAPI app
  |-- prediction router
  |-- monitoring router
  |-- Pydantic schemas
       |
       +--> ML inference layer
       |      |-- загружает cached artifacts
       |      |-- применяет feature engineering + preprocessor
       |      +-- возвращает probability, prediction, risk band
       |
       +--> DB layer
       |      |-- SQLAlchemy Core engine
       |      |-- insert/read helpers для prediction_logs
       |
       +--> Monitoring layer
              |-- PSI drift checks
              |-- prediction summary
              +-- quality metrics
```

## Основные папки

- `app/api`: FastAPI app и routers.
- `app/schemas`: Pydantic request/response models.
- `app/ml`: synthetic data, preprocessing, training, inference.
- `app/db`: SQL schema и helpers для prediction logs.
- `app/monitoring`: чистые функции мониторинга.
- `dashboard`: Streamlit UI и лёгкий API client.
- `data/sample`: demo request payloads.
- `artifacts`: trained model, preprocessor, metrics, feature importance.
- `tests`: unit и API tests.

## Artifact contract

Inference использует:

- `artifacts/trained_model.pkl`
- `artifacts/preprocessor.pkl`
- `artifacts/metrics.json`

Если artifacts отсутствуют, `/predict` возвращает HTTP 503 с командой для
обучения модели. API не переобучает модель во время request.

## Privacy boundary

API payload может содержать `user_id` для демо. Logging layer удаляет raw
`user_id` из сохранённых input features и пишет только `user_id_hash`.
