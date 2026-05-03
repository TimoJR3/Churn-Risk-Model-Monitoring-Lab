# AGENTS.md

## Project

Churn Risk & Model Monitoring Lab is a Python 3.11 ML service lab for churn prediction. It includes synthetic data generation, preprocessing, baseline model training, FastAPI, PostgreSQL schema, Streamlit dashboard, Docker Compose, and pytest coverage.

## Folders

- `app/api`: FastAPI app and HTTP endpoints.
- `app/core`: environment-driven settings.
- `app/db`: SQL schema and seed loading.
- `app/ml`: data generation, loading, feature engineering, preprocessing, training, EDA.
- `app/monitoring`: model quality and drift checks.
- `app/schemas`: Pydantic request/response schemas.
- `app/services`: API business logic and orchestration.
- `dashboard`: Streamlit dashboard.
- `data`: raw and processed local datasets.
- `artifacts`: trained model, preprocessor, metrics, feature importance.
- `docs`: architecture, data dictionary, EDA, model card.
- `tests`: pytest suite.

## Commands

```powershell
python -m pytest -q
python -m compileall app dashboard tests
python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
python -m app.ml.preprocessing --source csv --test-size 0.2
python -m app.ml.training --source csv --n-splits 3
uvicorn app.api.main:app --reload
streamlit run dashboard/app.py
docker compose up --build
```

## Code Change Rules

- Read existing files before editing; keep changes scoped to the task.
- Do not do broad refactors unless required by the task.
- Use Python type hints for public functions.
- Do not add heavy dependencies without a clear reason.
- Do not hardcode secrets.
- Do not log raw personal data or full user records.
- API errors should return clear JSON responses.
- Every new endpoint must have Pydantic request and response schemas plus a pytest test.
- Every monitoring or scoring algorithm must have unit tests for edge cases.
- Update tests when behavior changes.

## Definition of Done

- Required code and docs changes are complete and scoped.
- New or changed behavior is covered by pytest.
- `python -m compileall app dashboard tests` passes.
- `python -m pytest -q` passes.
- Docker config is checked when Docker files change.
- README/docs are updated when commands, API behavior, or architecture changes.
- No secrets, raw personal data, unrelated refactors, or generated cache files are committed.
