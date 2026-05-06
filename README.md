# Churn Risk & Model Monitoring Lab

Production-style ML portfolio project for churn prediction, inference logging,
model monitoring, and a Streamlit dashboard.

The project shows the full path from synthetic customer data to a working API:
synthetic data -> preprocessing -> training -> artifacts -> FastAPI inference ->
PostgreSQL prediction logs -> monitoring -> Streamlit dashboard.

## Business Problem

A subscription business wants to identify customers with elevated churn risk
before they leave. Early risk signals let the business prioritize retention
actions such as support outreach, product education, discounts, or account
review.

The model estimates churn probability from product usage, payment friction,
support activity, plan type, and recency features.

## What This Demonstrates

This is built as a junior Data Scientist / ML Engineer portfolio project. It
demonstrates:

- reproducible synthetic data generation;
- feature engineering and preprocessing outside notebooks;
- baseline model training with saved artifacts;
- FastAPI inference with Pydantic request/response schemas;
- prediction logging in PostgreSQL;
- privacy-aware logging with hashed user identifiers;
- model metadata, quality, and drift monitoring endpoints;
- Streamlit dashboard that consumes the API;
- pytest coverage, Docker Compose, and GitHub Actions CI.

## Architecture

```text
synthetic data
    -> preprocessing
    -> training
    -> artifacts
    -> FastAPI inference
    -> PostgreSQL logs
    -> monitoring
    -> Streamlit dashboard
```

Main docs:

- [Architecture](docs/architecture.md)
- [API examples](docs/api_examples.md)
- [Monitoring](docs/monitoring.md)
- [Demo script](docs/demo_script.md)
- [Data dictionary](docs/data_dictionary.md)
- [Model card](docs/model_card.md)

## Quickstart

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
```

Run the core workflow:

```powershell
python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
python -m app.ml.preprocessing --source csv --test-size 0.2
python -m app.ml.training --source csv --n-splits 3
uvicorn app.api.main:app --reload
```

In another terminal:

```powershell
streamlit run dashboard/app.py
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env
```

Run the core workflow:

```bash
python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
python -m app.ml.preprocessing --source csv --test-size 0.2
python -m app.ml.training --source csv --n-splits 3
uvicorn app.api.main:app --reload
```

In another terminal:

```bash
streamlit run dashboard/app.py
```

### Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

Windows PowerShell equivalent:

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Services:

- API: <http://localhost:8000>
- Swagger UI: <http://localhost:8000/docs>
- Dashboard: <http://localhost:8501>
- PostgreSQL: `localhost:5432`

## Commands

These commands match the `Makefile` targets.

| Task | Make target | Command |
| --- | --- | --- |
| Install dependencies | `make install` | `pip install -r requirements.txt` |
| Generate data | `make generate-data` | `python -m app.ml.generate_synthetic_data` |
| EDA report | `make eda-report` | `python -m app.ml.eda` |
| Preprocess data | `make prepare-data` | `python -m app.ml.preprocessing` |
| Train model | `make train` | `python -m app.ml.training` |
| Seed database | `make seed-db` | `python -m app.db.load_seed_data` |
| Run API | `make run-api` | `uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000` |
| Run dashboard | `make run-dashboard` | `streamlit run dashboard/app.py --server.port 8501` |
| Run tests | `make test` | `pytest -q` |
| Docker up | `make docker-up` | `docker compose up --build` |
| Docker down | `make docker-down` | `docker compose down` |

Common explicit commands:

```bash
python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
python -m app.ml.preprocessing --source csv --test-size 0.2
python -m app.ml.training --source csv --n-splits 3
uvicorn app.api.main:app --reload
streamlit run dashboard/app.py
python -m pytest -q
docker compose up --build
```

## API Endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `POST` | `/predict` | Single-user churn prediction |
| `POST` | `/predict/batch` | Batch churn prediction |
| `GET` | `/model/metadata` | Model metrics and artifact status |
| `GET` | `/predictions/recent` | Recent prediction logs without raw user IDs |
| `GET` | `/monitoring/summary` | Prediction monitoring summary |
| `POST` | `/monitoring/drift` | PSI drift check |
| `POST` | `/monitoring/quality` | Quality metrics from labels and scores |

## Example Curl Requests

Health:

```bash
curl http://localhost:8000/health
```

Single prediction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_sample.json
```

Batch prediction:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_batch_sample.json
```

Model metadata:

```bash
curl http://localhost:8000/model/metadata
```

Recent logs:

```bash
curl "http://localhost:8000/predictions/recent?limit=20"
```

Drift check:

```bash
curl -X POST http://localhost:8000/monitoring/drift \
  -H "Content-Type: application/json" \
  -d "{\"expected\":[1,2,3,4,5],\"actual\":[2,3,4,5,6],\"buckets\":5}"
```

Quality check:

```bash
curl -X POST http://localhost:8000/monitoring/quality \
  -H "Content-Type: application/json" \
  -d "{\"y_true\":[0,0,1,1],\"y_score\":[0.1,0.4,0.6,0.9],\"threshold\":0.5}"
```

More examples: [docs/api_examples.md](docs/api_examples.md).

## Testing and CI

Local checks:

```bash
python -m compileall app dashboard tests
python -m ruff check .
python -m pytest -q
docker compose config
docker build -t churn-lab:test .
```

GitHub Actions runs the same core checks on push and pull request to `main`.
On tags matching `v*`, CI publishes a Docker image to GitHub Container Registry:

- `ghcr.io/<owner>/<repo>:<tag>`
- `ghcr.io/<owner>/<repo>:latest`

## Model Monitoring

The monitoring layer exposes:

- prediction summary from logged inference results;
- Population Stability Index (PSI) for numeric feature drift;
- quality metrics from labels and prediction scores.

PSI compares the expected distribution of a numeric feature with the actual
distribution. In this project:

- `stable`: PSI `< 0.1`;
- `warning`: `0.1 <= PSI < 0.25`;
- `drift`: PSI `>= 0.25`.

Limitations: PSI is a univariate signal, bucket choices matter, and this demo
does not include scheduled monitoring jobs or alerting.

More details: [docs/monitoring.md](docs/monitoring.md).

## Security and Privacy

- No secrets are committed. Use `.env` locally and keep it ignored.
- Prediction logs never store raw `user_id`; only a SHA-256 hash is stored.
- The dataset is synthetic only.
- Sample payloads are fake and contain no real personal data.

## Limitations

- Synthetic dataset; metrics are not business benchmarks.
- No real production SLA, autoscaling, or incident response process.
- The default threshold is `0.5` and is not cost-optimized.
- Drift monitoring is simplified and request-driven.
- The dashboard is a local demo UI, not an authenticated production console.

## Roadmap

- Probability calibration.
- Cost-aware threshold optimization.
- Scheduled monitoring jobs.
- Evidently or custom HTML/PDF monitoring reports.
- Deployment with managed database and environment-specific secrets.
