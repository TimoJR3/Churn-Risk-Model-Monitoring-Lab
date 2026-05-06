# Demo Script

Use this script to walk through the portfolio project in 5-10 minutes.

## 1. Explain the Business Goal

The project predicts churn risk for a subscription product. The business wants
to identify users likely to leave so retention actions can happen earlier.

## 2. Show the Architecture

Open [architecture.md](architecture.md) and describe the flow:

```text
synthetic data -> preprocessing -> training -> artifacts -> FastAPI inference
-> PostgreSQL logs -> monitoring -> Streamlit dashboard
```

## 3. Run the ML Workflow

```bash
python -m app.ml.generate_synthetic_data --n-users 2000 --seed 42
python -m app.ml.preprocessing --source csv --test-size 0.2
python -m app.ml.training --source csv --n-splits 3
```

Point out the generated artifacts:

- `artifacts/trained_model.pkl`
- `artifacts/preprocessor.pkl`
- `artifacts/metrics.json`
- `docs/model_card.md`

## 4. Start the API

```bash
uvicorn app.api.main:app --reload
```

Check:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model/metadata
```

## 5. Run Predictions

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_sample.json
```

Then run batch scoring:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  --data @data/sample/predict_batch_sample.json
```

Mention that raw `user_id` is not stored in logs, only a hash.

## 6. Show Monitoring

```bash
curl http://localhost:8000/monitoring/summary
```

Explain PSI:

- `< 0.1`: stable;
- `0.1..0.25`: warning;
- `>= 0.25`: drift.

## 7. Show the Dashboard

```bash
streamlit run dashboard/app.py
```

Open <http://localhost:8501> and show:

- API health and model metadata;
- single prediction form;
- batch demo;
- monitoring tab;
- model tab.

## 8. Show Engineering Quality

```bash
python -m compileall app dashboard tests
python -m pytest -q
docker compose config
```

Mention GitHub Actions CI runs compile, Ruff, pytest, Docker Compose config,
and Docker build.
