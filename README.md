# Churn Risk & Model Monitoring Lab

Production-like pet project for churn prediction, model serving, and model
monitoring. This first stage contains only the project scaffold.

## Current Scope

- FastAPI service with `/health`.
- Streamlit dashboard placeholder.
- PostgreSQL service in Docker Compose.
- Basic pytest coverage.
- Documentation for the initial architecture.

Model training, drift detection, persistence of predictions, and dashboard
analytics are intentionally out of scope for stage 1.

## Quickstart

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.api.main:app --reload
```

Open:

- API health: http://localhost:8000/health
- API docs: http://localhost:8000/docs

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Run tests:

```bash
pytest -q
```

Run with Docker Compose:

```bash
cp .env.example .env
docker compose up --build
```

Services:

- API: http://localhost:8000
- Dashboard: http://localhost:8501
- PostgreSQL: localhost:5432

## Project Status

Stage 1 is focused on repository structure and local developer experience.
The next stages will add data loading, baseline training, prediction API,
database writes, monitoring checks, and dashboard metrics.
