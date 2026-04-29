.PHONY: install test generate-data eda-report prepare-data seed-db run-api run-dashboard docker-up docker-down

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -q

generate-data:
	python -m app.ml.generate_synthetic_data

eda-report:
	python -m app.ml.eda

prepare-data:
	python -m app.ml.preprocessing

seed-db:
	python -m app.db.load_seed_data

run-api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py --server.port 8501

docker-up:
	docker compose up --build

docker-down:
	docker compose down
