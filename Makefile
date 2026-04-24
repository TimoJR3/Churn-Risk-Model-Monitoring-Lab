.PHONY: install test run-api run-dashboard docker-up docker-down

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest -q

run-api:
	uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000

run-dashboard:
	streamlit run dashboard/app.py --server.port 8501

docker-up:
	docker compose up --build

docker-down:
	docker compose down
