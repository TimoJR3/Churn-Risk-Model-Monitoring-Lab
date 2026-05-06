FROM python:3.11-slim

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd --create-home --shell /usr/sbin/nologin appuser

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8000 8501

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
