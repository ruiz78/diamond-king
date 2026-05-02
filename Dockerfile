FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data models

ENV PYTHONPATH=/app

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
