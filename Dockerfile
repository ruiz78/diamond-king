FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data models
RUN chmod +x start.sh

ENV PYTHONPATH=/app

CMD ["/bin/bash", "/app/start.sh"]
