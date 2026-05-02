FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data models
RUN chmod +x start.sh
# Fix Windows line endings
RUN sed -i 's/\r//' start.sh

ENV PYTHONPATH=/app

CMD ["/bin/bash", "start.sh"]
