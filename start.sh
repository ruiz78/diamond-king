#!/bin/bash
cd /app
export PYTHONPATH=/app
exec uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}
