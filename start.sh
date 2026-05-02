#!/bin/bash
cd /app
export PYTHONPATH=/app

echo "👑 Diamond King starting up..."
echo "🔄 Running midnight engine to generate fresh projections..."
python pipeline/midnight_engine.py

echo "🚀 Starting API server..."
exec uvicorn api.server:app --host 0.0.0.0 --port ${PORT:-8000}
