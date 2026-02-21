#!/bin/bash
set -e

echo "Seeding demo documents..."
python scripts/seed_demo.py

echo "Starting FinRAG API..."
exec uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
