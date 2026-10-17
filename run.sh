#!/usr/bin/env bash
# Linux/macOS: set up, train and start TrapWire LK on http://localhost:8000
set -e
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/train.py
uvicorn app.main:app --port 8000
