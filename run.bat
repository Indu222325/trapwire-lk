@echo off
REM Windows: set up, train and start TrapWire LK on http://localhost:8000
if not exist .venv python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
python scripts\generate_data.py
python scripts\train.py
uvicorn app.main:app --port 8000
