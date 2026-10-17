"""SQLite prediction log, human-review queue and alert log."""
import json
import sqlite3
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "logs" / "trapwire.db"
JSONL_PATH = ROOT / "logs" / "predictions.jsonl"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL, text TEXT, lang TEXT, pred TEXT, confidence REAL, scam_prob REAL,
    verdict TEXT, unknown_rate REAL, model_version TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    prediction_id INTEGER PRIMARY KEY, status TEXT, label TEXT, reviewer TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, level TEXT, message TEXT, details TEXT
);
"""


def conn():
    DB_PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, conn() as c:
        c.executescript(SCHEMA)


def reset():
    with _lock, conn() as c:
        c.executescript("DROP TABLE IF EXISTS predictions; DROP TABLE IF EXISTS reviews; DROP TABLE IF EXISTS alerts;")
        c.executescript(SCHEMA)
    if JSONL_PATH.exists():
        JSONL_PATH.unlink()


def log_prediction(rec):
    with _lock, conn() as c:
        cur = c.execute(
            "INSERT INTO predictions (ts,text,lang,pred,confidence,scam_prob,verdict,unknown_rate,model_version,source)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rec["ts"], rec["text"], rec["lang"], rec["pred"], rec["confidence"], rec["scam_prob"],
             rec["verdict"], rec["unknown_rate"], rec["model_version"], rec["source"]))
        pid = cur.lastrowid
        if rec.get("needs_review"):
            c.execute("INSERT INTO reviews (prediction_id,status,ts) VALUES (?, 'pending', ?)", (pid, rec["ts"]))
    with open(JSONL_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({**rec, "id": pid}, ensure_ascii=False) + "\n")
    return pid


def recent(n, version=None):
    with conn() as c:
        if version:
            rows = c.execute("SELECT * FROM predictions WHERE model_version=? ORDER BY id DESC LIMIT ?",
                             (version, n)).fetchall()
        else:
            rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (n,)).fetchall()
    return [dict(r) for r in rows][::-1]


def all_predictions():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM predictions ORDER BY id").fetchall()]


def pending_reviews(limit=50):
    with conn() as c:
        rows = c.execute(
            "SELECT p.* FROM reviews r JOIN predictions p ON p.id = r.prediction_id "
            "WHERE r.status='pending' ORDER BY p.id LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def review_counts():
    with conn() as c:
        rows = c.execute("SELECT status, COUNT(*) n FROM reviews GROUP BY status").fetchall()
    return {r["status"]: r["n"] for r in rows}


def submit_review(pid, label, reviewer):
    with _lock, conn() as c:
        c.execute("INSERT OR REPLACE INTO reviews (prediction_id,status,label,reviewer,ts) VALUES (?,?,?,?,?)",
                  (pid, "done", label, reviewer, time.time()))


def reviewed_examples():
    with conn() as c:
        rows = c.execute(
            "SELECT p.text, r.label FROM reviews r JOIN predictions p ON p.id=r.prediction_id "
            "WHERE r.status='done' AND r.label IS NOT NULL").fetchall()
    return [dict(r) for r in rows]


def add_alert(level, message, details=None):
    with _lock, conn() as c:
        c.execute("INSERT INTO alerts (ts,level,message,details) VALUES (?,?,?,?)",
                  (time.time(), level, message, json.dumps(details or {})))


def alerts(limit=20):
    with conn() as c:
        rows = c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]
