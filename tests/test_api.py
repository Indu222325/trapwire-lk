"""Contract and behaviour tests: python -m pytest -q"""
import logging

from fastapi.testclient import TestClient

from app.main import app

logging.disable(logging.CRITICAL)
c = TestClient(app)


def setup_module():
    c.post("/api/demo/reset")


def test_health():
    r = c.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_detects_bank_scam_singlish():
    r = c.post("/api/check", json={"text": "Oyage LTB account eka block karala. Verify karanna http://secure-pay.example/12"}).json()
    assert r["verdict"] in ("Scam", "Suspicious") and r["pred"] != "legit"
    for key in ("confidence", "warning", "model_version", "lang", "id"):
        assert key in r


def test_legit_otp_is_safe():
    r = c.post("/api/check", json={"text": "Your OTP is 482913. Do not share it with anyone. The bank will never ask for your OTP."}).json()
    assert r["verdict"] == "Safe"


def test_sinhala_and_tamil_language_detected():
    si = c.post("/api/check", json={"text": "ඔබගේ විශ්‍රාම වැටුප නවත්වා ඇත. වහාම 071-xxx-1234 අමතන්න."}).json()
    ta = c.post("/api/check", json={"text": "உங்கள் ஓய்வூதியம் நிறுத்தப்பட்டுள்ளது. உடனே 071-xxx-1234 ஐ அழைக்கவும்."}).json()
    assert si["lang"] == "si" and ta["lang"] == "ta"


def test_empty_text_rejected():
    assert c.post("/api/check", json={"text": ""}).status_code == 422


def test_drift_detect_review_retrain_loop():
    c.post("/api/demo/reset")
    c.post("/api/demo/replay", json={"dataset": "test", "n": 60})
    assert c.get("/api/monitor").json()["status"] in ("OK", "WARNING")
    r = c.post("/api/demo/replay", json={"dataset": "shifted", "n": 80}).json()
    m = c.get("/api/monitor").json()
    assert m["status"] == "DRIFT" and m["safe_mode"] is True
    assert any(a["level"] == "CRITICAL" for a in m["alerts"])
    assert c.post("/api/demo/auto_review").json()["labelled"] >= 20
    out = c.post("/api/retrain").json()
    assert out["promoted"] is True
    assert out["unseen_shifted"]["challenger_accuracy"] > out["unseen_shifted"]["champion_accuracy"]
    c.post("/api/demo/reset")


def test_retrain_needs_new_labels():
    c.post("/api/demo/reset")
    c.post("/api/demo/replay", json={"dataset": "test", "n": 60})
    c.post("/api/demo/replay", json={"dataset": "shifted", "n": 80})
    c.post("/api/demo/auto_review")
    assert c.post("/api/retrain").status_code == 200
    # the same labels cannot trigger a second retraining
    assert c.post("/api/retrain").status_code == 400
    c.post("/api/demo/reset")
