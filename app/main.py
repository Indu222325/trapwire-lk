"""TrapWire LK service: prediction API, monitoring, human review and controlled retraining.

Run locally:  uvicorn app.main:app --port 8000
"""
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import monitor, store  # noqa: E402
from src.model import MODELS, evaluate, load_bundle, train_and_save, unknown_rate  # noqa: E402
from src.textprep import detect_lang  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("trapwire")

NORMAL_THRESHOLD = 0.60   # below this confidence -> Suspicious + human review
SAFE_MODE_THRESHOLD = 0.75  # stricter threshold while drift is active
MAX_F1_REGRESSION = 0.02  # a retrained model may not lose more than this on the held-out test set

WARNINGS = {
    "Scam": {
        "en": "This looks like a scam. Do not click links, share OTPs or PINs, or send money. Call your bank using the number on your card.",
        "si": "මෙය වංචාවක් ලෙස පෙනේ. සබැඳි ක්ලික් නොකරන්න, OTP හෝ PIN අංක ලබා නොදෙන්න, මුදල් නොයවන්න.",
        "ta": "இது மோசடி போல் தெரிகிறது. இணைப்புகளை அழுத்த வேண்டாம், OTP அல்லது PIN பகிர வேண்டாம், பணம் அனுப்ப வேண்டாம்.",
        "singlish": "Meka scam ekak wage. Link click karanna epa, OTP/PIN denna epa, salli ewanna epa.",
        "tanglish": "Idhu scam maadhiri irukku. Link click pannadheenga, OTP/PIN share pannadheenga, panam anupadheenga.",
    },
    "Suspicious": {
        "en": "We are not sure about this message. Do not act on it yet: verify by calling the official number. A reviewer will check it.",
        "si": "මෙම පණිවිඩය ගැන සහතික නැත. නිල අංකයට කතා කර තහවුරු කරගන්න.",
        "ta": "இந்த செய்தி பற்றி உறுதியாக தெரியவில்லை. அதிகாரப்பூர்வ எண்ணை அழைத்து உறுதிப்படுத்தவும்.",
        "singlish": "Me message eka gana sure na. Official number ekata call karala check karanna.",
        "tanglish": "Indha message pathi sure illa. Official number ku call panni check pannunga.",
    },
    "Safe": {
        "en": "No scam pattern found. Still, never share an OTP or PIN with anyone.",
        "si": "වංචා රටාවක් හමු නොවීය. එසේ වුවද OTP හෝ PIN කිසිවෙකුට නොදෙන්න.",
        "ta": "மோசடி முறை எதுவும் இல்லை. இருந்தாலும் OTP அல்லது PIN யாருக்கும் கொடுக்காதீர்கள்.",
        "singlish": "Scam pattern ekak na. Eth OTP/PIN kawuruwath ekka share karanna epa.",
        "tanglish": "Scam pattern illa. Aanaalum OTP/PIN yaarukkum kudukkadheenga.",
    },
}


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.load(self.active_version())
        self.last_status = "OK"

    @staticmethod
    def active_version():
        f = MODELS / "ACTIVE"
        return f.read_text().strip() if f.exists() else "v1"

    def load(self, version):
        b = load_bundle(version)
        self.model, self.vocab, self.meta = b["model"], b["vocab"], b["meta"]
        self.version = version
        (MODELS / "ACTIVE").write_text(version)
        log.info("Loaded model %s", version)


S = State()
store.init()
app = FastAPI(title="TrapWire LK", version="1.0")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


class CheckIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    source: str = "user"


class ReviewIn(BaseModel):
    id: int
    label: str
    reviewer: str = "reviewer"


class ReplayIn(BaseModel):
    dataset: str = "test"
    n: int = 60


def explain(text, cls):
    """Top word features pushing the message towards the predicted class."""
    feats = S.model.named_steps["features"]
    clf = S.model.named_steps["clf"]
    word = feats.transformer_list[0][1]
    x = word.transform([text])
    if x.nnz == 0:
        return []
    ci = list(clf.classes_).index(cls)
    coef = clf.coef_[ci][: len(word.vocabulary_)]
    names = word.get_feature_names_out()
    contrib = [(names[j], float(x[0, j] * coef[j])) for j in x.nonzero()[1]]
    contrib = [c for c in contrib if c[1] > 0 and "token" not in c[0]]
    return [w for w, _ in sorted(contrib, key=lambda c: -c[1])[:5]]


def predict(text, source="user"):
    with S.lock:
        proba = S.model.predict_proba([text])[0]
        classes = list(S.model.classes_)
        version, vocab = S.version, S.vocab
        cls = classes[int(np.argmax(proba))]
        reasons = explain(text, cls)
    conf = float(proba.max())
    scam_prob = 1.0 - float(proba[classes.index("legit")]) if "legit" in classes else conf
    safe_mode = S.last_status == "DRIFT"
    threshold = SAFE_MODE_THRESHOLD if safe_mode else NORMAL_THRESHOLD
    if conf < threshold:
        verdict = "Suspicious"
    elif cls == "legit":
        verdict = "Safe"
    else:
        verdict = "Scam"
    lang = detect_lang(text)
    rec = {
        "ts": time.time(), "text": text, "lang": lang, "pred": cls, "confidence": round(conf, 4),
        "scam_prob": round(scam_prob, 4), "verdict": verdict, "unknown_rate": round(unknown_rate(text, vocab), 4),
        "model_version": version, "source": source, "needs_review": verdict == "Suspicious",
    }
    rec["id"] = store.log_prediction(rec)
    log.info("prediction id=%s verdict=%s pred=%s conf=%.2f lang=%s model=%s",
             rec["id"], verdict, cls, conf, lang, version)
    return {**rec, "threshold": threshold, "safe_mode": safe_mode, "reasons": reasons,
            "warning": WARNINGS[verdict].get(lang, WARNINGS[verdict]["en"]),
            "top_classes": sorted([{"label": c, "p": round(float(p), 3)} for c, p in zip(classes, proba)],
                                  key=lambda d: -d["p"])[:3]}


def refresh_monitor():
    """Recompute drift; on a transition into DRIFT raise an alert (drift response)."""
    rows = store.recent(monitor.WINDOW, version=S.version)  # judge only the active model
    rep = monitor.compute(rows, S.meta["reference"])
    status = rep["status"]
    if status == "DRIFT" and S.last_status != "DRIFT":
        fired = [k for k, v in rep["signals"].items() if v["level"] == "drift"]
        store.add_alert("CRITICAL", f"Drift detected on model {S.version}: {', '.join(fired)}. "
                        "Safe mode ON: stricter threshold, uncertain messages routed to human review.", rep["signals"])
        log.warning("DRIFT ALERT model=%s signals=%s", S.version, fired)
    elif status == "OK" and S.last_status == "DRIFT":
        store.add_alert("INFO", f"Drift cleared on model {S.version}. Safe mode OFF.")
    if status in ("OK", "WARNING", "DRIFT"):
        S.last_status = status
    return rep


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": S.version}


@app.post("/api/check")
def check(body: CheckIn):
    out = predict(body.text, body.source)
    refresh_monitor()
    return out


@app.get("/api/monitor")
def get_monitor():
    rep = refresh_monitor()
    counts = store.review_counts()
    done = counts.get("done", 0)
    return {
        **rep,
        "safe_mode": S.last_status == "DRIFT",
        "model": {"version": S.version, "trained_at": S.meta["trained_at"],
                  "test_metrics": S.meta["test_metrics"], "notes": S.meta.get("notes", "")},
        "timeline": monitor.timeline(store.all_predictions()),
        "alerts": store.alerts(),
        "reviews": {"pending": counts.get("pending", 0), "done": done},
        "retrain_ready": done >= monitor.RETRAIN_MIN_REVIEWS,
        "retrain_min_reviews": monitor.RETRAIN_MIN_REVIEWS,
        "total_predictions": len(store.all_predictions()),
    }


@app.get("/api/review")
def get_reviews():
    return {"items": store.pending_reviews(), "labels": sorted(set(S.model.classes_) | {"digital_id_scam"})}


@app.post("/api/review")
def post_review(body: ReviewIn):
    store.submit_review(body.id, body.label.strip(), body.reviewer)
    return {"ok": True}


@app.post("/api/retrain")
def retrain():
    """Controlled retraining: train a challenger on train.csv + human-reviewed
    labels, and promote it only if it does not regress on the held-out test set."""
    reviewed = pd.DataFrame(store.reviewed_examples())
    if len(reviewed) < monitor.RETRAIN_MIN_REVIEWS:
        raise HTTPException(400, f"Need at least {monitor.RETRAIN_MIN_REVIEWS} reviewed messages, have {len(reviewed)}.")
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    shifted = pd.read_csv(ROOT / "data" / "shifted.csv")
    extra = reviewed.rename(columns={"label": "scam_type"})
    extra["lang"] = extra.text.map(detect_lang)
    # Reviewed messages are weighted up (repeated) so a small batch can teach a new pattern
    new_train = pd.concat([train, *[extra] * 3], ignore_index=True)

    n = len([p for p in MODELS.glob("v*.joblib")]) + 1
    version = f"v{n}"
    champion_test = S.meta["test_metrics"]
    challenger, meta = train_and_save(new_train, test, version,
                                      notes=f"Retrained with {len(reviewed)} human-reviewed messages")
    # Evaluate both on shifted messages that were NOT used for training
    unseen_shift = shifted[~shifted.text.isin(set(reviewed.text))]
    with S.lock:
        champ_shift, _, _ = evaluate(S.model, unseen_shift)
    chall_shift, _, _ = evaluate(challenger, unseen_shift)
    regression = champion_test["macro_f1"] - meta["test_metrics"]["macro_f1"]
    promote = regression <= MAX_F1_REGRESSION and chall_shift["accuracy"] >= champ_shift["accuracy"]
    result = {
        "challenger": version, "champion": S.version, "promoted": promote,
        "reviewed_used": int(len(reviewed)),
        "heldout_test": {"champion_macro_f1": champion_test["macro_f1"], "challenger_macro_f1": meta["test_metrics"]["macro_f1"]},
        "unseen_shifted": {"champion_accuracy": champ_shift["accuracy"], "challenger_accuracy": chall_shift["accuracy"],
                           "champion_scam_recall": champ_shift["scam_recall"], "challenger_scam_recall": chall_shift["scam_recall"]},
        "rule": f"Promote only if held-out test macro-F1 drops by at most {MAX_F1_REGRESSION} and accuracy on new patterns improves.",
    }
    if promote:
        old = S.version
        with S.lock:
            S.load(version)
        S.last_status = "OK"
        store.add_alert("INFO", f"Model {version} promoted (was {old}). Shifted-set accuracy "
                        f"{champ_shift['accuracy']:.0%} -> {chall_shift['accuracy']:.0%}.", result)
    else:
        store.add_alert("WARNING", f"Challenger {version} rejected by the regression guard; keeping {S.version}.", result)
    log.info("retrain result %s", result)
    return result


@app.get("/api/models")
def models():
    out = []
    for f in sorted(MODELS.glob("v*.joblib")):
        m = load_bundle(f.stem)["meta"]
        out.append({"version": m["version"], "trained_at": m["trained_at"], "train_rows": m["train_rows"],
                    "test_macro_f1": m["test_metrics"]["macro_f1"], "notes": m.get("notes", ""),
                    "active": m["version"] == S.version})
    return out


# ---------------------------------------------------------------- demo helpers
@app.post("/api/demo/replay")
def replay(body: ReplayIn):
    """Stream messages from a dataset through the live service (simulated traffic)."""
    if body.dataset not in ("test", "shifted"):
        raise HTTPException(400, "dataset must be 'test' or 'shifted'")
    df = pd.read_csv(ROOT / "data" / f"{body.dataset}.csv").sample(n=min(body.n, 400), random_state=int(time.time()))
    verdicts = {"Scam": 0, "Suspicious": 0, "Safe": 0}
    for t in df.text:
        verdicts[predict(t, source=f"replay:{body.dataset}")["verdict"]] += 1
        refresh_monitor()
    return {"sent": len(df), "verdicts": verdicts, "status": S.last_status}


@app.post("/api/demo/auto_review")
def auto_review():
    """Simulated reviewers: label pending items with the ground truth from the
    synthetic datasets (stands in for a human analyst during the demo)."""
    truth = {}
    for name in ("test", "shifted"):
        d = pd.read_csv(ROOT / "data" / f"{name}.csv")
        truth.update(dict(zip(d.text, d.scam_type)))
    n = 0
    for item in store.pending_reviews(limit=500):
        if item["text"] in truth:
            store.submit_review(item["id"], truth[item["text"]], "simulated-reviewer")
            n += 1
    return {"labelled": n}


@app.post("/api/demo/reset")
def reset():
    store.reset()
    with S.lock:
        S.load("v1")
    S.last_status = "OK"
    for f in MODELS.glob("v*.joblib"):
        if f.stem != "v1":
            f.unlink()
    return {"ok": True, "model_version": "v1"}
