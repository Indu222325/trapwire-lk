"""Drift monitoring: compares a rolling window of live predictions to the
reference profile saved with the active model."""
import numpy as np

from src.model import CONF_BINS

WINDOW = 60          # predictions per rolling window
MIN_WINDOW = 25      # need at least this many before judging drift
CONF_DROP_WARN, CONF_DROP_DRIFT = 0.06, 0.10
UNK_RISE_WARN, UNK_RISE_DRIFT = 0.08, 0.15
PSI_WARN, PSI_DRIFT = 0.10, 0.25
REVIEW_SHARE_DRIFT = 0.35
RETRAIN_MIN_REVIEWS = 20


def psi(expected, actual, eps=1e-4):
    e = np.clip(np.asarray(expected, float), eps, None)
    a = np.clip(np.asarray(actual, float), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def level(value, warn, drift):
    return "drift" if value >= drift else "warn" if value >= warn else "ok"


def compute(rows, ref):
    """Return the drift report for the most recent window of predictions."""
    win = rows[-WINDOW:]
    if len(win) < MIN_WINDOW:
        return {"status": "WARMING_UP", "window_size": len(win), "signals": {}, "needed": MIN_WINDOW}
    conf = np.array([r["confidence"] for r in win])
    unk = np.array([r["unknown_rate"] for r in win])
    review_share = float(np.mean([r["verdict"] == "Suspicious" for r in win]))
    hist, _ = np.histogram(conf, bins=CONF_BINS)
    hist = hist / hist.sum()

    conf_drop = ref["mean_confidence"] - float(conf.mean())
    unk_rise = float(unk.mean()) - ref["mean_unknown_rate"]
    p = psi(ref["conf_hist"], hist)
    signals = {
        "confidence": {"value": round(float(conf.mean()), 3), "reference": round(ref["mean_confidence"], 3),
                       "change": round(-conf_drop, 3), "level": level(conf_drop, CONF_DROP_WARN, CONF_DROP_DRIFT),
                       "label": "Average model confidence"},
        "unknown_words": {"value": round(float(unk.mean()), 3), "reference": round(ref["mean_unknown_rate"], 3),
                          "change": round(unk_rise, 3), "level": level(unk_rise, UNK_RISE_WARN, UNK_RISE_DRIFT),
                          "label": "Share of words never seen in training"},
        "psi": {"value": round(p, 3), "reference": 0.0, "change": round(p, 3),
                "level": level(p, PSI_WARN, PSI_DRIFT) if conf_drop > 0 else "ok", "label": "Confidence distribution shift (PSI)"},
        "review_share": {"value": round(review_share, 3), "reference": None, "change": None,
                         "level": "drift" if review_share >= REVIEW_SHARE_DRIFT else "ok",
                         "label": "Share of messages sent to human review"},
    }
    n_drift = sum(s["level"] == "drift" for s in signals.values())
    n_warn = sum(s["level"] == "warn" for s in signals.values())
    status = "DRIFT" if n_drift >= 2 else "WARNING" if (n_drift + n_warn) >= 1 else "OK"
    return {"status": status, "window_size": len(win), "signals": signals}


def timeline(rows, step=5):
    """Rolling series for the dashboard chart."""
    out = []
    for i in range(MIN_WINDOW, len(rows) + 1, step):
        w = rows[max(0, i - 30):i]
        out.append({
            "i": i,
            "confidence": round(float(np.mean([r["confidence"] for r in w])), 3),
            "unknown": round(float(np.mean([r["unknown_rate"] for r in w])), 3),
            "review": round(float(np.mean([r["verdict"] == "Suspicious" for r in w])), 3),
            "version": w[-1]["model_version"],
        })
    return out
