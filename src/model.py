"""Model definition, training, evaluation and drift reference statistics."""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline

from src.textprep import TOKEN_PATTERN, normalize, tokens

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
CONF_BINS = np.array([0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01])


def build_pipeline():
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, token_pattern=TOKEN_PATTERN, ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(2, 5),
                                 min_df=2, sublinear_tf=True, max_features=60000)),
    ])
    clf = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


def build_baseline():
    return Pipeline([("tfidf", TfidfVectorizer(preprocessor=normalize, token_pattern=TOKEN_PATTERN)), ("clf", MultinomialNB())])


def scam_metrics(y_true, y_pred):
    """Binary scam-vs-legit view of multi-class predictions."""
    bt = (np.asarray(y_true) != "legit").astype(int)
    bp = (np.asarray(y_pred) != "legit").astype(int)
    tp = int(((bt == 1) & (bp == 1)).sum())
    fn = int(((bt == 1) & (bp == 0)).sum())
    fp = int(((bt == 0) & (bp == 1)).sum())
    return {
        "scam_recall": round(tp / max(tp + fn, 1), 4),
        "scam_precision": round(tp / max(tp + fp, 1), 4),
        "missed_scams": fn,
        "false_alarms": fp,
    }


def evaluate(model, df):
    pred = model.predict(df.text)
    proba = model.predict_proba(df.text).max(axis=1)
    out = {
        "n": int(len(df)),
        "accuracy": round(accuracy_score(df.scam_type, pred), 4),
        "macro_f1": round(f1_score(df.scam_type, pred, average="macro"), 4),
        "mean_confidence": round(float(proba.mean()), 4),
        **scam_metrics(df.scam_type, pred),
    }
    by_lang = {}
    for lang, g in df.assign(pred=pred).groupby("lang"):
        by_lang[lang] = round(accuracy_score(g.scam_type, g.pred), 4)
    out["accuracy_by_language"] = by_lang
    return out, pred, proba


def reference_stats(model, df_ref):
    """Statistics of a trusted reference window, used later to detect drift."""
    proba = model.predict_proba(df_ref.text).max(axis=1)
    hist, _ = np.histogram(proba, bins=CONF_BINS)
    vocab = set()
    for t in df_ref.text:
        vocab.update(tokens(t))
    unk = [unknown_rate(t, vocab) for t in df_ref.text]
    return {
        "mean_confidence": float(proba.mean()),
        "conf_hist": (hist / hist.sum()).tolist(),
        "mean_unknown_rate": float(np.mean(unk)),
        "classes": list(model.classes_),
    }, vocab


def unknown_rate(text, vocab):
    toks = tokens(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t not in vocab) / len(toks)


def train_and_save(train_df, test_df, version, notes=""):
    """Train a model version, evaluate on the held-out test set and save it."""
    MODELS.mkdir(exist_ok=True)
    t0 = time.time()
    model = build_pipeline().fit(train_df.text, train_df.scam_type)
    metrics, _, _ = evaluate(model, test_df)
    # The reference vocabulary comes from training text; the reference
    # confidence profile comes from the held-out test set (unseen data).
    _, train_vocab = reference_stats(model, train_df)
    ref, _ = reference_stats(model, test_df)
    vocab = train_vocab
    ref["mean_unknown_rate"] = float(np.mean([unknown_rate(t, vocab) for t in test_df.text]))
    meta = {
        "version": version,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_rows": int(len(train_df)),
        "train_seconds": round(time.time() - t0, 1),
        "test_metrics": metrics,
        "reference": ref,
        "notes": notes,
    }
    joblib.dump({"model": model, "vocab": vocab, "meta": meta}, MODELS / f"{version}.joblib")
    return model, meta


def load_bundle(version):
    return joblib.load(MODELS / f"{version}.joblib")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
