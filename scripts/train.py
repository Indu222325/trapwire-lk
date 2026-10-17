"""
Train the first TrapWire model (v1), compare it with a baseline, and write the
evaluation report used in the README.

    python scripts/train.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402

from src.model import (MODELS, REPORTS, build_baseline, evaluate, train_and_save,  # noqa: E402
                       write_json)


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    shifted = pd.read_csv(ROOT / "data" / "shifted.csv")

    baseline = build_baseline().fit(train.text, train.scam_type)
    base_metrics, _, _ = evaluate(baseline, test)

    model, meta = train_and_save(train, test, "v1", notes="Initial model trained on train.csv")
    test_metrics = meta["test_metrics"]
    _, pred, _ = evaluate(model, test)

    # How v1 does on the shifted set it has never seen (the drift problem)
    shift_metrics, _, _ = evaluate(model, shifted)
    new_target = shifted[shifted.scam_type == "digital_id_scam"]
    caught = (model.predict(new_target.text) != "legit").mean()

    # Split the test score into wording seen in training vs wording never seen
    unseen_mask = ~test.template_id.isin(set(train.template_id))
    seen_m, _, _ = evaluate(model, test[~unseen_mask])
    unseen_m, _, _ = evaluate(model, test[unseen_mask])
    test_metrics["seen_wording"] = {k: seen_m[k] for k in ("n", "accuracy", "scam_recall")}
    test_metrics["unseen_wording"] = {k: unseen_m[k] for k in ("n", "accuracy", "scam_recall")}

    labels = sorted(test.scam_type.unique())
    report = {
        "baseline_naive_bayes_on_test": base_metrics,
        "v1_on_heldout_test": test_metrics,
        "v1_on_shifted_set": {k: v for k, v in shift_metrics.items() if k != "accuracy"},
        "v1_digital_id_scams_flagged_as_not_legit": round(float(caught), 4),
        "per_class_report_test": classification_report(test.scam_type, pred, output_dict=True),
        "confusion_matrix_test": {"labels": labels,
                                  "matrix": confusion_matrix(test.scam_type, pred, labels=labels).tolist()},
    }
    write_json(REPORTS / "evaluation.json", report)
    (MODELS / "ACTIVE").write_text("v1")

    print("Baseline (Naive Bayes) test :", {k: base_metrics[k] for k in ("accuracy", "macro_f1")})
    print("TrapWire v1 held-out test   :", {k: test_metrics[k] for k in
                                            ("accuracy", "macro_f1", "scam_recall", "scam_precision")})
    print("  by language              :", test_metrics["accuracy_by_language"])
    print("  seen wording             :", test_metrics["seen_wording"])
    print("  unseen wording           :", test_metrics["unseen_wording"])
    print("v1 on SHIFTED set           :", {k: shift_metrics[k] for k in ("mean_confidence", "scam_recall")},
          "(test mean_confidence was", test_metrics["mean_confidence"], ")")
    print(f"v1 flags {caught:.0%} of the new SL-UDI digital-ID scams as not legit")
    print("Saved models/v1.joblib and reports/evaluation.json")


if __name__ == "__main__":
    main()
