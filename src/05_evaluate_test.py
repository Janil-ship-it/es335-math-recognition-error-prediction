"""Single final evaluation on held-out test groups."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ALL_FEATURES, N_BOOTSTRAP, PREDICTIONS_PATH, RESULTS_DIR, REVIEW_RATES, SEED, SPLITS_PATH
from .metrics import brier_score, paired_bootstrap_ci, pr_auc, recall_at_review_rate, residual_ser
from .models import FixedConfidenceBaseline, LogisticRegressionL2, SimpleRandomForest

def load_all():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "features.csv")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    dev = df[df["group_id"].isin(splits["dev_groups"])].copy()
    test = df[df["group_id"].isin(splits["test_groups"])].copy()
    return dev, test, splits

def evaluate():
    dev, test, splits = load_all()
    X_dev = dev[ALL_FEATURES].values.astype(float)
    y_dev = dev["needs_correction"].values.astype(int)
    X_te = test[ALL_FEATURES].values.astype(float)
    y_te = test["needs_correction"].values.astype(int)
    ser_te = test["ser"].values.astype(float)
    groups_te = test["group_id"].values
    baseline = FixedConfidenceBaseline().fit(X_dev, y_dev, ALL_FEATURES)
    lr = LogisticRegressionL2(C=1.0).fit(X_dev, y_dev)
    rf = SimpleRandomForest(n_estimators=40, max_depth=4, seed=SEED).fit(X_dev, y_dev)
    models = {"baseline_confidence": baseline, "logistic_l2": lr, "random_forest": rf}
    preds = test[["region_id", "group_id", "needs_correction", "ser"]].copy()
    metrics = {}
    for name, model in models.items():
        scores = model.predict_proba(X_te)
        preds[f"score_{name}"] = scores
        prauc = pr_auc(y_te, scores)
        recalls = {f"recall@{int(r*100)}": recall_at_review_rate(y_te, scores, r) for r in REVIEW_RATES}
        res_ser = {f"residual_ser@{int(r*100)}": residual_ser(y_te, scores, ser_te, r) for r in REVIEW_RATES}
        brier = brier_score(y_te, scores)
        if name == "baseline_confidence":
            boot = paired_bootstrap_ci(pr_auc, y_te, scores, None, groups_te, n_boot=N_BOOTSTRAP, seed=SEED)
            metrics[name] = {"pr_auc": prauc, "pr_auc_ci": [boot["ci_low"], boot["ci_high"]], **recalls, **res_ser, "brier": brier}
        else:
            base_scores = preds["score_baseline_confidence"].values
            boot = paired_bootstrap_ci(pr_auc, y_te, scores, base_scores, groups_te, n_boot=N_BOOTSTRAP, seed=SEED)
            metrics[name] = {"pr_auc": prauc, "pr_auc_ci": [boot["ci_low_a"], boot["ci_high_a"]],
                             "delta_vs_baseline": boot["diff"], "delta_ci": [boot["ci_low_diff"], boot["ci_high_diff"]],
                             **recalls, **res_ser, "brier": brier}
    rng = np.random.default_rng(SEED)
    y_shuf = rng.permutation(y_dev)
    lr_shuf = LogisticRegressionL2(C=1.0).fit(X_dev, y_shuf)
    metrics["shuffled_label_control"] = {"pr_auc": pr_auc(y_te, lr_shuf.predict_proba(X_te)),
                                         "note": "Should be near prevalence if no leakage"}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds.to_csv(PREDICTIONS_PATH, index=False)
    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics, preds

if __name__ == "__main__":
    evaluate()
