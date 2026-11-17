"""Grouped 5-fold CV and model selection. Never touches test set."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import (ALL_FEATURES, CONFIDENCE_FEATURES, IMAGE_FEATURES,
                     STRUCTURE_FEATURES, N_FOLDS, RESULTS_DIR, SEED, SPLITS_PATH)
from .metrics import pr_auc
from .models import FixedConfidenceBaseline, LogisticRegressionL2, SimpleRandomForest, get_model, HAS_SKLEARN

def load_dev_data():
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "features.csv")
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    return df[df["group_id"].isin(splits["dev_groups"])].copy(), splits

def grouped_folds(groups, n_folds=N_FOLDS, seed=SEED):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    folds = np.array_split(unique, n_folds)
    for i in range(n_folds):
        val_g = folds[i]
        train_g = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        yield train_g, val_g

def run_cv():
    df, splits = load_dev_data()
    X = df[ALL_FEATURES].values.astype(float)
    y = df["needs_correction"].values.astype(int)
    groups = df["group_id"].values

    feature_sets = {
        "confidence_only": CONFIDENCE_FEATURES,
        "image_only": IMAGE_FEATURES,
        "structure_only": STRUCTURE_FEATURES,
        "all": ALL_FEATURES,
    }

    model_specs = [
        ("baseline_confidence", lambda feats: FixedConfidenceBaseline()),
        ("logistic_l2", lambda feats: LogisticRegressionL2(C=1.0)),
        ("random_forest", lambda feats: SimpleRandomForest(n_estimators=40, max_depth=4, seed=SEED)),
    ]
    if HAS_SKLEARN:
        model_specs.append(("svm_rbf", lambda feats: get_model("svm_rbf", seed=SEED)))

    results = []
    for fs_name, feats in feature_sets.items():
        idx = [ALL_FEATURES.index(f) for f in feats]
        Xf = X[:, idx]
        for model_name, ModelFactory in model_specs:
            if model_name == "baseline_confidence" and "mean_token_conf" not in feats:
                continue
            fold_prauc = []
            for train_g, val_g in grouped_folds(groups):
                tr = np.isin(groups, train_g)
                va = np.isin(groups, val_g)
                m = ModelFactory(feats)
                if model_name == "baseline_confidence":
                    m.fit(Xf[tr], y[tr], feats)
                else:
                    m.fit(Xf[tr], y[tr])
                fold_prauc.append(pr_auc(y[va], m.predict_proba(Xf[va])))
            results.append({
                "model": model_name,
                "features": fs_name,
                "val_prauc_mean": float(np.mean(fold_prauc)),
                "val_prauc_std": float(np.std(fold_prauc)),
                "fold_scores": [float(x) for x in fold_prauc],
            })

    all_res = [r for r in results if r["features"] == "all"]
    best = max(all_res, key=lambda r: r["val_prauc_mean"])
    se = best["val_prauc_std"] / np.sqrt(N_FOLDS)
    candidates = [r for r in all_res if r["val_prauc_mean"] >= best["val_prauc_mean"] - se]
    order = {"baseline_confidence": 0, "logistic_l2": 1, "svm_rbf": 2, "random_forest": 3}
    chosen = min(candidates, key=lambda r: order.get(r["model"], 9))

    summary = {
        "cv_results": results,
        "best_by_mean": best,
        "chosen_by_1se": chosen,
        "n_dev_regions": len(df),
        "n_dev_groups": int(df["group_id"].nunique()),
        "error_rate": float(y.mean()),
        "has_sklearn": HAS_SKLEARN,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "cv_results"}, indent=2))
    return summary

if __name__ == "__main__":
    run_cv()
