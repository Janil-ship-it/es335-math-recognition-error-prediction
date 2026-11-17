"""Grouped 5-fold CV, hyperparameter selection, feature ablation. Never touches test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import (
    ALL_FEATURES, CONFIDENCE_FEATURES, FEATURES_PATH, FORBIDDEN_FEATURES,
    IMAGE_FEATURES, N_FOLDS, RESULTS_DIR, SEED, SPLITS_PATH, STRUCTURE_FEATURES,
)
from .metrics import pr_auc
from .models import FixedConfidenceBaseline, get_model, HAS_SKLEARN


def load_dev():
    df = pd.read_csv(FEATURES_PATH)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    for col in FORBIDDEN_FEATURES:
        assert col not in ALL_FEATURES
    dev = df[df["group_id"].isin(splits["dev_groups"])].copy()
    assert set(dev["group_id"]).isdisjoint(splits["test_groups"])
    return dev, splits


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
    df, splits = load_dev()
    X = df[ALL_FEATURES].values.astype(float)
    y = df["needs_correction"].values.astype(int)
    groups = df["group_id"].values

    feature_sets = {
        "confidence_only": CONFIDENCE_FEATURES,
        "image_only": IMAGE_FEATURES,
        "structure_only": STRUCTURE_FEATURES,
        "all": ALL_FEATURES,
    }

    model_grids = {
        "baseline_confidence": [{}],
        "logistic_l2": [{"C": c} for c in (0.1, 1.0, 10.0)],
        "random_forest": [{"max_depth": d} for d in (3, 5, 8)],
    }
    if HAS_SKLEARN:
        model_grids["svm_rbf"] = [{"C": c} for c in (0.5, 1.0, 5.0)]

    results = []
    best_params = {}
    for model_name, grid in model_grids.items():
        for fs_name, feats in feature_sets.items():
            if model_name == "baseline_confidence" and "mean_token_conf" not in feats:
                continue
            idx = [ALL_FEATURES.index(f) for f in feats]
            Xf = X[:, idx]
            for params in grid:
                fold_scores = []
                for train_g, val_g in grouped_folds(groups):
                    tr = np.isin(groups, train_g)
                    va = np.isin(groups, val_g)
                    if model_name == "baseline_confidence":
                        m = FixedConfidenceBaseline().fit(Xf[tr], y[tr], feats)
                    elif model_name == "logistic_l2":
                        from .models import LogisticRegressionL2
                        m = LogisticRegressionL2(C=params.get("C", 1.0)).fit(Xf[tr], y[tr])
                    elif model_name == "random_forest":
                        from .models import SimpleRandomForest
                        m = SimpleRandomForest(n_estimators=50, max_depth=params.get("max_depth", 4), seed=SEED).fit(Xf[tr], y[tr])
                    else:
                        m = get_model("svm_rbf", seed=SEED)
                        if hasattr(m, "C"):
                            m.C = params.get("C", 1.0)
                        m.fit(Xf[tr], y[tr])
                    fold_scores.append(pr_auc(y[va], m.predict_proba(Xf[va])))
                mean_s = float(np.mean(fold_scores))
                results.append({
                    "model": model_name, "features": fs_name, "params": params,
                    "val_prauc_mean": mean_s, "val_prauc_std": float(np.std(fold_scores)),
                    "fold_scores": [float(x) for x in fold_scores],
                })

    all_res = [r for r in results if r["features"] == "all"]
    by_model = {}
    for r in all_res:
        key = r["model"]
        if key not in by_model or r["val_prauc_mean"] > by_model[key]["val_prauc_mean"]:
            by_model[key] = r
            best_params[key] = r["params"]

    ranked = sorted(by_model.values(), key=lambda r: -r["val_prauc_mean"])
    best = ranked[0]
    se = best["val_prauc_std"] / np.sqrt(N_FOLDS)
    candidates = [r for r in ranked if r["val_prauc_mean"] >= best["val_prauc_mean"] - se]
    order = {"baseline_confidence": 0, "logistic_l2": 1, "svm_rbf": 2, "random_forest": 3}
    chosen = min(candidates, key=lambda r: order.get(r["model"], 9))

    summary = {
        "cv_results": results,
        "best_params": best_params,
        "best_by_mean": {k: v for k, v in best.items() if k != "fold_scores"},
        "chosen_by_1se": {k: v for k, v in chosen.items() if k != "fold_scores"},
        "n_dev_regions": len(df),
        "n_dev_groups": int(df["group_id"].nunique()),
        "error_rate": float(y.mean()),
        "has_sklearn": HAS_SKLEARN,
        "ablation": [
            {k: v for k, v in r.items() if k != "fold_scores"}
            for r in results if r["model"] == chosen["model"]
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cv_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: summary[k] for k in ("best_params", "chosen_by_1se", "n_dev_regions", "error_rate")}, indent=2))
    return summary


if __name__ == "__main__":
    run_cv()
