"""Single final evaluation on held-out test groups. Touches test set once."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    ALL_FEATURES,
    FEATURES_PATH,
    FORBIDDEN_FEATURES,
    N_BOOTSTRAP,
    PREDICTIONS_PATH,
    RESULTS_DIR,
    REVIEW_RATES,
    SEED,
    SPLITS_PATH,
)
from .metrics import (
    brier_score,
    paired_bootstrap_ci,
    pr_auc,
    recall_at_review_rate,
    residual_ser,
    recall_curve,
)
from .models import (
    FixedConfidenceBaseline,
    LogisticRegressionL2,
    SimpleRandomForest,
    get_model,
    HAS_SKLEARN,
)


def load_all():
    df = pd.read_csv(FEATURES_PATH)

    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    for col in FORBIDDEN_FEATURES:
        assert col not in ALL_FEATURES, f"Forbidden feature included: {col}"

    dev = df[df["group_id"].isin(splits["dev_groups"])].copy()
    test = df[df["group_id"].isin(splits["test_groups"])].copy()

    assert set(dev["group_id"]).isdisjoint(set(test["group_id"]))

    return dev, test, splits


def _make_model(name, params, seed=SEED):
    if name == "baseline_confidence":
        return FixedConfidenceBaseline()

    if name == "logistic_l2":
        return LogisticRegressionL2(C=params.get("C", 1.0))

    if name == "random_forest":
        return SimpleRandomForest(
            n_estimators=50,
            max_depth=params.get("max_depth", 4),
            seed=seed,
        )

    if name == "svm_rbf":
        model = get_model("svm_rbf", seed=seed)
        if hasattr(model, "C"):
            model.C = params.get("C", 1.0)
        return model

    raise ValueError(f"Unknown model: {name}")


def evaluate():
    dev, test, splits = load_all()

    X_dev = dev[ALL_FEATURES].values.astype(float)
    y_dev = dev["needs_correction"].values.astype(int)

    X_te = test[ALL_FEATURES].values.astype(float)
    y_te = test["needs_correction"].values.astype(int)

    ser_te = test["ser"].values.astype(float)
    groups_te = test["group_id"].values

    # Load cross-validation results.
    # Model selection must happen on development data, never test metrics.
    cv_path = RESULTS_DIR / "cv_results.json"

    if not cv_path.exists():
        raise FileNotFoundError(
            f"Missing CV results: {cv_path}. Run src.04_train_cv first."
        )

    with open(cv_path) as f:
        cv_blob = json.load(f)

    best_params = cv_blob.get("best_params", {})
    chosen_by_1se = cv_blob.get("chosen_by_1se")

    if not chosen_by_1se or "model" not in chosen_by_1se:
        raise ValueError(
            "cv_results.json does not contain a valid chosen_by_1se model."
        )

    model_names = [
        "baseline_confidence",
        "logistic_l2",
        "random_forest",
    ]

    if HAS_SKLEARN:
        model_names.append("svm_rbf")

    selected_model = chosen_by_1se["model"]

    if selected_model not in model_names:
        raise ValueError(
            f"CV-selected model '{selected_model}' is not available. "
            f"Available models: {model_names}"
        )

    # Train models on development data.
    models = {}

    for name in model_names:
        params = best_params.get(name, {})
        model = _make_model(name, params)

        if name == "baseline_confidence":
            model.fit(X_dev, y_dev, ALL_FEATURES)
        else:
            model.fit(X_dev, y_dev)

        models[name] = model

    preds = test[
        [
            "region_id",
            "group_id",
            "needs_correction",
            "ser",
            "degradation",
            "reference_latex",
            "prediction_latex",
        ]
    ].copy()

    metrics = {}
    curves = {}
    base_scores = None

    # Evaluate all models on the held-out test set.
    for name, model in models.items():
        scores = model.predict_proba(X_te)
        preds[f"score_{name}"] = scores

        if name == "baseline_confidence":
            base_scores = scores

        prauc = pr_auc(y_te, scores)

        recalls = {
            f"recall@{int(r * 100)}": recall_at_review_rate(y_te, scores, r)
            for r in REVIEW_RATES
        }

        res_ser = {
            f"residual_ser@{int(r * 100)}": residual_ser(
                y_te, scores, ser_te, r
            )
            for r in REVIEW_RATES
        }

        brier = brier_score(y_te, scores)
        curve = recall_curve(y_te, scores, steps=100)
        curves[name] = curve

        if name == "baseline_confidence":
            boot = paired_bootstrap_ci(
                pr_auc,
                y_te,
                scores,
                None,
                groups_te,
                n_boot=N_BOOTSTRAP,
                seed=SEED,
            )

            metrics[name] = {
                "pr_auc": prauc,
                "pr_auc_ci": [boot["ci_low"], boot["ci_high"]],
                **recalls,
                **res_ser,
                "brier": brier,
            }

        else:
            boot = paired_bootstrap_ci(
                pr_auc,
                y_te,
                scores,
                base_scores,
                groups_te,
                n_boot=N_BOOTSTRAP,
                seed=SEED,
            )

            metrics[name] = {
                "pr_auc": prauc,
                "pr_auc_ci": [boot["ci_low_a"], boot["ci_high_a"]],
                "delta_vs_baseline": boot["diff"],
                "delta_ci": [
                    boot["ci_low_diff"],
                    boot["ci_high_diff"],
                ],
                **recalls,
                **res_ser,
                "brier": brier,
            }

    # Test-set bounds.
    n = len(y_te)
    n_pos = int(y_te.sum())

    metrics["bounds"] = {
        "review_nothing_recall": 0.0,
        "review_everything_recall": 1.0,
        "random_review_recall@30": 0.30,
        "n_test": n,
        "n_positive": n_pos,
        "prevalence": float(y_te.mean()),
    }

    # Shuffled-label control.
    rng = np.random.default_rng(SEED)
    shuffled_scores = []

    for _ in range(100):
        y_shuf = rng.permutation(y_dev)

        model = LogisticRegressionL2(
            C=best_params.get("logistic_l2", {}).get("C", 1.0)
        ).fit(X_dev, y_shuf)

        shuffled_scores.append(
            pr_auc(y_te, model.predict_proba(X_te))
        )

    metrics["shuffled_label_control"] = {
        "mean_pr_auc": float(np.mean(shuffled_scores)),
        "ci_95": [
            float(np.quantile(shuffled_scores, 0.025)),
            float(np.quantile(shuffled_scores, 0.975)),
        ],
        "test_prevalence": float(y_te.mean()),
        "n_perm": 100,
        "note": "Should be near prevalence if no leakage",
    }

    # Demonstration comparing random region-level and grouped splits.
    rng2 = np.random.default_rng(SEED + 1)

    all_df = pd.concat([dev, test], ignore_index=True)
    n_all = len(all_df)

    idx = rng2.permutation(n_all)
    n_te_random = max(1, int(0.2 * n_all))

    te_idx = idx[:n_te_random]
    tr_idx = idx[n_te_random:]

    X_all = all_df[ALL_FEATURES].values.astype(float)
    y_all = all_df["needs_correction"].values.astype(int)

    leakage_model = LogisticRegressionL2(C=1.0).fit(
        X_all[tr_idx],
        y_all[tr_idx],
    )

    leak_prauc = pr_auc(
        y_all[te_idx],
        leakage_model.predict_proba(X_all[te_idx]),
    )

    metrics["leakage_demo"] = {
        "region_level_split_prauc": float(leak_prauc),
        "grouped_split_logistic_prauc": metrics["logistic_l2"]["pr_auc"],
        "note": (
            "Region-level split can leak writer/expression structure; "
            "grouped is the valid estimate."
        ),
    }

    # Logistic regression feature coefficients.
    logistic_model = models["logistic_l2"]

    if (
        hasattr(logistic_model, "coef_")
        and logistic_model.coef_ is not None
    ):
        metrics["feature_importance_logistic"] = {
            feature: float(coef)
            for feature, coef in zip(
                ALL_FEATURES,
                logistic_model.coef_,
            )
        }

    # Per-group test statistics.
    per_group = []

    for group_id, subset in test.groupby("group_id"):
        per_group.append({
            "group_id": group_id,
            "n": len(subset),
            "error_rate": float(
                subset["needs_correction"].mean()
            ),
            "mean_ser": float(subset["ser"].mean()),
        })

    metrics["per_group"] = per_group

    # Per-degradation test statistics.
    by_degradation = {}

    for degradation, subset in test.groupby("degradation"):
        by_degradation[degradation] = {
            "n": len(subset),
            "error_rate": float(
                subset["needs_correction"].mean()
            ),
            "mean_ser": float(subset["ser"].mean()),
        }

    metrics["by_degradation"] = by_degradation

    # Use the model selected by CV's 1-SE rule.
    # Never select a model using held-out test PR-AUC.
    best_name = selected_model

    metrics["selected_for_review"] = {
        "model": best_name,
        "source": "cv_chosen_by_1se",
        "cv_chosen": chosen_by_1se,
    }

    scores_best = preds[f"score_{best_name}"].values

    # Select the top 30% of samples for human review.
    k30 = max(1, int(round(0.3 * n)))

    order = np.argsort(-scores_best)
    reviewed = set(order[:k30].tolist())

    missed_idx = [
        i
        for i in range(n)
        if y_te[i] == 1 and i not in reviewed
    ]

    preds["missed_at_30"] = [
        i in missed_idx for i in range(n)
    ]

    preds["reviewed_at_30"] = [
        i in reviewed for i in range(n)
    ]

    # Export missed error cases for manual inspection.
    review = test.iloc[missed_idx][
        [
            "region_id",
            "reference_latex",
            "prediction_latex",
            "ser",
        ]
    ].copy()

    review["cause_tag"] = ""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    review.to_csv(
        RESULTS_DIR / "error_review.csv",
        index=False,
    )

    preds.to_csv(PREDICTIONS_PATH, index=False)

    output = {
        "metrics": metrics,
        "recall_curves": curves,
    }

    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    print(json.dumps({
        "selected_for_review": metrics["selected_for_review"],
        "models": {
            key: {
                "pr_auc": value.get("pr_auc"),
                "recall@30": value.get("recall@30"),
            }
            for key, value in metrics.items()
            if isinstance(value, dict) and "pr_auc" in value
        },
        "shuffled": metrics["shuffled_label_control"],
        "bounds": metrics["bounds"],
    }, indent=2))

    return metrics, preds


if __name__ == "__main__":
    evaluate()
