"""Evaluation metrics, bootstrap CIs, and re-exports of latex normalisation."""
from __future__ import annotations
import numpy as np

from .latex_norm import normalise_latex, needs_correction  # noqa: F401
from .mathwriting import token_error_rate, tokenize_expression

try:
    from sklearn.metrics import average_precision_score
    _HAS_SK = True
except ImportError:
    _HAS_SK = False


def symbol_error_rate(pred: str, ref: str) -> float:
    return token_error_rate(pred, ref)


def pr_auc(y_true, scores) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_true.sum() == 0:
        return 0.0
    if _HAS_SK:
        return float(average_precision_score(y_true, scores))
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    recall = tp / max(tp[-1], 1)
    precision = tp / np.maximum(tp + fp, 1)
    return float(np.trapezoid(np.concatenate([[1.0], precision]), np.concatenate([[0.0], recall])))


def recall_at_review_rate(y_true, scores, rate: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    k = max(1, int(round(rate * n)))
    order = np.argsort(-scores)[:k]
    return float(y_true[order].sum()) / max(int(y_true.sum()), 1)


def residual_ser(y_true, scores, ser, rate):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    ser = np.asarray(ser, dtype=float)
    n = len(y_true)
    k = max(1, int(round(rate * n)))
    order = np.argsort(-scores)
    residual = ser.copy()
    residual[order[:k]] = 0.0
    return float(residual.mean())


def brier_score(y_true, probs):
    return float(np.mean((np.asarray(probs) - np.asarray(y_true, dtype=float)) ** 2))


def paired_bootstrap_ci(metric_fn, y_true, scores_a, scores_b, groups, n_boot=2000, seed=42, alpha=0.05):
    """Grouped bootstrap WITH multiplicity (fixed: no np.isin dedup)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    n_g = len(unique_groups)
    group_to_idx = {g: np.where(groups == g)[0] for g in unique_groups}

    def _stat(drawn_groups):
        parts = [group_to_idx[g] for g in drawn_groups]
        idx = np.concatenate(parts) if parts else np.array([], dtype=int)
        if len(idx) == 0:
            return (0.0, 0.0) if scores_b is not None else 0.0
        m_a = metric_fn(y_true[idx], scores_a[idx])
        if scores_b is None:
            return m_a
        m_b = metric_fn(y_true[idx], np.asarray(scores_b, dtype=float)[idx])
        return m_a, m_a - m_b

    orig = _stat(unique_groups)
    boots = []
    for _ in range(n_boot):
        drawn = rng.choice(unique_groups, size=n_g, replace=True)
        boots.append(_stat(drawn))
    if scores_b is None:
        boots = np.asarray(boots, dtype=float)
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        return {"estimate": float(orig), "ci_low": float(lo), "ci_high": float(hi)}
    else:
        boots = np.asarray(boots, dtype=float)
        est_a, est_diff = orig
        lo_a, hi_a = np.quantile(boots[:, 0], [alpha / 2, 1 - alpha / 2])
        lo_d, hi_d = np.quantile(boots[:, 1], [alpha / 2, 1 - alpha / 2])
        return {
            "estimate_a": float(est_a), "ci_low_a": float(lo_a), "ci_high_a": float(hi_a),
            "diff": float(est_diff), "ci_low_diff": float(lo_d), "ci_high_diff": float(hi_d),
        }


def recall_curve(y_true, scores, steps=100):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    total_pos = max(int(y_true.sum()), 1)
    curve = []
    for pct in range(1, steps + 1):
        k = max(1, int(round(pct / 100.0 * n)))
        rec = float(y_sorted[:k].sum()) / total_pos
        curve.append({"budget_pct": pct, "recall": rec, "n_reviewed": k})
    return curve
