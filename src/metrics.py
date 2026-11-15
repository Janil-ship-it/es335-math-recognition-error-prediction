"""Evaluation metrics and LaTeX normalisation."""
from __future__ import annotations
import re
import numpy as np

_SPACE_RE = re.compile(r"\s+")
_CMD_EQUIV = {r"\leq": r"\le", r"\geq": r"\ge", r"\neq": r"\ne", r"\cdot": r"\times", r"\to": r"\rightarrow"}

def normalise_latex(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    s = _SPACE_RE.sub("", s)
    for a, b in _CMD_EQUIV.items():
        s = s.replace(a, b)
    return s

def needs_correction(pred: str, ref: str) -> int:
    return int(normalise_latex(pred) != normalise_latex(ref))

def symbol_error_rate(pred: str, ref: str) -> float:
    p = list(normalise_latex(pred))
    r = list(normalise_latex(ref))
    if not p and not r:
        return 0.0
    n, m = len(r), len(p)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == p[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return float(dp[n, m]) / max(n, m, 1)

def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    recall = tp / max(tp[-1], 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))

def recall_at_review_rate(y_true: np.ndarray, scores: np.ndarray, rate: float) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    k = max(1, int(round(rate * n)))
    order = np.argsort(-scores)[:k]
    return float(y_true[order].sum()) / max(y_true.sum(), 1)

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
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_g = len(unique_groups)
    def _stat(idx):
        mask = np.isin(groups, unique_groups[idx])
        m_a = metric_fn(y_true[mask], scores_a[mask])
        if scores_b is None:
            return m_a
        m_b = metric_fn(y_true[mask], scores_b[mask])
        return m_a, m_a - m_b
    orig = _stat(np.arange(n_g))
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n_g, size=n_g, replace=True)
        boots.append(_stat(idx))
    if scores_b is None:
        boots = np.asarray(boots)
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        return {"estimate": float(orig), "ci_low": float(lo), "ci_high": float(hi)}
    else:
        boots = np.asarray(boots)
        est_a, est_diff = orig
        lo_a, hi_a = np.quantile(boots[:, 0], [alpha / 2, 1 - alpha / 2])
        lo_d, hi_d = np.quantile(boots[:, 1], [alpha / 2, 1 - alpha / 2])
        return {
            "estimate_a": float(est_a), "ci_low_a": float(lo_a), "ci_high_a": float(hi_a),
            "diff": float(est_diff), "ci_low_diff": float(lo_d), "ci_high_diff": float(hi_d),
        }
