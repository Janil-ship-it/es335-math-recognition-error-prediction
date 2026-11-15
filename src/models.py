"""Classical models with NumPy/SciPy."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def standardise(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std, mean, std

@dataclass
class FixedConfidenceBaseline:
    mode: str = "mean"
    feature_idx: dict = field(default_factory=dict)
    def fit(self, X, y, feature_names: Sequence[str]):
        from .metrics import pr_auc
        self.feature_idx = {n: i for i, n in enumerate(feature_names)}
        mean_scores = -X[:, self.feature_idx["mean_token_conf"]]
        min_scores = -X[:, self.feature_idx["min_token_conf"]]
        self.mode = "mean" if pr_auc(y, mean_scores) >= pr_auc(y, min_scores) else "min"
        return self
    def predict_proba(self, X):
        col = "mean_token_conf" if self.mode == "mean" else "min_token_conf"
        return 1.0 - X[:, self.feature_idx[col]]
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

@dataclass
class LogisticRegressionL2:
    C: float = 1.0
    max_iter: int = 400
    coef_: np.ndarray | None = None
    intercept_: float = 0.0
    mean_ = None
    std_ = None
    def fit(self, X, y, sample_weight=None):
        X, self.mean_, self.std_ = standardise(X)
        n, d = X.shape
        if sample_weight is None:
            sample_weight = np.ones(n)
        pos, neg = y.sum(), n - y.sum()
        w = sample_weight.copy()
        if pos > 0 and neg > 0:
            w[y == 1] *= n / (2 * pos)
            w[y == 0] *= n / (2 * neg)
        def loss(theta):
            intercept, coef = theta[0], theta[1:]
            logits = intercept + X @ coef
            neg_ll = -np.sum(w * (y * (-np.logaddexp(0, -logits)) + (1 - y) * (-np.logaddexp(0, logits))))
            return neg_ll + 0.5 / self.C * np.sum(coef ** 2)
        res = minimize(loss, np.zeros(d + 1), method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.intercept_ = float(res.x[0])
        self.coef_ = res.x[1:].copy()
        return self
    def predict_proba(self, X):
        Xs, _, _ = standardise(X, self.mean_, self.std_)
        return expit(self.intercept_ + Xs @ self.coef_)
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

def _best_split(X, y, min_leaf=5):
    n, d = X.shape
    best_gain, best_j, best_t = -1.0, 0, 0.0
    parent_gini = 1.0 - ((y == 0).mean() ** 2 + (y == 1).mean() ** 2)
    for j in range(d):
        vals = np.unique(X[:, j])
        if len(vals) < 2: continue
        for t in np.quantile(vals, [0.25, 0.5, 0.75]):
            left = y[X[:, j] <= t]
            right = y[X[:, j] > t]
            if len(left) < min_leaf or len(right) < min_leaf: continue
            g_l = 1.0 - ((left == 0).mean() ** 2 + (left == 1).mean() ** 2)
            g_r = 1.0 - ((right == 0).mean() ** 2 + (right == 1).mean() ** 2)
            gain = parent_gini - (len(left)/n)*g_l - (len(right)/n)*g_r
            if gain > best_gain:
                best_gain, best_j, best_t = gain, j, t
    return best_j, best_t, best_gain

@dataclass
class _TreeNode:
    j: int = -1
    t: float = 0.0
    left = None
    right = None
    pred: float = 0.5

def _build_tree(X, y, depth, max_depth, min_leaf):
    node = _TreeNode()
    node.pred = float(y.mean()) if len(y) else 0.5
    if depth >= max_depth or len(y) < 2*min_leaf or y.std() < 1e-6:
        return node
    j, t, gain = _best_split(X, y, min_leaf)
    if gain <= 0: return node
    node.j, node.t = j, t
    mask = X[:, j] <= t
    node.left = _build_tree(X[mask], y[mask], depth+1, max_depth, min_leaf)
    node.right = _build_tree(X[~mask], y[~mask], depth+1, max_depth, min_leaf)
    return node

def _predict_tree(node, x):
    if node.left is None: return node.pred
    return _predict_tree(node.left if x[node.j] <= node.t else node.right, x)

@dataclass
class SimpleRandomForest:
    n_estimators: int = 40
    max_depth: int = 4
    min_leaf: int = 8
    max_features: float = 0.7
    seed: int = 42
    trees: list = field(default_factory=list)
    mean_ = None
    std_ = None
    feat_idx_ = None
    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X, self.mean_, self.std_ = standardise(X)
        n, d = X.shape
        k = max(1, int(self.max_features * d))
        self.trees, self.feat_idx_ = [], []
        for _ in range(self.n_estimators):
            idx = rng.choice(n, size=n, replace=True)
            feats = rng.choice(d, size=k, replace=False)
            self.feat_idx_.append(feats)
            self.trees.append(_build_tree(X[idx][:, feats], y[idx], 0, self.max_depth, self.min_leaf))
        return self
    def predict_proba(self, X):
        Xs, _, _ = standardise(X, self.mean_, self.std_)
        preds = np.zeros(len(Xs))
        for tree, feats in zip(self.trees, self.feat_idx_):
            for i, x in enumerate(Xs):
                preds[i] += _predict_tree(tree, x[feats])
        return preds / len(self.trees)
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
