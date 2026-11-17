import numpy as np
from src.metrics import pr_auc, paired_bootstrap_ci, symbol_error_rate, recall_at_review_rate
from src.latex_norm import normalise_latex, needs_correction
from src.mathwriting import token_error_rate

def test_normalise():
    assert normalise_latex(r"a  +  b") == "a+b"
    assert normalise_latex(r"\leq") == r"\le"

def test_needs_correction_ws():
    assert needs_correction(r"a+b", r"a + b") == 0

def test_pr_auc_all_tied_near_prevalence():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0])
    scores = np.ones(len(y)) * 0.5
    ap = pr_auc(y, scores)
    assert abs(ap - y.mean()) < 0.05

def test_pr_auc_perfect():
    assert pr_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) > 0.95

def test_paired_bootstrap_preserves_multiplicity():
    y = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    groups = np.array(["A", "A", "B", "B"])
    def n_rows(yt, sc):
        return float(len(yt))
    out = paired_bootstrap_ci(n_rows, y, scores, None, groups, n_boot=200, seed=0)
    assert abs(out["estimate"] - 4.0) < 1e-9

def test_token_ser_not_char_level():
    assert token_error_rate(r"x^{10}", r"x^{2}") < 0.6

def test_symbol_error_rate_is_token():
    assert abs(symbol_error_rate(r"x^{10}", r"x^{2}") - token_error_rate(r"x^{10}", r"x^{2}")) < 1e-9

def test_recall_at_rate():
    y = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    assert abs(recall_at_review_rate(y, scores, 0.5) - 0.5) < 1e-9
