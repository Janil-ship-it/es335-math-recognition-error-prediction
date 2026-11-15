import numpy as np
from src.metrics import normalise_latex, needs_correction, pr_auc, recall_at_review_rate, symbol_error_rate

def test_normalise():
    assert normalise_latex(r"a  +  b") == "a+b"
    assert normalise_latex(r"\leq") == r"\le"

def test_needs_correction():
    assert needs_correction(r"a+b", r"a + b") == 0
    assert needs_correction(r"a+b", r"a+c") == 1

def test_ser():
    assert symbol_error_rate("a b c", "a b c") == 0.0
    assert abs(symbol_error_rate("a b c", "a x c") - 1/3) < 1e-6

def test_pr_auc_perfect():
    assert pr_auc(np.array([1,1,0,0]), np.array([0.9,0.8,0.2,0.1])) > 0.99

def test_recall():
    y = np.array([1,1,1,0,0,0,0,0,0,0])
    s = np.array([0.9,0.8,0.1,0.7,0.6,0.5,0.4,0.3,0.2,0.05])
    assert abs(recall_at_review_rate(y, s, 0.2) - 2/3) < 1e-6
