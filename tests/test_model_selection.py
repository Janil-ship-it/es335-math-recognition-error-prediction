"""Model selection must come from CV 1SE, never from test metrics."""
from __future__ import annotations
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_evaluate_source_contains_1se_logic():
    """Static check: 05_evaluate_test.py must reference chosen_by_1se and must not
    select the review model by maximising test PR-AUC."""
    src = (ROOT / "src" / "05_evaluate_test.py").read_text()
    assert "chosen_by_1se" in src
    assert "cv_chosen_by_1se" in src or "chosen_by_1se" in src
    # The old anti-pattern
    assert "key=lambda n: metrics[n][\"pr_auc\"]" not in src
    assert "key=lambda n: metrics[n]['pr_auc']" not in src


def test_cv_script_emits_chosen_by_1se():
    src = (ROOT / "src" / "04_train_cv.py").read_text()
    assert "chosen_by_1se" in src
    assert "val_prauc_mean" in src
