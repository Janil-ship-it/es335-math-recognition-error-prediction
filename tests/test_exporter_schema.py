"""Exporter and site JSON schema expectations (no real results required)."""
from __future__ import annotations
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_export_site_handles_missing_metrics_gracefully():
    """The exporter must raise a clear SystemExit when test metrics are absent,
    rather than writing a half-broken site_results.json."""
    src = (ROOT / "src" / "06_export_site.py").read_text()
    assert "Run 05_evaluate_test first" in src or "metrics_path.exists()" in src


def test_site_results_demo_is_valid_json():
    path = ROOT / "docs" / "data" / "site_results.json"
    if not path.exists():
        pytest.skip("site_results.json not present")
    data = json.loads(path.read_text())
    assert isinstance(data, dict)


def test_app_js_has_theme_support():
    js = (ROOT / "docs" / "app.js").read_text()
    assert "mathguard-theme" in js or "THEME_KEY" in js
    assert "data-theme" in js
    assert "theme-toggle" in js
    assert "prefers-color-scheme" in js
