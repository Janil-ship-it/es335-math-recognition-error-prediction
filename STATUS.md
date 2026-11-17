# Implementation status

Pipeline code is switched to real MathWriting data (preparation, recognition, features, CV, evaluation, site export).
Committed metrics and site_results come from the last successful development-proxy / transitional run (n_test ≈ 146).
A fresh Colab / local run against the official MathWriting archive is required to regenerate cv_results.json, full features, and final numbers.

## Quick Colab sequence

```bash
python -m src.00_prepare_mathwriting --root mathwriting-2024 --n 2000 --seed 42 --force
python -m src.02_recognise --real --model DGurgurov/im2latex --limit 30 --dev-only --verify
python -m src.decide_target
python -m src.02_recognise --real --model DGurgurov/im2latex
python -m src.03_features
python -m src.04_train_cv
python -m src.05_evaluate_test
python -m src.06_export_site
pytest tests/ -q
```

## Post-merge notes (2026-11-18)

- Website pages (problem, data, methods, results, explore, errors, limitations, reproduce) restored with full academic content after the merge had reduced them to stubs.
- `src/config.py` now falls back to `data/splits.json` when `data/mathwriting/splits.json` is absent, so committed splits remain usable.
- Interactive results charts, review-lab budget slider, and region explorer are wired to `docs/data/site_results.json` and `demo_regions.json`.
- All existing tests pass (OpenCV-dependent tests require opencv-python-headless in the environment).
