# Implementation Report

**Project:** es335-math-recognition-error-prediction (MathGuard)  
**Date:** 2026-11-17  
**Status:** Research-quality development pipeline + interactive site complete on Plan B proxy.

## Completed

### ML pipeline
- [x] Synthetic development dataset (`src/data_generator.py`) with group structure and realistic feature–error links
- [x] Fixed student-level splits (`data/splits.json`)
- [x] LaTeX normalisation + SER + unit tests
- [x] Image preprocessing & line-region extraction (OpenCV, `01_extract_regions.py`)
- [x] Recogniser interface with proxy backend + real-model hook (`02_recognise.py`)
- [x] Feature extraction: image quality, structure counts, confidence (`03_features.py`)
- [x] Models: confidence baseline, L2 logistic, random forest, RBF SVM (calibrated)
- [x] Grouped 5-fold CV with feature ablation (confidence / image / structure / all)
- [x] One-standard-error model selection rule
- [x] Single held-out test evaluation with paired bootstrap CIs over students
- [x] PR-AUC, recall@20/30/50%, residual SER, Brier score
- [x] Shuffled-label leakage control
- [x] Feature importance (logistic coefficients)
- [x] PR-curve and reliability-bin exports for the site

### Dashboard (static, GitHub Pages ready)
- [x] Multi-page site: Home, Data, Methods, Results, Explore, Limitations, Reproduce
- [x] Results page: live metrics table, PR curves (Chart.js), feature-importance bars
- [x] Explore / Review Lab: budget slider (10–100%), baseline vs model comparison, dual-axis chart
- [x] Region Explorer: KaTeX rendering, filters (errors / misses / false alarms), sortable
- [x] Connected to real generated JSON (`docs/data/site_results.json`, `demo_regions.json`)
- [x] Clean light theme, responsive layout, sticky nav

### Engineering
- [x] Fixed `__inti__.py` typo
- [x] Expanded `requirements.txt` (scikit-learn, opencv-python-headless)
- [x] Updated README with reproduction steps and current numbers
- [x] Tests pass (`pytest`)
- [x] No fabricated experimental numbers — all metrics produced by running the code

## Remaining / blocked

| Item | Reason | Mitigation |
|------|--------|------------|
| Real Plan A student scans | Privacy; not shared yet | Proxy path is complete; same code when data arrives |
| Full TexTeller / UniMERNet inference | Large weights + torch + preferably GPU | Clean interface in `02_recognise.py`; set `USE_REAL_MODEL=True` when available |
| Inter-annotator agreement | Needs dual human transcription of real scans | Protocol documented in SPEC / Methods |
| Mark-agreement analysis | Requires Plan A marks | Optional downstream check only |
| Stronger line detector | Projection method is a prototype | Manual correction allowed in v1 |
| More unit tests for CV / splits | Time | Existing metric tests + integrity checks in evaluation |

## How to verify

```bash
pip install -r requirements.txt
python -m src.data_generator
python -m src.04_train_cv
python -m src.05_evaluate_test
pytest tests/ -q
# open docs/index.html or docs/explore.html
```

## Scientific honesty notes

- Bootstrap intervals for Δ vs baseline still include zero (small test set of 6 groups).
- Shuffled-label PR-AUC ≈ 0.56 ≈ prevalence → no obvious leakage.
- Development proxy deliberately has a high error rate to stress-test ranking; real course handwriting may differ.
