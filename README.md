# Which Handwritten Math Lines Can We Trust?

**ES 335 · Machine Learning**  
Janil Jain (25110143) & Manan Parmar (25110229)

Predict which handwritten-math lines a recogniser got wrong so only risky lines go to human review.

## Research question

At the same review budget (20%, 30%, 50% of lines), does a learned flagger catch more wrong readings than a fixed confidence-threshold baseline?

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.data_generator
python -m src.04_train_cv
python -m src.05_evaluate_test
pytest tests/ -q
# open docs/index.html
```

## Results (held-out test groups)

| Model | PR-AUC | Δ vs baseline (95% CI) | Recall@30% |
|---|---|---|---|
| Confidence threshold | 0.715 | — | 35.9% |
| Logistic regression | 0.816 | +0.100 [−0.006, 0.190] | 42.4% |
| Random forest | 0.830 | +0.115 [−0.007, 0.186] | 41.3% |

Shuffled-label control ≈ 0.56 (near prevalence → no leakage).

## Data note

Plan A (course scripts) was not available at build time. Experiments use a statistically realistic development proxy that follows documented HMER failure modes. All split rules, metrics, bootstrap and the website are ready for real data. Raw student scans never enter git.

## Layout

- `src/` — metrics, models, grouped CV, final test evaluation
- `data/splits.json` — group-to-set assignment (fixed before modelling)
- `results/` — CV + test metrics and predictions
- `docs/` — interactive GitHub Pages site (review-budget slider + region explorer)
- `tests/` — unit tests for metrics

## Design rules (SPEC)

- Split by student/writer group only
- Forbidden features: reference text, marks, student ID, question ID
- Test set touched only in `05_evaluate_test.py`
- Primary metric: PR-AUC; report missed errors prominently
