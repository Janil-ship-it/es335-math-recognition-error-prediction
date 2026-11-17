# Which Handwritten Math Lines Can We Trust?

**ES 335 · Machine Learning · IIT Gandhinagar**  
Janil Jain (25110143) & Manan Parmar (25110229)

**Research question**

> Can a learned error-prediction model identify more incorrect handwritten mathematical recognitions than a confidence-based baseline at the same human review budget?

**Website (full reasoning + interactive laboratory):** open [`docs/index.html`](docs/index.html) or enable GitHub Pages on `/docs`.  
A grader should be able to assess the project from the website alone; this repository is for verification and reproduction.

| Page | Content |
|------|---------|
| [Home](docs/index.html) | Research question & overview |
| [Problem](docs/problem.html) | Problem formulation |
| [Data](docs/data.html) | Dataset & sampling |
| [Methods](docs/methods.html) | Features, models, evaluation design |
| [Results](docs/results.html) | Metrics, charts, review-budget analysis |
| [Review lab](docs/explore.html) | Interactive exploration of regions |
| [Errors](docs/errors.html) | Error analysis |
| [Limitations](docs/limitations.html) | Limitations & threats to validity |
| [Reproduce](docs/reproduce.html) | Commands & environment |

---

## What one row represents

One **row** = one rendered human-written mathematical expression (from MathWriting ink) together with its recogniser output.

- **Grouping key:** normalised mathematical expression string (`normalizedLabel`).  
  Writer IDs are unavailable in the current pipeline, so writer-level leakage cannot be excluded.
- **Label:** `needs_correction` after LaTeX normalisation (reference vs prediction).
- **Images:** rendered from original handwritten ink with *simulated* blur, noise, rotation and shading. Degradation is stored for analysis only and is never used as a model feature.

## Dataset

- **Source:** [MathWriting](https://arxiv.org/abs/2404.10690) (arXiv:2404.10690), human-written expressions only.  
  Licence: **CC BY-NC-SA 4.0** (see `data/LICENSE-DATA.md`).
- Download the official archive, extract it, then run the preparation script (see Pipeline below).  
  The raw MathWriting files and derived regions are **not** committed to git (they are large and covered by the dataset licence terms).

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional (real HMER inference):

```bash
pip install torch transformers
```

The default recogniser candidate is `DGurgurov/im2latex` (see `src/config.py`). Other candidates can be selected at recognition time.

## Pipeline (do not re-run recognition while Colab is active)

Correct order:

```
00_prepare_mathwriting
  → 02_recognise (pilot)
  → decide_target
  → 02_recognise (full / resume)
  → 03_features
  → 04_train_cv
  → 05_evaluate_test
  → 06_export_site
  → pytest
```

### 1. Prepare MathWriting regions

```bash
python -m src.00_prepare_mathwriting \
  --root /path/to/mathwriting-2024 \
  --n 2000 \
  --seed 42
```

- Writes under `data/mathwriting/` (regions, metadata, splits).  
- Refuses to overwrite an existing `splits.json` unless `--force` is given.  
- **Do not re-run this while recognition is in progress.**

### 2. Recognition (running in Colab — do not start another copy)

```bash
# Pilot / smoke
python -m src.02_recognise --pilot

# Decide the binary target from pilot statistics
python -m src.decide_target

# Full or resume
python -m src.02_recognise --resume
```

Outputs: `results/recognitions.jsonl`, `results/target_decision.json`, run metadata.  
**Never delete or overwrite these files while the Colab job is running.**

### 3. Features

```bash
python -m src.03_features
```

Produces `data/features.csv`.

### 4. Cross-validation (development groups only)

```bash
python -m src.04_train_cv
```

- Grouped 5-fold CV, hyper-parameter grids, feature ablations.  
- Writes `results/cv_results.json` including `chosen_by_1se` (one-standard-error rule).  
- **Never touches the test set.**

### 5. Final test evaluation

```bash
python -m src.05_evaluate_test
```

- Trains models on all development groups with the CV-selected hyper-parameters.  
- Selects the model used for error-review examples from **`cv_results.json → chosen_by_1se`**, never from test PR-AUC.  
- Writes `results/test_metrics.json`, `results/predictions.csv`, bootstrap CIs, residual SER, etc.

### 6. Website export

```bash
python -m src.06_export_site
```

Writes `docs/data/site_results.json` and (when predictions exist) `docs/data/regions.json`.

### 7. Tests

```bash
pytest tests/ -q
```

Only fixtures and pure logic are exercised; the suite does not download models, run recognition, or regenerate the dataset.

---

## Models & evaluation

| Model | Role |
|-------|------|
| Confidence baseline | Fixed ranking by recogniser confidence |
| Logistic Regression (L2) | Linear model on all features |
| RBF SVM | Non-linear kernel (requires scikit-learn) |
| Random Forest | Tree ensemble |

**Metrics (held-out test groups):**

- PR-AUC (primary)
- Recall at 20 %, 30 %, 50 % review budget
- Residual symbol error rate (SER) after review
- Brier score
- Bootstrap confidence intervals (grouped)
- Shuffled-label control & leakage demo

**Model selection:** one-standard-error rule on grouped CV (`chosen_by_1se`). The test set is used only once, for final reporting.

## Results table

*(Populated automatically from `results/test_metrics.json` once the final evaluation finishes. Until then the cells show “—” on the website.)*

| Model | PR-AUC | Δ vs baseline (95 % CI) | Recall@30 % |
|-------|--------|--------------------------|-------------|
| Confidence baseline | — | — | — |
| Logistic regression | — | — | — |
| SVM (RBF) | — | — | — |
| Random forest | — | — | — |

Full interactive charts and the review-budget laboratory live on the [Results](docs/results.html) and [Review lab](docs/explore.html) pages.

## Design rules (must hold)

- Split by expression group (`normalizedLabel`) only; never by individual region.
- Forbidden features: reference LaTeX, marks, student/writer ID, question ID, degradation label, prediction LaTeX, SER, target itself.
- Test set is touched only inside `05_evaluate_test.py`.
- Model chosen for qualitative error review comes from CV (`chosen_by_1se`), never from test metrics.

## Limitations (high level)

- Writer IDs are absent → possible writer-level leakage.
- Degradation is simulated, not real scanner noise.
- Recogniser is used off-the-shelf (no fine-tuning).
- Sample size and group count are limited by the MathWriting subset and compute budget.
- Bootstrap CIs may still include zero; we do not claim a statistically conclusive win until the final numbers are in.

See the [Limitations](docs/limitations.html) page for the full discussion.

## Citations

- **MathWriting** — Gervais, P., et al. *MathWriting: A Dataset For Handwritten Mathematical Expression Recognition*. arXiv:2404.10690, 2024. CC BY-NC-SA 4.0. https://arxiv.org/abs/2404.10690
- Candidate recognisers (inference only, not fine-tuned): models such as `DGurgurov/im2latex` and TexTeller variants.

## AI disclosure

Portions of the website, documentation and static code improvements were assisted by Grok (xAI). All experimental design decisions, evaluation methodology, and final claims remain the responsibility of the authors. Recognition is performed with publicly available pretrained models; no student data is used.

## Licence

Code: see `LICENSE`.  
MathWriting data: CC BY-NC-SA 4.0 (see `data/LICENSE-DATA.md`).
