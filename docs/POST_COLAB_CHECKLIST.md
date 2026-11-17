# Post-Colab Checklist

Use this checklist **only after** the Google Colab recognition job has finished successfully.  
Do **not** run these steps while recognition is still writing `results/recognitions.jsonl`.

## 0. Stop recognition & back up

- [ ] Confirm the Colab runtime has stopped (or the local process has exited cleanly).
- [ ] Copy the entire `results/` directory to a safe backup location, e.g.

```bash
cp -a results results_backup_$(date +%Y%m%d_%H%M%S)
```

- [ ] Also back up `data/mathwriting/` if it lives on the same machine.
- [ ] **Never delete** `results/recognitions.jsonl` or `results/target_decision.json`.

## 1. Verify recognition outputs

Expected files after a successful run:

| File | Purpose |
|------|---------|
| `results/recognitions.jsonl` | One JSON object per region (prediction, confidence, …) |
| `results/target_decision.json` | Binary target rule chosen from pilot statistics |
| `results/run_metadata.json` | Recogniser id, revision, seed, timestamps |

Quick sanity checks:

```bash
# Count completed recognitions
wc -l results/recognitions.jsonl

# Inspect target decision
python -c "import json; print(json.load(open('results/target_decision.json')))"

# Confirm splits are still the ones used for recognition
python -c "import json; s=json.load(open('data/mathwriting/splits.json')); print(len(s['dev_groups']), len(s['test_groups']))"
```

- [ ] Recognition count matches the expected number of regions.
- [ ] `target_decision.json` exists and is consistent with the pilot.
- [ ] Train / development / test group lists are unchanged.

## 2. Feature extraction

```bash
python -m src.03_features
```

Expected output:

- `data/features.csv`

- [ ] `features.csv` has one row per recognised region.
- [ ] All columns in `ALL_FEATURES` (see `src/config.py`) are present and finite.
- [ ] No forbidden columns (reference, prediction LaTeX, SER, target, group ids used as features, …) appear in the feature matrix.

## 3. Cross-validation (development groups only)

```bash
python -m src.04_train_cv
```

Expected output:

- `results/cv_results.json` containing at least:
  - `best_params`
  - `chosen_by_1se`  ← **this is the only source of truth for model selection**
  - `ablation`
  - `n_dev_regions`, `error_rate`

- [ ] `chosen_by_1se.model` is one of the trained models.
- [ ] Test groups never appear in any CV fold.

## 4. Final held-out evaluation

```bash
python -m src.05_evaluate_test
```

**Critical:** the script must select the model used for error-review examples from `cv_results.json → chosen_by_1se`.  
It must **never** choose the model by maximising test PR-AUC.

Expected outputs:

- `results/test_metrics.json`
- `results/predictions.csv`
- `results/error_review.csv` (qualitative missed-error examples at 30 % budget)

- [ ] `test_metrics.json` contains PR-AUC, recall@20/30/50, residual SER, Brier, bootstrap CIs for every model.
- [ ] `metrics.selected_for_review.source` is `"cv_chosen_by_1se"`.
- [ ] Shuffled-label control is near prevalence.

## 5. Website export

```bash
python -m src.06_export_site
```

Expected outputs:

- `docs/data/site_results.json`
- `docs/data/regions.json` (and optional crop thumbnails under `docs/data/crops/`)

- [ ] Open `docs/index.html` locally; numbers and charts load without console errors.
- [ ] Theme toggle (sun/moon) works and persists across pages.
- [ ] Review-budget slider and region explorer populate from the real JSON files.
- [ ] Missing keys display “—” / “Awaiting final evaluation”, never fabricated numbers.

## 6. Tests

```bash
pytest tests/ -q
```

- [ ] All tests pass.
- [ ] No test downloaded a model, ran recognition, or rewrote real results.

## 7. Documentation & final numbers

- [ ] Update the results table in `README.md` from the real `test_metrics.json` (or leave the “—” placeholders if you prefer the website to be the single source of truth).
- [ ] Remove any remaining “synthetic / proxy” language that no longer applies.
- [ ] Confirm `SPEC.md`, `STATUS.md` and `IMPLEMENTATION_REPORT.md` do not contradict the final methodology.

## 8. Commit, push & deploy

```bash
git status
git add -A
git commit -m "Final evaluation results and site export after Colab recognition"
git push origin <your-branch-or-main>
```

- [ ] Enable or refresh GitHub Pages (source = `/docs`).
- [ ] Visit the live site and verify every page, the theme switcher, charts and the interactive laboratory.

## Generated artefacts that must exist after each stage

| Stage | Required artefacts |
|-------|--------------------|
| Prepare | `data/mathwriting/splits.json`, region images + metadata |
| Recognise | `results/recognitions.jsonl`, `results/target_decision.json`, `results/run_metadata.json` |
| Features | `data/features.csv` |
| CV | `results/cv_results.json` (with `chosen_by_1se`) |
| Test eval | `results/test_metrics.json`, `results/predictions.csv` |
| Site export | `docs/data/site_results.json`, `docs/data/regions.json` |

Once all boxes above are checked, the project is ready for grading.
