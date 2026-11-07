# CLAUDE.md

Read SPEC.md before doing anything. It defines the task, split rules, metrics and claims. If a request conflicts with SPEC.md, stop and ask.

## Project in one line

Predict which handwritten-maths lines a pre-trained recogniser got wrong, so only risky lines go to human review. Compare learned flaggers against a fixed confidence-threshold baseline at equal review budgets.

## Stack

- Python 3.11, scikit-learn, numpy, pandas, scipy, matplotlib
- Pre-trained recogniser (TexTeller or UniMERNet), inference only, never fine-tuned
- pytest for tests, ruff for lint
- Pin every dependency in `requirements.txt`

## Repo layout

```
handwritten-math-routing/
  README.md
  requirements.txt
  SPEC.md
  CLAUDE.md
  data/
    sample/            permitted sample regions only
    splits.json        group-to-set assignment, written once before modelling
    references.csv     region_id, group_id, reference_latex, transcriber
  src/
    metrics.py         normalisation, SER, PR-AUC, recall@k, residual error, bootstrap
    01_extract_regions.py
    02_recognise.py    recogniser output and per-token confidences
    03_features.py
    04_train_cv.py     grouped CV and tuning
    05_evaluate_test.py  the ONLY place the test set is read
  tests/
  results/             precomputed predictions and metrics
  docs/                GitHub Pages site
```

## Hard rules (never break these)

1. **Test set is off limits** everywhere except `05_evaluate_test.py`, and that script runs once per frozen model set. Never load test rows for exploration, tuning, plotting or "just a quick check".
2. **Split by group (student or writer), never by region.** Every region of a group lives in exactly one set. Assert this in code and in a test.
3. **`data/splits.json` is written once, with a fixed seed, before any modelling.** Never regenerate it after results exist. Never edit it by hand.
4. **Fit on training folds only:** scalers, class weights, imputers, the mean-vs-min confidence choice, hyperparameters. Use sklearn `Pipeline` so fitting inside CV is automatic.
5. **Forbidden features:** reference text, marks, student/group ID, question ID. Features come only from the allowed list in SPEC.md section 5. If you add a feature, say so and explain why it is available at prediction time.
6. **Never copy recogniser output into references.** References come from human transcription only.
7. **Validation and test keep natural class rates.** Class weights apply in training only.
8. **Fixed seeds everywhere** (numpy, sklearn, any torch). Log the seed with every result.
9. **No tuning on test, no peeking at test metrics to choose among models.** Model selection uses validation PR-AUC and the one-standard-error rule from SPEC.md.
10. **Raw student scans and anything identifying a student never go in git.** Keep them in a gitignored folder. No names, roll numbers or API keys in code, logs, commits or the site.
11. **Do not fine-tune the recogniser.**

## How to work

- **Plan first.** For any non-trivial task, write a short plan and wait for approval before coding.
- **One thin slice per session.** Do not build the whole project at once. Finish, test, commit, then stop.
- **Tests with the code.** Every metric function, normalisation rule and split function gets a unit test. Metric tests must use known answers (for example random scores give recall near the review rate, perfect scores give near 100%).
- **Run before claiming done.** Run the code and the tests. Paste real output. Never say "should work".
- **Print shapes and distributions** after any data step: row counts per set, class balance, group counts, missing values.
- **Never invent APIs, model arguments or paper results.** If unsure how a library or the recogniser works, read its docs or source and say what you verified. If you cannot verify, say so.
- **Small diffs.** Change only what the task needs. No drive-by refactors. No new dependencies without asking.
- **Config over constants.** Review rates (0.2, 0.3, 0.5), seeds, paths and bootstrap count live in one config file.
- **Commit after each working slice** with a clear message. Do not commit large data or model weights.

## Code style

- Type hints and short docstrings on public functions.
- Small pure functions for anything the tests touch. Keep I/O at the edges.
- Plots: label axes, include units and the number of students and regions in the title or caption.
- Report every result with 95% bootstrap intervals over students, never a single bare number.

## Reporting rules

- Report ALL models, not just the winner.
- Report missed errors prominently, with examples. A confident miss is the costliest failure.
- Token savings are an estimate unless measured with logged settings. Label them as such.
- State limitations plainly: small test set, one recogniser, one assignment format, references made by us.
- If the learned models do not beat the baseline, say so. Do not search for a flattering metric.

## Things to flag to the humans immediately

- Anything that looks like leakage: suspiciously high scores, a feature that correlates almost perfectly with the target, a group appearing in two sets.
- The recogniser being nearly always right or nearly always wrong (the target then carries no signal).
- Any need to touch the test set or splits.json.
- Missing writer IDs in Plan B data.

## Before ending a session

1. Tests pass. 2. Lint clean. 3. Work committed. 4. Summarise what changed, what was verified, what is still unverified, and the next slice.