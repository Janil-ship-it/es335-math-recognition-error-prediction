# SPEC: Which Handwritten Math Lines Can We Trust?

Course: ES 335. Team: Janil Jain, Manan Parmar. Final submission: Nov 17, 2026.
Source of truth: the revised proposal (Oct 26, 2026). If this file and the proposal disagree, ask the humans.

## 1. Goal

Given a cropped image of ONE line of handwritten maths and the LaTeX that a pre-trained recogniser produced for it, predict whether that LaTeX contains an error. Flag risky lines for human review. Unflagged lines keep the recogniser's text.

Success is NOT accuracy. Success is: at the same review budget (20%, 30%, 50% of lines), does a learned flagger catch more wrong lines than the fixed confidence-threshold baseline?

A modest or negative result is acceptable. If the baseline matches the learned models, we report that.

## 2. Definitions

| Element | Definition |
|---|---|
| Unit (one row) | One region: a cropped line/step of one student's working, plus the recogniser's output |
| Target | `needs_correction` = 1 if recogniser LaTeX != reference LaTeX after normalisation, else 0. Also record symbol error rate (SER) |
| Prediction time | Right after the recogniser runs, before grading or review |
| Allowed inputs (features) | Image-quality measures of the crop, recogniser confidence values, structure counts. Optional: a second recogniser's output |
| FORBIDDEN as input | Reference text, marks, student ID, question ID |
| Group | Student (Plan A) or writer (Plan B). Used for splitting only, never as a feature |

## 3. Data

- **Plan A (preferred):** anonymised course scans from Prof. Bhatia, with checked references and marks. Access not yet confirmed. Target: about 25 to 30 students, 500+ regions.
- **Plan B (development and fallback):** MathWriting (arXiv:2404.10690, CC BY-NC-SA 4.0). Use a random subset of a few thousand human-written samples, rendered to images with the authors' code. Check whether writer IDs exist. If they do not, no LaTeX expression may appear in both train and test, and the writer-leakage limitation must be stated in results.
- Plan B ink is cleaner than paper scans. Add simulated blur, noise and skew. Never pool Plan A and Plan B without saying so.
- Raw scans are private. Never commit them. Publish nothing identifiable.

## 4. Pipeline

1. Capture check: flag blank, cut-off or very blurry scans. Report them, do not silently drop.
2. Region extraction: line-level crops (manual correction allowed for the first version). Diagrams are kept as linked images, not transcribed.
3. Light cleaning: grayscale, deskew, pad and resize to the recogniser's input size.
4. Recognition: pre-trained recogniser, NOT fine-tuned. Candidates: TexTeller, UniMERNet. Choose in a pilot on about 30 lines. It must return per-token scores and run on our hardware.
5. Feature extraction.
6. Scaling: standardise with training-fold statistics only.
7. Class balance: class weights in training only. Validation and test keep natural rates.

## 5. Features

- Confidence: mean token confidence, minimum token confidence, output length.
- Image quality: blur, contrast, ink density, stroke width, skew, crop size, background noise.
- Structure counts: fractions, powers, roots, subscripts in the output LaTeX.

## 6. Labelling and normalisation

- Both team members transcribe the same subset independently into normalised LaTeX. Measure agreement. Resolve differences by discussion.
- The recogniser output is NEVER copied into the reference.
- Normalise before comparing: spacing, bracket style, equivalent commands. Every normalisation rule must have a unit test.

## 7. Models

| Model | Notes |
|---|---|
| Fixed confidence threshold (BASELINE) | Rank by mean or min token confidence. Choose mean vs min on training folds |
| Review nothing / review everything | Bounds |
| Logistic regression | L2, standardised features |
| SVM | RBF kernel, calibrated probabilities |
| Random forest | |

Selection rule: pick the simplest model whose validation PR-AUC is within one standard error of the best. Report ALL models.

Ablation: image-quality only, confidence only, all features.

## 8. Evaluation design

Fixed BEFORE any model is compared. Do not change after seeing results.

- Test set: about 20% of students. One final evaluation of frozen models.
- Development set: the remaining students. Training, tuning and selection by 5-fold CV grouped by student.
- Student-to-set assignment uses a fixed seed and is saved to `data/splits.json` before modelling.
- After CV, retrain the chosen models on all development students and score ONCE on test.
- Robustness check (never used for selection): hold out one question at a time.

### Metrics

| Metric | Notes |
|---|---|
| PR-AUC (primary) | Threshold-free |
| Recall of wrong regions at 20%, 30%, 50% review | Matches a marker's real budget |
| Residual symbol error vs review rate | Simulate review by replacing flagged outputs with the reference |
| Missed errors and the worst misses | Report prominently, with examples |
| Correction effort | Regions reviewed, edit distance fixed |
| Calibration | Brier score, reliability plot |
| Mark agreement | Plan A only, if marks are shared |
| Token estimate | Secondary. Label as an estimate unless measured with logged settings |

### Uncertainty

Paired bootstrap over TEST STUDENTS (about 2,000 resamples, 95% intervals) for each metric and for each model's difference from the baseline. No other statistical tests.

### Integrity checks (all required)

1. Shuffled-label control: should score near chance.
2. Leakage demo: random region-level split vs student-level split, to show how much a careless split inflates scores.
3. Per-student results, to check that a few students do not drive the average.

## 9. Claims

We WILL claim: how well the flagger finds wrong readings for this recogniser, this kind of handwriting and these scans, on unseen students.

We will NOT claim: results hold for other recognisers, scanners or subjects; that digitisation alone makes grading correct; that tokens were saved before we measured or clearly estimated them.

## 10. Known risks and mitigations

| Risk | Mitigation |
|---|---|
| Few students and regions | Grouped CV, limit complexity, report wide intervals |
| Recogniser nearly always right or wrong | Pilot first. If needed, use SER as a continuous target |
| Leakage through handwriting style | Split by student/writer, show leakage demo |
| Ambiguous references | Normalisation, two transcribers, report agreement |
| Plan B differs from real scans | Simulated degradation, never pool silently |

## 11. Deliverables

- Repository with the layout in CLAUDE.md, a README with reproduction steps, and pinned `requirements.txt`.
- Static GitHub Pages site (HTML, CSS, JS) with sections for question, data, preprocessing, baseline and methods, evaluation, results, errors, limitations, conclusion, reproduction.
- Two interactions, running in-browser from precomputed predictions:
  1. Review-budget slider: for baseline vs best model, show wrong lines caught, missed, residual error, effort.
  2. Region explorer: crop, rendered recogniser text, reference, predicted risk, features. Filter to misses and false alarms.

## 12. Milestones

| Dates | Milestone |
|---|---|
| Now to Nov 8 | Eval harness, Plan B loader, splits.json, 30-line recogniser pilot, baseline |
| Nov 9 to 11 | Features, models, grouped CV, model selection |
| Nov 12 to 13 | Single test evaluation, ablation, error analysis, integrity checks |
| Nov 14 to 16 | Website, README, viva practice, publish and test the site |
| Nov 17 | Final submission |

## 13. Definition of done for any slice

- Code runs end to end from the command line.
- Tests pass (`pytest`).
- Seeds fixed, outputs reproducible.
- Committed with a clear message.
- A human has read the diff.