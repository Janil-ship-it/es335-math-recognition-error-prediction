# Implementation status

Pipeline code switched to real MathWriting data.
No fabricated numbers — run on Colab to produce results.

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
