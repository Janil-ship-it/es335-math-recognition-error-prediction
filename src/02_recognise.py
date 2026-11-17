"""Recogniser wrapper for pre-trained HMER models (TexTeller / UniMERNet).

Development status:
- Real model weights are large and require torch + GPU for practical speed.
- This module provides a clean interface and a deterministic proxy for the
  development dataset so the rest of the pipeline remains testable offline.
- When a real model is available, implement `_run_real_model` and set
  USE_REAL_MODEL = True.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .config import DATA_DIR, SEED

USE_REAL_MODEL = False  # set True only when torch + model weights are present


def _run_proxy(crop: np.ndarray, rng: np.random.Generator) -> dict:
    """Deterministic-ish proxy that returns LaTeX + token confidences."""
    h, w = crop.shape[:2]
    complexity = min(8, max(1, (h * w) // 8000))
    tokens = ["x", "^", "{", "2", "}", "+", "\\frac", "{", "a", "}", "{", "b", "}"][: 4 + complexity]
    confs = rng.uniform(0.35, 0.97, size=len(tokens)).tolist()
    latex = "".join(tokens)
    return {
        "latex": latex,
        "token_confidences": confs,
        "mean_token_conf": float(np.mean(confs)),
        "min_token_conf": float(np.min(confs)),
        "output_length": len(tokens),
        "backend": "proxy",
    }


def recognise_crop(crop: np.ndarray, seed: int = SEED) -> dict:
    """Run recognition on a single grayscale crop. Returns dict with latex + confidences."""
    rng = np.random.default_rng(seed + int(crop.sum()) % 10000)
    if USE_REAL_MODEL:
        return _run_real_model(crop)
    return _run_proxy(crop, rng)


def _run_real_model(crop: np.ndarray) -> dict:
    """Placeholder for TexTeller / UniMERNet inference."""
    raise NotImplementedError(
        "Real HMER backend not configured. Install torch + model weights and "
        "implement _run_real_model, then set USE_REAL_MODEL=True."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=Path, default=DATA_DIR / "regions")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "recognitions.json")
    args = parser.parse_args()
    if not args.regions.exists() or not any(args.regions.glob("*.png")):
        print("02_recognise: no region images found — using development proxy path.")
        print("  Real recognition requires cropped line images and a loaded HMER model.")
        return
    results = []
    for p in sorted(args.regions.glob("*.png")):
        import cv2
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        out = recognise_crop(img)
        out["region_path"] = str(p)
        results.append(out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} recognitions to {args.out}")


if __name__ == "__main__":
    main()
