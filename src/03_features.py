"""Feature extraction from region crops + recogniser output.

Image-quality features use OpenCV when available.
Structure features are counted from the predicted LaTeX string.
Confidence features come from the recogniser.
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .config import ALL_FEATURES, DATA_DIR, IMAGE_FEATURES

FRAC_RE = re.compile(r"\\frac|\\dfrac|\\tfrac")
POWER_RE = re.compile(r"\^")
ROOT_RE = re.compile(r"\\sqrt")
SUB_RE = re.compile(r"_")


def image_quality_features(crop: np.ndarray) -> dict:
    """Compute blur, contrast, ink density, stroke width, skew, crop size, bg noise."""
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if HAS_CV2 else crop.mean(axis=2).astype(np.uint8)
    else:
        gray = crop
    h, w = gray.shape[:2]
    if HAS_CV2:
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    else:
        lap_var = float(np.var(np.diff(gray.astype(float), axis=0)))
    blur = float(np.clip(1.0 / (1.0 + lap_var / 200.0), 0.05, 0.95))
    contrast = float(np.clip(gray.std() / 128.0, 0.05, 0.95))
    ink_density = float((gray < 128).mean())
    if HAS_CV2:
        bin_inv = (gray < 128).astype(np.uint8)
        if bin_inv.sum() > 10:
            dist = cv2.distanceTransform(bin_inv, cv2.DIST_L2, 3)
            stroke_width = float(np.median(dist[dist > 0]) * 2) if (dist > 0).any() else 1.5
        else:
            stroke_width = 1.5
    else:
        stroke_width = 1.5
    stroke_width = float(np.clip(stroke_width, 0.5, 8.0))
    if HAS_CV2:
        bin_inv = ((gray < 128).astype(np.uint8) * 255)
        m = cv2.moments(bin_inv)
        if abs(m["mu02"]) > 1e-6:
            skew = float(np.degrees(0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])))
        else:
            skew = 0.0
    else:
        skew = 0.0
    skew = float(np.clip(abs(skew), 0, 45))
    crop_size = int(h * w)
    light = gray[gray > 180]
    bg_noise = float(np.clip(light.std() / 64.0 if len(light) > 20 else 0.05, 0, 0.5))
    return {
        "blur": blur,
        "contrast": contrast,
        "ink_density": ink_density,
        "stroke_width": stroke_width,
        "skew": skew,
        "crop_size": crop_size,
        "bg_noise": bg_noise,
    }


def structure_features(latex: str) -> dict:
    s = latex or ""
    return {
        "n_frac": len(FRAC_RE.findall(s)),
        "n_power": len(POWER_RE.findall(s)),
        "n_root": len(ROOT_RE.findall(s)),
        "n_sub": len(SUB_RE.findall(s)),
    }


def extract_features(crop: np.ndarray, recognition: dict) -> dict:
    feats = {}
    feats.update(image_quality_features(crop))
    feats.update(structure_features(recognition.get("latex", "")))
    feats["mean_token_conf"] = float(recognition.get("mean_token_conf", 0.5))
    feats["min_token_conf"] = float(recognition.get("min_token_conf", 0.5))
    feats["output_length"] = int(recognition.get("output_length", len(recognition.get("latex", ""))))
    return feats


def main():
    print("03_features: development data already materialised in data/features.csv")
    print("  For real crops, call extract_features(crop, recognition_dict).")
    if HAS_CV2:
        crop = np.random.randint(30, 220, (48, 200), dtype=np.uint8)
        crop[10:38, 20:180] = 40
        rec = {"latex": r"x^{2}+\frac{a}{b}", "mean_token_conf": 0.72,
               "min_token_conf": 0.41, "output_length": 12}
        f = extract_features(crop, rec)
        print("  sample features:", {k: round(v, 3) if isinstance(v, float) else v for k, v in f.items()})


if __name__ == "__main__":
    main()
