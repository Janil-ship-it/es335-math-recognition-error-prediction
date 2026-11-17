"""Region extraction / rendering helpers.

MathWriting pipeline uses src/00_prepare_mathwriting.py (render_ink).
This module keeps a page-level line detector with the dropped-last-line fix.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .config import DATA_DIR


def preprocess_image(img: np.ndarray) -> np.ndarray:
    if not HAS_CV2:
        raise RuntimeError("opencv required")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 > p2:
        gray = np.clip((gray.astype(float) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    return gray


def extract_line_regions(page_img: np.ndarray, min_height: int = 20) -> list:
    """Horizontal projection line detector. Flushes trailing region (last-line fix)."""
    if not HAS_CV2:
        raise RuntimeError("opencv required")
    gray = preprocess_image(page_img)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10)
    proj = binary.sum(axis=1)
    threshold = proj.max() * 0.08
    in_line, regions, start = False, [], 0
    n_rows = len(proj)

    def _emit(y0, y1):
        h = y1 - y0
        if h < min_height:
            return
        crop = gray[y0:y1, :]
        cols = np.where((crop < 240).sum(axis=0) > 0)[0]
        if len(cols) <= 10:
            return
        x0, x1 = int(cols[0]), int(cols[-1] + 1)
        regions.append({"bbox": (x0, int(y0), x1 - x0, int(h)), "crop": crop[:, x0:x1]})

    for i, v in enumerate(proj):
        if not in_line and v > threshold:
            in_line, start = True, i
        elif in_line and v <= threshold:
            in_line = False
            _emit(start, i)
    if in_line:  # FIX: flush last line
        _emit(start, n_rows)
    return regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "regions")
    args = ap.parse_args()
    if args.input is None:
        print("01_extract_regions: no input — MathWriting uses 00_prepare_mathwriting.")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [args.input] if args.input.is_file() else sorted(list(args.input.glob("*.png")) + list(args.input.glob("*.jpg")))
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        regions = extract_line_regions(img)
        print(f"{p.name}: {len(regions)} regions")
        for i, r in enumerate(regions):
            cv2.imwrite(str(args.out / f"{p.stem}_r{i:03d}.png"), r["crop"])


if __name__ == "__main__":
    main()
