"""Region extraction from page images / PDFs.

For Plan A (real scans): detect line-level crops.
For development proxy: no-op (features already materialised).

Blocked components:
- Real student scans are private and not in the repository.
- Full PDF page segmentation requires annotated ground-truth line boxes.
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
    """Grayscale, mild denoise, contrast stretch. Returns uint8."""
    if not HAS_CV2:
        raise RuntimeError("opencv-python required for image preprocessing")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    p2, p98 = np.percentile(gray, (2, 98))
    if p98 > p2:
        gray = np.clip((gray.astype(float) - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
    return gray


def extract_line_regions(page_img: np.ndarray, min_height: int = 20) -> list[dict]:
    """Simple horizontal projection-based line detector (baseline implementation)."""
    if not HAS_CV2:
        raise RuntimeError("opencv-python required")
    gray = preprocess_image(page_img)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 31, 10)
    proj = binary.sum(axis=1)
    threshold = proj.max() * 0.08
    in_line = False
    regions = []
    start = 0
    for i, v in enumerate(proj):
        if not in_line and v > threshold:
            in_line = True
            start = i
        elif in_line and v <= threshold:
            in_line = False
            h = i - start
            if h >= min_height:
                crop = gray[start:i, :]
                col_sum = (crop < 240).sum(axis=0)
                cols = np.where(col_sum > 0)[0]
                if len(cols) > 10:
                    x0, x1 = cols[0], cols[-1] + 1
                    crop = crop[:, x0:x1]
                    regions.append({
                        "bbox": (int(x0), int(start), int(x1 - x0), int(h)),
                        "crop": crop,
                    })
    return regions


def main():
    parser = argparse.ArgumentParser(description="Extract line regions from page images")
    parser.add_argument("--input", type=Path, help="Input image or directory")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "regions")
    args = parser.parse_args()
    if args.input is None:
        print("01_extract_regions: no input provided — no-op for development proxy.")
        print("  Real scans are private (Plan A). Use --input path/to/page.png for pilot.")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    paths = [args.input] if args.input.is_file() else sorted(args.input.glob("*.png")) + sorted(args.input.glob("*.jpg"))
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"Could not read {p}")
            continue
        regions = extract_line_regions(img)
        print(f"{p.name}: {len(regions)} regions")
        for i, r in enumerate(regions):
            out_path = args.out / f"{p.stem}_r{i:03d}.png"
            cv2.imwrite(str(out_path), r["crop"])


if __name__ == "__main__":
    main()
