"""Feature extraction from region crops + recogniser output.

Blur fix: Tenengrad on median-denoised image so score is monotonic with
simulated degradation. ink_orientation replaces skew (ink-blob axis, not slant).
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from .config import (
    ALL_FEATURES, FEATURES_PATH, FORBIDDEN_FEATURES, REGIONS_DIR,
    REGIONS_META_PATH, RECOGNITIONS_PATH, RESULTS_DIR,
)
from .latex_norm import needs_correction
from .mathwriting import token_error_rate

FRAC_RE = re.compile(r"\\frac|\\dfrac|\\tfrac")
POWER_RE = re.compile(r"\^")
ROOT_RE = re.compile(r"\\sqrt")
SUB_RE = re.compile(r"_")


def image_quality_features(crop: np.ndarray) -> dict:
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if HAS_CV2 else crop.mean(axis=2).astype(np.uint8)
    else:
        gray = crop.copy()
    h, w = gray.shape[:2]
    if HAS_CV2:
        den = cv2.medianBlur(gray, 3)
        gx = cv2.Sobel(den, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(den, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = float((gx ** 2 + gy ** 2).mean())
    else:
        tenengrad = float(np.var(np.diff(gray.astype(float), axis=0)))
    blur = float(np.clip(1.0 / (1.0 + tenengrad / 200.0), 0.01, 0.99))
    contrast = float(np.clip(gray.std() / 128.0, 0.01, 0.99))
    ink_density = float((gray < 128).mean())
    if HAS_CV2:
        bin_inv = (gray < 128).astype(np.uint8)
        if bin_inv.sum() > 10:
            dist = cv2.distanceTransform(bin_inv, cv2.DIST_L2, 3)
            stroke_width = float(np.median(dist[dist > 0]) * 2) if (dist > 0).any() else 1.5
        else:
            stroke_width = 1.5
        m = cv2.moments((bin_inv * 255).astype(np.uint8))
        if abs(m["mu02"]) > 1e-6:
            ink_orientation = float(np.degrees(0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])))
        else:
            ink_orientation = 0.0
    else:
        stroke_width, ink_orientation = 1.5, 0.0
    stroke_width = float(np.clip(stroke_width, 0.5, 12.0))
    ink_orientation = float(np.clip(abs(ink_orientation), 0, 45))
    light = gray[gray > 180]
    bg_noise = float(np.clip(light.std() / 64.0 if len(light) > 20 else 0.05, 0, 0.5))
    return {
        "blur": blur, "contrast": contrast, "ink_density": ink_density,
        "stroke_width": stroke_width, "ink_orientation": ink_orientation,
        "crop_size": int(h * w), "bg_noise": bg_noise,
    }


def structure_features(latex: str) -> dict:
    s = latex or ""
    return {
        "n_frac": len(FRAC_RE.findall(s)), "n_power": len(POWER_RE.findall(s)),
        "n_root": len(ROOT_RE.findall(s)), "n_sub": len(SUB_RE.findall(s)),
    }


def extract_features(crop: np.ndarray, recognition: dict) -> dict:
    feats = {}
    feats.update(image_quality_features(crop))
    feats.update(structure_features(recognition.get("latex", "")))
    feats["mean_token_conf"] = float(recognition.get("mean_token_conf", 0.5))
    feats["min_token_conf"] = float(recognition.get("min_token_conf", 0.5))
    feats["output_length"] = int(recognition.get("output_length", len(recognition.get("latex", "") or "")))
    return feats


def _load_recognitions(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("region_id"):
                out[rec["region_id"]] = rec
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=REGIONS_META_PATH)
    ap.add_argument("--regions", type=Path, default=REGIONS_DIR)
    ap.add_argument("--recognitions", type=Path, default=RECOGNITIONS_PATH)
    ap.add_argument("--out", type=Path, default=FEATURES_PATH)
    ap.add_argument("--target", choices=["exact", "token"], default=None)
    args = ap.parse_args()
    if not args.meta.exists():
        raise SystemExit(f"Missing {args.meta}")
    meta = pd.read_csv(args.meta)
    recs = _load_recognitions(args.recognitions)
    if not recs:
        raise SystemExit(f"No recognitions in {args.recognitions}")
    target_path = RESULTS_DIR / "target_decision.json"
    target_mode, token_tau = args.target, 0.0
    if target_mode is None and target_path.exists():
        td = json.load(open(target_path))
        target_mode = td.get("target", "exact")
        token_tau = float(td.get("tau", 0.0) or 0.0)
    if target_mode is None:
        target_mode = "exact"
    rows, missing = [], 0
    import cv2 as _cv2
    for _, r in meta.iterrows():
        rid = r["region_id"]
        crop_path = args.regions / f"{rid}.png"
        if rid not in recs or not crop_path.exists():
            missing += 1
            continue
        crop = _cv2.imread(str(crop_path), _cv2.IMREAD_GRAYSCALE)
        rec = recs[rid]
        feats = extract_features(crop, rec)
        pred, ref = rec.get("latex", ""), r["reference_latex"]
        ser = token_error_rate(pred, ref)
        y = needs_correction(pred, ref) if target_mode == "exact" else int(ser > token_tau)
        rows.append({
            "region_id": rid, "group_id": r["group_id"], "set": r["set"],
            "sample_id": r.get("sample_id", ""), "source_split": r.get("source_split", ""),
            "degradation": r.get("degradation", "clean"),
            "reference_latex": ref, "prediction_latex": pred, "ser": ser,
            "needs_correction": y, **feats,
        })
    df = pd.DataFrame(rows)
    for col in FORBIDDEN_FEATURES:
        assert col not in ALL_FEATURES, col
    assert set(ALL_FEATURES).issubset(df.columns), set(ALL_FEATURES) - set(df.columns)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out} (missing={missing})")
    print(f"target={target_mode} tau={token_tau} error_rate={df['needs_correction'].mean():.3f}")
    print(f"ink_density mean={df['ink_density'].mean():.4f}")
    if "degradation" in df.columns:
        print(df.groupby("degradation")["blur"].agg(["mean", "median", "count"]))


if __name__ == "__main__":
    main()
