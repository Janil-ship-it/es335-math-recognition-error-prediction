"""Build the real-data region set from MathWriting (human-written inks only).

Steps: sample inks (fixed seed) -> render -> assign SIMULATED degradation level -> write
metadata + group-level train/test split. No synthetic inks, no 'symbols' folder.

  python -m src.00_prepare_mathwriting --root /path/to/mathwriting-2024 --n 2000 --seed 42

Groups = normalizedLabel (MathWriting has no writer id), so one expression never appears in
both dev and test. Writer-level leakage cannot be excluded (some writers contributed to several splits).
Refuses to overwrite an existing splits file unless --force.
"""
from __future__ import annotations
import argparse, json
import importlib
from pathlib import Path
import numpy as np
import pandas as pd

from .config import DATA_DIR, SEED, TEST_FRACTION
mw = importlib.import_module("src.mathwriting")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="extracted mathwriting-2024 (or -excerpt) folder")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--from-splits", nargs="+", default=["valid", "test"], choices=list(mw.HUMAN_SPLITS))
    ap.add_argument("--out", type=Path, default=DATA_DIR / "mathwriting")
    ap.add_argument("--max-label-chars", type=int, default=120)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    splits_path = a.out / "splits.json"
    if splits_path.exists() and not a.force:
        raise SystemExit(f"{splits_path} exists; refusing to overwrite (use --force).")
    rng = np.random.default_rng(a.seed)

    cands = []
    for split, p in mw.list_human_inks(a.root, a.from_splits):
        ink = mw.read_inkml_file(p)
        ann = ink.annotations
        if ann.get("inkCreationMethod") != "human" or not ann.get("normalizedLabel"):
            continue
        if len(ann["normalizedLabel"]) > a.max_label_chars or not ink.strokes:
            continue
        cands.append((split, p, ann))
    print(f"eligible human inks: {len(cands)} (from {a.from_splits}, label<= {a.max_label_chars} chars)")
    order = rng.permutation(len(cands))[: min(a.n, len(cands))]

    (a.out / "regions").mkdir(parents=True, exist_ok=True)
    import cv2
    rows = []
    for k, idx in enumerate(order):
        split, p, ann = cands[idx]
        ink = mw.read_inkml_file(p)
        level = mw.DEGRADATION_LEVELS[int(rng.integers(0, 3))]
        dseed = int(rng.integers(0, 2**31 - 1))
        img = mw.degrade(mw.render_ink(ink), level, dseed)
        rid = f"R{k:05d}"
        cv2.imwrite(str(a.out / "regions" / f"{rid}.png"), img)
        rows.append(dict(region_id=rid, sample_id=ann["sampleId"], source_split=split,
                         group_id="G_" + ann["normalizedLabel"], reference_latex=ann["normalizedLabel"],
                         degradation=level, degradation_seed=dseed, n_strokes=len(ink.strokes),
                         height=img.shape[0], width=img.shape[1]))
    meta = pd.DataFrame(rows)
    groups = np.array(sorted(meta["group_id"].unique())); rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * TEST_FRACTION)))
    test_g, dev_g = sorted(groups[:n_test].tolist()), sorted(groups[n_test:].tolist())
    assert not set(test_g) & set(dev_g)
    meta["set"] = np.where(meta["group_id"].isin(test_g), "test", "dev")
    meta.to_csv(a.out / "regions_meta.csv", index=False)
    json.dump({"seed": a.seed, "group_type": "normalizedLabel", "from_splits": a.from_splits,
               "n_regions": len(meta), "n_groups": len(groups), "test_groups": test_g, "dev_groups": dev_g,
               "simulated_degradation": True, "source": "MathWriting arXiv:2404.10690 (CC BY-NC-SA 4.0)"},
              open(splits_path, "w"), indent=1)
    print(meta["set"].value_counts().to_dict(), "| degradation:", meta["degradation"].value_counts().to_dict())


if __name__ == "__main__":
    main()
