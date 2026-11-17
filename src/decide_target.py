"""Decide target definition from a pilot recognition run (dev only)."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .config import RECOGNITIONS_PATH, REGIONS_META_PATH, RESULTS_DIR, TARGET_DECISION_PATH
from .latex_norm import needs_correction
from .mathwriting import token_error_rate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recognitions", type=Path, default=RECOGNITIONS_PATH)
    ap.add_argument("--meta", type=Path, default=REGIONS_META_PATH)
    args = ap.parse_args()
    meta = pd.read_csv(args.meta)
    meta = meta[meta["set"] == "dev"]
    recs = {}
    with open(args.recognitions) as f:
        for line in f:
            r = json.loads(line)
            recs[r["region_id"]] = r
    exact_errs, sers = [], []
    for _, row in meta.iterrows():
        rid = row["region_id"]
        if rid not in recs:
            continue
        pred = recs[rid].get("latex", "")
        ref = row["reference_latex"]
        exact_errs.append(needs_correction(pred, ref))
        sers.append(token_error_rate(pred, ref))
    exact_errs = np.array(exact_errs)
    sers = np.array(sers)
    rate = float(exact_errs.mean()) if len(exact_errs) else 0.0
    if 0.15 <= rate <= 0.85:
        decision = {"target": "exact", "tau": None, "exact_error_rate": rate,
                    "n": int(len(exact_errs)), "reason": "exact-match error rate in [0.15, 0.85]"}
    else:
        bad = sers[exact_errs == 1]
        tau = float(np.median(bad)) if len(bad) else float(np.median(sers)) if len(sers) else 0.0
        decision = {"target": "token", "tau": tau, "exact_error_rate": rate,
                    "n": int(len(exact_errs)),
                    "reason": f"exact rate {rate:.3f} outside [0.15,0.85]; tau=median SER on errors"}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TARGET_DECISION_PATH, "w") as f:
        json.dump(decision, f, indent=2)
    print(json.dumps(decision, indent=2))

if __name__ == "__main__":
    main()
