"""Generate realistic development dataset for Plan B methodology (unit tests only)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import DATA_DIR, REFERENCES_PATH, SEED, SPLITS_PATH, TEST_FRACTION

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def generate_development_data(n_students=28, regions_per_student=(18, 32), seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    region_id = 0
    for sid in range(n_students):
        n_reg = int(rng.integers(*regions_per_student))
        student_blur_bias = rng.normal(0, 0.35)
        student_conf_bias = rng.normal(0, 0.25)
        for _ in range(n_reg):
            blur = np.clip(rng.normal(0.45 + student_blur_bias, 0.22), 0.05, 0.95)
            contrast = np.clip(rng.normal(0.55 - 0.3 * blur, 0.18), 0.1, 0.95)
            ink_density = np.clip(rng.normal(0.5, 0.15), 0.1, 0.9)
            stroke_width = np.clip(rng.normal(1.8, 0.6), 0.5, 4.0)
            skew = np.clip(np.abs(rng.normal(0, 4.5)), 0, 18)
            crop_size = int(rng.integers(80, 420))
            bg_noise = np.clip(rng.exponential(0.08), 0, 0.45)
            n_frac = int(rng.poisson(0.7))
            n_power = int(rng.poisson(0.9))
            n_root = int(rng.poisson(0.25))
            n_sub = int(rng.poisson(0.4))
            complexity = n_frac*1.4 + n_power*0.8 + n_root*1.6 + n_sub*0.6
            logit = (-1.1 + 2.4*blur - 1.8*contrast + 0.9*bg_noise + 0.35*complexity
                     + 0.08*skew + student_blur_bias*0.4 + rng.normal(0, 0.55))
            p_err = _sigmoid(logit)
            needs_corr = int(rng.random() < p_err)
            mean_conf = np.clip(0.78 - 0.45*p_err + student_conf_bias + rng.normal(0, 0.12), 0.15, 0.98)
            min_conf = np.clip(mean_conf - rng.uniform(0.05, 0.35), 0.05, mean_conf)
            output_length = int(rng.integers(4, 28) + complexity*2)
            ser = 0.0 if needs_corr == 0 else float(np.clip(rng.beta(2, 5) + 0.1*complexity/5, 0.05, 0.9))
            ref = f"x^{{{n_power}}}+\\frac{{a}}{{b}}" if n_frac else "a+b"
            pred = ref if needs_corr == 0 else ref.replace("a", "c")
            rows.append({
                "region_id": f"R{region_id:05d}", "group_id": f"S{sid:03d}", "student_id": f"S{sid:03d}",
                "blur": blur, "contrast": contrast, "ink_density": ink_density, "stroke_width": stroke_width,
                "skew": skew, "crop_size": crop_size, "bg_noise": bg_noise,
                "n_frac": n_frac, "n_power": n_power, "n_root": n_root, "n_sub": n_sub,
                "mean_token_conf": mean_conf, "min_token_conf": min_conf, "output_length": output_length,
                "needs_correction": needs_corr, "ser": ser,
                "reference_latex": ref, "recogniser_latex": pred, "transcriber": "dev-proxy",
            })
            region_id += 1
    return pd.DataFrame(rows)

def make_splits(df, test_fraction=TEST_FRACTION, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = np.array(df["group_id"].unique())
    rng.shuffle(groups)
    n_test = max(1, int(round(len(groups) * test_fraction)))
    test_groups = sorted(groups[:n_test].tolist())
    dev_groups = sorted(groups[n_test:].tolist())
    return {"seed": seed, "test_fraction": test_fraction, "test_groups": test_groups,
            "dev_groups": dev_groups, "n_test_groups": len(test_groups), "n_dev_groups": len(dev_groups)}

def save_development_data(df, splits):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df[["region_id", "group_id", "reference_latex", "transcriber"]].to_csv(REFERENCES_PATH, index=False)
    with open(SPLITS_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    df.to_csv(DATA_DIR / "features.csv", index=False)
    print(f"Saved {len(df)} regions, {df['group_id'].nunique()} groups, error rate={df['needs_correction'].mean():.3f}")

if __name__ == "__main__":
    df = generate_development_data()
    splits = make_splits(df)
    save_development_data(df, splits)
