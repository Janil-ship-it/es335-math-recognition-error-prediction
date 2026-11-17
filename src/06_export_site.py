"""Export docs/data/site_results.json and regions.json for the website."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .config import (
    DOCS_DIR, FEATURES_PATH, PREDICTIONS_PATH, RESULTS_DIR, SPLITS_PATH,
    REGIONS_DIR, RUN_METADATA_PATH,
)


def main():
    docs_data = DOCS_DIR / "data"
    docs_data.mkdir(parents=True, exist_ok=True)
    crops_dir = docs_data / "crops"
    crops_dir.mkdir(exist_ok=True)

    metrics_path = RESULTS_DIR / "test_metrics.json"
    if not metrics_path.exists():
        raise SystemExit("Run 05_evaluate_test first")
    blob = json.load(open(metrics_path))
    metrics = blob.get("metrics", blob)
    curves = blob.get("recall_curves", {})

    splits = json.load(open(SPLITS_PATH)) if SPLITS_PATH.exists() else {}
    run_meta = json.load(open(RUN_METADATA_PATH)) if RUN_METADATA_PATH.exists() else {}

    meta = {
        "data_kind": "real_mathwriting_simulated_degradation",
        "data_label": "Real handwriting (MathWriting), simulated scan degradation",
        "n_regions": int(splits.get("n_regions", 0)),
        "n_groups": int(splits.get("n_groups", 0)),
        "group_type": splits.get("group_type", "normalizedLabel"),
        "writer_leakage_note": "MathWriting has no writer id; writer-level leakage cannot be excluded.",
        "recogniser_id": run_meta.get("model_id", "unknown"),
        "recogniser_revision": run_meta.get("revision", "unknown"),
        "target": json.load(open(RESULTS_DIR / "target_decision.json")) if (RESULTS_DIR / "target_decision.json").exists() else {},
        "simulated_degradation": True,
        "degradation_note": "Degradation (blur, noise, rotation, shading) is SIMULATED and stored for analysis only; it is never a model feature.",
        "source": splits.get("source", "MathWriting arXiv:2404.10690 (CC BY-NC-SA 4.0)"),
    }

    site = {
        "meta": meta,
        "metrics": {k: v for k, v in metrics.items() if k not in (
            "per_group", "by_degradation", "feature_importance_logistic",
            "leakage_demo", "shuffled_label_control", "bounds")},
        "bounds": metrics.get("bounds", {}),
        "recall_curve": curves,
        "pr_curves": {},
        "reliability": {},
        "per_group": metrics.get("per_group", []),
        "ablation": [],
        "feature_importance": metrics.get("feature_importance_logistic", {}),
        "leakage_demo": metrics.get("leakage_demo", {}),
        "shuffled_control": metrics.get("shuffled_label_control", {}),
        "by_degradation": metrics.get("by_degradation", {}),
    }
    if (RESULTS_DIR / "cv_results.json").exists():
        cv = json.load(open(RESULTS_DIR / "cv_results.json"))
        site["ablation"] = cv.get("ablation", [])

    with open(docs_data / "site_results.json", "w") as f:
        json.dump(site, f, indent=2)
    print("Wrote", docs_data / "site_results.json")

    if PREDICTIONS_PATH.exists() and FEATURES_PATH.exists():
        preds = pd.read_csv(PREDICTIONS_PATH)
        feats = pd.read_csv(FEATURES_PATH)
        test = feats[feats["set"] == "test"].head(80)
        regions = []
        for _, r in test.iterrows():
            rid = r["region_id"]
            src = REGIONS_DIR / f"{rid}.png"
            thumb = crops_dir / f"{rid}.png"
            if src.exists() and not thumb.exists():
                try:
                    import cv2
                    img = cv2.imread(str(src), 0)
                    h, w = img.shape
                    scale = min(1.0, 320 / max(w, 1))
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                    cv2.imwrite(str(thumb), img)
                except Exception:
                    pass
            prow = preds[preds["region_id"] == rid]
            scores = {}
            if len(prow):
                for c in prow.columns:
                    if c.startswith("score_"):
                        scores[c.replace("score_", "")] = float(prow.iloc[0][c])
            regions.append({
                "id": rid,
                "thumbnail": f"data/crops/{rid}.png" if thumb.exists() else None,
                "reference": r.get("reference_latex", ""),
                "prediction": r.get("prediction_latex", ""),
                "scores": scores,
                "features": {k: float(r[k]) for k in ("mean_token_conf", "min_token_conf", "blur", "ink_density") if k in r},
                "missed_at_30": bool(prow.iloc[0]["missed_at_30"]) if len(prow) and "missed_at_30" in prow.columns else False,
                "needs_correction": int(r["needs_correction"]),
            })
        with open(docs_data / "regions.json", "w") as f:
            json.dump(regions, f, indent=2)
        print(f"Wrote {len(regions)} regions to regions.json")


if __name__ == "__main__":
    main()
