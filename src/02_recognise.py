"""Recogniser wrapper for pre-trained HMER models.

Candidates:
  - DGurgurov/im2latex  (arXiv:2408.04015)
  - OleehyO/TexTeller   (arXiv:2508.09220; note v4 withdrawn — record revision hash)

Per-token probabilities via generate(return_dict_in_generate=True, output_scores=True)
+ compute_transition_scores.

  python -m src.02_recognise --real --model DGurgurov/im2latex --limit 30 --dev-only
  python -m src.02_recognise --real --model DGurgurov/im2latex
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

from .config import (
    DEFAULT_RECOGNISER, LOGS_DIR, REGIONS_DIR, REGIONS_META_PATH,
    RECOGNITIONS_PATH, SEED,
)

USE_REAL_MODEL = False


def _run_proxy(crop, rng):
    h, w = crop.shape[:2]
    complexity = min(8, max(1, (h * w) // 8000))
    tokens = ["x", "^", "{", "2", "}", "+", "\\frac", "{", "a", "}", "{", "b", "}"][: 4 + complexity]
    confs = rng.uniform(0.35, 0.97, size=len(tokens)).tolist()
    return {
        "latex": "".join(tokens), "token_confidences": confs,
        "mean_token_conf": float(np.mean(confs)), "min_token_conf": float(np.min(confs)),
        "output_length": len(tokens), "backend": "proxy", "model_id": "proxy", "revision": None,
    }


def recognise_crop(crop, seed=SEED):
    rng = np.random.default_rng(seed + int(crop.sum()) % 10000)
    if USE_REAL_MODEL:
        raise RuntimeError("Use CLI --real which loads the model once.")
    return _run_proxy(crop, rng)


class RealRecogniser:
    def __init__(self, model_id=DEFAULT_RECOGNISER, device=None):
        import torch
        from transformers import VisionEncoderDecoderModel, AutoTokenizer, ViTImageProcessor
        self.torch = torch
        self.model_id = model_id
        print(f"Loading {model_id} ...")
        t0 = time.time()
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            self.processor = ViTImageProcessor.from_pretrained(model_id)
        except Exception:
            from transformers import AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model.eval()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(device)
        try:
            from huggingface_hub import model_info
            self.revision = model_info(model_id).sha
        except Exception:
            self.revision = "unknown"
        print(f"Loaded {model_id} rev={self.revision[:12]} on {device} in {time.time()-t0:.1f}s")

    def predict(self, crop, max_new_tokens=96):
        from PIL import Image
        img = Image.fromarray(crop).convert("RGB") if crop.ndim == 2 else Image.fromarray(crop).convert("RGB")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                pixel_values, max_new_tokens=max_new_tokens, num_beams=1, do_sample=False,
                return_dict_in_generate=True, output_scores=True,
            )
        text = self.tokenizer.batch_decode(out.sequences, skip_special_tokens=True)[0]
        confs = []
        if out.scores:
            try:
                trans = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                confs = self.torch.exp(trans[0]).detach().cpu().tolist()
            except Exception as e:
                print(f"  warn: transition scores failed ({e})")
        return {
            "latex": text, "token_confidences": [float(c) for c in confs],
            "mean_token_conf": float(np.mean(confs)) if confs else 0.5,
            "min_token_conf": float(np.min(confs)) if confs else 0.5,
            "output_length": len(confs) if confs else len(text),
            "backend": "transformers", "model_id": self.model_id, "revision": self.revision,
        }


def _already_done(path):
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["region_id"])
                    except Exception:
                        pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--model", default=DEFAULT_RECOGNISER)
    ap.add_argument("--meta", type=Path, default=REGIONS_META_PATH)
    ap.add_argument("--regions", type=Path, default=REGIONS_DIR)
    ap.add_argument("--out", type=Path, default=RECOGNITIONS_PATH)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dev-only", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.real:
        print("02_recognise: proxy mode (pass --real for HMER).")
        return
    if not args.meta.exists():
        raise SystemExit(f"Missing {args.meta}")
    import pandas as pd
    meta = pd.read_csv(args.meta)
    if args.dev_only:
        meta = meta[meta["set"] == "dev"].copy()
    if args.limit > 0:
        meta = meta.head(args.limit)
    recogniser = RealRecogniser(args.model)
    done = _already_done(args.out)
    print(f"Resuming: {len(done)} already done; {len(meta)} planned")
    times, n_err, n_done = [], 0, 0
    import cv2
    from .latex_norm import needs_correction
    with open(args.out, "a") as fout:
        for _, row in meta.iterrows():
            rid = row["region_id"]
            if rid in done:
                continue
            crop_path = args.regions / f"{rid}.png"
            if not crop_path.exists():
                continue
            crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
            t0 = time.time()
            try:
                rec = recogniser.predict(crop, max_new_tokens=args.max_new_tokens)
            except Exception as e:
                print(f"  fail {rid}: {e}")
                continue
            dt = time.time() - t0
            times.append(dt)
            rec["region_id"] = rid
            rec["seconds"] = round(dt, 3)
            err = needs_correction(rec["latex"], row["reference_latex"])
            n_err += err
            n_done += 1
            rec["needs_correction_exact"] = err
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if args.verify and n_done <= 3:
                print(f"\n=== VERIFY {rid} ===")
                print("ref :", str(row["reference_latex"])[:120])
                print("pred:", rec["latex"][:120])
                print("conf:", [round(c, 4) for c in rec["token_confidences"][:12]])
            if n_done % 5 == 0:
                print(f"  [{n_done}] last={dt:.1f}s mean={np.mean(times):.1f}s err={n_err}/{n_done}")
            if args.verify and n_done >= 3:
                break
    if times:
        print(f"Finished: {n_done} new, mean {np.mean(times):.2f}s/img, exact-err={n_err}/{n_done}")
        log = {
            "model_id": recogniser.model_id, "revision": recogniser.revision,
            "n_done": n_done, "n_err_exact": n_err,
            "exact_error_rate": n_err / max(n_done, 1),
            "mean_seconds": float(np.mean(times)), "device": recogniser.device,
        }
        with open(LOGS_DIR / "recognise_pilot.json", "w") as f:
            json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
