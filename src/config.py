"""Central configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
LOGS_DIR = RESULTS_DIR / "logs"

MATHWRITING_DIR = DATA_DIR / "mathwriting"
REGIONS_DIR = MATHWRITING_DIR / "regions"
REGIONS_META_PATH = MATHWRITING_DIR / "regions_meta.csv"
# Prefer MathWriting splits when present (after 00_prepare); fall back to committed data/splits.json
_SPLITS_MW = MATHWRITING_DIR / "splits.json"
_SPLITS_LEGACY = DATA_DIR / "splits.json"
SPLITS_PATH = _SPLITS_MW if _SPLITS_MW.exists() else _SPLITS_LEGACY
FEATURES_PATH = DATA_DIR / "features.csv"
RECOGNITIONS_PATH = RESULTS_DIR / "recognitions.jsonl"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
TARGET_DECISION_PATH = RESULTS_DIR / "target_decision.json"
RUN_METADATA_PATH = RESULTS_DIR / "run_metadata.json"

SEED = 42
N_BOOTSTRAP = 2000
REVIEW_RATES = (0.20, 0.30, 0.50)
TEST_FRACTION = 0.20
N_FOLDS = 5

CONFIDENCE_FEATURES = ["mean_token_conf", "min_token_conf", "output_length"]
IMAGE_FEATURES = [
    "blur", "contrast", "ink_density", "stroke_width",
    "ink_orientation", "crop_size", "bg_noise",
]
STRUCTURE_FEATURES = ["n_frac", "n_power", "n_root", "n_sub"]
ALL_FEATURES = CONFIDENCE_FEATURES + IMAGE_FEATURES + STRUCTURE_FEATURES

FORBIDDEN_FEATURES = {
    "reference_latex", "marks", "student_id", "group_id", "question_id",
    "degradation", "source_split", "sample_id", "region_id", "set",
    "degradation_seed", "n_strokes", "height", "width",
    "prediction_latex", "ser", "needs_correction",
}

DEFAULT_RECOGNISER = "DGurgurov/im2latex"
RECOGNISER_CANDIDATES = ("DGurgurov/im2latex", "OleehyO/TexTeller")
