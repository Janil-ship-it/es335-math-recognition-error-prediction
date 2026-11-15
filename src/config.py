"""Central configuration."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
SPLITS_PATH = DATA_DIR / "splits.json"
REFERENCES_PATH = DATA_DIR / "references.csv"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.csv"
SEED = 42
N_BOOTSTRAP = 2000
REVIEW_RATES = (0.20, 0.30, 0.50)
TEST_FRACTION = 0.20
N_FOLDS = 5
CONFIDENCE_FEATURES = ["mean_token_conf", "min_token_conf", "output_length"]
IMAGE_FEATURES = ["blur", "contrast", "ink_density", "stroke_width", "skew", "crop_size", "bg_noise"]
STRUCTURE_FEATURES = ["n_frac", "n_power", "n_root", "n_sub"]
ALL_FEATURES = CONFIDENCE_FEATURES + IMAGE_FEATURES + STRUCTURE_FEATURES
FORBIDDEN_FEATURES = {"reference_latex", "marks", "student_id", "group_id", "question_id"}
