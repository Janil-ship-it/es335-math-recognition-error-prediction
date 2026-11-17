import importlib
import numpy as np
from src.mathwriting import degrade, DEGRADATION_LEVELS
_feat = importlib.import_module("src.03_features")
image_quality_features = _feat.image_quality_features

def _synthetic_ink(h=80, w=240):
    g = np.full((h, w), 255, np.uint8)
    g[25:55, 20:220] = 20
    g[30:32, 30:200] = 0
    g[48:50, 40:180] = 0
    return g

def test_blur_monotonic_with_degradation():
    clean = _synthetic_ink()
    scores = {}
    for level in DEGRADATION_LEVELS:
        img = degrade(clean, level, seed=7)
        f = image_quality_features(img)
        scores[level] = f["blur"]
    assert scores["clean"] <= scores["mild"] + 0.05, scores
    assert scores["mild"] <= scores["strong"] + 0.05, scores
    assert scores["strong"] > scores["clean"], scores

def test_ink_orientation_bounded():
    f = image_quality_features(_synthetic_ink())
    assert 0 <= f["ink_orientation"] <= 45
