"""MathWriting I/O, ink rendering, scan-degradation simulation and token metrics.

Reader, Cairo rendering convention and tokenizer follow the dataset authors' example
notebook (google-research/google-research, mathwriting/mathwriting_code_examples.ipynb,
Apache-2.0). Dataset: Gervais, Fadeeva, Maksai, arXiv:2404.10690 (CC BY-NC-SA 4.0).
Degradation is SIMULATED and must be described as such everywhere it is reported.
"""
from __future__ import annotations
import dataclasses
import re
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

HUMAN_SPLITS = ("train", "valid", "test")      # 'symbols' is also tagged human but holds single glyphs
INKML_NS = "{http://www.w3.org/2003/InkML}"
TARGET_WIDTH_PX = 800        # ink is scaled so its width is ~this (scale clipped to SCALE_RANGE)
SCALE_RANGE = (0.5, 3.0)
STROKE_PX = 4.0
MARGIN_PX = 24
DEGRADATION_LEVELS = ("clean", "mild", "strong")


@dataclasses.dataclass
class Ink:
    strokes: list            # each array shape (3, n): x, y, t
    annotations: dict


def read_inkml_file(path) -> Ink:
    root = ElementTree.fromstring(Path(path).read_text())
    strokes, ann = [], {}
    for el in root:
        tag = el.tag.removeprefix(INKML_NS)
        if tag == "annotation":
            ann[el.attrib.get("type")] = el.text
        elif tag == "trace":
            xs, ys, ts = [], [], []
            for pt in el.text.split(","):
                x, y, t = pt.strip().split(" ")
                xs.append(float(x)); ys.append(float(y)); ts.append(float(t))
            strokes.append(np.array((xs, ys, ts)))
    return Ink(strokes=strokes, annotations=ann)


def list_human_inks(root, splits=HUMAN_SPLITS):
    """Paths of human-written inks that have a normalizedLabel (never synthetic, never symbols)."""
    out = []
    for s in splits:
        for p in sorted((Path(root) / s).glob("*.inkml")):
            out.append((s, p))
    return out


def render_ink(ink: Ink):
    """Render to an 8-bit grayscale numpy array (ink dark on white), width-normalised."""
    import cairo
    mins = np.vstack([s[:2].min(axis=1) for s in ink.strokes]).min(axis=0)
    maxs = np.vstack([s[:2].max(axis=1) for s in ink.strokes]).max(axis=0)
    w_units = max(float(maxs[0] - mins[0]), 1.0)
    scale = float(np.clip(TARGET_WIDTH_PX / w_units, *SCALE_RANGE))
    width = int(w_units * scale + 2 * MARGIN_PX)
    height = int(max(float(maxs[1] - mins[1]), 1.0) * scale + 2 * MARGIN_PX)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surf)
    ctx.set_source_rgb(1, 1, 1); ctx.paint()
    ctx.set_source_rgb(0, 0, 0)
    ctx.set_line_width(STROKE_PX); ctx.set_line_cap(cairo.LINE_CAP_ROUND); ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    for s in ink.strokes:
        xs = (s[0] - mins[0]) * scale + MARGIN_PX
        ys = (s[1] - mins[1]) * scale + MARGIN_PX
        ctx.move_to(xs[0], ys[0])
        if len(xs) == 1:
            ctx.line_to(xs[0] + 0.01, ys[0])
        for x, y in zip(xs[1:], ys[1:]):
            ctx.line_to(x, y)
        ctx.stroke()
    buf = np.frombuffer(surf.get_data(), dtype=np.uint8).reshape(height, surf.get_stride() // 4, 4)[:, :width]
    return buf[:, :, 1].copy()      # BGRX -> green channel == gray for black-on-white


def degrade(gray: np.ndarray, level: str, seed: int) -> np.ndarray:
    """SIMULATED scan degradation, deterministic given (level, seed). Returns uint8 gray."""
    import cv2
    if level not in DEGRADATION_LEVELS:
        raise ValueError(level)
    if level == "clean":
        return gray.copy()
    rng = np.random.default_rng(seed)
    p = {"mild":   dict(blur=(0.6, 1.2), rot=2.0, noise=6.0,  shade=10.0, ink=(0.85, 1.0), specks=0.0),
         "strong": dict(blur=(1.5, 2.5), rot=5.0, noise=18.0, shade=35.0, ink=(0.55, 0.8), specks=0.0008)}[level]
    h, w = gray.shape
    ink = 1.0 - gray.astype(np.float32) / 255.0
    ang = float(rng.uniform(-p["rot"], p["rot"]))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    ink = cv2.warpAffine(ink, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0.0)
    sigma = float(rng.uniform(*p["blur"]))
    ink = cv2.GaussianBlur(ink, (0, 0), sigma)
    strength = float(rng.uniform(*p["ink"]))
    paper = 245.0 - rng.uniform(0, 8)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    gx, gy = rng.uniform(-1, 1, 2)
    shade = p["shade"] * (gx * xs / w + gy * ys / h) * 0.5
    img = paper + shade - strength * ink * paper
    img += rng.normal(0, p["noise"], img.shape).astype(np.float32)
    if p["specks"] > 0:
        mask = rng.random(img.shape) < p["specks"]
        img[mask] -= rng.uniform(40, 120, mask.sum())
    return np.clip(img, 0, 255).astype(np.uint8)


# ---- tokenizer: copied in spirit from the authors' notebook (Apache-2.0) ----
_COMMAND_RE = re.compile(r"\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)")


def tokenize_expression(s: str) -> list[str]:
    tokens = []
    while s:
        if s[0] == "\\":
            m = _COMMAND_RE.match(s)
            tokens.append(m.group(0) if m else s[0])
        else:
            tokens.append(s[0])
        s = s[len(tokens[-1]):]
    return tokens


def token_edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def token_error_rate(pred: str, ref: str) -> float:
    """Token-level edit distance / reference token count (the dataset authors' CER convention)."""
    r, p = tokenize_expression(ref), tokenize_expression(pred)
    return token_edit_distance(p, r) / max(len(r), 1)
