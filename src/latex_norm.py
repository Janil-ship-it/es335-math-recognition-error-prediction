"""Regex-based LaTeX normalisation for fair exact-match comparison.

Rules (each has a unit test):
- map \\leq/\\geq/\\neq to short forms BEFORE collapsing whitespace
  (with (?![a-zA-Z]) lookaheads so \\geqslant is never turned into \\geslant)
- do NOT map \\cdot -> \\times
- strip \\left / \\right
- unify \\dfrac/\\tfrac -> \\frac
- collapse whitespace and spacing commands
"""
from __future__ import annotations
import re

_CMD_MAP = [
    (re.compile(r"\\leq(?![a-zA-Z])"), r"\\le"),
    (re.compile(r"\\geq(?![a-zA-Z])"), r"\\ge"),
    (re.compile(r"\\neq(?![a-zA-Z])"), r"\\ne"),
    (re.compile(r"\\to(?![a-zA-Z])"), r"\\rightarrow"),
    (re.compile(r"\\dfrac(?![a-zA-Z])"), r"\\frac"),
    (re.compile(r"\\tfrac(?![a-zA-Z])"), r"\\frac"),
]
_LEFT_RIGHT_RE = re.compile(r"\\left(?![a-zA-Z])|\\right(?![a-zA-Z])")
_SPACING_RE = re.compile(r"\\[,\;\!]|\\quad|\\qquad|\\hspace\{[^}]*\}")
_SPACE_RE = re.compile(r"\s+")


def normalise_latex(s: str) -> str:
    if not isinstance(s, str):
        s = str(s) if s is not None else ""
    s = s.strip()
    for pat, repl in _CMD_MAP:
        s = pat.sub(repl, s)
    s = _LEFT_RIGHT_RE.sub("", s)
    s = _SPACING_RE.sub("", s)
    s = _SPACE_RE.sub("", s)
    return s


def needs_correction(pred: str, ref: str) -> int:
    return int(normalise_latex(pred) != normalise_latex(ref))
