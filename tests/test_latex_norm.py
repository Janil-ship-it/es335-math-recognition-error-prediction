from src.latex_norm import normalise_latex, needs_correction

def test_collapse_whitespace():
    assert normalise_latex(r"a  +  b") == "a+b"

def test_leq_geq_neq_with_lookahead():
    assert normalise_latex(r"a\leq b") == r"a\leb"
    assert normalise_latex(r"a\geq b") == r"a\geb"
    assert normalise_latex(r"a\neq b") == r"a\neb"
    assert "geqslant" in normalise_latex(r"a\geqslant b")

def test_cdot_not_mapped_to_times():
    s = normalise_latex(r"a\cdot b")
    assert r"\cdot" in s
    assert r"\times" not in s

def test_left_right_stripped():
    assert normalise_latex(r"\left(x\right)") == "(x)"

def test_dfrac_tfrac_to_frac():
    assert normalise_latex(r"\dfrac{1}{2}") == r"\frac{1}{2}"
    assert normalise_latex(r"\tfrac{1}{2}") == r"\frac{1}{2}"

def test_needs_correction():
    assert needs_correction(r"a\leq b", r"a\le b") == 0
    assert needs_correction(r"a+b", r"a-b") == 1
