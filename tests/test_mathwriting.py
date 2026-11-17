import numpy as np
from src.mathwriting import tokenize_expression, token_error_rate, degrade, DEGRADATION_LEVELS

def test_tokenizer_matches_authors_example():
    t = tokenize_expression(r'\frac{\alpha}{2}\begin{matrix}1&0\\0&1\end{matrix}\not\in\mathbb{R}')
    assert t[0] == r'\frac' and r'\alpha' in t and r'\begin{matrix}' in t and r'\mathbb{R}' in t

def test_token_error_rate_matches_authors_numbers():
    # authors' example: 5 mistakes over 19 reference tokens
    ex = [(r'\sqrt{2}', r'\sqrt{2}'), (r'\frac{1}{2}', r'\frac{i}{2}'), (r'\alpha^{2}', 'a^{2}'), ('abc', 'def')]
    errs = sum(token_error_rate(p, r) * len(tokenize_expression(r)) for p, r in ex)
    total = sum(len(tokenize_expression(r)) for _, r in ex)
    assert round(errs) == 5 and total == 19

def test_x10_vs_x2_is_token_level():
    assert token_error_rate('x^{10}', 'x^{2}') < 0.5   # char-level SER gave 0.33; here 1 sub + 1 ins over 5 tokens

def test_degradation_is_deterministic_and_levels_differ():
    g = np.full((80, 200), 255, np.uint8); g[30:50, 20:180] = 0
    a, b = degrade(g, 'strong', 7), degrade(g, 'strong', 7)
    assert (a == b).all()
    assert not (degrade(g, 'clean', 7) == degrade(g, 'mild', 7)).all()
    assert set(DEGRADATION_LEVELS) == {'clean', 'mild', 'strong'}
