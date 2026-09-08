"""The Bernstein bracket must decide exactly what the Sturm chain decides.

The chain is the authority; the bracket is only allowed to be faster.  These
pin that, and pin the two places where a subdivision test is easy to get wrong:
the de Casteljau split must be exact, and a polynomial that TOUCHES zero must
not be reported strictly positive.
"""
import math
import random
import sys
from fractions import Fraction as Fr

import pytest

sys.path.insert(0, "src")
import exact_check as E


def _bern_eval(coeffs, x):
    n = len(coeffs) - 1
    return sum(Fr(coeffs[i]) * math.comb(n, i) * x ** i * (1 - x) ** (n - i)
               for i in range(n + 1))


def test_decasteljau_halves_are_exact():
    rng = random.Random(3)
    for _ in range(120):
        d = rng.randint(1, 12)
        c = [rng.randint(-30, 30) for _ in range(d + 1)]
        lo, hi = E._decasteljau_halves_int(c)
        scale = Fr(1, 1 << d)
        for t in (Fr(0), Fr(1, 8), Fr(1, 4), Fr(3, 8), Fr(1, 2)):
            assert _bern_eval(c, t) == scale * _bern_eval(lo, 2 * t)
        for t in (Fr(1, 2), Fr(5, 8), Fr(3, 4), Fr(1)):
            assert _bern_eval(c, t) == scale * _bern_eval(hi, 2 * t - 1)


def test_power_to_bern_inverts_bern_to_power():
    rng = random.Random(5)
    for _ in range(120):
        d = rng.randint(1, 16)
        b = [Fr(rng.randint(-40, 40), rng.randint(1, 9)) for _ in range(d + 1)]
        assert E.power_to_bern(E.bern_to_power(list(b), d), d) == b


def test_fast_decision_equals_the_sturm_authority():
    rng = random.Random(23)
    fell_back = 0
    for _ in range(300):
        d = rng.randint(1, 16)
        p = [Fr(rng.randint(-25, 25), rng.randint(1, 7)) for _ in range(d + 1)]
        for tau in (Fr(0), Fr(1, 100), Fr(-1, 50)):
            ref = E.positive_on_unit_interval(p, tau)
            fast = E.positive_on_unit_interval_fast(p, tau)
            assert bool(ref) == bool(fast), (d, tau)
            q = E._trim(E._add([Fr(v) for v in p], [Fr(tau)]))
            if E._deg(q) >= 0:
                decided, _ = E.positive_on_unit_interval_bernstein_int(
                    E.power_to_bern(q, E._deg(q)))
                fell_back += not decided
    # not a requirement, but a regression tripwire: the bracket settled all of
    # these when the test was written
    assert fell_back == 0


def test_a_polynomial_touching_zero_is_not_strictly_positive():
    # (2s - 1)^2 in the degree-2 Bernstein basis
    coeffs = [Fr(1), Fr(-1), Fr(1)]
    assert _bern_eval(coeffs, Fr(1, 2)) == 0
    decided, value = E.positive_on_unit_interval_bernstein_int(coeffs)
    assert decided and value is False
    assert E.positive_on_unit_interval_bernstein(coeffs, 2, 0.0) is False


def test_all_positive_coefficients_settle_in_one_pass():
    coeffs = [Fr(3), Fr(1), Fr(7), Fr(2)]
    decided, value = E.positive_on_unit_interval_bernstein_int(coeffs)
    assert decided and value is True


@pytest.mark.parametrize("height,expected", [(0., False), (1., True)])
def test_certify_grid_flag_does_not_move_the_verdict(height, expected):
    """The optional sampled diagnostic cannot change the exact decision."""
    import numpy as np
    from types import SimpleNamespace
    from bernstein import bd
    curve = np.array([np.linspace(0., 1., 6), np.full(6, height)])
    center = np.array([.5, .05])
    def clearances(curves):
        points = np.array([curves[0] @ bd(t, 5) for t in np.linspace(0., 1., 2001)])
        return {"obstacle": float(np.min(np.sum((points - center)**2, axis=1) - .2**2))}
    problem = SimpleNamespace(d=5, n=2, obs=[[(center, .2)]], rr=0., pairs=[],
                              clearances=clearances)
    a = E.certify(problem, [curve], tol=1e-6, grid=True)
    b = E.certify(problem, [curve], tol=1e-6, grid=False)
    for key in ("ok", "obstacle", "pair", "linear", "deriv", "n_polys"):
        assert a[key] == b[key]
    assert a["ok"] is expected
    assert b["worst_grid"] is None and a["worst_grid"] is not None
