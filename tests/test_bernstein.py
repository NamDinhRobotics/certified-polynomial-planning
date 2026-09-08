"""Gate 1a -- Bernstein basis identities.

These are identities, not results: a failure here is a regression in the
basis code, not a discovery.  Tolerances are 1e-9; the implementation actually
achieves ~1e-16.
"""
import numpy as np
import pytest
from math import factorial

from bernstein import bd, bd_derivs, Dk, gram_deriv, S_map, T_map, apply_map

DEGREES = [1, 2, 3, 5, 7]
TOL = 1e-9
SS = np.linspace(0.0, 1.0, 41)


@pytest.mark.parametrize("d", DEGREES)
def test_partition_of_unity(d):
    """sum_i B^d_i(s) = 1 for all s  (eq. 31: u = e)."""
    err = max(abs(bd(s, d).sum() - 1.0) for s in SS)
    assert err < TOL, f"d={d} partition of unity err={err:.3e}"


@pytest.mark.parametrize("d", DEGREES)
def test_cost_identity(d):
    """int_0^1 ||gamma^(k)||^2 ds = tr(G^(k) Gamma^T Gamma)  (eq. 3).

    Checked against 40-point Gauss-Legendre, which is exact for these degrees.
    """
    rng = np.random.default_rng(d)
    xg, wg = np.polynomial.legendre.leggauss(40)
    xg = 0.5 * (xg + 1.0)
    wg = 0.5 * wg
    for k in range(1, min(d, 4) + 1):
        G = rng.standard_normal((3, d + 1))
        Gk = gram_deriv(d, k)
        scale = factorial(d) / factorial(d - k)
        lhs = 0.0
        for x, w in zip(xg, wg):
            v = G @ (Dk(d, k) @ bd(x, d - k)) * scale
            lhs += w * float(v @ v)
        rhs = float(np.trace(Gk @ G.T @ G))
        rel = abs(lhs - rhs) / max(1.0, abs(rhs))
        assert rel < 1e-8, f"d={d} k={k} cost identity rel err={rel:.3e}"


@pytest.mark.parametrize("d", DEGREES)
def test_S_map(d):
    """b_d(s)^T M b_d(s) = S_d(M)^T b_2d(s)  (eq. 32)."""
    rng = np.random.default_rng(100 + d)
    M = rng.standard_normal((d + 1, d + 1))
    M = M + M.T
    coef = apply_map(S_map(d), M)
    err = max(abs(bd(s, d) @ M @ bd(s, d) - coef @ bd(s, 2 * d)) for s in SS)
    assert err < TOL, f"d={d} S_d err={err:.3e}"


@pytest.mark.parametrize("d", DEGREES)
def test_T_map(d):
    """s(1-s) b_{d-1}(s)^T M b_{d-1}(s) = T_d(M)^T b_2d(s)  (eq. 33)."""
    rng = np.random.default_rng(200 + d)
    M = rng.standard_normal((d, d))
    M = M + M.T
    coef = apply_map(T_map(d), M)
    err = max(abs(s * (1 - s) * (bd(s, d - 1) @ M @ bd(s, d - 1))
                  - coef @ bd(s, 2 * d)) for s in SS)
    assert err < TOL, f"d={d} T_d err={err:.3e}"


@pytest.mark.parametrize("d", [2, 3, 5, 7])
def test_second_derivative(d):
    """bd_derivs against a central finite difference."""
    rng = np.random.default_rng(300 + d)
    G = rng.standard_normal((2, d + 1))
    h, s0 = 1e-5, 0.37
    num = (G @ bd(s0 + h, d) - 2 * G @ bd(s0, d) + G @ bd(s0 - h, d)) / h ** 2
    ana = G @ bd_derivs(s0, d, 2)[2]
    rel = np.max(np.abs(num - ana)) / max(1.0, np.max(np.abs(ana)))
    assert rel < 1e-5, f"d={d} 2nd derivative rel err={rel:.3e}"
