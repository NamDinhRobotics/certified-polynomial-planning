"""Bernstein / Bezier utilities for the Graesdal-Amice-Parrilo-Tedrake relaxation.

Standard Bernstein basis:  B^d_i(s) = C(d,i) s^i (1-s)^(d-i),  i = 0..d
All equation numbers refer to arXiv:2606.14063.
"""
import numpy as np
from math import comb, factorial


def bd(s, d):
    """Basis vector b_d(s) in R^{d+1}."""
    s = np.asarray(s, dtype=float)
    i = np.arange(d + 1)
    c = np.array([comb(d, k) for k in i], dtype=float)
    return c * (s ** i) * ((1.0 - s) ** (d - i))


def diff_matrix(d):
    """D_d in R^{(d+1) x d}: -1 on diagonal, +1 on subdiagonal (eq. 28)."""
    D = np.zeros((d + 1, d))
    for i in range(d):
        D[i, i] = -1.0
        D[i + 1, i] = 1.0
    return D


def Dk(d, k):
    """D_d^{(k)} = D_d D_{d-1} ... D_{d-k+1}  in R^{(d+1) x (d-k+1)}."""
    M = np.eye(d + 1)
    for j in range(k):
        M = M @ diff_matrix(d - j)
    return M


def gram(d):
    """G_d with [G_d]_ij = C(d,i)C(d,j) / ((2d+1) C(2d,i+j))   (eq. 29)."""
    G = np.zeros((d + 1, d + 1))
    for i in range(d + 1):
        for j in range(d + 1):
            G[i, j] = comb(d, i) * comb(d, j) / ((2 * d + 1) * comb(2 * d, i + j))
    return G


def gram_deriv(d, k):
    """G^{(k)} = (d!/(d-k)!)^2 D_d^{(k)} G_{d-k} (D_d^{(k)})^T   (eq. 30)."""
    if k == 0:
        return gram(d)
    scale = (factorial(d) / factorial(d - k)) ** 2
    Dm = Dk(d, k)
    return scale * (Dm @ gram(d - k) @ Dm.T)


def S_map(d):
    """S_d : S^{d+1} -> R^{2d+1},  [S_d(M)]_k = sum_{i+j=k} C(d,i)C(d,j)/C(2d,k) M_ij  (eq. 32).

    Returned as a (2d+1, d+1, d+1) tensor T so that S_d(M) = einsum('kij,ij->k', T, M).
    """
    T = np.zeros((2 * d + 1, d + 1, d + 1))
    for i in range(d + 1):
        for j in range(d + 1):
            k = i + j
            T[k, i, j] = comb(d, i) * comb(d, j) / comb(2 * d, k)
    return T


def T_map(d):
    """T_d : S^d -> R^{2d+1} (eq. 33), shifted+weighted version of S_{d-1}."""
    assert d >= 1
    T = np.zeros((2 * d + 1, d, d))
    Sm1 = S_map(d - 1)                      # (2d-1, d, d)
    for k in range(1, 2 * d):
        w = k * (2 * d - k) / (2 * d * (2 * d - 1))
        T[k] = w * Sm1[k - 1]
    return T


def apply_map(T, M):
    return np.einsum('kij,ij->k', T, M)


def to_monomial(d):
    """Rows i = monomial coefficients (ascending powers) of B^d_i.

    B^d_i(s) = C(d,i) s^i (1-s)^(d-i), so row i is C(d,i) * conv(e_i, (1-s)^(d-i)).
    Lets a Bezier curve be turned into an exact polynomial, which is what the
    exact feasibility check in relaxation.exact_clearance needs.
    """
    M = np.zeros((d + 1, d + 1))
    for i in range(d + 1):
        # (1-s)^(d-i) = sum_t C(d-i,t) (-1)^t s^t
        for t in range(d - i + 1):
            M[i, i + t] = comb(d, i) * comb(d - i, t) * ((-1.0) ** t)
    return M


def bd_derivs(s, d, order):
    """Rows 0..order of  d^i/ds^i b_d(s), each in R^{d+1}."""
    out = []
    for i in range(order + 1):
        if i == 0:
            out.append(bd(s, d))
        else:
            scale = factorial(d) / factorial(d - i)
            out.append(scale * (Dk(d, i) @ bd(s, d - i)))
    return np.array(out)
