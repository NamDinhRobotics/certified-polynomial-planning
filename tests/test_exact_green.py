"""Pin the exact-arithmetic certificate, including where it does NOT certify.

The point of exact arithmetic here is that a sign is either negative or it is
not; a test suite that only asserted the successes would pass just as well for a
routine that certified everything.  So the failures are pinned too: the
single-segment configurations must stay strictly negative, and welded knots must
stay strictly negative, in exact arithmetic.

The other thing worth pinning is that the exact computation answers the same
question as the floating-point pipeline.  It uses a different null-space basis,
and that is only harmless because the tested block is basis-independent --- so
that invariance is asserted directly rather than assumed.
"""
import os
import sys
from fractions import Fraction as F

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments"))

from exact_green import (certify_exact, coefficient_blocks,      # noqa: E402
                         constraint_matrix, elevated_min, matmul,
                         nullspace, q_gram, q_gram_deriv)

TABLE1 = [("velocity", 1, 1, 0, 12, 0), ("acceleration", 2, 3, 2, 6, 2),
          ("jerk", 3, 5, 3, 4, 3), ("snap", 4, 7, 4, 3, 4)]
SINGLE = ((3, 1, 0), (5, 1, 0), (7, 1, 0), (9, 1, 0),
          (5, 2, 1), (7, 2, 1), (7, 3, 2))


# ---- the arithmetic itself -------------------------------------------
def test_gram_is_exact_rational():
    G = q_gram(3)
    assert all(isinstance(x, F) for row in G for x in row)
    # G_d entries are C(d,i)C(d,j)/((2d+1)C(2d,i+j)); check one by hand
    assert G[0][0] == F(1, 7 * 1)
    Gf = np.array([[float(x) for x in row] for row in G])
    assert np.allclose(Gf, Gf.T)


def test_gram_deriv_matches_float_implementation():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from bernstein import gram_deriv
    for d, k in ((3, 1), (5, 2), (7, 3)):
        E = np.array([[float(x) for x in row] for row in q_gram_deriv(d, k)])
        assert np.abs(E - gram_deriv(d, k)).max() < 1e-9


def test_nullspace_is_exact_and_correct():
    A = constraint_matrix(d=3, l=0, N=2, eta=1)
    B = nullspace(A)
    assert B, "expected a nontrivial null space"
    for v in B:
        for row in A:
            assert sum(a * b for a, b in zip(row, v)) == 0   # exactly zero


# ---- the property that licenses using a different basis ---------------
def test_block_is_independent_of_the_nullspace_basis():
    """`N -> N T` sends `K` to `T^T K T` and leaves `N K^{-1} N^T` fixed.

    Without this the exact computation would be answering a question about a
    basis rather than about the curve, and comparing it to the float pipeline
    would be meaningless.
    """
    d, k, l, N, eta = 3, 1, 0, 2, 1
    blocks, r = coefficient_blocks(d, k, l, N, eta)
    # re-derive with a deliberately skewed basis of the same null space
    import exact_green as eg
    orig = eg.nullspace

    def skewed(A):
        B = orig(A)
        T = [[F(1) if i == j else F(1, 2 + (i + j) % 3) for j in range(len(B))]
             for i in range(len(B))]
        cols = matmul([[B[c][i] for c in range(len(B))]
                       for i in range(len(B[0]))], T)
        return [[cols[i][c] for i in range(len(cols))] for c in range(len(B))]

    eg.nullspace = skewed
    try:
        blocks2, r2 = coefficient_blocks(d, k, l, N, eta)
    finally:
        eg.nullspace = orig
    assert r2 == r
    for key in blocks:
        assert blocks[key] == blocks2[key], "block moved with the basis"


def test_exact_matches_float_pipeline():
    from multisegment import MultiSegment
    for _, k, d, l, N, eta in TABLE1:
        blocks, _ = coefficient_blocks(d, k, l, N, eta)
        b0 = [[-2., 0.]] + [[0., 0.]] * l
        b1 = [[2., 0.]] + [[0., 0.]] * l
        ms = MultiSegment(2, d, k, l, N, [(np.zeros(2), .4)], b0, b1, eta=eta)
        maps = [ms.Nperp[ms.slice_(i)] for i in range(N)]
        K = ms.time_scale * sum(M.T @ ms.Gk @ M for M in maps)
        Ki = np.linalg.inv((K + K.T) / 2)
        fl = [maps[i] @ Ki @ maps[j].T for i in range(N) for j in range(N)]
        for i in range(N):
            for j in range(N):
                E = np.array([[float(x) for x in row] for row in blocks[(i, j)]])
                assert np.abs(E - fl[i * N + j]).max() < 1e-12


# ---- what is certified, and what is not -------------------------------
@pytest.mark.parametrize("name,k,d,l,N,eta", TABLE1)
def test_table1_rows_are_exactly_zero_not_negative(name, k, d, l, N, eta):
    """The claim the paper makes, and the one it deliberately does not.

    Nonnegative: yes, exactly.  Strictly positive: NO -- the minimum is exactly
    zero, which is why Theorem 9 needs the "no block vanishes identically"
    hypothesis rather than a strict bound.  Asserting `not strict` keeps a later
    change from quietly upgrading the claim.
    """
    r = certify_exact(d, k, l, N, eta, D=96)
    assert r["elev_min"] == 0
    assert r["nonneg"] is True
    assert r["strict"] is False


@pytest.mark.parametrize("d,k,l", SINGLE)
def test_single_segment_is_strictly_negative(d, k, l):
    """The negative control: one polynomial never certifies, exactly."""
    r = certify_exact(d, k, l, N=1, eta=0, D=96)
    assert r["elev_min"] < 0
    assert r["nonneg"] is False


@pytest.mark.parametrize("N,eta", [(2, 3), (3, 3), (4, 3)])
def test_welded_knots_stay_negative(N, eta):
    """`eta = d` is a single polynomial in disguise and must behave like one."""
    r = certify_exact(3, 1, 0, N, eta, D=96)
    assert r["elev_min"] < 0
    assert r["nonneg"] is False


@pytest.mark.parametrize("N", [2, 3, 4])
def test_genuine_knots_certify(N):
    r = certify_exact(3, 1, 0, N, eta=1, D=96)
    assert r["nonneg"] is True


# ---- the irreducibility hypothesis ------------------------------------
@pytest.mark.parametrize("name,k,d,l,N,eta", TABLE1)
def test_no_block_vanishes_and_pinned_rows_are_the_boundary(name, k, d, l,
                                                            N, eta):
    blocks, r = coefficient_blocks(d, k, l, N, eta)
    Np = nullspace(constraint_matrix(d, l, N, eta))
    pinned = [i for i in range(N * (d + 1))
              if all(Np[c][i] == 0 for c in range(r))]
    assert len(pinned) == 2 * (l + 1)
    for key, B in blocks.items():
        assert any(x > 0 for row in B for x in row), \
            "block %s vanishes: Gr would be reducible" % (key,)


def test_elevation_is_monotone():
    """Elevating must not make the bound worse, or the test would be unsound."""
    blocks, _ = coefficient_blocks(3, 1, 0, 1, 0)
    B = blocks[(0, 0)]
    prev = None
    for D in (3, 8, 16, 32):
        cur = elevated_min(B, 3, D)
        if prev is not None:
            assert cur >= prev - F(1, 10 ** 30)
        prev = cur


# ---- the closed square: junctions, not just interior parameters --------
@pytest.mark.parametrize("name,k,d,l,N,eta", TABLE1)
def test_closed_square_hypothesis_holds(name, k, d, l, N, eta):
    """The open-square argument leaves the boundary, and the boundary is not
    empty: an interior junction has `u != 0` and is reachable.  The extra
    hypothesis is a finite sign check on rows `0` and `d`; if it ever failed,
    Corollary 13 would have to carry an interiority caveat again."""
    from exact_green import endpoint_structure
    s = endpoint_structure(d, k, l, N, eta)
    assert s["n_live_endpoints"] == 2 * (N - 1), \
        "the live boundary indices should be exactly the interior junctions"
    assert s["rows_ok"] and s["corners_ok"]
    assert s["closed_square_ok"]


def test_pinned_ends_are_not_live():
    """The two ends of the curve are pinned, so they are vacuous rather than
    boundary cases needing the extra hypothesis."""
    from exact_green import endpoint_structure
    s = endpoint_structure(3, 1, 0, 3, 1)
    assert [0, 0] not in s["live_endpoints"]        # start of segment 0
    assert [2, 3] not in s["live_endpoints"]        # end of the last segment
