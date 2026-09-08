"""Markov-Lukacs: constructive LOW-RANK SOS certificates of nonnegativity on [0,1].

The relaxation's per-obstacle certificate is

    S_d(M_j) = S_d(Q0_j) + T_d(Q1_j),      Q0_j >= 0 in S^{d+1}, Q1_j >= 0 in S^d,

which in polynomial form says  p_j(s) = sigma_0(s) + s(1-s) sigma_1(s)  with
sigma_0 = b_d^T Q0 b_d  and  sigma_1 = b_{d-1}^T Q1 b_{d-1}.

The classical Markov-Lukacs theorem says more than "sigma_0, sigma_1 are SOS":
for p >= 0 on [0,1] of degree <= 2d the certificate is attained with a SINGLE
square in each term,

    p = q(s)^2 + s(1-s) r(s)^2,     deg q <= d,  deg r <= d-1,

so Q0 = u u^T and Q1 = v v^T are RANK ONE (u, v the Bernstein coefficients of
q, r).  That is one rank lower than the "univariate SOS = sum of two squares"
bound, which gives rank <= 2 -- see docs/rank2_lemma.md.  Q0 and Q1 do not
enter the objective, so restricting them to rank 1 cannot change the optimal
value: the projection of the feasible set onto (Gamma, X) is untouched.

`ml_decompose` is the constructive proof, run as an algorithm:

  * factor p over C;
  * write each factor -- a conjugate pair, a pair of real roots outside (0,1),
    a double root inside, or a leftover linear factor -- in the form
    q^2 + s(1-s) r^2 by solving three scalar equations;
  * multiply the pieces with the Brahmagupta-Fibonacci identity for the form
    N(q, r) = q^2 + w r^2, which is multiplicatively closed:
        (q1^2 + w r1^2)(q2^2 + w r2^2)
            = (q1 q2 - w r1 r2)^2 + w (q1 r2 + q2 r1)^2.

`polish_rank1` then runs Newton on the resulting (u, v) against the exact
Bernstein-coefficient target.  The system is square -- (d+1) + d = 2d+1
unknowns for 2d+1 equations -- so from the constructive initial point it
converges to machine precision whenever p really is nonnegative on [0,1].
Root finding is therefore only ever used as an initialiser; the reported
residual is Newton's.
"""
import numpy as np

from bernstein import S_map, T_map, to_monomial

_pm = np.polynomial.polynomial
_W = np.array([0.0, 1.0, -1.0])          # w(s) = s(1-s), ascending


# ----------------------------------------------------------------------
# one factor at a time
# ----------------------------------------------------------------------
def _factor_rep(L, B, C):
    """Represent F(s) = L s^2 + B s + C (>= 0 on [0,1]) as q^2 + w r^2.

    With q = a s + b and r = c (constant),
        q^2 + w c^2 = (a^2 - c^2) s^2 + (2ab + c^2) s + b^2,
    so  b^2 = F(0),  a^2 - c^2 = L,  and eliminating c^2 gives
        a^2 + 2ba - (B + L) = 0,     B + L + b^2 = F(1).
    Taking a = -b - sqrt(F(1)) maximises |a|, which is what keeps
    c^2 = a^2 - L nonnegative: for L = 1 it needs |a| >= 1, i.e.
    sqrt(F(0)) + sqrt(F(1)) >= 1, which holds with equality exactly when F is
    a perfect square with its root in [0,1].  L in {1, -1, 0} are all handled
    by the same three lines; L = 0 is a leftover linear factor.
    """
    F0 = C
    F1 = L + B + C
    b = np.sqrt(max(F0, 0.0))
    a = -b - np.sqrt(max(F1, 0.0))
    c2 = a * a - L
    # A perfect square with its root inside [0,1] hits c2 = 0 exactly, so tiny
    # negative values are rounding, not a violation; only a real dip counts.
    clipped = c2 < -1e-9 * max(1.0, abs(L), abs(B), abs(C))
    q = np.array([b, a])
    r = np.array([np.sqrt(max(c2, 0.0))])
    return q, r, bool(clipped)


def _mul(q1, r1, q2, r2):
    """Brahmagupta-Fibonacci for N(q, r) = q^2 + w r^2."""
    q = _pm.polymul(q1, q2)
    t = _pm.polymul(_pm.polymul(r1, r2), _W)
    q = _pm.polyadd(q, -t)
    r = _pm.polyadd(_pm.polymul(q1, r2), _pm.polymul(q2, r1))
    return q, r


def _classify(rts, tol):
    """Split roots into conjugate pairs / inner reals / left reals / right reals."""
    pairs, inner, left, right = [], [], [], []
    used = np.zeros(rts.size, bool)
    for i in range(rts.size):
        if used[i]:
            continue
        z = rts[i]
        if abs(z.imag) > tol:
            # find the conjugate partner
            best, bd_ = -1, np.inf
            for j in range(rts.size):
                if used[j] or j == i:
                    continue
                dd = abs(rts[j] - np.conj(z))
                if dd < bd_:
                    best, bd_ = j, dd
            if best >= 0:
                used[i] = used[best] = True
                pairs.append(z)
                continue
        used[i] = True
        x = float(z.real)
        if x < tol:
            left.append(x)
        elif x > 1.0 - tol:
            right.append(x)
        else:
            inner.append(x)
    return pairs, sorted(inner), sorted(left), sorted(right, reverse=True)


def ml_decompose(p_mono, tol=1e-7):
    """p >= 0 on [0,1] (ascending monomial coeffs) -> (q, r) with p = q^2 + w r^2.

    Returns (q, r, info).  `info['ok']` is False when the root pattern is not
    the one a nonnegative polynomial must have (an odd number of roots strictly
    inside (0,1), or a wrong overall sign); the caller should still try Newton,
    because a p that dips a few 1e-10 below zero lands here routinely.
    """
    p = np.asarray(p_mono, float)
    scale = float(np.max(np.abs(p))) if p.size else 0.0
    if scale == 0.0:
        return np.zeros(1), np.zeros(1), dict(ok=True, n_clipped=0, deg=0)
    nz = np.nonzero(np.abs(p) > 1e-13 * scale)[0]
    p = p[:nz[-1] + 1]
    info = dict(ok=True, n_clipped=0, deg=int(p.size - 1))
    if p.size == 1:
        if p[0] < 0.0:
            info["ok"] = False
        return np.array([np.sqrt(max(p[0], 0.0))]), np.zeros(1), info

    lead = float(p[-1])
    rts = np.asarray(_pm.polyroots(p), complex)
    pairs, inner, left, right = _classify(rts, tol)

    factors, n_flip = [], 0
    for z in pairs:                                   # (s-z)(s-zbar)
        factors.append((1.0, -2.0 * float(z.real), float(abs(z) ** 2)))
    for a in range(0, len(inner) - 1, 2):             # double root inside
        x, y = inner[a], inner[a + 1]
        factors.append((1.0, -(x + y), x * y))
    if len(inner) % 2:
        info["ok"] = False                            # p changes sign in (0,1)
        x = inner[-1]
        factors.append((1.0, -2.0 * x, x * x))
    nlr = min(len(left), len(right))
    for a in range(nlr):                              # (s-xi)(eta-s)
        xi, eta = left[a], right[a]
        factors.append((-1.0, xi + eta, -xi * eta))
        n_flip += 1
    rest_l, rest_r = left[nlr:], right[nlr:]
    for grp in (rest_l, rest_r):
        for a in range(0, len(grp) - 1, 2):
            x, y = grp[a], grp[a + 1]
            factors.append((1.0, -(x + y), x * y))
    if len(rest_l) % 2:                               # leftover (s - xi)
        factors.append((0.0, 1.0, -rest_l[-1]))
    if len(rest_r) % 2:                               # leftover (eta - s)
        factors.append((0.0, -1.0, rest_r[-1]))
        n_flip += 1

    kappa = lead * (-1.0) ** n_flip
    if kappa <= 0.0:
        info["ok"] = False
        kappa = abs(kappa) if kappa != 0.0 else 1.0

    q, r = np.array([np.sqrt(kappa)]), np.zeros(1)
    for L, B, C in factors:
        qf, rf, clipped = _factor_rep(L, B, C)
        info["n_clipped"] += int(clipped)
        q, r = _mul(q, r, qf, rf)
    return q, r, info


# ----------------------------------------------------------------------
# Bernstein coordinates + Newton polish
# ----------------------------------------------------------------------
def mono_to_bernstein(c_mono, deg):
    """Bernstein-`deg` coefficients u with sum_i u_i B^deg_i(s) = c_mono(s)."""
    c = np.zeros(deg + 1)
    c_mono = np.asarray(c_mono, float)
    if c_mono.size > deg + 1:
        if np.max(np.abs(c_mono[deg + 1:])) > 1e-8 * max(1.0, np.max(np.abs(c_mono))):
            raise ValueError(f"polynomial of degree {c_mono.size - 1} > {deg}")
        c_mono = c_mono[:deg + 1]
    c[:c_mono.size] = c_mono
    return np.linalg.solve(to_monomial(deg).T, c)


def rank1_residual(u, v, target, Sd, Td):
    """S_d(u u^T) + T_d(v v^T) - target, in the B^{2d} coefficient basis."""
    return (np.einsum('kij,i,j->k', Sd, u, u)
            + np.einsum('kij,i,j->k', Td, v, v) - target)


#: The compiled Newton, when the kernel is built.  It is 58 % of
#: `rank1_certificate`, which is nearly all of `Factored.init_cold`, which
#: dominates a seed hunt -- so this is the one hot spot worth compiling.
#:
#: It was written before `ChildActiveSet` seeded itself tie-aware, and back
#: then switching it on changed an answer: the two agree to 1e-10, but that
#: was enough to flip which of three near-equal violations `argmin` picked,
#: and a80's chord3 child then cost 13.7 % more.  The fix belonged in the
#: active set, not here.  With that fix the two arms return identical costs on
#: all four spatial scenes, which `tests/test_ml_polish_c.py` pins.
_C_POLISH = None


def _c_polish():
    global _C_POLISH
    if _C_POLISH is None:
        try:
            import bm_fast as _bf
            _bf._try_import_kernel()
            if _bf.HAVE_KERNEL and hasattr(_bf._knl, "ml_polish_rank1"):
                _C_POLISH = (_bf._knl, _bf._knl_ffi)
            else:
                _C_POLISH = False
        except Exception:
            _C_POLISH = False
    return _C_POLISH or None


def polish_rank1(u, v, target, Sd, Td, iters=60, tol=1e-14):
    """Newton on the square system rank1_residual(u, v) = 0."""
    knl = _c_polish()
    if knl is not None:
        lib, ffi = knl
        uc = np.ascontiguousarray(u, float).copy()
        vc = np.ascontiguousarray(v, float).copy()
        S = np.ascontiguousarray(Sd, float)
        T = np.ascontiguousarray(Td, float)
        g = np.ascontiguousarray(target, float)
        rel = ffi.new("double *")
        dp = lambda a: ffi.cast("double *", ffi.from_buffer(a))   # noqa: E731
        rc = lib.ml_polish_rank1(uc.size - 1, dp(S), dp(T), dp(g), dp(uc),
                                 dp(vc), int(iters), float(tol), rel)
        if rc == 0:
            return uc, vc, float(rel[0])
    return _polish_rank1_py(u, v, target, Sd, Td, iters, tol)


def _polish_rank1_py(u, v, target, Sd, Td, iters=60, tol=1e-14):
    """The reference implementation; the C above must agree with it."""
    u, v = np.array(u, float), np.array(v, float)
    scale = max(1.0, float(np.max(np.abs(target))))
    best = (u.copy(), v.copy(), float(np.max(np.abs(rank1_residual(u, v, target, Sd, Td)))))
    for _ in range(iters):
        res = rank1_residual(u, v, target, Sd, Td)
        nres = float(np.max(np.abs(res)))
        if nres < best[2]:
            best = (u.copy(), v.copy(), nres)
        if nres <= tol * scale:
            break
        J = np.concatenate([2.0 * np.einsum('kij,j->ki', Sd, u),
                            2.0 * np.einsum('kij,j->ki', Td, v)], axis=1)
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        t = 1.0
        for _ls in range(30):                       # plain backtracking
            un = u + t * step[:u.size]
            vn = v + t * step[u.size:]
            if float(np.max(np.abs(rank1_residual(un, vn, target, Sd, Td)))) < nres:
                break
            t *= 0.5
        else:
            break
        u, v = un, vn
    res = rank1_residual(u, v, target, Sd, Td)
    if float(np.max(np.abs(res))) > best[2]:
        u, v, _ = best
    return u, v, float(np.max(np.abs(rank1_residual(u, v, target, Sd, Td))) / scale)


def rank1_certificate(target, d, Sd=None, Td=None, tol=1e-7):
    """Rank-1 (Q0, Q1) matching `target` = S_d(M) in the B^{2d} basis.

    Returns (u, v, info) with Q0 = outer(u, u), Q1 = outer(v, v).
    info['rel_residual'] is Newton's, relative to max|target|.
    """
    Sd = S_map(d) if Sd is None else Sd
    Td = T_map(d) if Td is None else Td
    target = np.asarray(target, float)
    p_mono = target @ to_monomial(2 * d)
    q, r, info = ml_decompose(p_mono, tol=tol)
    try:
        u = mono_to_bernstein(q, d)
        v = mono_to_bernstein(r, d - 1)
    except ValueError as exc:
        info = dict(info, ok=False, degree_error=str(exc))
        u = np.full(d + 1, np.sqrt(max(float(np.max(target)), 0.0)))
        v = np.zeros(d)
    r0 = float(np.max(np.abs(rank1_residual(u, v, target, Sd, Td)))
               / max(1.0, float(np.max(np.abs(target)))))
    u, v, rel = polish_rank1(u, v, target, Sd, Td)
    info.update(rel_residual=rel, rel_residual_construct=r0)
    return u, v, info


def numeric_rank(A, rtol=1e-7):
    w = np.linalg.eigvalsh(0.5 * (A + A.T))
    scale = max(1.0, float(np.abs(w).max()) if w.size else 1.0)
    return int(np.sum(w > rtol * scale))
