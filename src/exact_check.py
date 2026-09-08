"""An EXACT continuous-time verifier for the curves the recovery procedure
executes, in place of the finite grid the loop actually gates on.

`iso_planner_scene._clear_ok` (and `iso_swing_planner._clear_ok`, the same
function) accepts a trajectory when `MultiRobot.clearances` and
`MultiRobotLinear.linear_margins` -- both of which SAMPLE 2001 parameters --
are all `>= -1e-6`.  A finite grid is not a continuous-time certificate: a
degree-`2d` polynomial can dip below the tolerance strictly between two
samples, and the sampling that would prove otherwise does not exist.  Every
constraint the two classes carry is, however, a UNIVARIATE POLYNOMIAL
inequality on `[0, 1]`, and that is exactly decidable.

    obstacle   p(s) = ||gamma(s) - c||^2 - r^2                degree 2d
    pair       p(s) = ||gamma_i(s) - gamma_j(s)||^2 - (2rr)^2 degree 2d
    linear     p(s) = sum_i a_i^T Gamma_i W b_d(s) - beta(s)  degree d
    axis       p(s) = a -+ u^T gamma^(j)(s)                   degree d - j
    deriv      p(s) = a^2 - ||gamma^(j)(s)||^2                degree 2(d - j)
    point cut  side * (u^T gamma(s*) - b)  at ONE parameter -- not a
               continuum question at all, so it is evaluated directly

The decision is arithmetic, not numerical.  Every float that enters (the
control points `Gamma`, the centres `c`, the radii, the tolerance `tau`) is a
binary rational, so `fractions.Fraction(x)` is its EXACT value; the Bernstein
to power basis change uses integer binomials; and the number of distinct real
roots of `q = p + tau` in `(0, 1]` is counted by a Sturm sequence over exact
`Fraction`s.  `q > 0` on `[0, 1]` iff `q(0) > 0` and that count is zero, and
that is the verdict this module reports.  Nothing is sampled and nothing is
rounded, so a verdict here is a statement about the continuum.

The test is STRICT (`q > 0`, not `q >= 0`) by construction: `q(0) = 0` or
`q(1) = 0` is a root and is reported as NOT certified.  Against the grid test
at the same `tau` this can differ only on a curve that touches `-tau` exactly,
which no float trajectory produced by an interior-point solver does.

Not covered, and raised as an error rather than passed silently: MOVING
obstacles (`MultiRobot.moving`).  They are the same polynomial family, but
`clearances` does not read them either, so certifying them here would compare
two different constraint sets.  None of the instances this module is used on
carries one.
"""
from fractions import Fraction
from math import comb, factorial


import numpy as np

#: 2d for the closed loop's d = 5 is 10; the cap is far above anything the
#: relaxation builds and exists so a mis-shaped input fails loudly.
MAX_DEGREE = 64
#: hard cap on the size of any Sturm-chain coefficient.  Exact remainder
#: sequences can blow up superexponentially in pathological cases; at the
#: degrees here they do not, and this turns "hangs forever" into an error.
MAX_BITS = 1 << 16

ZERO = Fraction(0)


class ExactCheckError(RuntimeError):
    """The exact check cannot be run on this input (shape, degree, or cost)."""


# ------------------------------------------------------------------ polynomials
# A polynomial is a list of `Fraction` in the POWER basis, ascending powers.

def _deg(p):
    for i in range(len(p) - 1, -1, -1):
        if p[i] != 0:
            return i
    return -1                                        # the zero polynomial


def _trim(p):
    d = _deg(p)
    return list(p[:d + 1])


def _add(a, b):
    n = max(len(a), len(b))
    out = [ZERO] * n
    for i, v in enumerate(a):
        out[i] += v
    for i, v in enumerate(b):
        out[i] += v
    return out


def _scale(a, k):
    k = Fraction(k)
    return [k * v for v in a]


def _mul(a, b):
    da, db = _deg(a), _deg(b)
    if da < 0 or db < 0:
        return [ZERO]
    out = [ZERO] * (da + db + 1)
    for i in range(da + 1):
        ai = a[i]
        if ai == 0:
            continue
        for j in range(db + 1):
            if b[j] != 0:
                out[i + j] += ai * b[j]
    return out


def _eval(p, x):
    x = Fraction(x)
    acc = ZERO
    for c in reversed(_trim(p) or [ZERO]):
        acc = acc * x + c
    return acc


def _deriv(p):
    if _deg(p) <= 0:
        return [ZERO]
    return [p[i] * i for i in range(1, _deg(p) + 1)]


def bern_to_power(coeffs, d):
    """Bernstein coefficients on the degree-`d` basis -> power basis, exactly.

    `B^d_i(s) = C(d, i) s^i (1 - s)^(d - i) = sum_t C(d,i) C(d-i,t) (-1)^t
    s^(i+t)`, all binomials integer, so the change of basis is exact.
    """
    if len(coeffs) != d + 1:
        raise ExactCheckError("degree %d needs %d Bernstein coefficients, got %d"
                              % (d, d + 1, len(coeffs)))
    out = [ZERO] * (d + 1)
    for i in range(d + 1):
        ci = coeffs[i]
        if ci == 0:
            continue
        cb = comb(d, i)
        for t in range(d - i + 1):
            out[i + t] += ci * (cb * comb(d - i, t) * (-1) ** t)
    return out


# ---------------------------------------------------------------------- Sturm
def _bits(p):
    m = 0
    for c in p:
        if c:
            m = max(m, c.numerator.bit_length(), c.denominator.bit_length())
    return m


def _primitive(p):
    """`p` rescaled by a POSITIVE rational to integer coefficients, content 1.

    Sign changes are what Sturm counts, so any positive rescaling is free;
    keeping the chain integral and content-free is what keeps it cheap.
    """
    p = _trim(p)
    if _deg(p) < 0:
        return [ZERO]
    den = 1
    for c in p:
        den = _lcm(den, c.denominator)
    ints = [int(c * den) for c in p]
    g = 0
    for v in ints:
        g = _gcd(g, abs(v))
    if g > 1:
        ints = [v // g for v in ints]
    return [Fraction(v) for v in ints]


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _lcm(a, b):
    return a * b // _gcd(a, b)


def _rem(a, b):
    """Remainder of `a / b` over the rationals (b nonzero)."""
    a = _trim(a)
    db = _deg(b)
    if db < 0:
        raise ExactCheckError("division by the zero polynomial")
    r = list(a)
    lb = b[db]
    while True:
        dr = _deg(r)
        if dr < db:
            return _trim(r)
        f = r[dr] / lb
        for i in range(db + 1):
            r[dr - db + i] -= f * b[i]
        r[dr] = ZERO                                  # exact cancellation


def sturm_chain(p):
    """`p0 = p`, `p1 = p'`, `p_{i+1} = -rem(p_{i-1}, p_i)`, kept primitive."""
    c0 = _primitive(p)
    if _deg(c0) < 0:
        raise ExactCheckError("Sturm chain of the zero polynomial")
    chain = [c0]
    d1 = _deriv(c0)
    if _deg(d1) < 0:
        return chain
    chain.append(_primitive(d1))
    while True:
        r = _rem(chain[-2], chain[-1])
        if _deg(r) < 0:
            break
        nxt = _primitive([-v for v in r])
        if _bits(nxt) > MAX_BITS:
            raise ExactCheckError(
                "Sturm chain coefficients exceeded %d bits at chain index %d "
                "(degree %d); the exact check refuses to continue"
                % (MAX_BITS, len(chain), _deg(p)))
        chain.append(nxt)
    return chain


def _sign_changes(chain, x):
    prev, v = 0, 0
    for q in chain:
        s = _eval(q, x)
        s = (s > 0) - (s < 0)
        if s == 0:
            continue
        if prev != 0 and s != prev:
            v += 1
        prev = s
    return v


def _deflate_at_one(p):
    """Strip the factor `(s - 1)^m`; returns `(m, p / (s-1)^m)`."""
    m = 0
    q = _trim(p)
    while _deg(q) > 0 and _eval(q, 1) == 0:
        # synthetic division by (s - 1), ascending-order coefficients
        n = _deg(q)
        out = [ZERO] * n
        acc = ZERO
        for i in range(n, 0, -1):
            acc = q[i] + acc
            out[i - 1] = acc
        q, m = _trim(out), m + 1
    return m, q


def _deflate_at_zero(p):
    """Strip the factor `s^m`; returns `(m, p / s^m)`."""
    q = _trim(p)
    if _deg(q) < 0:
        return 0, q
    m = 0
    while m <= _deg(q) and q[m] == 0:
        m += 1
    return m, q[m:]


def count_roots_in_half_open_unit(p):
    """Number of DISTINCT real roots of `p` in `(0, 1]`, exactly.

    The polynomial is first deflated at both endpoints so Sturm's theorem is
    applied where its hypotheses hold (`p(0) != 0`, `p(1) != 0`), and the root
    at `1`, if any, is added back.  Multiple roots need no special handling:
    the chain's last element is `gcd(p, p')` and dividing through by it, which
    is what the sign-change count is insensitive to away from its own roots,
    leaves the chain of the squarefree part.
    """
    p = _trim(p)
    if _deg(p) < 0:
        raise ExactCheckError("every parameter is a root: the polynomial is 0")
    m1, q = _deflate_at_one(p)
    _m0, q = _deflate_at_zero(q)
    n_open = 0
    if _deg(q) >= 1:
        chain = sturm_chain(q)
        n_open = _sign_changes(chain, 0) - _sign_changes(chain, 1)
    return n_open + (1 if m1 else 0)


def positive_on_unit_interval(p, tau=0.0):
    """`True` iff `p(s) + tau > 0` for every `s` in `[0, 1]` -- exactly.

    `p` is a power-basis list of `Fraction` (or anything `Fraction` accepts).
    """
    q = _trim(_add([Fraction(v) for v in p], [Fraction(tau)]))
    if _deg(q) < 0:
        return False                                  # q == 0 everywhere
    if _deg(q) > MAX_DEGREE:
        raise ExactCheckError("degree %d exceeds the cap %d"
                              % (_deg(q), MAX_DEGREE))
    if _eval(q, 0) <= 0:
        return False
    return count_roots_in_half_open_unit(q) == 0


# ------------------------------------------------------------- the constraints
def _F(x):
    """A float (or int) as its EXACT binary value."""
    return Fraction(float(x))


def _Fmat(A):
    A = np.asarray(A, float)
    return [[_F(v) for v in row] for row in A]


def _Fvec(v):
    return [_F(x) for x in np.asarray(v, float).ravel()]


def _sq_norm_poly(rows, d):
    """`sum_a (rows[a]^T b_d(s))^2` in the power basis, rows in Bernstein."""
    acc = [ZERO]
    for r in rows:
        pa = bern_to_power(list(r), d)
        acc = _add(acc, _mul(pa, pa))
    return acc


def obstacle_polys(mr, Gammas):
    """`||gamma_i(s) - c||^2 - r^2` for every (robot, obstacle), power basis."""
    d = int(mr.d)
    out = []
    for i, obs in enumerate(mr.obs):
        G = _Fmat(Gammas[i])
        for (c, r) in obs:
            cf = _Fvec(c)
            rows = [[G[a][k] - cf[a] for k in range(d + 1)]
                    for a in range(len(cf))]
            p = _sq_norm_poly(rows, d)
            out.append(("obstacle", _add(p, [-_F(r) * _F(r)])))
    return out


def certify_obstacle_curves(Gammas, obstacles, d, tol=1e-9):
    """Certify p(s) + tol > 0 for every static obstacle and curve segment.

    All finite input floats are interpreted as exact binary rationals. This
    checks obstacle clearance only, not boundary data, dynamics or cuts.
    Resource/degree failures raise ExactCheckError; callers must report an
    unknown result, never accept it. The tolerance is in squared scene units.
    """
    from types import SimpleNamespace
    if not isinstance(d, (int, np.integer)) or d < 0 or 2 * d > MAX_DEGREE:
        raise ExactCheckError("unsupported polynomial degree")
    if not np.isfinite(tol) or tol < 0:
        raise ExactCheckError("tolerance must be finite and nonnegative")
    curves = [np.asarray(G, float) for G in Gammas]
    if not curves:
        raise ExactCheckError("no curve segments")
    n = curves[0].shape[0] if curves[0].ndim == 2 else 0
    if not n or any(G.shape != (n, d + 1) or not np.all(np.isfinite(G))
                    for G in curves):
        raise ExactCheckError("invalid or nonfinite curve coefficients")
    obs = [(np.asarray(c, float), float(r)) for c, r in obstacles]
    if any(c.shape != (n,) or not np.all(np.isfinite(c))
           or not np.isfinite(r) or r < 0 for c, r in obs):
        raise ExactCheckError("invalid static ball")
    model = SimpleNamespace(d=d, obs=[obs for _ in curves])
    polys = obstacle_polys(model, curves)
    outcomes = [bool(positive_on_unit_interval_fast(p, _F(tol)))
                for _, p in polys]
    return dict(ok=all(outcomes), n_polys=len(outcomes),
                failed_indices=[i for i, ok in enumerate(outcomes) if not ok],
                tolerance=float(tol), arithmetic="exact_binary_rational",
                predicate="p(s) + tolerance > 0 on [0,1]")


def pair_polys(mr, Gammas):
    """`||gamma_i(s) - gamma_j(s)||^2 - (2 rr)^2` for every pair."""
    d = int(mr.d)
    rr = _F(mr.rr)
    out = []
    for (i, j) in mr.pairs:
        Gi, Gj = _Fmat(Gammas[i]), _Fmat(Gammas[j])
        rows = [[Gi[a][k] - Gj[a][k] for k in range(d + 1)]
                for a in range(len(Gi))]
        p = _sq_norm_poly(rows, d)
        out.append(("pair", _add(p, [-(2 * rr) * (2 * rr)])))
    return out


def _deriv_bernstein(mr, Gamma, order):
    """Bernstein coefficients, on the degree-`d - order` basis, of every
    component of `gamma^(order)(s)`.

    `MultiRobotLinear.deriv_profile` computes `scale * B_{d-j} Dj^T Gamma^T`
    with `scale = d! / (d - j)!` and `Dj = bernstein.Dk(d, j)`, i.e.
    `gamma^(j)(s) = scale * Gamma Dj b_{d-j}(s)`.  The same `scale` and the
    same `Dj` are used here, converted to exact rationals.
    """
    from bernstein import Dk
    d = int(mr.d)
    scale = _F(factorial(d) / factorial(d - order))
    Dj = _Fmat(Dk(d, order))                         # (d+1, d-order+1)
    G = _Fmat(Gamma)
    m = d - order
    rows = []
    for a in range(len(G)):
        rows.append([scale * sum((G[a][k] * Dj[k][t] for k in range(d + 1)),
                                 ZERO) for t in range(m + 1)])
    return rows, m


def linear_polys(mr, Gammas):
    """`sum_i a_i^T Gamma_i W b_d(s) - beta(s)`, exactly as `linear_margins`
    assembles it: `cvec = -bvec(b) + sum_i (Gamma_i W)^T a_i`."""
    d = int(mr.d)
    out = []
    for entry in getattr(mr, "linear_polys", []):
        coeffs, b = entry[0], entry[1]
        W = entry[2] if len(entry) > 2 else None
        cvec = [-v for v in _Fvec(mr._bvec(b))]
        for rb, a in coeffs.items():
            G = _Fmat(Gammas[int(rb)])
            av = _Fvec(a)
            if W is None:
                q = [[G[x][k] for k in range(d + 1)] for x in range(len(G))]
            else:
                Wf = _Fmat(W)
                q = [[sum((G[x][k] * Wf[k][t] for k in range(d + 1)), ZERO)
                      for t in range(d + 1)] for x in range(len(G))]
            for t in range(d + 1):
                cvec[t] += sum((av[x] * q[x][t] for x in range(len(q))), ZERO)
        out.append(("linear", bern_to_power(cvec, d)))
    return out


def axis_deriv_polys(mr, Gammas):
    """`a - u^T gamma^(j)(s) >= 0` AND `a + u^T gamma^(j)(s) >= 0`."""
    out = []
    for rb, order, u, a in getattr(mr, "axis_deriv_bounds", []):
        rows, m = _deriv_bernstein(mr, Gammas[int(rb)], int(order))
        uv = _Fvec(u)
        cvec = [sum((uv[x] * rows[x][t] for x in range(len(rows))), ZERO)
                for t in range(m + 1)]
        pa = bern_to_power(cvec, m)
        av = _F(a)
        out.append(("axis", _add([av], [-v for v in pa])))
        out.append(("axis", _add([av], pa)))
    return out


def deriv_polys(mr, Gammas):
    """`a^2 - ||gamma^(j)(s)||^2 >= 0`, degree `2 (d - j)`."""
    out = []
    for rb, order, a in getattr(mr, "deriv_bounds", []):
        rows, m = _deriv_bernstein(mr, Gammas[int(rb)], int(order))
        p = _sq_norm_poly(rows, m)
        av = _F(a)
        out.append(("deriv", _add([av * av], [-v for v in p])))
    return out


def point_cut_values(mr, Gammas):
    """`side * (u^T gamma(s*) - b)` at each cut's single parameter, exactly.

    A point cut is not a continuum question -- the grid test reads it at the
    one parameter it constrains and so does this -- so it is EVALUATED, and
    at the tolerance `_clear_ok` uses (`>= -tol`), not the strict test the
    polynomial families get.
    """
    d = int(mr.d)
    out = []
    for rb, s, u, b, side in getattr(mr, "point_cuts", []):
        G = _Fmat(Gammas[int(rb)])
        sf = _F(s)
        basis = [Fraction(comb(d, i)) * sf ** i * (1 - sf) ** (d - i)
                 for i in range(d + 1)]
        uv = _Fvec(u)
        val = sum((uv[x] * sum((G[x][k] * basis[k] for k in range(d + 1)), ZERO)
                   for x in range(len(G))), ZERO) - _F(b)
        out.append(("point", _F(side) * val))
    return out


# ------------------------------------------------------------------ public API
def certify(mr, Gammas, tol=1e-6, grid=True):
    """EXACTLY decide the constraint set `mr` carries, on the curves `Gammas`.

    `mr` is a `MultiRobot` / `MultiRobotLinear`; `Gammas` the list of
    `(n, d+1)` float control-point arrays, exactly as `mr.solve()["Gamma"]`
    returns them (the solver's own frame and units -- for the closed loop, the
    chord frame in decimetres, where `tol = 1e-6` is the loop's tolerance).

    Returns `dict(ok, obstacle, pair, linear, deriv, n_polys, worst_grid)`:
    each family's exact verdict, the number of univariate decisions taken, and
    the SAMPLED minimum the deployed test reads, for comparison.

    `worst_grid` is a diagnostic and nothing in the verdict depends on it, but
    computing it costs a 2001-point sweep -- measured at 10 ms per call, which
    was two thirds of `certify` on the Go2 swing.  `grid=False` skips it and
    reports `worst_grid=None`; the exact verdict is unchanged.
    """
    if getattr(mr, "moving", None):
        raise ExactCheckError(
            "moving obstacles are not covered: `MultiRobot.clearances` does "
            "not read them either, so a verdict here would compare a "
            "different constraint set")
    d = int(mr.d)
    if 2 * d > MAX_DEGREE:
        raise ExactCheckError("2d = %d exceeds the degree cap %d"
                              % (2 * d, MAX_DEGREE))
    Gammas = [np.asarray(G, float) for G in Gammas]
    for i, G in enumerate(Gammas):
        if G.shape != (mr.n, d + 1):
            raise ExactCheckError(
                "Gamma[%d] must have shape (n, d+1) = %s, got %s"
                % (i, (mr.n, d + 1), G.shape))
    tau = _F(tol)

    fam = {}
    fam["obstacle"] = obstacle_polys(mr, Gammas)
    fam["pair"] = pair_polys(mr, Gammas)
    fam["linear"] = linear_polys(mr, Gammas) + axis_deriv_polys(mr, Gammas)
    fam["deriv"] = deriv_polys(mr, Gammas)

    verdict, n_polys = {}, 0
    for k, polys in fam.items():
        ok = True
        for _kind, p in polys:
            n_polys += 1
            if not positive_on_unit_interval_fast(p, tau):
                ok = False
        verdict[k] = ok
    # the pointwise family rides with the linear one, as `_clear_ok` reads it
    for _kind, v in point_cut_values(mr, Gammas):
        n_polys += 1
        if v + tau < 0:
            verdict["linear"] = False

    if grid:
        gridvals = mr.clearances(Gammas)
        vals = [v for v in gridvals.values() if np.isfinite(v)]
        if hasattr(mr, "linear_margins"):
            vals += [v for _k, v in mr.linear_margins(Gammas) if np.isfinite(v)]
        worst_grid = float(min(vals)) if vals else float("inf")
    else:
        worst_grid = None

    return dict(ok=bool(all(verdict.values())), n_polys=int(n_polys),
                worst_grid=worst_grid, **{k: bool(v) for k, v in verdict.items()})


# ----------------------------------------------------------------------
# Bernstein subdivision, the fast path.
#
# The obstacle polynomial arrives in the Bernstein basis and only ever has to
# be DECIDED, not solved.  Bernstein coefficients bracket the polynomial:
# `min(c) <= p(s) <= max(c)` on `[0, 1]`, so all-positive coefficients settle
# it in one pass, and `p(0) = c[0]`, `p(1) = c[d]` settle the negative case at
# an endpoint.  Otherwise de Casteljau splits `[0,1]` in half -- exactly, the
# only division is by 2 -- and the brackets tighten quadratically.
#
# The common case in practice is a curve with real clearance, where the first
# test already answers.  A curve TANGENT to an obstacle is the case
# subdivision cannot finish: the bracket has to separate a gap that is zero,
# so the recursion is capped and Sturm decides those.  Sturm stays the
# authority; this only removes it from the easy majority.
#
# Everything is integer: `to_int_bernstein` clears denominators once, and de
# Casteljau on integers is done by scaling each level by 2 instead of halving,
# which keeps the sign structure and never introduces a fraction.


def to_int_bernstein(coeffs):
    """Bernstein coefficients as integers, content 1, sign preserved."""
    cs = [Fraction(c) for c in coeffs]
    den = 1
    for c in cs:
        den = _lcm(den, c.denominator)
    ints = [int(c * den) for c in cs]
    g = 0
    for v in ints:
        g = _gcd(g, abs(v))
    return [v // g for v in ints] if g > 1 else ints


def _decasteljau_halves_int(c):
    """Both halves at `s = 1/2`, each scaled by `2**d` -- integral, sign-exact.

    The left half's coefficients are the de Casteljau diagonal and the right
    half's the last column; scaling every level by 2 rather than halving turns
    the usual `(a+b)/2` into `a+b` and multiplies the whole half by `2**d`,
    which is positive and so changes no sign.
    """
    d = len(c) - 1
    rows = [list(c)]
    for k in range(1, d + 1):
        prev = rows[-1]
        # level k of the doubled recursion; earlier levels are scaled up to
        # match so that one common factor 2**d comes out of the whole half
        rows.append([prev[i] + prev[i + 1] for i in range(len(prev) - 1)])
    left = [rows[k][0] * (1 << (d - k)) for k in range(d + 1)]
    right = [rows[d - k][k] * (1 << k) for k in range(d + 1)]
    return left, right


def positive_on_unit_interval_bernstein_int(coeffs, max_depth=24):
    """`(decided, value)` for `p > 0` on `[0,1]`, from Bernstein coefficients.

    A leaf is accepted only when its coefficients are STRICTLY positive: they
    bound `p` on that subinterval, so strict positivity on every leaf is strict
    positivity on `[0,1]`.  `p(0) = c[0]` and `p(1) = c[d]` are exact, so a
    non-positive endpoint answers immediately.  A curve that TOUCHES zero can
    never make every leaf strictly positive, so it runs to `max_depth` and
    comes back undecided -- which is the honest answer, and Sturm settles it.
    """
    c0 = to_int_bernstein(coeffs)
    if c0[0] <= 0 or c0[-1] <= 0:
        return True, False
    stack = [(c0, 0)]
    while stack:
        c, depth = stack.pop()
        if min(c) > 0:
            continue
        if c[0] <= 0 or c[-1] <= 0:
            return True, False
        if depth >= max_depth:
            return False, None
        lo, hi = _decasteljau_halves_int(c)
        stack.append((lo, depth + 1))
        stack.append((hi, depth + 1))
    return True, True


def power_to_bern(p, d):
    """Power-basis coefficients as degree-`d` Bernstein coefficients, exactly.

    `b_k = sum_{i <= k} C(k,i)/C(d,i) a_i`, the inverse of `bern_to_power`.
    """
    a = [Fraction(v) for v in p]
    if len(a) > d + 1:
        raise ExactCheckError("a degree-%d polynomial does not fit a degree-%d "
                              "Bernstein basis" % (_deg(p), d))
    a = a + [ZERO] * (d + 1 - len(a))
    out = []
    for k in range(d + 1):
        acc = ZERO
        for i in range(k + 1):
            if a[i]:
                acc += Fraction(comb(k, i), comb(d, i)) * a[i]
        out.append(acc)
    return out


def positive_on_unit_interval_fast(p, tau=0.0, max_depth=24):
    """`p(s) + tau > 0` on `[0, 1]`, exactly -- subdivision first, Sturm after.

    The same decision `positive_on_unit_interval` makes, which stays the
    authority: this only tries the cheap bracket first.  Measured on the
    frozen_barrier and chord curves, 200x to 550x on a curve with real
    clearance and break-even on one tangent to its obstacle, which is the case
    the bracket cannot settle and hands over.
    """
    q = _trim(_add([Fraction(v) for v in p], [Fraction(tau)]))
    dq = _deg(q)
    if dq < 0:
        return False                                  # q == 0 everywhere
    decided, val = positive_on_unit_interval_bernstein_int(
        power_to_bern(q, dq), max_depth=max_depth)
    if decided:
        return bool(val)
    return positive_on_unit_interval(p, tau)


def positive_on_unit_interval_bernstein(coeffs, d, tau=0.0, max_depth=24):
    """`p(s) + tau > 0` on `[0, 1]`, exactly, from Bernstein coefficients.

    The subdivision bracket answers first; the Sturm chain, which stays the
    authority, is consulted only when the bracket cannot settle it.
    """
    if len(coeffs) != d + 1:
        raise ExactCheckError("expected %d Bernstein coefficients, got %d"
                              % (d + 1, len(coeffs)))
    decided, val = positive_on_unit_interval_bernstein_int(
        [Fraction(v) + Fraction(tau) for v in coeffs], max_depth=max_depth)
    if decided:
        return bool(val)
    return positive_on_unit_interval(
        bern_to_power([Fraction(v) for v in coeffs], d), tau)
