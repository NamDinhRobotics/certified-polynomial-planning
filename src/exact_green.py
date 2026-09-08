"""The Bernstein positivity certificate in EXACT rational arithmetic.

WHY THIS EXISTS.  The floating-point pipeline reports elevated Bernstein
coefficient minima like `-1.7e-18` and `-4.6e-23` and reads them as roundoff.
That reading is an assumption, not a result: the test as stated requires
NONNEGATIVE coefficients, and those numbers are negative, so in floating point
the certificate formally fails on exactly the configurations the paper claims.
Everything entering the test is rational, so the question is decidable.

    G_d entries        C(d,i)C(d,j) / ((2d+1) C(2d,i+j))          rational
    D_d                -1 / +1                                    integer
    b_d(0), b_d(1)     e_0, e_d                                   integer
    time scale         N^(2k-1)                                   integer
    elevation          C(d,a)C(D-d,A-a) / C(D,A)                  rational

so the whole chain is exact over Q and the sign of every elevated coefficient
is decidable.

THE ONE STEP THAT LOOKS LIKE AN OBSTACLE, AND IS NOT.  `MultiSegment` gets its
null-space basis from an SVD, which is irrational.  It does not matter: the
tested block is basis-independent.  For any invertible `T`, replacing
`N -> N T` sends `K = tau N^T G N` to `T^T K T`, hence

    N' K'^{-1} N'^T = N T (T^T K T)^{-1} T^T N^T = N K^{-1} N^T ,

so the block, and therefore every coefficient of it, is unchanged.  We may
compute in ANY exact basis of the same null space and get the same answer as
the float pipeline is approximating -- which `a12` checks numerically.

WHAT EXACTNESS BUYS, PRECISELY.  A nonnegative coefficient array certifies
`Gr >= 0`, and a coefficient array with a POSITIVE minimum certifies
`Gr >= min > 0`, because the Bernstein coefficients bound the polynomial from
below by their minimum on the whole square.  The second is what the corrected
Theorem needs -- entrywise nonnegativity alone leaves the Gram matrix possibly
reducible, and Perron-Frobenius then gives no simple top eigenvalue.
"""
from fractions import Fraction as F
from math import comb, factorial


# ---------------------------------------------------------------- matrices
def zeros(p, q):
    return [[F(0)] * q for _ in range(p)]


def matmul(A, B):
    p, s, q = len(A), len(B), len(B[0])
    out = [[F(0)] * q for _ in range(p)]
    for i in range(p):
        Ai, Oi = A[i], out[i]
        for t in range(s):
            a = Ai[t]
            if a:
                Bt = B[t]
                for j in range(q):
                    b = Bt[j]
                    if b:
                        Oi[j] += a * b
    return out


def transpose(A):
    return [list(col) for col in zip(*A)]


# ------------------------------------------------------------- Bernstein
def q_gram(d):
    """`G_d` with entries `C(d,i)C(d,j) / ((2d+1) C(2d,i+j))`, exactly."""
    return [[F(comb(d, i) * comb(d, j), (2 * d + 1) * comb(2 * d, i + j))
             for j in range(d + 1)] for i in range(d + 1)]


def q_diff(d):
    """`D_d`: `-1` on the diagonal, `+1` on the subdiagonal, in `Z^{(d+1)xd}`."""
    D = zeros(d + 1, d)
    for j in range(d):
        D[j][j] = F(-1)
        D[j + 1][j] = F(1)
    return D


def q_Dk(d, k):
    M = [[F(1) if i == j else F(0) for j in range(d + 1)] for i in range(d + 1)]
    for j in range(k):
        M = matmul(M, q_diff(d - j))
    return M


def q_gram_deriv(d, k):
    if k == 0:
        return q_gram(d)
    scale = F(factorial(d), factorial(d - k)) ** 2
    Dm = q_Dk(d, k)
    G = matmul(matmul(Dm, q_gram(d - k)), transpose(Dm))
    return [[scale * x for x in row] for row in G]


def q_bd_endpoint(s, d):
    """`b_d(s)` at `s = 0` or `s = 1` only, where it is `e_0` or `e_d`."""
    assert s in (0, 1)
    v = [F(0)] * (d + 1)
    v[0 if s == 0 else d] = F(1)
    return v


def q_bd_derivs(s, d, order):
    """Rows `0..order` of `d^i/ds^i b_d(s)` at an endpoint, exactly."""
    out = [q_bd_endpoint(s, d)]
    for i in range(1, order + 1):
        scale = F(factorial(d), factorial(d - i))
        col = [[x] for x in q_bd_endpoint(s, d - i)]
        v = matmul(q_Dk(d, i), col)
        out.append([scale * r[0] for r in v])
    return out


# ----------------------------------------------------------- linear algebra
def rref(A):
    """Reduced row echelon form over Q; returns (R, pivot columns)."""
    R = [row[:] for row in A]
    rows, cols = len(R), len(R[0])
    piv, r = [], 0
    for c in range(cols):
        p = next((i for i in range(r, rows) if R[i][c] != 0), None)
        if p is None:
            continue
        R[r], R[p] = R[p], R[r]
        inv = F(1) / R[r][c]
        R[r] = [x * inv for x in R[r]]
        for i in range(rows):
            if i != r and R[i][c] != 0:
                f = R[i][c]
                R[i] = [a - f * b for a, b in zip(R[i], R[r])]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return R, piv


def nullspace(A):
    """Exact basis of `{x : A x = 0}`, as a list of columns."""
    R, piv = rref(A)
    cols = len(A[0])
    free = [c for c in range(cols) if c not in piv]
    basis = []
    for fc in free:
        v = [F(0)] * cols
        v[fc] = F(1)
        for r, pc in enumerate(piv):
            v[pc] = -R[r][fc]
        basis.append(v)
    return basis


def solve(A, B):
    """Exact `A^{-1} B` by Gauss-Jordan on `[A | B]`; `A` must be invertible."""
    n = len(A)
    M = [A[i][:] + B[i][:] for i in range(n)]
    for c in range(n):
        p = next((i for i in range(c, n) if M[i][c] != 0), None)
        if p is None:
            raise ValueError("singular matrix")
        M[c], M[p] = M[p], M[c]
        inv = F(1) / M[c][c]
        M[c] = [x * inv for x in M[c]]
        for i in range(n):
            if i != c and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[c])]
    return [row[n:] for row in M]


# ------------------------------------------------------- the tested block
def constraint_matrix(d, l, N, eta):
    """Boundary and continuity constraints, as rows of `A^T` (one per equation).

    Mirrors `MultiSegment.__init__` exactly: derivatives `0..l` pinned at the
    start of segment `0` and the end of segment `N-1`, and `C^eta` matching at
    each junction.  Only the LEFT-hand side matters here -- the null space does
    not see the boundary VALUES, which is the same reason the certificate is
    independent of the obstacles.
    """
    M = N * (d + 1)
    rows = []
    db0, db1 = q_bd_derivs(0, d, l), q_bd_derivs(1, d, l)
    for i in range(l + 1):
        a = [F(0)] * M
        a[0:d + 1] = db0[i]
        rows.append(a)
        a = [F(0)] * M
        a[(N - 1) * (d + 1):N * (d + 1)] = db1[i]
        rows.append(a)
    dbe1, dbe0 = q_bd_derivs(1, d, eta), q_bd_derivs(0, d, eta)
    for j in range(N - 1):
        for m in range(eta + 1):
            a = [F(0)] * M
            a[j * (d + 1):(j + 1) * (d + 1)] = dbe1[m]
            a[(j + 1) * (d + 1):(j + 2) * (d + 1)] = [-x for x in dbe0[m]]
            rows.append(a)
    return rows


def coefficient_blocks(d, k, l, N, eta):
    """The blocks `N_{perp,i} K^{-1} N_{perp,j}^T`, exactly.

    Returns `(blocks, r)` with `blocks[(i,j)]` a `(d+1)x(d+1)` rational matrix.
    """
    Np = nullspace(constraint_matrix(d, l, N, eta))      # columns, length M
    r = len(Np)
    if r < 1:
        raise ValueError("no free parameters (N=%d d=%d l=%d eta=%d)"
                         % (N, d, l, eta))
    Nperp = [[Np[c][i] for c in range(r)] for i in range(N * (d + 1))]

    Gk = q_gram_deriv(d, k)
    tau = F(N) ** (2 * k - 1)
    # K = tau * sum_i Nperp_i^T G Nperp_i
    K = zeros(r, r)
    for i in range(N):
        blk = Nperp[i * (d + 1):(i + 1) * (d + 1)]
        piece = matmul(transpose(blk), matmul(Gk, blk))
        for a in range(r):
            for b in range(r):
                K[a][b] += tau * piece[a][b]

    Y = solve(K, transpose(Nperp))                       # r x M
    blocks = {}
    for i in range(N):
        Bi = Nperp[i * (d + 1):(i + 1) * (d + 1)]
        for j in range(N):
            cols = range(j * (d + 1), (j + 1) * (d + 1))
            Yj = [[Y[a][c] for c in cols] for a in range(r)]
            blocks[(i, j)] = matmul(Bi, Yj)
    return blocks, r


# ------------------------------------------------------------- elevation
def elevated_min(B, d, D):
    """Exact minimum of `E B E^T`, the degree-`D` elevated coefficient array.

    `E[A,a] = C(d,a) C(D-d,A-a) / C(D,A)`.  The denominators `C(D,A)` are
    positive and depend on the row (resp. column) only, so the work is done
    with the INTEGER numerators `Etil[A,a] = C(d,a) C(D-d,A-a)` and the
    division is applied once, at the end, to the winning entry.  Comparing
    `x/p` against `y/q` with `p,q>0` is `x*q` against `y*p`, so the search
    itself never leaves the integers.
    """
    Et = [[comb(d, a) * comb(D - d, A - a)
           if 0 <= A - a <= D - d else 0 for a in range(d + 1)]
          for A in range(D + 1)]
    den = [comb(D, A) for A in range(D + 1)]

    # clear denominators of B once: the common factor is positive
    L = 1
    for row in B:
        for x in row:
            L = L * x.denominator // __import__("math").gcd(L, x.denominator)
    Bi = [[int(x * L) for x in row] for row in B]

    EB = [[sum(Et[A][a] * Bi[a][b] for a in range(d + 1))
           for b in range(d + 1)] for A in range(D + 1)]

    best_num, best_den = None, None
    for A in range(D + 1):
        EBA, dA = EB[A], den[A]
        for C in range(D + 1):
            EtC = Et[C]
            v = 0
            for b in range(d + 1):
                e = EtC[b]
                if e:
                    v += EBA[b] * e
            p = dA * den[C]                     # positive
            if best_num is None or v * best_den < best_num * p:
                best_num, best_den = v, p
    return F(best_num, best_den * L)


def endpoint_structure(d, k, l, N, eta):
    """The finite extra check that removes the "interior parameters" caveat.

    On the OPEN square every Bernstein basis function is positive, so a
    nonnegative block that is not identically zero already gives `Gr > 0`.  The
    boundary is where that argument runs out: at `s = 0` only `beta_0` survives
    and at `s = 1` only `beta_d`, so `Gr` there is read off a single ROW (or,
    for two boundary parameters, a single ENTRY) of the block.

    A boundary parameter is only worth checking if it is LIVE, i.e.
    `u = N_perp,i^T b_d(s) != 0`.  Dead ones are the control points pinned by
    the boundary conditions: such a contact contributes `w_a u_a u_a^T = 0` to
    `M_lambda` and drops out of the Gram matrix instead of disconnecting it.
    The live ones are exactly the interior junctions, where `s = 1` on segment
    `i` and `s = 0` on segment `i+1` name the same point of the curve.

    Returns the two conditions, checked exactly:
      `rows_ok`    every live boundary row of every block is not identically zero
                   (covers boundary parameter against interior parameter)
      `corners_ok` every entry pairing two live boundary parameters is positive
                   (covers boundary against boundary)
    """
    blocks, r = coefficient_blocks(d, k, l, N, eta)
    Np = nullspace(constraint_matrix(d, l, N, eta))
    M = N * (d + 1)

    def live(i, row):
        idx = i * (d + 1) + row
        return any(Np[c][idx] != 0 for c in range(r))

    ends = [(i, e) for i in range(N) for e in (0, d) if live(i, e)]
    bad_rows, bad_corners = [], []
    for (i, e) in ends:
        for j in range(N):
            if all(x == 0 for x in blocks[(i, j)][e]):
                bad_rows.append([i, e, j])
    for (i, e) in ends:
        for (j, e2) in ends:
            if blocks[(i, j)][e][e2] <= 0:
                bad_corners.append([i, e, j, e2])
    return dict(n_live_endpoints=len(ends),
                live_endpoints=[[i, e] for i, e in ends],
                n_bad_rows=len(bad_rows), bad_rows=bad_rows[:8],
                n_bad_corners=len(bad_corners), bad_corners=bad_corners[:8],
                rows_ok=not bad_rows, corners_ok=not bad_corners,
                closed_square_ok=bool(not bad_rows and not bad_corners))


def certify_exact(d, k, l, N, eta, D=96):
    """Exact verdict for one configuration.

    `strict` is the property the corrected Theorem needs: a POSITIVE minimum
    bounds the Green's function away from zero on the whole square, hence gives
    an entrywise positive Gram matrix at any contact set, hence irreducibility,
    hence a simple top eigenvalue.  `nonneg` alone leaves reducibility open.
    """
    blocks, r = coefficient_blocks(d, k, l, N, eta)
    scale = max([F(1)] + [abs(x) for B in blocks.values() for row in B
                          for x in row])
    raw = min(x for B in blocks.values() for row in B for x in row)
    ele = min(elevated_min(B, d, D) for B in blocks.values())
    return dict(d=d, k=k, l=l, N=N, eta=eta, r=r, D=D,
                f_single=d + 1 - 2 * (l + 1),
                scale=scale, raw_min=raw, elev_min=ele,
                raw_min_rel=raw / scale, elev_min_rel=ele / scale,
                nonneg=bool(ele >= 0), strict=bool(ele > 0))


# ------------------------------------------- the invertibility condition
def energy_nullity(d, k, l, N=1, eta=None):
    """Exact nullity of `K` (or `K_multi`), i.e. how far it is from `> 0`.

    `x^T K x` is the `k`-th derivative energy of the curve whose free control
    points are `x` and whose pinned ones are zero, so `Kx = 0` says every
    segment is a polynomial of degree `<= k-1`.  Such a curve is admissible
    only if it also meets the `C^eta` junctions and the order-`l` boundary
    conditions, and the pinned control points being zero means a zero of
    multiplicity `>= l+1` at each end.

    Single segment: a nonzero polynomial of degree `<= k-1` has at most `k-1`
    zeros with multiplicity, so one exists iff `2(l+1) <= k-1`.  Hence

        K > 0   <=>   k <= 2(l+1) = 2l+2 ,

    and the nullity is `max(0, k - 2(l+1))`.  Multisegment: the piecewise space
    has dimension `Nk - (N-1) min(eta+1, k)` and the boundary removes
    `2 min(l+1, k)`, giving the count returned as `predicted` below.

    This matters because `K^{-1}` is used throughout -- in the pencil, in `Gr`,
    and in the tested block `B^{ij}`, which does not even exist when
    `K_multi` is singular.
    """
    eta = (d - 1) if eta is None else eta
    if N == 1:
        Fi = list(range(l + 1, d - l))
        if not Fi:
            return dict(r=0, rank=0, nullity=0, predicted=0, definite=True)
        G = q_gram_deriv(d, k)
        K = [[G[i][j] for j in Fi] for i in Fi]
        r = len(Fi)
        predicted = max(0, k - 2 * (l + 1))
    else:
        Np = nullspace(constraint_matrix(d, l, N, eta))
        r = len(Np)
        if r < 1:
            return dict(r=0, rank=0, nullity=0, predicted=0, definite=True)
        Nperp = [[Np[c][i] for c in range(r)] for i in range(N * (d + 1))]
        Gk = q_gram_deriv(d, k)
        tau = F(N) ** (2 * k - 1)
        K = zeros(r, r)
        for i in range(N):
            blk = Nperp[i * (d + 1):(i + 1) * (d + 1)]
            piece = matmul(transpose(blk), matmul(Gk, blk))
            for a in range(r):
                for b in range(r):
                    K[a][b] += tau * piece[a][b]
        predicted = max(0, N * k - (N - 1) * min(eta + 1, k)
                        - 2 * min(l + 1, k))
    _, piv = rref([row[:] for row in K])
    rank = len(piv)
    return dict(r=r, rank=rank, nullity=r - rank, predicted=predicted,
                definite=bool(rank == r))


# ------------------------------------------ polynomials over Q, for a_0
def poly_mul(a, b):
    out = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def poly_add(a, b):
    out = [F(0)] * max(len(a), len(b))
    for i, x in enumerate(a):
        out[i] += x
    for i, x in enumerate(b):
        out[i] += x
    return out


def poly_trim(a):
    while len(a) > 1 and a[-1] == 0:
        a = a[:-1]
    return a


def poly_divmod(a, b):
    a, b = poly_trim(a[:]), poly_trim(b[:])
    q = [F(0)] * max(1, len(a) - len(b) + 1)
    while len(a) >= len(b) and any(a):
        if a[-1] == 0:
            a = poly_trim(a)
            continue
        c = a[-1] / b[-1]
        sh = len(a) - len(b)
        q[sh] = c
        for i, y in enumerate(b):
            a[i + sh] -= c * y
        a = poly_trim(a)
        if len(a) < len(b):
            break
    return poly_trim(q), poly_trim(a)


def bernstein_poly(i, d):
    """`B_{i,d}(s) = C(d,i) s^i (1-s)^{d-i}` as exact coefficients."""
    p = [F(comb(d, i))]
    for _ in range(i):
        p = poly_mul(p, [F(0), F(1)])
    for _ in range(d - i):
        p = poly_mul(p, [F(1), F(-1)])
    return p


def reflect(p):
    """`p(1-s)`."""
    out = [F(0)]
    for i, c in enumerate(p):
        term = [c]
        for _ in range(i):
            term = poly_mul(term, [F(1), F(-1)])
        out = poly_add(out, term)
    return poly_trim(out)


def gr12_polynomial(d, k, l):
    """`Gr_12(sigma, 1-sigma)` as an exact polynomial in `sigma`, for `f = 2`.

    At `f = 2` the certificate can fail only where this vanishes
    (Remark on the `m = 2` case), so its roots are the nodal points -- and
    since the coefficients are rational the roots are algebraic numbers we can
    name rather than bisect.
    """
    Fi = list(range(l + 1, d - l))
    if len(Fi) != 2:
        raise ValueError("gr12_polynomial is the f = 2 case (need d = 3 + 2l)")
    G = q_gram_deriv(d, k)
    K = [[G[i][j] for j in Fi] for i in Fi]
    Ki = solve(K, [[F(1), F(0)], [F(0), F(1)]])
    u = [bernstein_poly(i, d) for i in Fi]
    v = [reflect(p) for p in u]
    P = [F(0)]
    for a in range(2):
        for b in range(2):
            P = poly_add(P, [Ki[a][b] * x for x in poly_mul(u[a], v[b])])
    return poly_trim(P), K, Ki
