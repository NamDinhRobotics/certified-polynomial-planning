"""Semidefinite relaxation for collision-free motion planning (arXiv:2606.14063).

Single polynomial segment, point robot, spherical obstacles.  Implements (SDP)
of section V together with the facial reduction of section V-C: the 2(l+1)
boundary control points are eliminated in closed form, so only the
f = d+1-2(l+1) free control points remain.  Without the reduction the SDP is
not strictly feasible and solver failures masquerade as modelling bugs.

Two interchangeable backends:

  'cvxpy'    CVXPY + Clarabel (Gate 0 stack, default)
  'barrier'  the dependency-free primal log-barrier method in sdpsolver.py.
             Kept as a cross-check oracle, not because it is a good solver.

Both are exercised against each other in tests/test_relaxation.py.
"""
import numpy as np
from bernstein import bd, bd_derivs, gram_deriv, S_map, T_map, apply_map, to_monomial
from sdpsolver import Block, ConeProblem


# ----------------------------------------------------------------------
# Numerical, non-sampled feasibility
#
# p_j(s) = ||gamma(s)-c_j||^2 - r_j^2 is a polynomial of degree 2d, so its
# minimum over [0,1] is attained at an endpoint or a real root of p'.
# Computing those roots in floating point is not a rational certificate.
# Sampling instead is not merely inaccurate, it is biased in the dangerous
# direction: a curve may dip into an obstacle *between* samples, which makes
# the sampled ground-truth cost an under-estimate of c*_P and fires the
# Theorem-2 tripwire (c*_SDP <= c*_P) on a perfectly correct relaxation.
# ----------------------------------------------------------------------
def obstacle_poly(G, cj, rj, d):
    """Monomial coefficients (ascending) of p_j(s) = ||gamma(s)-c_j||^2 - r_j^2."""
    Bm = to_monomial(d)                      # (d+1, d+1)
    coef = G @ Bm                            # (n, d+1): gamma_a(s) coefficients
    p = np.zeros(2 * d + 1)
    for a in range(G.shape[0]):
        q = coef[a].copy()
        q[0] -= cj[a]
        p[:2 * d + 1] += np.convolve(q, q)
    p[0] -= rj ** 2
    return p


def poly_min_on_unit(p, tol=1e-12):
    """Numerical stationary-point minimum on [0,1], ascending coefficients.

    A failed or nonfinite computation returns (-inf, nan): existing clearance
    gates then reject it. This is a numerical estimate, not a proof of sign;
    use exact_check for independent rational certification. ``tol`` is kept
    for API compatibility, but never discards a small nonconstant polynomial.
    """
    p = np.asarray(p, dtype=float)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
        return -np.inf, np.nan
    cand = [0.0, 1.0]
    dp = np.polynomial.polynomial.polyder(p)
    if dp.size and np.any(dp != 0):
        try:
            rts = np.polynomial.polynomial.polyroots(dp)
        except Exception:
            return -np.inf, np.nan
        if not np.all(np.isfinite(rts)):
            return -np.inf, np.nan
        for z in np.atleast_1d(rts):
            if abs(np.imag(z)) < 1e-9:
                x = float(np.real(z))
                if -1e-12 <= x <= 1.0 + 1e-12:
                    cand.append(min(1.0, max(0.0, x)))
    vals = np.polynomial.polynomial.polyval(np.array(cand), p)
    if not np.all(np.isfinite(vals)):
        return -np.inf, np.nan
    k = int(np.argmin(vals))
    return float(vals[k]), float(cand[k])


def exact_clearance(G, obs, d):
    """Legacy name for numerical, non-sampled obstacle clearance.

    -inf means the computation failed and must never be accepted as clear.
    """
    if not np.all(np.isfinite(G)):
        return -np.inf, (None, None)
    best, arg = np.inf, (None, None)
    for j, (cj, rj) in enumerate(obs):
        v, s = poly_min_on_unit(obstacle_poly(G, cj, rj, d))
        if v < best:
            best, arg = v, (j, s)
    if best is np.inf:
        return np.inf, (None, None)
    return best, arg


def sym_basis(k):
    """Basis of S^k, upper-triangular parameterisation (E_ij + E_ji, i<j)."""
    Zi, idx = [], []
    for i in range(k):
        for j in range(i, k):
            E = np.zeros((k, k))
            E[i, j] = 1.0
            E[j, i] = 1.0
            Zi.append(E)
            idx.append((i, j))
    return np.array(Zi), idx


class Segment:
    """One polynomial segment of degree d in R^n with l-th order boundary data."""

    def __init__(self, n, d, k, l, obstacles, bc0, bc1, trace_bound=1e4):
        """obstacles: list of (center (n,), radius). bc0/bc1: (l+1, n) arrays."""
        self.n, self.d, self.k, self.l = n, d, k, l
        self.trace_bound = None if trace_bound is None else float(trace_bound)
        self.obs = [(np.asarray(c, float), float(r)) for c, r in obstacles]
        self.f = d + 1 - 2 * (l + 1)
        if self.f < 1:
            raise ValueError(f"no free control points (d={d}, l={l})")

        self.Kidx = list(range(l + 1)) + list(range(d - l, d + 1))
        self.Fidx = list(range(l + 1, d - l))
        assert len(self.Fidx) == self.f

        # bc0/bc1 are (l+1, n): one row per derivative order.  Passing a bare
        # (n,) vector for l=0 silently broadcasts to the wrong endpoint instead
        # of failing, so check the shape rather than trusting the caller.
        for nm, bc in (("bc0", bc0), ("bc1", bc1)):
            arr = np.asarray(bc, float)
            if arr.shape != (l + 1, n):
                raise ValueError(
                    f"{nm} must have shape (l+1, n) = {(l + 1, n)}, got "
                    f"{arr.shape}; for l=0 pass [[x, y]] not [x, y]")

        # boundary conditions  Gamma A = B
        db0 = bd_derivs(0.0, d, l)
        db1 = bd_derivs(1.0, d, l)
        A = np.zeros((d + 1, 2 * (l + 1)))
        B = np.zeros((n, 2 * (l + 1)))
        for i in range(l + 1):
            A[:, 2 * i] = db0[i]
            A[:, 2 * i + 1] = db1[i]
            B[:, 2 * i] = np.asarray(bc0[i], float)
            B[:, 2 * i + 1] = np.asarray(bc1[i], float)
        # section V-C: the rows of A at the free indices vanish identically,
        # which is exactly what makes the closed-form elimination possible.
        assert np.allclose(A[self.Fidx, :], 0.0), "free rows of A must vanish"
        self.Gfix = B @ np.linalg.inv(A[self.Kidx, :])      # (n, 2(l+1))
        self.A_bc, self.B_bc = A, B

        # permutation taking [fixed cols | free cols] back to natural order
        self.perm = list(self.Kidx) + list(self.Fidx)
        self.P = np.zeros((d + 1, d + 1))
        for a, p in enumerate(self.perm):
            self.P[p, a] = 1.0

        self.Gk = gram_deriv(d, k)
        self.Sd, self.Td = S_map(d), T_map(d)
        self.Sflat = self.Sd.reshape(2 * d + 1, -1)
        self.Tflat = self.Td.reshape(2 * d + 1, -1)
        self._layout()

    # ------------------------------------------------------------------
    # shared helpers (numeric)
    # ------------------------------------------------------------------
    def full_Gamma(self, Gfree):
        G = np.zeros((self.n, self.d + 1))
        G[:, self.Kidx] = self.Gfix
        G[:, self.Fidx] = Gfree
        return G

    def full_X(self, Gfree, Xfree):
        """X with the free-free block replaced by Xfree (everything else exact)."""
        d = self.d
        X = np.zeros((d + 1, d + 1))
        Gf = self.Gfix
        X[np.ix_(self.Kidx, self.Kidx)] = Gf.T @ Gf
        cross = Gf.T @ Gfree
        X[np.ix_(self.Kidx, self.Fidx)] = cross
        X[np.ix_(self.Fidx, self.Kidx)] = cross.T
        X[np.ix_(self.Fidx, self.Fidx)] = Xfree
        return X

    def M_obstacle(self, X, G, cj, rj):
        """M_j = X - (Gamma^T c_j) e^T - e (Gamma^T c_j)^T + (c^Tc - r^2) e e^T."""
        e = np.ones(self.d + 1)
        Gc = G.T @ cj
        return X - np.outer(Gc, e) - np.outer(e, Gc) + (cj @ cj - rj ** 2) * np.outer(e, e)

    # ------------------------------------------------------------------
    # CVXPY backend
    # ------------------------------------------------------------------
    def build_cvxpy(self):
        """Return (problem, vars) for the facially-reduced (SDP)."""
        import cvxpy as cp
        n, d, f = self.n, self.d, self.f

        Gfree = cp.Variable((n, f), name="Gfree")
        Xfree = cp.Variable((f, f), symmetric=True, name="Xfree")

        Gfix = self.Gfix
        Gamma = cp.hstack([cp.Constant(Gfix), Gfree]) @ self.P.T       # (n, d+1)
        cross = Gfix.T @ Gfree                                          # (2(l+1), f)
        Xperm = cp.bmat([[cp.Constant(Gfix.T @ Gfix), cross],
                         [cross.T,                    Xfree]])
        X = self.P @ Xperm @ self.P.T                                   # (d+1, d+1)

        cons = []
        # X >= Gamma^T Gamma, in facially-reduced form (Schur complement).
        cons.append(cp.bmat([[np.eye(n), Gfree],
                             [Gfree.T,   Xfree]]) >> 0)

        Qvars = []
        e_row = np.ones((1, d + 1))
        for cj, rj in self.obs:
            Q0 = cp.Variable((d + 1, d + 1), symmetric=True)
            Q1 = cp.Variable((d, d), symmetric=True)
            cons += [Q0 >> 0, Q1 >> 0]
            Qvars.append((Q0, Q1))

            Gc = Gamma.T @ cj.reshape(n, 1)                             # (d+1, 1)
            ge = Gc @ e_row
            M = X - ge - ge.T + (float(cj @ cj) - rj ** 2) * np.ones((d + 1, d + 1))

            # Markov-Lukacs in the Bernstein basis:  S_d(M - Q0) - T_d(Q1) = 0
            resid = [cp.sum(cp.multiply(self.Sd[kk], M - Q0))
                     - cp.sum(cp.multiply(self.Td[kk], Q1))
                     for kk in range(2 * d + 1)]
            cons.append(cp.hstack(resid) == 0)

        cost = cp.sum(cp.multiply(self.Gk, X))       # tr(G^(k) X), both symmetric
        prob = cp.Problem(cp.Minimize(cost), cons)
        self._cvx = dict(prob=prob, Gfree=Gfree, Xfree=Xfree, Qvars=Qvars,
                         Gamma=Gamma, X=X)
        return self._cvx

    # Tolerance ladder.  The tightest setting is tried first; instances whose
    # optimum sits exactly on the boundary of the PSD cone (lam_max(S) ~ 1e-10)
    # cannot be *certified* at 1e-11 and come back 'optimal_inaccurate' even
    # though the iterate is fine.  Loosening the certificate requirement one
    # notch resolves those without touching the answer.  The tolerance actually
    # used is reported in the result dict -- it is never silently softened.
    CLARABEL_LADDER = (1e-11, 1e-9, 1e-8)
    SCS_LADDER = (1e-10, 1e-8)

    def solve_cvxpy(self, solver="CLARABEL", tol=None, **kw):
        import cvxpy as cp
        import warnings
        c = getattr(self, "_cvx", None) or self.build_cvxpy()
        if tol is not None:
            ladder = (tol,)
        elif solver == "CLARABEL":
            ladder = self.CLARABEL_LADDER
        elif solver == "SCS":
            ladder = self.SCS_LADDER
        else:
            ladder = (None,)

        last = dict(converged=False, status="not attempted",
                    backend=f"cvxpy/{solver}")
        for tol_i in ladder:
            opts = dict(verbose=False)
            if solver == "CLARABEL" and tol_i is not None:
                opts.update(tol_gap_abs=tol_i, tol_gap_rel=tol_i,
                            tol_feas=tol_i, max_iter=500)
            elif solver == "SCS" and tol_i is not None:
                opts.update(eps=tol_i, max_iters=200000)
            opts.update(kw)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c["prob"].solve(solver=solver, **opts)
            except cp.error.SolverError as exc:
                last = dict(converged=False, status=f"SolverError: {exc}",
                            backend=f"cvxpy/{solver}", tol=tol_i)
                continue
            status = c["prob"].status
            if status not in ("optimal", "optimal_inaccurate") or c["Gfree"].value is None:
                last = dict(converged=False, status=status,
                            backend=f"cvxpy/{solver}", tol=tol_i)
                continue

            Gfree = np.asarray(c["Gfree"].value, float)
            Xfree = np.asarray(c["Xfree"].value, float)
            Xfree = 0.5 * (Xfree + Xfree.T)
            out = self._package(Gfree, Xfree, float(c["prob"].value))
            out.update(status=status, backend=f"cvxpy/{solver}", tol=tol_i,
                       converged=(status == "optimal"),
                       tr_max=max((float(np.trace(Q0.value) + np.trace(Q1.value))
                                   for Q0, Q1 in c["Qvars"]), default=0.0))
            if status == "optimal":
                return out
            last = out            # keep the inaccurate iterate, try one notch looser
        return last

    # ------------------------------------------------------------------
    # barrier backend, kept as an independent cross-check oracle
    # ------------------------------------------------------------------
    def _layout(self):
        n, d, f, m = self.n, self.d, self.f, len(self.obs)
        self.nG = n * f
        self.symX, self.idxX = sym_basis(f)
        self.nX = len(self.idxX)
        self.symQ0, self.idxQ0 = sym_basis(d + 1)
        self.symQ1, self.idxQ1 = sym_basis(d)
        self.nQ0, self.nQ1 = len(self.idxQ0), len(self.idxQ1)
        self.oG = 0
        self.oX = self.nG
        self.oQ = self.oX + self.nX
        self.per_obs = self.nQ0 + self.nQ1
        self.N = self.oQ + m * self.per_obs

    def unpack(self, z):
        n, f, d = self.n, self.f, self.d
        Gfree = z[self.oG:self.oG + self.nG].reshape(n, f)
        Xfree = np.zeros((f, f))
        for v, (i, j) in zip(z[self.oX:self.oX + self.nX], self.idxX):
            Xfree[i, j] = v
            Xfree[j, i] = v
        Qs = []
        for jo in range(len(self.obs)):
            o = self.oQ + jo * self.per_obs
            Q0 = np.zeros((d + 1, d + 1))
            for v, (i, j) in zip(z[o:o + self.nQ0], self.idxQ0):
                Q0[i, j] = v
                Q0[j, i] = v
            Q1 = np.zeros((d, d))
            for v, (i, j) in zip(z[o + self.nQ0:o + self.per_obs], self.idxQ1):
                Q1[i, j] = v
                Q1[j, i] = v
            Qs.append((Q0, Q1))
        return Gfree, Xfree, Qs

    def _affine(self, fn, out_shape):
        z0 = np.zeros(self.N)
        c0 = np.asarray(fn(z0), float)
        coef = np.zeros((self.N,) + tuple(out_shape))
        for i in range(self.N):
            e = np.zeros(self.N)
            e[i] = 1.0
            coef[i] = np.asarray(fn(e), float) - c0
        return c0, coef

    def build_barrier(self):
        d, n, f = self.d, self.n, self.f
        e = np.ones(d + 1)

        def Xfun(z):
            Gfree, Xfree, _ = self.unpack(z)
            return self.full_X(Gfree, Xfree)

        X0, Xc = self._affine(Xfun, (d + 1, d + 1))
        c_vec = np.einsum('ij,kji->k', self.Gk, Xc)
        self.cost_const = float(np.trace(self.Gk @ X0))

        rows, rhs = [], []
        for jo, (cj, rj) in enumerate(self.obs):
            def res(z, cj=cj, rj=rj, jo=jo):
                Gfree, Xfree, Qs = self.unpack(z)
                G = self.full_Gamma(Gfree)
                X = self.full_X(Gfree, Xfree)
                M = self.M_obstacle(X, G, cj, rj)
                Q0, Q1 = Qs[jo]
                return apply_map(self.Sd, M) - apply_map(self.Sd, Q0) - apply_map(self.Td, Q1)
            r0, rc = self._affine(res, (2 * d + 1,))
            rows.append(rc.T)
            rhs.append(-r0)
        Aeq = np.vstack(rows) if rows else np.zeros((0, self.N))
        beq = np.concatenate(rhs) if rhs else np.zeros(0)

        blocks = []
        sz = n + f
        Z0 = np.zeros((sz, sz))
        Z0[:n, :n] = np.eye(n)
        Zi = np.zeros((self.N, sz, sz))
        for a in range(n):
            for bcol in range(f):
                i = self.oG + a * f + bcol
                Zi[i, a, n + bcol] = 1.0
                Zi[i, n + bcol, a] = 1.0
        for s, (i, j) in enumerate(self.idxX):
            Zi[self.oX + s, n + i, n + j] = 1.0
            Zi[self.oX + s, n + j, n + i] = 1.0
        blocks.append(Block(Z0, Zi))

        for jo in range(len(self.obs)):
            o = self.oQ + jo * self.per_obs
            Zi = np.zeros((self.N, d + 1, d + 1))
            Zi[o:o + self.nQ0] = self.symQ0
            blocks.append(Block(np.zeros((d + 1, d + 1)), Zi))
            Zi = np.zeros((self.N, d, d))
            Zi[o + self.nQ0:o + self.per_obs] = self.symQ1
            blocks.append(Block(np.zeros((d, d)), Zi))
            # Q0,Q1 do not enter the cost, so -logdet Q -> -inf without this.
            Zi = np.zeros((self.N, 1, 1))
            for s_, (i, j) in enumerate(self.idxQ0):
                if i == j:
                    Zi[o + s_, 0, 0] = -1.0
            for s_, (i, j) in enumerate(self.idxQ1):
                if i == j:
                    Zi[o + self.nQ0 + s_, 0, 0] = -1.0
            if self.trace_bound is not None:
                blocks.append(Block(np.array([[self.trace_bound]]), Zi))

        self.prob = ConeProblem(self.N, c_vec, Aeq, beq, blocks)
        return self.prob

    build = build_barrier          # backwards-compatible alias

    def solve_barrier(self, eps=1e-9, verbose=False):
        if not hasattr(self, 'prob'):
            self.build_barrier()
        z, obj, gap = self.prob.solve(eps=eps, verbose=verbose)
        Gfree, Xfree, Qs = self.unpack(z)
        tr_max = max((np.trace(Q0) + np.trace(Q1) for Q0, Q1 in Qs), default=0.0)
        nfail = getattr(self.prob, 'n_centering_failures', 0)
        out = self._package(Gfree, Xfree, obj + self.cost_const)
        out.update(z=z, tr_max=float(tr_max), duality_gap=gap,
                   n_centering_failures=nfail, converged=(nfail == 0),
                   status='barrier', backend='barrier')
        return out

    def solve_robust(self, eps=1e-8, ladder=(1e4, 3e3, 3e4, 1e3)):
        """Ladder over the compactness bound R; accept the first converged run."""
        best = None
        for R in ladder:
            self.trace_bound = float(R)
            if hasattr(self, 'prob'):
                del self.prob
            try:
                r = self.solve_barrier(eps=eps)
            except Exception:
                continue
            r['trace_bound'] = R
            if best is None or r['n_centering_failures'] < best['n_centering_failures']:
                best = r
            if r['converged'] and r['tr_max'] < 0.2 * R:
                return r
        if best is None:
            raise RuntimeError("all trace bounds failed")
        return best

    # ------------------------------------------------------------------
    def solve(self, backend="cvxpy", **kw):
        if backend == "cvxpy":
            return self.solve_cvxpy(**kw)
        if backend == "barrier":
            return self.solve_barrier(**kw)
        if backend == "barrier_robust":
            return self.solve_robust(**kw)
        raise ValueError(f"unknown backend {backend!r}")

    # ------------------------------------------------------------------
    def _package(self, Gfree, Xfree, cost):
        """Common post-processing: gap matrix, its spectrum, rho, lifted V."""
        S = Xfree - Gfree.T @ Gfree
        S = 0.5 * (S + S.T)
        w = np.linalg.eigvalsh(S)                      # ascending
        scale = max(1.0, float(np.abs(w).max()) if w.size else 1.0)
        rho = int(np.sum(w > 1e-6 * scale))
        wdesc = w[::-1]
        return dict(Gfree=Gfree, Xfree=Xfree, Gamma=self.full_Gamma(Gfree),
                    X=self.full_X(Gfree, Xfree), gap=S, gap_eigs=w,
                    lam1=float(wdesc[0]) if wdesc.size else 0.0,
                    lam2=float(wdesc[1]) if wdesc.size > 1 else 0.0,
                    rho=rho, cost=float(cost), rank_scale=scale)

    # ------------------------------------------------------------------
    def curve(self, G, ns=400):
        ss = np.linspace(0, 1, ns)
        return ss, np.array([G @ bd(s, self.d) for s in ss])

    def min_clearance(self, G, ns=2001):
        """min_s min_j (||gamma(s)-c_j|| - r_j); negative means collision."""
        _, P = self.curve(G, ns)
        best = np.inf
        for cj, rj in self.obs:
            best = min(best, float(np.min(np.linalg.norm(P - cj, axis=1) - rj)))
        return best

    def lifted_V(self, res, tol=1e-6):
        """Rows of V in R^{rho x (d+1)} from V^T V = X - Gamma^T Gamma (Thm 3)."""
        w, U = np.linalg.eigh(res['gap'])
        scale = max(1.0, float(np.abs(w).max()) if w.size else 1.0)
        keep = w > tol * scale
        Vf = (U[:, keep] * np.sqrt(np.maximum(w[keep], 0.0))).T          # (rho, f)
        V = np.zeros((Vf.shape[0], self.d + 1))
        V[:, self.Fidx] = Vf
        return V

    def lifted_clearance(self, res, ns=2001):
        """min_{s,j} ||gamma(s)-c_j||^2 + ||v(s)||^2 - r_j^2  (Thm 3; must be >= 0)."""
        G, V = res['Gamma'], self.lifted_V(res)
        ss = np.linspace(0, 1, ns)
        Bs = np.array([bd(s, self.d) for s in ss])       # (ns, d+1)
        P = Bs @ G.T                                     # (ns, n)
        vv = np.sum((Bs @ V.T) ** 2, axis=1) if V.size else np.zeros(ns)
        best = np.inf
        for cj, rj in self.obs:
            best = min(best, float(np.min(np.sum((P - cj) ** 2, axis=1) + vv - rj ** 2)))
        return best

    def projected_violation(self, res, ns=2001):
        """min_{s,j} ||gamma(s)-c_j||^2 - r_j^2 for the projected curve."""
        G = res['Gamma']
        ss = np.linspace(0, 1, ns)
        P = np.array([G @ bd(s, self.d) for s in ss])
        best = np.inf
        for cj, rj in self.obs:
            best = min(best, float(np.min(np.sum((P - cj) ** 2, axis=1) - rj ** 2)))
        return best

    def true_cost(self, G):
        return float(np.trace(self.Gk @ G.T @ G))
