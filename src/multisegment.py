"""N-segment version of the relaxation: C^eta continuity, cross-Gram blocks,
and the joint block LMI.

Stack the segments' control points side by side,

    Gamma = [ Gamma_1 | ... | Gamma_N ]  in R^{n x M},   M = N (d+1),

and let X in S^M carry all the Gram blocks at once -- the diagonal blocks
X_{ii} = Gamma_i^T Gamma_i drive the cost and the obstacle certificates, and
the off-diagonal cross-Grams X_{i,i+1} are what the continuity constraints act
on.  The relaxation is X >= Gamma^T Gamma, i.e. the single block LMI

    [[ I_n , Gamma ], [ Gamma^T , X ]]  >=  0,

whose size is n + M and therefore still grows only additively in n.

FACIAL REDUCTION, generalised.  Every linear constraint on Gamma is of the form
Gamma A = B (boundary conditions have B != 0, continuity has B = 0), and each
carries the implied constraint X A = Gamma^T B (requirement R3).  Together they
force

    S A = X A - Gamma^T Gamma A = Gamma^T B - Gamma^T B = 0,        S := X - Gamma^T Gamma,

so S is supported on ker(A^T) and the LMI can never be strictly feasible while
any constraint is present.  At N segments there are 2(l+1) + (N-1)(eta+1)
constraint columns, so this bites much harder than in the single-segment
case.

The cure is to parameterise instead of constrain.  With N_perp an orthonormal
basis of ker(A^T) (dimension r = M - rank(A)) and Gamma_0 any particular
solution of Gamma_0 A = B,

    Gamma = Gamma_0 + Y N_perp^T,                       Y in R^{n x r} free,
    X     = Gamma_0^T Gamma_0 + Gamma_0^T Y N_perp^T
            + N_perp Y^T Gamma_0 + N_perp W N_perp^T,   W in S^r,

which is AFFINE in (Y, W), and the whole relaxation reduces to W >= Y^T Y, i.e.

    [[ I_n , Y ], [ Y^T , W ]] >= 0.

For N = 1 this collapses to exactly the section V-C reduction already in
relaxation.py: A has zero rows at the free indices, so ker(A^T) is spanned by
those coordinates, N_perp is the free-index selector, and (Y, W) are
(Gamma_free, X_free).
"""
import numpy as np

from bernstein import bd, bd_derivs, gram_deriv, S_map, T_map
from relaxation import exact_clearance, obstacle_poly, poly_min_on_unit


class MultiSegment:
    """N polynomial segments of degree d in R^n, joined with C^eta continuity.

    Each segment is parameterised on s in [0,1]; the trajectory is the
    concatenation, which corresponds to a uniform time allocation.
    """

    def __init__(self, n, d, k, l, N, obstacles, bc0, bc1, eta=None,
                 normalise_time=True, cuts=None):
        self.n, self.d, self.k, self.l, self.N = n, d, k, l, N
        # A homotopy COMMITMENT, imposed on the spline the same way it is
        # imposed on a single segment: one direction `a`, one offset `beta`,
        # one sense, held at a set of parameters.  Each entry is
        #   (segment index i, local s in [0,1], a in R^n, beta, sense in {-1,+1})
        # and means  sense * (a . Gamma_i b_d(s) - beta) >= 0.  This is what
        # lets Section VI's mechanism run on a CERTIFIED (N>1) family instead
        # of the single-segment one the grid does not cover.
        self.cuts = [] if cuts is None else [
            (int(i), float(sl), np.asarray(a, float), float(b), int(sg))
            for (i, sl, a, b, sg) in cuts]
        # Each segment carries s in [0,1], so N segments span N units of
        # parameter.  Holding the TOTAL duration at 1 instead means each
        # segment occupies 1/N of it, and d^k/dt^k = N^k d^k/ds^k, giving a
        # cost factor N^(2k-1).  Without this the obstacle-free optimum reads
        # 16/N and "more segments" looks like a free cost reduction, which is
        # a units artefact rather than a better trajectory.
        self.time_scale = float(N) ** (2 * k - 1) if normalise_time else 1.0
        self.eta = (d - 1) if eta is None else eta
        self.obs = [(np.asarray(c, float), float(r)) for c, r in obstacles]
        self.M = N * (d + 1)
        self.Gk = gram_deriv(d, k)
        self.Sd, self.Td = S_map(d), T_map(d)

        A_cols, B_cols = [], []

        # --- boundary conditions: derivatives 0..l at the very start and end
        db0 = bd_derivs(0.0, d, l)
        db1 = bd_derivs(1.0, d, l)
        for i in range(l + 1):
            a = np.zeros(self.M)
            a[self.slice_(0)] = db0[i]
            A_cols.append(a)
            B_cols.append(np.asarray(bc0[i], float))

            a = np.zeros(self.M)
            a[self.slice_(N - 1)] = db1[i]
            A_cols.append(a)
            B_cols.append(np.asarray(bc1[i], float))

        # --- C^eta continuity at each junction: Gamma_i b^(m)(1) = Gamma_{i+1} b^(m)(0)
        dbe1 = bd_derivs(1.0, d, self.eta)
        dbe0 = bd_derivs(0.0, d, self.eta)
        for j in range(N - 1):
            for m in range(self.eta + 1):
                a = np.zeros(self.M)
                a[self.slice_(j)] = dbe1[m]
                a[self.slice_(j + 1)] = -dbe0[m]
                A_cols.append(a)
                B_cols.append(np.zeros(n))

        self.A = np.array(A_cols).T                      # (M, q)
        self.B = np.array(B_cols).T                      # (n, q)

        # particular solution and the null space of A^T
        self.Gamma0 = self.B @ np.linalg.pinv(self.A)    # (n, M)
        resid = np.linalg.norm(self.Gamma0 @ self.A - self.B)
        if resid > 1e-8 * max(1.0, np.linalg.norm(self.B)):
            raise ValueError(f"boundary/continuity data inconsistent (resid {resid:.2e})")

        U, sv, _ = np.linalg.svd(self.A)
        rank = int(np.sum(sv > 1e-10 * max(1.0, sv.max())))
        self.Nperp = U[:, rank:]                          # (M, r)
        self.r = self.Nperp.shape[1]
        if self.r < 1:
            raise ValueError(f"no free parameters (N={N}, d={d}, l={l}, eta={self.eta})")
        # sanity: the parameterisation must satisfy the constraints identically
        assert np.allclose(self.Nperp.T @ self.A, 0.0, atol=1e-9)

    # ------------------------------------------------------------------
    def slice_(self, i):
        return slice(i * (self.d + 1), (i + 1) * (self.d + 1))

    def full_Gamma(self, Y):
        return self.Gamma0 + np.asarray(Y, float) @ self.Nperp.T

    def full_X(self, Y, W):
        Y = np.asarray(Y, float)
        G0, Np = self.Gamma0, self.Nperp
        cross = G0.T @ Y @ Np.T
        return G0.T @ G0 + cross + cross.T + Np @ np.asarray(W, float) @ Np.T

    def seg_Gamma(self, G, i):
        return G[:, self.slice_(i)]

    # ------------------------------------------------------------------
    def build_cvxpy(self):
        import cvxpy as cp
        n, d, r = self.n, self.d, self.r

        Y = cp.Variable((n, r), name="Y")
        W = cp.Variable((r, r), symmetric=True, name="W")
        G0, Np = self.Gamma0, self.Nperp

        Gamma = G0 + Y @ Np.T
        cross = G0.T @ Y @ Np.T
        X = cp.Constant(G0.T @ G0) + cross + cross.T + Np @ W @ Np.T

        cons = [cp.bmat([[np.eye(n), Y], [Y.T, W]]) >> 0]
        sos_blocks = []

        e_row = np.ones((1, d + 1))
        for i in range(self.N):
            sl = self.slice_(i)
            Gi = Gamma[:, sl]
            Xi = X[sl, sl]
            for cj, rj in self.obs:
                Q0 = cp.Variable((d + 1, d + 1), symmetric=True)
                Q1 = cp.Variable((d, d), symmetric=True)
                cons += [Q0 >> 0, Q1 >> 0]
                sos_blocks.append((i, Q0, Q1))
                Gc = Gi.T @ cj.reshape(n, 1)
                ge = Gc @ e_row
                Mij = (Xi - ge - ge.T
                       + (float(cj @ cj) - rj ** 2) * np.ones((d + 1, d + 1)))
                cons.append(cp.hstack(
                    [cp.sum(cp.multiply(self.Sd[t], Mij - Q0))
                     - cp.sum(cp.multiply(self.Td[t], Q1))
                     for t in range(2 * d + 1)]) == 0)

        for (i, s_loc, a, beta, sense) in self.cuts:
            b = bd(s_loc, d)                       # (d+1,)
            cons.append(sense * (a @ (Gamma[:, self.slice_(i)] @ b) - beta)
                        >= 0)

        cost = self.time_scale * sum(
            cp.sum(cp.multiply(self.Gk, X[self.slice_(i), self.slice_(i)]))
            for i in range(self.N))
        prob = cp.Problem(cp.Minimize(cost), cons)
        self._cvx = dict(prob=prob, Y=Y, W=W, Gamma=Gamma, X=X, cost=cost,
                         cons=cons, sos=sos_blocks)
        return self._cvx

    LADDER = (1e-11, 1e-9, 1e-8)

    def solve(self, solver="CLARABEL", tol=None):
        import cvxpy as cp
        import warnings
        c = getattr(self, "_cvx", None) or self.build_cvxpy()
        ladder = (tol,) if tol is not None else self.LADDER
        last = dict(converged=False, status="not attempted")
        for t in ladder:
            opts = dict(verbose=False)
            if solver == "CLARABEL":
                opts.update(tol_gap_abs=t, tol_gap_rel=t, tol_feas=t, max_iter=600)
            elif solver == "SCS":
                opts.update(eps=t, max_iters=200000)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    c["prob"].solve(solver=solver, **opts)
            except cp.error.SolverError as exc:
                last = dict(converged=False, status=f"SolverError: {exc}")
                continue
            st = c["prob"].status
            if st not in ("optimal", "optimal_inaccurate") or c["Y"].value is None:
                last = dict(converged=False, status=st)
                continue
            Y = np.asarray(c["Y"].value, float)
            W = np.asarray(c["W"].value, float)
            W = 0.5 * (W + W.T)
            out = self._package(Y, W, float(c["prob"].value))
            out.update(status=st, converged=(st == "optimal"), tol=t)
            if st == "optimal":
                return out
            last = out
        return last

    # ------------------------------------------------------------------
    def _package(self, Y, W, cost):
        Sr = W - Y.T @ Y
        Sr = 0.5 * (Sr + Sr.T)
        w = np.linalg.eigvalsh(Sr)
        scale = max(1.0, float(np.abs(w).max()) if w.size else 1.0)
        rho = int(np.sum(w > 1e-6 * scale))
        wd = w[::-1]
        return dict(Y=Y, W=W, Gamma=self.full_Gamma(Y), X=self.full_X(Y, W),
                    gap=Sr, gap_eigs=w, rho=rho, cost=float(cost),
                    lam1=float(wd[0]) if wd.size else 0.0,
                    lam2=float(wd[1]) if wd.size > 1 else 0.0)

    def lifted_V(self, res, tol=1e-6):
        """V with V^T V = X - Gamma^T Gamma, in the stacked coordinates."""
        w, U = np.linalg.eigh(res["gap"])
        scale = max(1.0, float(np.abs(w).max()) if w.size else 1.0)
        keep = w > tol * scale
        Vr = (U[:, keep] * np.sqrt(np.maximum(w[keep], 0.0))).T      # (rho, r)
        return Vr @ self.Nperp.T                                     # (rho, M)

    # ------------------------------------------------------------------
    def exact_clearance(self, G):
        """min over segments and obstacles of the exact polynomial minimum."""
        best, arg = np.inf, (None, None, None)
        for i in range(self.N):
            v, (j, s) = exact_clearance(self.seg_Gamma(G, i), self.obs, self.d)
            if v < best:
                best, arg = v, (i, j, s)
        return best, arg

    def lifted_clearance(self, res, ns=801):
        """Theorem 3 in the multi-segment setting: the lifted curve must be
        collision-free on every segment."""
        G, V = res["Gamma"], self.lifted_V(res)
        ss = np.linspace(0.0, 1.0, ns)
        Bs = np.array([bd(s, self.d) for s in ss])
        best = np.inf
        for i in range(self.N):
            P = Bs @ self.seg_Gamma(G, i).T
            vv = (np.sum((Bs @ V[:, self.slice_(i)].T) ** 2, axis=1)
                  if V.size else np.zeros(ns))
            for cj, rj in self.obs:
                best = min(best, float(np.min(np.sum((P - cj) ** 2, axis=1)
                                              + vv - rj ** 2)))
        return best

    def true_cost(self, G):
        return float(self.time_scale
                     * sum(np.trace(self.Gk @ self.seg_Gamma(G, i).T
                                    @ self.seg_Gamma(G, i))
                           for i in range(self.N)))

    def curve(self, G, ns_per_seg=200):
        """Concatenated sample points of the whole trajectory."""
        out = []
        ss = np.linspace(0.0, 1.0, ns_per_seg)
        for i in range(self.N):
            Gi = self.seg_Gamma(G, i)
            out.append(np.array([Gi @ bd(s, self.d) for s in ss]))
        return np.vstack(out)

    def M_obstacle(self, X, G, i, cj, rj):
        """The Markov--Lukacs target for obstacle j on segment i.

        Identical in form to the single-segment case, read off the block of
        (X, Gamma) that belongs to segment i -- which is what lets the
        rank-one reconstruction of Section VII be run unchanged on a
        multisegment, and therefore certifiable, family.
        """
        sl = self.slice_(i)
        Gi = np.asarray(G, float)[:, sl]
        Xi = np.asarray(X, float)[sl, sl]
        cj = np.asarray(cj, float)
        e_row = np.ones((1, self.d + 1))
        ge = (Gi.T @ cj.reshape(self.n, 1)) @ e_row
        return (Xi - ge - ge.T
                + (float(cj @ cj) - float(rj) ** 2)
                * np.ones((self.d + 1, self.d + 1)))

    def rank_profile(self, tol=1e-7):
        """Ranks of the SOS blocks AS THE INTERIOR-POINT SOLVER RETURNED THEM.

        This is emphatically NOT a test of the (n+rho,1,1) profile.  An
        interior-point method converges to the MAXIMUM-rank point of the
        optimal face, so these ranks are expected to exceed one and a high
        reading refutes nothing.  The profile claim is about RECONSTRUCTION --
        hold (Gamma*, X*), discard these blocks, and rebuild a rank-one pair
        satisfying the same coefficient equalities -- which is what
        `experiments/g1_rank_lemma.py` does.  Kept here only as the input that
        reconstruction starts from.
        """
        c = getattr(self, "_cvx", None)
        if c is None or not c.get("sos"):
            return None
        def _rank(M):
            if M is None:
                return None
            w = np.linalg.eigvalsh(0.5 * (np.asarray(M) + np.asarray(M).T))
            top = max(float(w.max()), 1e-300)
            return int(np.sum(w > tol * top))
        r0 = [_rank(Q0.value) for (_i, Q0, _Q1) in c["sos"]]
        r1 = [_rank(Q1.value) for (_i, _Q0, Q1) in c["sos"]]
        r0 = [x for x in r0 if x is not None]
        r1 = [x for x in r1 if x is not None]
        if not r0:
            return None
        return dict(sos0_max=max(r0), sos1_max=max(r1),
                    sos0=r0, sos1=r1, n_blocks=len(r0))

    def continuity_residual(self, G):
        """Max violation of the C^eta junction conditions (must be ~0)."""
        worst = 0.0
        d1 = bd_derivs(1.0, self.d, self.eta)
        d0 = bd_derivs(0.0, self.d, self.eta)
        for j in range(self.N - 1):
            for m in range(self.eta + 1):
                lhs = self.seg_Gamma(G, j) @ d1[m]
                rhs = self.seg_Gamma(G, j + 1) @ d0[m]
                worst = max(worst, float(np.max(np.abs(lhs - rhs))))
        return worst


# ----------------------------------------------------------------------
class MultiGroundTruth:
    """Brute-force reference for the N-segment nonconvex problem.

    Multi-start SLSQP over the free parameters Y, with collision enforced on a
    grid per segment and then driven to *continuous-time* feasibility by the
    same semi-infinite cutting plane groundtruth.py uses.  Continuity and the
    boundary conditions are exact by construction: they are baked into the
    parameterisation Gamma = Gamma_0 + Y N_perp^T, so SLSQP cannot violate them.
    """

    def __init__(self, ms, ns=120):
        self.ms = ms
        self.n, self.d, self.r, self.N = ms.n, ms.d, ms.r, ms.N
        self._set_points(np.linspace(0.0, 1.0, ns))

    def _set_points(self, ss):
        ms = self.ms
        self.ss = np.asarray(ss, float)
        self.Bs = np.array([bd(s, ms.d) for s in self.ss])          # (ns, d+1)
        # per segment, the map from Y to the sampled positions
        self.U = [self.Bs @ ms.Nperp[ms.slice_(i), :] for i in range(ms.N)]  # (ns, r)
        self.P0 = [self.Bs @ ms.seg_Gamma(ms.Gamma0, i).T for i in range(ms.N)]

    def _G(self, x):
        return self.ms.full_Gamma(x.reshape(self.n, self.r))

    def cost(self, x):
        return self.ms.true_cost(self._G(x))

    def cost_grad(self, x):
        ms = self.ms
        G = self._G(x)
        g = np.zeros((self.n, self.r))
        for i in range(ms.N):
            g += 2.0 * ms.seg_Gamma(G, i) @ ms.Gk @ ms.Nperp[ms.slice_(i), :]
        return (ms.time_scale * g).ravel()

    def cons(self, x):
        Y = x.reshape(self.n, self.r)
        out = []
        for i in range(self.ms.N):
            P = self.P0[i] + self.U[i] @ Y.T                        # (ns, n)
            for c, rr in self.ms.obs:
                out.append(np.sum((P - c) ** 2, axis=1) - rr ** 2)
        return np.concatenate(out) if out else np.zeros(0)

    def cons_jac(self, x):
        Y = x.reshape(self.n, self.r)
        blocks = []
        for i in range(self.ms.N):
            P = self.P0[i] + self.U[i] @ Y.T
            for c, _rr in self.ms.obs:
                R = P - c                                            # (ns, n)
                J = 2.0 * np.einsum('ta,tb->tab', R, self.U[i])
                blocks.append(J.reshape(len(self.ss), self.n * self.r))
        return np.vstack(blocks) if blocks else np.zeros((0, self.n * self.r))

    def exact_slack(self, x):
        return self.ms.exact_clearance(self._G(x))[0]

    def _touch_points(self, x):
        G = self._G(x)
        pts = set()
        for i in range(self.ms.N):
            Gi = self.ms.seg_Gamma(G, i)
            for c, rr in self.ms.obs:
                pts.add(round(poly_min_on_unit(obstacle_poly(Gi, c, rr, self.d))[1], 12))
        return sorted(pts)

    def _polish(self, x0, feas_tol, max_rounds=10):
        from scipy.optimize import minimize
        cons = [{'type': 'ineq', 'fun': self.cons, 'jac': self.cons_jac}]
        x = x0
        for _ in range(max_rounds):
            res = minimize(self.cost, x, jac=self.cost_grad, constraints=cons,
                           method='SLSQP', options={'maxiter': 600, 'ftol': 1e-14})
            if not res.success:
                return None
            x = res.x
            if self.exact_slack(x) >= -feas_tol:
                return float(res.fun), x.copy()
            new = [s for s in self._touch_points(x)
                   if np.min(np.abs(self.ss - s)) > 1e-9]
            if not new:
                return None
            self._set_points(np.sort(np.concatenate([self.ss, new])))
        return None

    def solve(self, ntry=60, seed=0, feas_tol=1e-9, dedup_tol=1e-4):
        rng = np.random.default_rng(seed)
        base = self.ss.copy()
        # start from the obstacle-free optimum, then random detours
        starts = [np.zeros(self.n * self.r)]
        for i in range(ntry - 1):
            scale = 0.4 + 3.0 * (i % 7) / 6.0
            starts.append(rng.standard_normal(self.n * self.r) * scale)

        minima = []
        for x0 in starts:
            self._set_points(base)
            got = self._polish(x0, feas_tol)
            if got is not None:
                minima.append(got)
        self._set_points(base)

        minima.sort(key=lambda t: t[0])
        distinct = []
        for c_, x_ in minima:
            G = self._G(x_)
            if all(np.linalg.norm(G - self._G(xd)) > dedup_tol for _, xd in distinct):
                distinct.append((c_, x_))
        if not minima:
            return dict(ok=False, cost=None, x=None, minima=[], n_distinct=0)
        return dict(ok=True, cost=minima[0][0], x=minima[0][1],
                    minima=distinct, n_distinct=len(distinct))
