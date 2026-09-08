"""Fixed-rank optimization of the *unchanged* multisegment polynomial SDP.

The rank-one gap is justified only for a family satisfying the Green-kernel
theorem and an attained complementary primal/dual pair. Local Newton steps
are never a global certificate: a separate dual bound and residual gate is
required, with an independent conic fallback. SOS factors have one row by
the univariate Markov--Lukacs representation, not by the gap-rank theorem.

This implementation uses an exact rational null basis (stored as integers
for the measured family). Numerical bounds are diagnostics; exact verification
of exported multipliers is implemented separately in exact_sdp_bounds.py.
"""
from fractions import Fraction as F
import time
import warnings
import numpy as np
import scipy.linalg as la

from exact_green import constraint_matrix, nullspace, rref, q_gram_deriv, solve as qsolve
from mlukacs import rank1_certificate


def rational_coordinates(ms):
    rows = constraint_matrix(ms.d, ms.l, ms.N, ms.eta)
    Q = np.asarray(nullspace(rows), dtype=object).T
    G0 = []
    for b in ms.B:
        aug = [a + [F(float(y))] for a, y in zip(rows, b)]
        R, piv = rref(aug)
        if ms.M in piv:
            raise ValueError('Inconsistent affine constraints')
        g = [F(0)] * ms.M
        for row, col in zip(R, piv):
            g[col] = row[-1]
        G0.append(g)
    G0 = np.asarray(G0, dtype=object)
    H = np.asarray(q_gram_deriv(ms.d,ms.k),dtype=object)
    K = sum((Q[ms.slice_(i)].T@H@Q[ms.slice_(i)] for i in range(ms.N)))
    C = sum((G0[:,ms.slice_(i)]@H@Q[ms.slice_(i)] for i in range(ms.N)))
    # Exact minimum-energy particular solution eliminates the linear cost.
    correction = np.asarray(qsolve(K.tolist(),C.T.tolist()),dtype=object).T
    G0 = G0-correction@Q.T
    return G0, Q


class CertifiedSpline:
    """129 variables / 99 equalities for F2, independent of conic Gram ranks."""
    def __init__(self, ms):
        if ms.cuts:
            raise ValueError('Root SDP only; point cuts are not implemented here')
        self.ms = ms
        self.G0q, self.Qq = rational_coordinates(ms)
        ms.Gamma0 = np.asarray(self.G0q, float)
        ms.Nperp = np.asarray(self.Qq, float)
        self.G0, self.Q = ms.Gamma0, ms.Nperp
        self.n, self.r, self.d = ms.n, ms.r, ms.d
        self.m = len(ms.obs)
        self.b = ms.N * self.m
        self.h = 2 * ms.d + 1
        self.ny = self.n * self.r
        self.nv = self.r
        self.a0 = self.ny + self.nv
        self.a1 = self.a0 + self.b * (self.d + 1)
        self.nz = self.a1 + self.b * self.d
        self.nr = self.b * self.h
        self.K = ms.time_scale * sum(
            self.Q[ms.slice_(i)].T @ ms.Gk @ self.Q[ms.slice_(i)]
            for i in range(ms.N))
        self.C = ms.time_scale * sum(
            self.G0[:, ms.slice_(i)] @ ms.Gk @ self.Q[ms.slice_(i)]
            for i in range(ms.N))
        self.c0 = ms.true_cost(self.G0)
        self.A = np.stack([np.einsum('ia,kij,jb->kab', self.Q[ms.slice_(i)],
                                    ms.Sd, self.Q[ms.slice_(i)])
                           for i in range(ms.N) for _ in ms.obs])
        self.update(ms.obs)
        self._cvx = None
        self.z = self.lam = None

    def update(self, obstacles):
        if len(obstacles) != self.m:
            raise ValueError('The obstacle count is fixed')
        self.ms.obs = [(np.asarray(c, float), float(r)) for c, r in obstacles]
        self.B, self.c = [], []
        for i in range(self.ms.N):
            qi = self.Q[self.ms.slice_(i)]
            gi = self.G0[:, self.ms.slice_(i)]
            for c, rad in self.ms.obs:
                delta = gi - c[:, None]
                self.B.append(np.einsum('ai,kij,jr->kar', delta, self.ms.Sd, qi))
                self.c.append(np.einsum('kij,ai,aj->k', self.ms.Sd, delta, delta) - rad**2)
        self.B, self.c = np.asarray(self.B), np.asarray(self.c)

    def unpack(self, z):
        z = np.asarray(z, float)
        return (z[:self.ny].reshape(self.n, self.r), z[self.ny:self.a0],
                z[self.a0:self.a1].reshape(self.b, self.d+1),
                z[self.a1:].reshape(self.b, self.d))

    def cost(self, z):
        Y, v, _, _ = self.unpack(z)
        return float(self.c0 + 2*np.sum(self.C*Y) + np.sum((Y@self.K)*Y) + v@self.K@v)

    def grad(self, z):
        Y, v, _, _ = self.unpack(z)
        g = np.zeros(self.nz)
        g[:self.ny] = (2*(Y@self.K + self.C)).ravel()
        g[self.ny:self.a0] = 2*self.K@v
        return g

    def target(self, z):
        Y, v, _, _ = self.unpack(z)
        W = Y.T@Y + np.outer(v, v)
        return self.c + 2*np.einsum('bkar,ar->bk', self.B, Y) + np.einsum('bkij,ij->bk', self.A, W)

    def residual(self, z):
        _, _, a, b = self.unpack(z)
        return (self.target(z) - np.einsum('kiJ,bi,bJ->bk', self.ms.Sd, a, a)
                - np.einsum('kiJ,bi,bJ->bk', self.ms.Td, b, b)).ravel()

    def jac(self, z):
        Y, v, a, b = self.unpack(z)
        J = np.zeros((self.b, self.h, self.nz))
        J[:, :, :self.ny] = (2*np.einsum('aj,bkji->bkai', Y, self.A) + 2*self.B).reshape(self.b, self.h, self.ny)
        J[:, :, self.ny:self.a0] = 2*np.einsum('bkij,j->bki', self.A, v)
        for i in range(self.b):
            J[i, :, self.a0+i*(self.d+1):self.a0+(i+1)*(self.d+1)] = -2*np.einsum('kij,j->ki', self.ms.Sd, a[i])
            J[i, :, self.a1+i*self.d:self.a1+(i+1)*self.d] = -2*np.einsum('kij,j->ki', self.ms.Td, b[i])
        return J.reshape(self.nr, self.nz)

    def slacks(self, lam):
        l = np.asarray(lam).reshape(self.b, self.h)
        Z = self.K + np.einsum('bk,bkij->ij', l, self.A)
        U = -np.einsum('bk,kij->bij', l, self.ms.Sd)
        T = -np.einsum('bk,kij->bij', l, self.ms.Td)
        B = self.C + np.einsum('bk,bkar->ar', l, self.B)
        return Z, U, T, B

    def hess(self, lam):
        Z, U, T, _ = self.slacks(lam)
        return 2*la.block_diag(*([Z]*(self.n+1)), *U, *T)

    def dual_bound(self, lam, margin=1e-9):
        """Repair Gram slacks, then scale multipliers to make Z positive definite.

        Returned lower bound is evaluated in floating point. It is independent
        of KKT stationarity; an exact replay is required for a rigorous bound.
        """
        l = np.asarray(lam, float).reshape(self.b, self.h).copy()
        if not np.isfinite(l).all():
            return dict(ok=False)
        _, U, T, _ = self.slacks(l)
        Uref = self.ms.Sd.sum(axis=0)/self.h
        Tref = self.ms.Td.sum(axis=0)/self.h
        for i in range(self.b):
            shift = max(0., margin-la.eigvalsh(U[i], Uref)[0], margin-la.eigvalsh(T[i], Tref)[0])
            l[i] -= shift/self.h
        Z, _, _, _ = self.slacks(l)
        e = la.eigvalsh(Z-self.K, self.K)[0]
        alpha = min(1., (1-margin)/(-e)) if e < 0 else 1.
        # A nearly singular Z amplifies a small stationarity error. Select the
        # best valid bound from a fixed, declared list of interior scalings.
        best = None
        for shrink in (0., 1e-8, 1e-7, 1e-6, 1e-5, 1e-4):
            a = alpha*(1-shrink)
            ll = a*l
            Z, U, T, B = self.slacks(ll)
            try:
                val = self.c0 + np.sum(ll*self.c) - np.sum(B*la.solve(Z, B.T, assume_a='pos').T)
                mineig = min(la.eigvalsh(Z)[0], np.linalg.eigvalsh(U).min(), np.linalg.eigvalsh(T).min())
            except (la.LinAlgError, ValueError):
                continue
            if np.isfinite(val) and mineig>0 and (best is None or val>best['lower']):
                best = dict(ok=True, lower=float(val), min_eig=float(mineig),
                            multipliers=ll.ravel(), scale=float(a))
        return best or dict(ok=False)

    def assess(self, z, lam):
        if not np.isfinite(z).all() or not np.isfinite(lam).all():
            return dict(ok=False,reason='nonfinite_candidate')
        cost = self.cost(z)
        res = float(np.max(np.abs(self.residual(z))))
        stat = float(np.max(np.abs(self.grad(z)+self.jac(z).T@lam)))
        bound = self.dual_bound(lam)
        gap = (cost-bound['lower'])/max(1., abs(cost)) if bound['ok'] else float('inf')
        return dict(ok=bool(np.isfinite(z).all() and res <= 1e-7 and stat <= 1e-6 and -1e-8 <= gap <= 1e-5),
                    residual=res, stationarity=stat, relative_gap=float(gap), cost=cost, bound=bound)

    def newton(self, z, lam, maxiter=12):
        z, lam = z.copy(), lam.copy()
        for it in range(maxiter):
            J = self.jac(z)
            f = np.r_[self.grad(z)+J.T@lam, self.residual(z)]
            norm = la.norm(f)
            if np.max(np.abs(f)) < 5e-9:
                return z, lam, it, 'stationary'
            KKT = np.block([[self.hess(lam), J.T], [J, np.zeros((self.nr, self.nr))]])
            with warnings.catch_warnings():
                warnings.simplefilter('error', la.LinAlgWarning)
                try:
                    step = la.solve(KKT, -f, assume_a='sym')
                except (la.LinAlgError, la.LinAlgWarning):
                    step = la.lstsq(KKT, -f, cond=1e-11, lapack_driver='gelsy')[0]
            accepted = False
            for fraction in (1., .5, .25, .125, .0625):
                zz = z + fraction*step[:self.nz]
                ll = lam + fraction*step[self.nz:]
                if not np.isfinite(zz).all() or la.norm(zz)>1e6:
                    continue
                ff = np.r_[self.grad(zz)+self.jac(zz).T@ll, self.residual(zz)]
                if la.norm(ff) < (1-1e-4*fraction)*norm:
                    z, lam, accepted = zz, ll, True
                    break
            if not accepted:
                return z, lam, it+1, 'line_search_failed'
        return z, lam, maxiter, 'iteration_limit'

    def build_conic(self):
        import cvxpy as cp
        Y = cp.Variable((self.n, self.r))
        W = cp.Variable((self.r, self.r), symmetric=True)
        # Parameters contain only affine constraint data, so this graph is DPP.
        bp = cp.Parameter((self.nr, self.ny))
        cp0 = cp.Parameter(self.nr)
        cons = [cp.bmat([[np.eye(self.n), Y], [Y.T, W]]) >> 0]
        eq, sos = [], []
        target = cp0 + 2*bp@cp.vec(Y, order='C') + self.A.reshape(self.nr, -1)@cp.vec(W, order='C')
        for i in range(self.b):
            Q0 = cp.Variable((self.d+1,self.d+1), symmetric=True)
            Q1 = cp.Variable((self.d,self.d), symmetric=True)
            e = (target[i*self.h:(i+1)*self.h]
                 - self.ms.Sd.reshape(self.h,-1)@cp.vec(Q0,order='C')
                 - self.ms.Td.reshape(self.h,-1)@cp.vec(Q1,order='C') == 0)
            cons.extend([Q0 >> 0, Q1 >> 0, e]); eq.append(e)
            sos.append((Q0,Q1))
        cost = self.c0 + 2*cp.sum(cp.multiply(self.C,Y)) + cp.sum(cp.multiply(self.K,W))
        problem = cp.Problem(cp.Minimize(cost), cons)
        assert problem.is_dpp()
        self._cvx = dict(problem=problem, Y=Y, W=W, B=bp, c=cp0, eq=eq, sos=sos)

    def conic(self):
        if self._cvx is None:
            self.build_conic()
        c = self._cvx
        c['B'].value = self.B.reshape(self.nr,self.ny)
        c['c'].value = self.c.ravel()
        import cvxpy as cp
        last = dict(ok=False,status='not attempted')
        for attempt,tol in enumerate((1e-9,1e-11),1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    c['problem'].solve(solver='CLARABEL', tol_gap_abs=tol,
                        tol_gap_rel=tol, tol_feas=tol, max_iter=600, warm_start=True)
            except cp.error.SolverError as e:
                last = dict(ok=False,status=str(e),attempts=attempt)
                continue
            if c['problem'].status != 'optimal':
                last = dict(ok=False,status=c['problem'].status,attempts=attempt)
                continue
            Y,W = c['Y'].value.copy(),c['W'].value.copy()
            lam = np.concatenate([e.dual_value for e in c['eq']])
            bound = self.dual_bound(lam)
            cost = float(c['problem'].value)
            relative_gap = (cost-bound['lower'])/max(1.,abs(cost)) if bound['ok'] else float('inf')
            residual = max(float(np.max(np.abs(e.expr.value))) for e in c['eq'])
            psd_min = min(np.linalg.eigvalsh(W-Y.T@Y).min(),
                          min(np.linalg.eigvalsh(q.value).min() for pair in c['sos'] for q in pair))
            ok = residual<=1e-7 and psd_min>=-1e-7 and -1e-8<=relative_gap<=1e-5
            last = dict(ok=bool(ok),status=c['problem'].status,Y=Y,W=W,cost=cost,
                        lam=lam,bound=bound,residual=residual,psd_min=float(psd_min),
                        relative_gap=float(relative_gap),attempts=attempt,tolerance=tol,
                        solve_ms=1000*c['problem'].solver_stats.solve_time)
            if ok:
                return last
        return last

    def seed(self, solution):
        Y = solution['Y']
        gap = solution['W'] - Y.T@Y
        e, U = la.eigh((gap+gap.T)/2)
        v = U[:,-1]*np.sqrt(max(e[-1],0.))
        # Keep a nonzero gap seed when moving between rho=0 and rho=1.
        if la.norm(v)<1e-5:
            v = U[:,-1]*1e-5
        z = np.r_[Y.ravel(), v, np.zeros(self.nz-self.a0)]
        target = self.target(z)
        for i in range(self.b):
            a,b,_ = rank1_certificate(target[i], self.d, self.ms.Sd, self.ms.Td)
            z[self.a0+i*(self.d+1):self.a0+(i+1)*(self.d+1)] = a
            z[self.a1+i*self.d:self.a1+(i+1)*self.d] = b
        return z, solution['lam'].copy()

    def solve(self, obstacles, local_iterations=12, use_warm_start=True):
        """Diagnose a local candidate, then dispatch to an independent SDP.

        A valid conic answer survives failure to construct the next warm seed.
        In that case ``ok`` is true, ``conic`` holds the answer and ``z`` is None.
        A numerical conic failure returns an explicit no-answer, never stale z.
        """
        begin = time.perf_counter()
        from cvxpy.error import SolverError
        from exact_check import ExactCheckError
        numerical = (np.linalg.LinAlgError, la.LinAlgWarning, ValueError,
                     FloatingPointError, OverflowError, SolverError, ExactCheckError)
        self.update(obstacles)
        update_ms = 1000*(time.perf_counter()-begin)
        local_ms, it, status = 0., 0, 'cold'
        verdict = point = None
        failed, gates = [], {}
        if self.z is not None and use_warm_start and local_iterations:
            t = time.perf_counter()
            try:
                z, lam, it, status = (self.newton(self.z, self.lam) if local_iterations == 12 else
                                      self.newton(self.z, self.lam, maxiter=local_iterations))
                # Retain the returned point even if its later assessment fails.
                point = dict(z=z.copy(), lam=lam.copy())
                verdict = self.assess(z, lam)
                finite = bool(np.isfinite(z).all() and np.isfinite(lam).all())
                gates['finite_candidate'] = finite
                if verdict.get('reason') == 'nonfinite_candidate' or not finite:
                    failed.append('nonfinite_candidate')
                    gates.update(equality_residual=None, stationarity=None,
                                 dual_bound=None, objective_gap=None)
                else:
                    residual, stationarity = verdict.get('residual'), verdict.get('stationarity')
                    gap, bound = verdict.get('relative_gap'), verdict.get('bound')
                    gates.update(
                        equality_residual=None if residual is None else bool(np.isfinite(residual) and residual<=1e-7),
                        stationarity=None if stationarity is None else bool(np.isfinite(stationarity) and stationarity<=1e-6),
                        dual_bound=None if bound is None else bool(bound['ok']),
                        objective_gap=None if gap is None else bool(np.isfinite(gap) and -1e-8<=gap<=1e-5))
                    failed.extend(name for key,name in (
                        ('equality_residual','equality_residual'),('stationarity','stationarity'),
                        ('dual_bound','dual_bound_failure'),('objective_gap','objective_gap')) if gates[key] is False)
                    if any(value is None for value in gates.values()):
                        failed.append('local_assessment_incomplete')
            except numerical as exc:
                status = 'local_numerical_failure'
                verdict = dict(ok=False, reason=status, exception_type=type(exc).__name__, detail=str(exc))
                failed = [status]
                gates = dict(finite_candidate=None, equality_residual=None,
                             stationarity=None, dual_bound=None, objective_gap=None)
            local_ms = 1000*(time.perf_counter()-t)
            if verdict['ok'] and not failed:
                self.z, self.lam = z, lam
                return dict(ok=True, source='factor', z=z, lam=lam,
                            total_ms=1000*(time.perf_counter()-begin), local_ms=local_ms,
                            update_ms=update_ms, conic_ms=0., initialization_ms=0.,
                            iterations=it, local_status=status, local_assessment=verdict,
                            local_gate_results=gates, rejected_local_point=None,
                            dispatch_reason='local_accepted', failed_gates=[],
                            seed_status='local_candidate',
                            **{k:v for k,v in verdict.items() if k!='ok'})
            if not failed:
                failed = ['local_assessment_rejected']
        else:
            failed = ['cold_initialization' if self.z is None else
                      'warm_start_disabled' if not use_warm_start else 'local_budget_zero']
        t = time.perf_counter()
        try:
            fallback = self.conic()
        except numerical as exc:
            fallback = dict(ok=False, status='conic_numerical_failure',
                            exception_type=type(exc).__name__, detail=str(exc))
        conic_ms = 1000*(time.perf_counter()-t)
        t = time.perf_counter()
        # A rejected or stale warm point must never be returned as an answer.
        self.z = self.lam = None
        seed_status, seed_error = 'not_attempted', None
        if fallback['ok']:
            try:
                z, lam = self.seed(fallback)
                if not np.isfinite(z).all() or not np.isfinite(lam).all():
                    raise ValueError('nonfinite warm seed')
                self.z, self.lam = z, lam
                seed_status = 'ready'
            except numerical as exc:
                seed_status = 'numerical_failure'
                seed_error = dict(exception_type=type(exc).__name__, detail=str(exc))
        initialization_ms = 1000*(time.perf_counter()-t)
        return dict(ok=fallback['ok'], source='conic', total_ms=1000*(time.perf_counter()-begin),
                    local_ms=local_ms, update_ms=update_ms, conic_ms=conic_ms,
                    initialization_ms=initialization_ms, iterations=it, local_status=status,
                    local_assessment=verdict, rejected_local_point=point,
                    local_gate_results=gates, dispatch_reason=failed[0], failed_gates=failed,
                    seed_status=seed_status, seed_error=seed_error, conic=fallback,
                    z=None if self.z is None else self.z.copy())
