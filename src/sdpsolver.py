"""A small self-contained SDP solver.

Solves      min  c^T z
            s.t. A z = b
                 Z_p(z) = Z_p^0 + sum_i z_i Z_p^i  >= 0,   p = 1..P

Primal log-barrier path-following with equality-constrained Newton steps.
Phase I introduces a slack tau with blocks Z_p(z) + tau*I and minimises tau.

Not competitive with Mosek; it is here so the whole pipeline has no
external dependencies and every number can be traced.
"""
import numpy as np


class Block:
    """Affine matrix function Z(z) = Z0 + sum_i z_i Zi, stored densely."""

    def __init__(self, Z0, Zi):
        self.Z0 = np.asarray(Z0, float)
        Zi = np.asarray(Zi, float)               # (N, sz, sz)
        self.N = Zi.shape[0]
        self.sz = self.Z0.shape[0]
        # most variables do not appear in most blocks: keep only those that do
        self.idx = np.flatnonzero(np.any(Zi != 0.0, axis=(1, 2)))
        self.Zc = np.ascontiguousarray(Zi[self.idx])

    def dense_Zi(self):
        Z = np.zeros((self.N, self.sz, self.sz))
        Z[self.idx] = self.Zc
        return Z

    def value(self, z):
        if self.idx.size == 0:
            return self.Z0.copy()
        return self.Z0 + np.tensordot(z[self.idx], self.Zc, axes=(0, 0))


def _pd_margin(M, frac=1e-12):
    """Cheap strict-positive-definiteness test (no eigendecomposition)."""
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return False
    dmin = np.min(np.diag(L))
    dmax = np.max(np.diag(L))
    return dmin * dmin > frac * max(1.0, dmax * dmax)


def _chol_ok(M):
    try:
        np.linalg.cholesky(M)
        return True
    except np.linalg.LinAlgError:
        return False


class ConeProblem:
    def __init__(self, n_var, c, A, b, blocks):
        self.N = n_var
        self.c = np.asarray(c, float)
        self.A = np.asarray(A, float).reshape(-1, n_var)
        self.b = np.asarray(b, float).ravel()
        self.blocks = blocks

    # ---------- barrier value / gradient / Hessian ----------
    def _phi(self, z, t, tau=None, blocks=None):
        blocks = blocks if blocks is not None else self.blocks
        val = t * float(self.c @ z)
        for blk in blocks:
            Z = blk.value(z)
            L = np.linalg.cholesky(Z)          # raises if not PD
            val -= 2.0 * np.sum(np.log(np.diag(L)))
        return val

    def _grad_hess(self, z, t, blocks=None):
        blocks = blocks if blocks is not None else self.blocks
        g = t * self.c.copy()
        H = np.zeros((len(z), len(z)))
        for blk in blocks:
            Z = blk.value(z)
            try:
                L = np.linalg.cholesky(Z)
                W = np.linalg.inv(L)
                W = W.T @ W
            except np.linalg.LinAlgError:
                W = np.linalg.pinv(Z, hermitian=True)
            if blk.idx.size == 0:
                continue
            Ai = np.einsum('pq,iqr->ipr', W, blk.Zc)      # W @ Zi, active vars only
            g[blk.idx] -= np.einsum('ipp->i', Ai)
            Hb = np.einsum('ipq,jqp->ij', Ai, Ai)
            H[np.ix_(blk.idx, blk.idx)] += Hb
        return g, H

    # ---------- one centering solve ----------
    def _newton(self, z, t, blocks, max_iter=60, tol=1e-10,
                stop_fn=None, verbose=False):
        A, N = self.A, len(z)
        m = A.shape[0]
        for it in range(max_iter):
            if stop_fn is not None and stop_fn(z):
                return z, True
            g, H = self._grad_hess(z, t, blocks)
            reg = 1e-11 * max(1.0, np.max(np.abs(np.diag(H))))
            for _ in range(8):
                K = np.zeros((N + m, N + m))
                K[:N, :N] = H + reg * np.eye(N)
                if m:
                    K[:N, N:] = A.T
                    K[N:, :N] = A
                rhs = np.concatenate([-g, np.zeros(m)])
                try:
                    sol = np.linalg.solve(K, rhs)
                    break
                except np.linalg.LinAlgError:
                    reg *= 100
            else:
                sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
            dz = sol[:N]
            lam2 = float(-g @ dz)
            if not np.isfinite(lam2):
                return z, False
            if lam2 / 2.0 <= tol:
                return z, True
            # backtracking line search, staying inside the cone
            f0 = self._phi(z, t, blocks=blocks)
            step = 1.0
            for _ in range(60):
                zt = z + step * dz
                if all(_pd_margin(blk.value(zt)) for blk in blocks):
                    try:
                        if self._phi(zt, t, blocks=blocks) <= f0 - 0.25 * step * lam2:
                            break
                    except np.linalg.LinAlgError:
                        pass
                step *= 0.5
            else:
                return z, False
            z = z + step * dz
        return z, False

    # ---------- phase I ----------
    def _phase1(self, verbose=False):
        N = self.N
        z0 = np.linalg.lstsq(self.A, self.b, rcond=None)[0] if self.A.size else np.zeros(N)
        if self.A.size and np.linalg.norm(self.A @ z0 - self.b) > 1e-8 * max(1, np.linalg.norm(self.b)):
            raise RuntimeError("equality constraints inconsistent")
        lam_min = min(np.linalg.eigvalsh(blk.value(z0)).min() for blk in self.blocks)
        tau0 = max(0.0, -lam_min) + 1.0

        # augmented variable vector [z, tau]
        aug_blocks = []
        for blk in self.blocks:
            Zi = np.concatenate([blk.dense_Zi(), np.eye(blk.sz)[None, :, :]], axis=0)
            aug_blocks.append(Block(blk.Z0, Zi))
        Aa = np.hstack([self.A, np.zeros((self.A.shape[0], 1))]) if self.A.size else np.zeros((0, N + 1))
        ca = np.zeros(N + 1); ca[-1] = 1.0
        sub = ConeProblem(N + 1, ca, Aa, self.b, aug_blocks)

        za = np.concatenate([z0, [tau0]])
        t = 1.0
        stop = lambda v: v[-1] < -1e-7
        for _ in range(60):
            za, _ = sub._newton(za, t, aug_blocks, max_iter=50, stop_fn=stop)
            if za[-1] < -1e-7:
                return za[:N]
            t *= 8.0
        raise RuntimeError(f"phase I failed, tau={za[-1]:.3e}")

    # ---------- driver ----------
    def solve(self, eps=1e-9, mu=8.0, t0=1.0, verbose=False):
        z = self._phase1()
        m_tot = sum(blk.sz for blk in self.blocks)
        t = t0
        n_fail = 0
        while True:
            z, okc = self._newton(z, t, self.blocks, max_iter=200)
            if not okc:
                n_fail += 1
            if verbose:
                print(f"  t={t:9.2e}  gap<={m_tot/t:9.2e}  obj={float(self.c@z):.9f}")
            if m_tot / t < eps:
                break
            t *= mu
        self.n_centering_failures = n_fail
        return z, float(self.c @ z), m_tot / t
