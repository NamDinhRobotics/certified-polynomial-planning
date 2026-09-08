"""Rational, same-instance SDP bounds for CertifiedSpline exports.

All geometry, coordinates and multipliers are interpreted as exact binary
rationals. The affine spline basis and Bernstein/energy maps are constructed
over Q. The returned interval bounds the SDP optimum, not a collision-free
projected trajectory. Verification is offline in the timing experiment.
"""
from fractions import Fraction as F
from math import comb, nextafter, inf
import time
import numpy as np
from exact_api import rational_scalar, exact_service

from exact_green import q_gram_deriv, matmul, transpose, zeros, solve
from exact_check import (_sq_norm_poly, _add, _scale,
                         positive_on_unit_interval_fast)


def rational_array(a):
    return np.asarray([rational_scalar(v) for v in np.asarray(a,dtype=object).ravel()],
                      dtype=object).reshape(np.asarray(a).shape)


def positive_definite(A):
    """Exact unpivoted LDL test; a nonpositive pivot rejects the certificate."""
    A = [list(row) for row in A]
    n = len(A)
    if any(A[i][j] != A[j][i] for i in range(n) for j in range(i)):
        return False
    for k in range(n):
        p = A[k][k]
        if p <= 0:
            return False
        for i in range(k+1,n):
            for j in range(i,n):
                A[j][i] -= A[i][k]*A[j][k]/p
                A[i][j] = A[j][i]
    return True


def fraction_record(x):
    return [str(x.numerator), str(x.denominator)]


class ExactSDPBounds:
    def __init__(self, factor):
        self.factor = factor
        self.ms = factor.ms
        self.Q = factor.Qq
        self.G0 = factor.G0q
        d = factor.d
        self.S = np.zeros((2*d+1,d+1,d+1),dtype=object)
        self.T = np.zeros((2*d+1,d,d),dtype=object)
        for k in range(2*d+1):
            for i in range(d+1):
                j = k-i
                if 0 <= j <= d:
                    self.S[k,i,j] = F(comb(d,i)*comb(d,j),comb(2*d,k))
            for i in range(d):
                j = k-1-i
                if 0 <= j < d:
                    self.T[k,i,j] = F(comb(d-1,i)*comb(d-1,j),comb(2*d,k))
        G = np.asarray(q_gram_deriv(d,self.ms.k),dtype=object)
        tau = F(self.ms.N)**(2*self.ms.k-1)
        self.K = np.zeros((factor.r,factor.r),dtype=object)
        self.C = np.zeros((factor.n,factor.r),dtype=object)
        self.c0 = F(0)
        self.A = []
        for i in range(self.ms.N):
            Q = self.Q[self.ms.slice_(i)]
            G0 = self.G0[:,self.ms.slice_(i)]
            self.K += tau*(Q.T@G@Q)
            self.C += tau*(G0@G@Q)
            self.c0 += tau*np.sum((G0@G)*G0)
            self.A.extend([np.asarray([Q.T@S@Q for S in self.S],dtype=object)]*factor.m)
        self.A = np.asarray(self.A,dtype=object)
        if not positive_definite(self.K):
            raise ValueError('The exact reduced energy is not positive definite')

    def data(self, obstacles):
        B,c = [],[]
        for i in range(self.ms.N):
            qi = self.Q[self.ms.slice_(i)]
            gi = self.G0[:,self.ms.slice_(i)]
            for center,radius in obstacles:
                delta = gi-rational_array(center)[:,None]
                B.append(np.asarray([delta@S@qi for S in self.S],dtype=object))
                c.append([np.sum((delta@S)*delta)-rational_scalar(radius)**2 for S in self.S])
        return np.asarray(B,dtype=object),np.asarray(c,dtype=object)

    def lower(self, multipliers, obstacles):
        l = rational_array(multipliers).reshape(self.factor.b,self.factor.h)
        Z = self.K.copy()
        U,T = [],[]
        for b in range(self.factor.b):
            U.append(-sum((l[b,k]*self.S[k] for k in range(self.factor.h))))
            T.append(-sum((l[b,k]*self.T[k] for k in range(self.factor.h))))
            Z += sum((l[b,k]*self.A[b,k] for k in range(self.factor.h)))
        if not all(positive_definite(M) for M in [Z]+U+T):
            return None
        B,c = self.data(obstacles)
        h = self.C + sum((l[b,k]*B[b,k] for b in range(self.factor.b)
                         for k in range(self.factor.h)))
        invh = np.asarray(solve(Z.tolist(),h.T.tolist()),dtype=object).T
        return self.c0+np.sum(l*c)-np.sum(h*invh)

    def upper(self, z, obstacles, shift):
        """A strictly feasible lifted curve with W=Y'Y+vv'+shift*I.

        Adding shift changes only the gap, preserving the exact affine boundary
        data and objective functional; its objective value increases by shift*trace(K).
        It need not preserve a rank-one gap, and the
        amount and its exact energy cost are reported, never hidden.
        """
        Y,v,_,_ = self.factor.unpack(z)
        return self.upper_point(Y,v[None,:],obstacles,shift)

    def upper_point(self, Y, V, obstacles, shift):
        Y,V = rational_array(Y),rational_array(V)
        shift = rational_scalar(shift)
        if shift < 0:
            return None
        Gamma = self.G0+Y@self.Q.T
        lift_coeffs = V@self.Q.T
        n_polys = 0
        for i in range(self.ms.N):
            sl = self.ms.slice_(i)
            lift = _sq_norm_poly(lift_coeffs[:,sl].tolist(),self.factor.d)
            if shift:
                lift = _add(lift,_scale(_sq_norm_poly(self.Q[sl].T.tolist(),self.factor.d),shift))
            for center,radius in obstacles:
                delta = Gamma[:,sl]-rational_array(center)[:,None]
                p = _add(_sq_norm_poly(delta.tolist(),self.factor.d),lift)
                p[0] -= rational_scalar(radius)**2
                if not positive_on_unit_interval_fast(p):
                    return None
                n_polys += 1
        cost = self.c0+2*np.sum(self.C*Y)+np.sum((Y@self.K)*Y)+np.sum((V@self.K)*V)+shift*np.trace(self.K)
        return cost,n_polys

    @exact_service
    def verify(self, z, multipliers, obstacles, max_relative_gap=1e-5, point=None):
        start = time.perf_counter()
        lower = self.lower(multipliers,obstacles)
        if lower is None:
            return dict(ok=False,reason='dual_not_strictly_feasible',ms=1000*(time.perf_counter()-start))
        for shift in (0.,1e-12,1e-10,1e-8,1e-6):
            upper = self.upper(z,obstacles,shift) if point is None else self.upper_point(*point,obstacles,shift)
            if upper is not None:
                value,npolys = upper
                gap = (value-lower)/max(F(1),abs(value))
                return dict(ok=bool(0 <= gap <= rational_scalar(max_relative_gap)),
                            reason='bounds' if 0 <= gap <= rational_scalar(max_relative_gap) else 'gap_too_large',
                            lower_exact=fraction_record(lower),upper_exact=fraction_record(value),
                            lower=nextafter(float(lower),-inf),upper=nextafter(float(value),inf),
                            relative_gap_upper=nextafter(float(gap),inf),shift=shift,
                            n_polys=npolys,n_pd_blocks=1+2*self.factor.b,
                            ms=1000*(time.perf_counter()-start))
        return dict(ok=False,reason='no_strict_primal_witness',ms=1000*(time.perf_counter()-start))
