"""Constructive sufficient recovery by sign-definite lift and safe endpoint lines.

Exact rational Bernstein arithmetic throughout construction and verification.
This is a sufficient-condition procedure; a budget failure means UNKNOWN.
No claim of universal fold completeness, bounded control effort, or fixed latency.
"""
from fractions import Fraction as F
from math import comb
import time
import numpy as np
from exact_api import rational_scalar, exact_service
from exact_check import _sq_norm_poly, bern_to_power, positive_on_unit_interval_fast


def rational(a):
    a = np.asarray(a, dtype=object)
    return np.asarray([rational_scalar(v) for v in a.flat],dtype=object).reshape(a.shape)


def split(a):
    """Exact midpoint de Casteljau, last axis is Bernstein degree."""
    a = np.asarray(a,dtype=object).copy()
    left,right = [a[...,0]],[a[...,-1]]
    while a.shape[-1]>1:
        a = (a[...,:-1]+a[...,1:])/2
        left.append(a[...,0]);right.append(a[...,-1])
    return np.stack(left,axis=-1),np.stack(right[::-1],axis=-1)


def positive_lift(z):
    """Certify z>0 except permitted zeros at the two global endpoints."""
    z=rational(z)
    for i,c in enumerate(z):
        d=len(c)-1
        nz=[k for k,x in enumerate(c) if x]
        if not nz:return False
        a,b=nz[0],d-nz[-1]
        if (a and i!=0) or (b and i!=len(z)-1):return False
        q=[c[k+a]*F(comb(d,k+a),comb(d-a-b,k)) for k in range(d-a-b+1)]
        if not positive_on_unit_interval_fast(bern_to_power(q,d-a-b)):return False
    return True


def physical_certificate(Gamma,obstacles):
    """Strict distance squared minus radius squared > 0, with ZERO tolerance."""
    Gamma=rational(Gamma)
    for g in Gamma:
        for center,radius in obstacles:
            p=_sq_norm_poly((g-rational(center)[:,None]).tolist(),g.shape[-1]-1)
            p[0]-=rational_scalar(radius)**2
            if not positive_on_unit_interval_fast(p):return False
    return True


def _box_distance_squared(coeffs,axes):
    out=F(0)
    for j in axes:
        lo,hi=min(coeffs[j]),max(coeffs[j])
        out+= (lo if lo>0 else (-hi if hi<0 else F(0)))**2
    return out


@exact_service
def escape_fold(Gamma,z,obstacles,axis=1,sign=1,margin=1e-8,max_depth=24,max_nodes=10000):
    """Return rational amplitude, normalized lift, curve and finite cell proof.

    Gamma shape (segments, ambient dimension, degree+1), z (segments,degree+1).
    Input must already satisfy required affine constraints. Adding z e preserves
    them when z belongs to their homogeneous nullspace. The caller checks this.
    Endpoint transverse separation is tested using inflated radii r+margin.
    """
    start=time.perf_counter()
    def fail(reason,**kw):return dict(ok=False,reason=reason,ms=1000*(time.perf_counter()-start),**kw)
    Gamma,z=rational(Gamma),rational(z)
    if Gamma.ndim!=3 or z.shape!=(Gamma.shape[0],Gamma.shape[2]):raise ValueError('shape')
    if sign not in (-1,1) or not 0<=axis<Gamma.shape[1] or margin<=0:raise ValueError('direction/margin')
    # Orientation uses the value at one segment midpoint; exact verification follows.
    d=z.shape[1]-1
    mid=sum(z[0,k]*comb(d,k) for k in range(d+1))
    if mid<0:z=-z
    scale=max(abs(x) for x in z.flat)
    if not scale:return fail('zero_lift')
    z=z/scale
    if not positive_lift(z):return fail('lift_sign_not_certified')
    axes=[j for j in range(Gamma.shape[1]) if j!=axis]
    for center,radius in obstacles:
        c=rational(center);R=rational_scalar(radius)+rational_scalar(margin)
        for p in (Gamma[0,:,0],Gamma[-1,:,-1]):
            if sum((p[j]-c[j])**2 for j in axes)<=R*R:return fail('endpoint_lines_not_separated')
    amplitude=F(0);nodes=0;leaves=0;depth_used=0
    for i,g in enumerate(Gamma):
        for center,radius in obstacles:
            delta=g-rational(center)[:,None];R=rational_scalar(radius)+rational_scalar(margin)
            stack=[(delta,z[i],0)]
            while stack:
                dd,zz,depth=stack.pop();nodes+=1;depth_used=max(depth_used,depth)
                if nodes>max_nodes:return fail('budget_unknown',nodes=nodes)
                if _box_distance_squared(dd,axes)>R*R:
                    leaves+=1;continue
                lower=min(zz)
                if lower>0:
                    amplitude=max(amplitude,(R-min(sign*dd[axis]))/lower)
                    leaves+=1;continue
                if depth>=max_depth:return fail('budget_unknown',nodes=nodes)
                dl,dr=split(dd);zl,zr=split(zz)
                stack.extend([(dr,zr,depth+1),(dl,zl,depth+1)])
    folded=Gamma.copy();folded[:,axis,:]+=sign*amplitude*z
    ok=physical_certificate(folded,obstacles)
    return dict(ok=ok,reason='strict_physical_certificate' if ok else 'internal_verification_failure',
                amplitude=amplitude,z=z,Gamma=folded,axis=axis,sign=sign,
                nodes=nodes,leaves=leaves,max_depth=depth_used,ms=1000*(time.perf_counter()-start))


def green_representer(Q,K,segment,degree,parameter=F(1,2)):
    """Exact unit-height minimum-energy homogeneous mode at a live parameter.

    Global sign follows from a previously certified strictly positive Green
    kernel; callers still check the actual reconstructed polynomial sign.
    Q contains stacked Bernstein rows and K=Q.T H Q must be positive definite.
    """
    from exact_green import solve
    Q,K=rational(Q),rational(K)
    s=F(parameter);b=np.asarray([comb(degree,i)*s**i*(1-s)**(degree-i) for i in range(degree+1)],dtype=object)
    u=Q[segment*(degree+1):(segment+1)*(degree+1)].T@b
    v=np.asarray(solve(K.tolist(),u[:,None].tolist()),dtype=object)[:,0]
    diagonal=u@v
    if diagonal<=0:raise ValueError('nonpositive Green diagonal')
    v=v/diagonal
    return v,(v@Q.T).reshape(-1,degree+1)
