"""Positive-mode recovery with explicit exact cell certificates and budgets.

This module adds a new declared protocol. It does not change frozen experiments.
UNKNOWN concerns this sufficient construction, never original infeasibility.
"""
from fractions import Fraction as F
from math import comb
import time
import numpy as np
from exact_api import rational_scalar, exact_service
from escape_fold import rational, split, positive_lift, physical_certificate, green_representer
from constructive_recovery import encode, decode, fraction, exact_energy, coefficients


def bubble_mode(N, d):
    if d < 2:
        raise ValueError('Quadratic bubble requires degree at least two')
    out=[]
    for i in range(N):
        a,b,c=F(4*i,N)-F(4*i*i,N*N),F(4,N)-F(8*i,N*N),F(-4,N*N)
        out.append([a+b*F(j,d)+c*F(j*(j-1),d*(d-1)) for j in range(d+1)])
    return np.array(out,dtype=object)


def normalize(z):
    z=rational(z);N,d=z.shape[0],z.shape[1]-1
    v=sum(z[N//2,j]*F(comb(d,j),2**d) for j in range(d+1))
    if not v:return None
    z=z/v
    return z if positive_lift(z) else None


@exact_service
def safe_tail(G,z,obs,axis=1,sign=1,max_depth=24,max_nodes=10000):
    """Return alpha_safe and dyadic cell proof, with z(midpoint)=1.

    Caller establishes z is homogeneous and G affine feasible. A separate
    output checker verifies affine constraints. The returned cell cover proves
    all alpha >= alpha_safe safe for the original radii, using inflated radii.
    """
    start=time.perf_counter();G=rational(G)
    def fail(status,reason,**kw):
        return dict(ok=False,status=status,reason=reason,ms=1000*(time.perf_counter()-start),**kw)
    if axis not in range(G.shape[1]) or sign not in (-1,1):raise ValueError('direction')
    if z is None:return fail('UNKNOWN_SIGN','mode_unavailable')
    z=normalize(z)
    if z is None:return fail('UNKNOWN_SIGN','positive_mode_not_certified')
    axes=[a for a in range(G.shape[1]) if a!=axis]
    endpoints=[G[0,:,0],G[-1,:,-1]]
    if any(sum((p[a]-rational_scalar(c[a]))**2 for a in axes)<=rational_scalar(r)**2
           for c,r in obs for p in endpoints):
        return fail('UNKNOWN_GEOMETRY','prescribed_endpoint_line_intersects_ball')
    eps=F(1,2)
    for _ in range(256):
        if all(sum((p[a]-rational_scalar(c[a]))**2 for a in axes)>(rational_scalar(r)+eps)**2
               for c,r in obs for p in endpoints):break
        eps/=2
    else:return fail('UNKNOWN_BUDGET','inflation_bit_budget')
    proof=[];T=F(0);nodes=0
    for i,g in enumerate(G):
        for j,(c,r) in enumerate(obs):
            R=rational_scalar(r)+eps;stack=[(g-rational(c)[:,None],z[i],0,0)]
            while stack:
                dd,zz,depth,index=stack.pop();nodes+=1
                if nodes>max_nodes:return fail('UNKNOWN_BUDGET','cell_node_budget',nodes=nodes)
                transverse=sum((min(dd[a]) if min(dd[a])>0 else -max(dd[a]) if max(dd[a])<0 else F(0))**2 for a in axes)
                if transverse>R*R:
                    proof.append(dict(segment=i,obstacle=j,depth=depth,index=index,kind='transverse',bound=fraction(transverse)))
                    continue
                lower=min(zz)
                if lower>0:
                    longitudinal=min(sign*dd[axis]);t=max(F(0),(R-longitudinal)/lower);T=max(T,t)
                    proof.append(dict(segment=i,obstacle=j,depth=depth,index=index,kind='positive_mode',mode_lower=fraction(lower),longitudinal_lower=fraction(longitudinal),threshold=fraction(t)))
                    continue
                if depth>=max_depth:return fail('UNKNOWN_BUDGET','cell_depth_budget',nodes=nodes)
                dl,dr=split(dd);zl,zr=split(zz)
                stack.extend([(dr,zr,depth+1,2*index+1),(dl,zl,depth+1,2*index)])
    folded=G.copy();folded[:,axis,:]+=sign*T*z
    if not physical_certificate(folded,obs):
        raise AssertionError('Cell construction produced an unsafe curve')
    return dict(ok=True,status='CERTIFIED_TAIL',reason='all_amplitudes_at_least_threshold',z=z,Gamma=folded,
                amplitude=T,margin=eps,axis=axis,sign=sign,reference=G,proof=proof,nodes=nodes,
                ms=1000*(time.perf_counter()-start))


def workspace_interval(G,z,axis,sign,Fmat,gvec):
    """Exact closed amplitude interval from Bernstein control-hull inequalities."""
    G,z,Fmat,gvec=map(rational,(G,z,Fmat,gvec));lo,hi=F(0),None
    for segment,mode in zip(G,z):
        for point,height in zip(segment.T,mode):
            for normal,bound in zip(Fmat,gvec):
                slope=sign*height*normal[axis];rhs=bound-normal@point
                if not slope:
                    if rhs<0:return None
                elif slope>0:hi=rhs/slope if hi is None else min(hi,rhs/slope)
                else:lo=max(lo,rhs/slope)
    return None if hi is not None and lo>hi else (lo,hi)


def serialize_tail(t):
    if not t['ok']:return dict(t)
    return dict(ok=True,status=t['status'],axis=t['axis'],sign=t['sign'],amplitude=fraction(t['amplitude']),
                margin=fraction(t['margin']),z=encode(t['z']),reference=encode(t['reference']),
                cells=t['proof'],nodes=t['nodes'],ms=t['ms'])


@exact_service
def recover(G,modes,obs,k=1,axes=(1,),polish_steps=12,workspace=None):
    """Return least energy among exactly checked candidates actually tested.

    Includes strict physical checks; callers must not charge them a second
    time when reporting integrated totals. All modes use unit midpoint height.
    """
    begin=time.perf_counter();G=rational(G);J0=exact_energy(G,k)
    attempts=[];best=None
    if physical_certificate(G,obs):
        best=dict(Gamma=G,cost_exact=fraction(J0),cost=float(J0),method='plain',amplitude=fraction(F(0)))
        # The same plain-reference early return policy applies to every arm.
        if workspace is None or all(np.all(rational(workspace[0])@g<=rational(workspace[1])[:,None]) for g in G):
            return dict(ok=True,status='CERTIFIED_PHYSICAL',**{**best,'Gamma':encode(G)},attempts=[],plain=True,ms=1000*(time.perf_counter()-begin))
        best=None
    for name,z in modes:
        for axis in axes:
            for sign in (-1,1):
                t=safe_tail(G,z,obs,axis,sign);record=serialize_tail(t);record['mode']=name
                attempts.append(record)
                if not t['ok']:continue
                zz=t['z'];T=t['amplitude'];chosen=t['Gamma'];J=exact_energy(chosen,k);alpha=T
                if workspace is not None:
                    interval=workspace_interval(G,zz,axis,sign,*workspace)
                    if interval is None:
                        record['workspace_status']='UNKNOWN_WORKSPACE';continue
                    lo,hi=interval;lo=max(lo,T)
                    if hi is not None and lo>hi:
                        record['workspace_status']='UNKNOWN_WORKSPACE';continue
                    # Exact quadratic coefficients from three represented curves.
                    one=G.copy();one[:,axis,:]+=sign*zz
                    two=G.copy();two[:,axis,:]+=2*sign*zz
                    J1,J2=exact_energy(one,k),exact_energy(two,k)
                    a=(J2-2*J1+J0)/2;b=(J1-J0-a)/2
                    if a<=0:raise AssertionError('Positive-energy homogeneous mode required')
                    alpha=max(lo,-b/a)
                    if hi is not None:alpha=min(alpha,hi)
                    chosen=G.copy();chosen[:,axis,:]+=sign*alpha*zz;J=exact_energy(chosen,k)
                    record['workspace_interval']=[fraction(lo),None if hi is None else fraction(hi)]
                    assert all(np.all(rational(workspace[0])@g<=rational(workspace[1])[:,None]) for g in chosen)
                    assert physical_certificate(chosen,obs)
                else:
                    lo,hi=F(0),T
                    record['checkpoints']=[dict(steps=0,cost=float(J),elapsed_ms=1000*(time.perf_counter()-begin))]
                    for step in range(polish_steps):
                        x=(lo+hi)/2;candidate=G.copy();candidate[:,axis,:]+=sign*x*zz
                        if physical_certificate(candidate,obs):
                            hi=x;j=exact_energy(candidate,k)
                            if j<J:J,chosen,alpha=j,candidate,x
                        else:lo=x
                        if step+1 in (4,8,12):record['checkpoints'].append(dict(steps=step+1,cost=float(J),elapsed_ms=1000*(time.perf_counter()-begin)))
                record.update(candidate_cost=float(J),candidate_amplitude=fraction(alpha),elapsed_ms=1000*(time.perf_counter()-begin))
                if best is None or J<F(*map(int,best['cost_exact'])):
                    best=dict(Gamma=chosen,cost_exact=fraction(J),cost=float(J),method=name,axis=axis,sign=sign,amplitude=fraction(alpha))
    if best is None:
        statuses=[a['status'] for a in attempts];status='UNKNOWN_EXACT_CHECK' if 'UNKNOWN_EXACT_CHECK' in statuses else 'UNKNOWN_WORKSPACE' if workspace is not None else 'UNKNOWN_BUDGET' if 'UNKNOWN_BUDGET' in statuses else 'UNKNOWN_GEOMETRY' if 'UNKNOWN_GEOMETRY' in statuses else 'UNKNOWN_SIGN'
        return dict(ok=False,status=status,attempts=attempts,ms=1000*(time.perf_counter()-begin))
    assert physical_certificate(best['Gamma'],obs)
    best['Gamma']=encode(best['Gamma'])
    return dict(ok=True,status='CERTIFIED_PHYSICAL',**best,attempts=attempts,plain=False,ms=1000*(time.perf_counter()-begin))


def modes_for(f,exact,v=None):
    _,green=green_representer(f.Qq,exact.K,f.ms.N//2,f.d)
    gap=None if v is None else (rational(v)@f.Qq.T).reshape(f.ms.N,f.d+1)
    return dict(green=green,gap=gap,bubble=bubble_mode(f.ms.N,f.d) if f.ms.l==0 else None)
