"""Auditable physical-recovery harness for frozen new-geometry experiments.

The gap fold and Green correction are different coefficient-space modes.
All returned curves are exact rational splines and use a strict p>0 gate.
Polishing retains its feasible incumbent; it does not assume a connected
amplitude-feasible set and makes no global minimum-fold claim.
"""
from fractions import Fraction as F
from math import comb
import time
import numpy as np
from escape_fold import (rational,escape_fold,physical_certificate,green_representer)
from exact_green import q_gram_deriv


def encode(a):
    a=rational(a)
    return dict(shape=list(a.shape),values=[[str(x.numerator),str(x.denominator)] for x in a.flat])


def decode(a):
    return np.asarray([F(int(n),int(d)) for n,d in a['values']],dtype=object).reshape(a['shape'])


def fraction(x):return [str(x.numerator),str(x.denominator)]


def exact_energy(G,k=1):
    G=rational(G);N,n,h=G.shape
    H=F(N)**(2*k-1)*np.asarray(q_gram_deriv(h-1,k),dtype=object)
    return sum(np.sum((g@H)*g) for g in G)


def coefficients(f,Y):
    return (f.G0q+rational(Y)@f.Qq.T).reshape(f.n,f.ms.N,f.d+1).transpose(1,0,2)


def outcome(G,obs,k=1,**info):
    if G is None:return dict(ok=False,reason='no_candidate',**info)
    if not physical_certificate(G,obs):return dict(ok=False,reason='strict_gate_rejected',**info)
    J=exact_energy(G,k)
    return dict(ok=True,reason='strict_physical_certificate',Gamma=encode(G),cost=float(J),cost_exact=fraction(J),**info)


def endpoint_separation(G,obs,axis=1,margin=1e-8):
    G=rational(G);axes=[j for j in range(G.shape[1]) if j!=axis]
    margins=[]
    for center,radius in obs:
        c=rational(center);R=F(float(radius))+F(float(margin))
        for p in (G[0,:,0],G[-1,:,-1]):margins.append(sum((p[j]-c[j])**2 for j in axes)-R**2)
    return min(margins) if margins else F(1)


def recover_modes(f,exact,Y,v,obs,polish_steps=12,axis=1,margin=1e-8):
    """One measured call, including sign tests, both modes and exact checks."""
    start=time.perf_counter();G=coefficients(f,Y);plain=outcome(G,obs,f.ms.k,method='plain')
    if plain['ok']:
        plain.update(ms=1000*(time.perf_counter()-start),plain_strict=True,modes=[])
        return plain
    vg,zg=green_representer(f.Qq,exact.K,f.ms.N//2,f.d)
    modes=[('gap',None if v is None else (rational(v)@f.Qq.T).reshape(f.ms.N,f.d+1)),('green',zg)]
    candidates=[];records=[];J0=exact_energy(G,f.ms.k)
    H=F(f.ms.N)**(2*f.ms.k-1)*np.asarray(q_gram_deriv(f.d,f.ms.k),dtype=object)
    for name,z in modes:
        for sign in (-1,1):
            if z is None:
                records.append(dict(mode=name,sign=sign,ok=False,reason='no_gap'));continue
            res=escape_fold(G,z,obs,axis=axis,sign=sign,margin=margin)
            info={k:res[k] for k in ('ok','reason','ms','nodes','leaves','max_depth') if k in res}
            info.update(mode=name,sign=sign);records.append(info)
            if not res['ok']:continue
            zz=res['z'];T=res['amplitude']
            A=sum(G[i,axis]@H@zz[i] for i in range(f.ms.N))
            alpha=sum(zz[i]@H@zz[i] for i in range(f.ms.N))
            cost=lambda t:J0+2*sign*t*A+alpha*t*t
            best=(cost(T),res['Gamma'],T);lo,hi=F(0),T
            info['guaranteed_cost']=float(best[0]);info['amplitude_bound']=fraction(T)
            for _ in range(polish_steps):
                t=(lo+hi)/2;candidate=G.copy();candidate[:,axis,:]+=sign*t*zz
                if physical_certificate(candidate,obs):
                    hi=t
                    if cost(t)<best[0]:best=(cost(t),candidate,t)
                else:lo=t
            info['polished_cost']=float(best[0])
            candidates.append((best[0],best[1],name,sign,best[2]))
    if candidates:
        J,gg,name,sign,t=min(candidates,key=lambda x:x[0])
        ans=outcome(gg,obs,f.ms.k,method=name,sign=sign,amplitude=fraction(t))
        assert F(*map(int,ans['cost_exact']))==J
    else:ans=dict(ok=False,reason='sufficient_recovery_unknown',method='unknown')
    ans.update(ms=1000*(time.perf_counter()-start),plain_strict=False,modes=records)
    return ans
