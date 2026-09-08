"""Disclosed adaptation of iterative separating-cut/QP post-processing.

Based on Graesdal et al. (arXiv:2606.14063v1, VIII-B), not their full RRT
implementation. Fixed grid refinement, spherical normals, exact final gate.
"""
import time
from fractions import Fraction as F
import numpy as np
import cvxpy as cp
from bernstein import bd
from constructive_recovery import coefficients,outcome


def refine(f,Y,obs,grids=(21,41,81)):
 start=time.perf_counter();current=coefficients(f,Y);best=outcome(current,obs,f.ms.k,method='plain');trace=[]
 if not best['ok']:best=None
 for count in grids:
  B=np.array([bd(s,f.d) for s in np.linspace(0,1,count)]).T
  V=cp.Variable((f.n,f.r));G=f.G0+V@f.Q.T;rows=[];rhs=[]
  for i in range(f.ms.N):
   points=np.asarray(current[i],float)@B
   for c,r in obs:
    c=np.asarray(c);normals=points-c[:,None];length=np.linalg.norm(normals,axis=0)
    normals[:,length<1e-12]=np.eye(f.n)[:,1,None]
    normals/=np.linalg.norm(normals,axis=0)
    for j,b in enumerate(B.T):
     row=np.zeros((f.n,f.ms.M));row[:,f.ms.slice_(i)]=normals[:,j,None]*b
     rows.append(row.ravel());rhs.append(float(r)+1e-8+normals[:,j]@c)
  obj=f.c0+2*cp.sum(cp.multiply(f.C,V))+sum(cp.quad_form(V[j],cp.psd_wrap(f.K)) for j in range(f.n))
  problem=cp.Problem(cp.Minimize(obj),[np.array(rows)@cp.vec(G,order='C')>=rhs])
  try:problem.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=600)
  except cp.error.SolverError:
   trace.append(dict(grid=count,status='solver_error',ok=False,elapsed_ms=1000*(time.perf_counter()-start)));continue
  a=dict(ok=False)
  if problem.status=='optimal' and V.value is not None:
   current=coefficients(f,V.value);a=outcome(current,obs,f.ms.k,method='iterated_cut_qp')
   if a['ok'] and (best is None or F(*map(int,a['cost_exact']))<F(*map(int,best['cost_exact']))):best=a
  trace.append(dict(grid=count,status=problem.status,ok=a['ok'],cost=a.get('cost'),incumbent_cost=None if best is None else best['cost'],elapsed_ms=1000*(time.perf_counter()-start)))
 return dict(**(best or {'ok':False,'reason':'NO_CERTIFIED_CANDIDATE'}),trace=trace,ms=1000*(time.perf_counter()-start))
