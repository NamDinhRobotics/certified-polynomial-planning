"""Offline six-candidate diagnostics; loss is relative to best candidate, not optimum."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import scipy.linalg as la
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from review_numpy_solver import model
from revision_common import write_json,provenance

def scan(f,lam):
 l=np.asarray(lam,float).reshape(f.b,f.h).copy();_,U,T,_=f.slacks(l)
 Uref=f.ms.Sd.sum(axis=0)/f.h;Tref=f.ms.Td.sum(axis=0)/f.h
 shifts=[]
 for i in range(f.b):
  shift=max(0.,1e-9-la.eigvalsh(U[i],Uref)[0],1e-9-la.eigvalsh(T[i],Tref)[0]);shifts.append(float(shift));l[i]-=shift/f.h
 Z,*_=f.slacks(l);e=la.eigvalsh(Z-f.K,f.K)[0];beta=min(1.,(1-1e-9)/(-e)) if e<0 else 1.;rows=[]
 for delta in (0.,1e-8,1e-7,1e-6,1e-5,1e-4):
  ll=beta*(1-delta)*l;Z,U,T,B=f.slacks(ll)
  try:
   val=f.c0+np.sum(ll*f.c)-np.sum(B*la.solve(Z,B.T,assume_a='pos').T)
   emin=[float(np.linalg.eigvalsh(a)[0]) for a in [Z]+list(U)+list(T)];valid=bool(np.isfinite(val) and min(emin)>0)
   rows.append(dict(delta=delta,scale=beta*(1-delta),lower=float(val),valid=valid,min_eigenvalues=emin,condition_Z=float(np.linalg.cond(Z))))
  except (la.LinAlgError,ValueError) as exc:rows.append(dict(delta=delta,valid=False,reason=type(exc).__name__))
 good=[(i,x) for i,x in enumerate(rows) if x['valid']];best=max(good,key=lambda ix:ix[1]['lower'])[0] if good else None
 if best is not None:
  for x in rows:
   if x['valid']:x['loss_from_best']=rows[best]['lower']-x['lower']
 return dict(shifts=shifts,beta=beta,candidates=rows,selected_index=best)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise FileExistsError(a.out)
 data=json.loads(a.input.read_text());f=model(data['rows'][0]['obstacles']);rows=[]
 for row in data['rows']:
  f.update(row['obstacles']);n=row['numerical'];points=[]
  if n['ok']:
   p=n.get('conic',n);points.append(('returned_root',p['lam'],p['bound']))
  rejected=n.get('rejected_local_point');v=n.get('local_assessment')
  if rejected and v and 'bound' in v:points.append(('rejected_local',rejected['lam'],v['bound']))
  for kind,lam,bound in points:
   s=scan(f,lam);i=s['selected_index']
   if bound['ok']:
    assert i is not None and abs(s['candidates'][i]['lower']-bound['lower'])<=1e-10*max(1,abs(bound['lower']))
   rows.append(dict(id=row['id'],kind=kind,**s))
 write_json(a.out,dict(complete=True,provenance=provenance([Path(__file__)]),definition='nonnegative numerical loss relative to largest valid bound among the same fixed six candidates; not true optimum',rows=rows));print(len(rows))
if __name__=='__main__':main()
