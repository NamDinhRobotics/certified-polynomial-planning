"""Declared serial whole-service return timing, including required exact checks."""
from pathlib import Path
from fractions import Fraction as F
import argparse,json,sys,time,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from review_numpy_solver import model,conic_point
from exact_sdp_bounds import ExactSDPBounds,fraction_record
from constructive_recovery import coefficients
from q1_recovery import recover,modes_for
from q1_independent import affine,decode
from revision_common import clean,write_json,provenance

def service(f,exact,green,obs,budget=12,warm=True):
 start=time.perf_counter();stages={}
 numerical=f.solve(obs,local_iterations=budget,use_warm_start=warm)
 now=time.perf_counter();stages['root_ms']=1000*(now-start)
 a=dict(ok=False,status='NO_NUMERICAL_RESULT');bound=None
 if numerical['ok']:
  if numerical['source']=='factor':Y,v,*_=f.unpack(numerical['z']);point=None
  else:
   Y,V=conic_point(numerical['conic']);point=(Y,V);v=None
  a=recover(coefficients(f,Y),[('green',green)],obs,k=f.ms.k,axes=(1,))
  if a['ok']:affine(decode(a['Gamma']),f.ms.bc0 if hasattr(f.ms,'bc0') else [[-2,0]],f.ms.bc1 if hasattr(f.ms,'bc1') else [[2,0]],f.ms.eta)
 t=time.perf_counter();stages['physical_recovery_ms']=1000*(t-now)
 # A separately serialized physical-only return is a concrete service boundary.
 physical_payload=json.dumps(clean(dict(physical=a,numerical=numerical)),separators=(',',':'),allow_nan=False)
 physical_return=1000*(time.perf_counter()-start)
 exact_start=time.perf_counter()
 if numerical['ok']:
  c=numerical.get('conic',numerical)
  bound=exact.verify(numerical.get('z'),c['bound']['multipliers'],obs,point=point)
 exact_end=time.perf_counter();stages['exact_sdp_ms']=1000*(exact_end-exact_start)
 if a['ok'] and bound and bound['ok']:
  L=F(*map(int,bound['lower_exact']));J=F(*map(int,a['cost_exact']))
  if J<L:raise AssertionError('Physical energy below exact SDP lower bound')
  a['normalized_gap_upper']=float((J-L)/max(F(1),abs(L)))
 objective_ok=bool(a['ok'] and bound and bound['ok'])
 payload=json.dumps(clean(dict(physical=a,numerical=numerical,exact=bound)),separators=(',',':'),allow_nan=False)
 total=1000*(time.perf_counter()-start)
 overhead=total-sum(stages.values())
 if overhead < -1e-8 or physical_return>total:raise AssertionError('Disjoint timer accounting')
 return dict(numerical=numerical,arms={'green':a},exact=bound,physical_ok=a['ok'],objective_ok=objective_ok,
             physical_return_ms=physical_return,objective_return_ms=total,stages=stages,packing_overhead_ms=overhead,payload_bytes=len(payload.encode()))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--seeds',type=int,default=20);ap.add_argument('--steps',type=int,default=50);ap.add_argument('--ablation',action='store_true');args=ap.parse_args()
 if args.out.exists() or args.out.with_suffix('.protocol.json').exists():raise FileExistsError(args.out)
 import mlukacs;mlukacs._C_POLISH=False
 p=ROOT/'artifacts/revision_streaming_F2.json';old=json.loads(p.read_text())
 inputs=[r for r in old['rows'] if r['seed']<args.seeds and r['step']<args.steps]
 protocol=dict(scope='Exploratory serial replay of existing frozen F2 geometry; no held-out or executed-MPC claim',input_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),instances=len(inputs),mode='Green only, unit middle-segment midpoint; both vertical signs; 12 polishing steps',services=['strict physical coefficients + exact affine check + serialized payload','same physical output + full exact original-SDP interval + serialized payload'],setup='imports excluded; per-seed model/exact basis/Green measured separately; cold root compilation charged to first call',budget=[0,4,12] if args.ablation else [12],warm=[False,True] if args.ablation else [True],backend='NumPy SOS polish; BLAS single thread',provenance=provenance(list((ROOT/'experiments').glob('*.py'))))
 write_json(args.out.with_suffix('.protocol.json'),protocol);rows=[];setup=[]
 for budget in protocol['budget']:
  for warm in protocol['warm']:
   f=exact=green=None;seed=None
   for row in inputs:
    obs=row['obstacles']
    if seed!=row['seed']:
     seed=row['seed'];t=time.perf_counter();f=model(obs);exact=ExactSDPBounds(f);green=modes_for(f,exact)['green'];setup.append(dict(seed=seed,budget=budget,warm=warm,ms=1000*(time.perf_counter()-t)))
    result=service(f,exact,green,obs,budget,warm)
    rows.append(dict(id=f"b{budget}_w{int(warm)}_s{seed}_t{row['step']}",seed=seed,step=row['step'],budget=budget,warm=warm,obstacles=obs,family=dict(d=5,k=1,l=0,N=3,eta=2),bc0=[[-2,0]],bc1=[[2,0]],**result))
    if len(rows)%50==0:print('integrated',len(rows),flush=True)
 write_json(args.out,dict(protocol=protocol,complete=True,setup=setup,rows=rows));print(args.out)
if __name__=='__main__':main()
