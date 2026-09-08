"""Frozen exploratory reference/mode/direction study on all 240 existing cases."""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from revision_recovery_generalization import root_problem
from exact_sdp_bounds import ExactSDPBounds
from constructive_recovery import decode,coefficients
from q1_recovery import recover,modes_for
from q1_prior import refine
from revision_common import write_json,provenance

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--limit',type=int);args=ap.parse_args()
 if args.out.exists() or args.out.with_suffix('.protocol.json').exists():raise FileExistsError(args.out)
 import mlukacs;mlukacs._C_POLISH=False
 names=['src/q1_recovery.py','src/q1_prior.py','experiments/q1_factorial.py']
 priorpath=ROOT/'artifacts/revision_recovery_generalization.json';data=json.loads(priorpath.read_text());inputs=data['rows'][:args.limit]
 protocol=dict(scope='Exploratory on previously inspected frozen geometries; no held-out claim',source_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},input_sha256=hashlib.sha256(priorpath.read_bytes()).hexdigest(),instances=len(inputs),normalization='unit value at middle-segment midpoint',directions='signed vertical in factorial; signed axes ordered vertical,x,then z in axis arm',polishing=12,cell_depth=24,cell_nodes=10000,construction_margin='dyadic halving from 1/2 until exact endpoint-line tests pass; max256 decreases',plain_policy='same exact plain-reference early return in each arm',arms=['affine_bubble','affine_green','sdp_bubble','sdp_green','sdp_gap','sdp_selector','sdp_green_allaxes','iterated_cut_qp','workspace_sdp_green'],workspace='Bernstein box: x in [-3,3], other coordinates [-2,2]; only guaranteed tail, exact energy minimization in certified interval',prior='Section VIII-B-inspired adaptation: grids21,41,81; sampled radial halfspaces; original energy; exact gate and incumbent; no full RRT/original-implementation claim',timing='serial rotated arms; archived root reference acquisition excluded and separately identified; all recovery/gate costs included; no original-vs-new latency ratio')
 protocol['provenance']=provenance(list((ROOT/'experiments').glob('*.py')))
 write_json(args.out.with_suffix('.protocol.json'),protocol);rows=[]
 for idx,old in enumerate(inputs):
  t=time.perf_counter();f=root_problem(old);exact=ExactSDPBounds(f);Y=decode(old['reference_Y']);v=decode(old['gap_v']) if old['gap_v'] else None
  modes=modes_for(f,exact,v);gsdp=coefficients(f,Y);gaff=coefficients(f,np.zeros((f.n,f.r)))
  setup=1000*(time.perf_counter()-t);arms={};order=protocol['arms'][idx%9:]+protocol['arms'][:idx%9]
  for arm in order:
   if arm=='iterated_cut_qp':a=refine(f,Y,old['obstacles'])
   else:
    reference=gaff if arm.startswith('affine') else gsdp
    selected=[('gap',modes['gap']),('green',modes['green'])] if arm=='sdp_selector' else [(arm.split('_')[-1] if arm!='sdp_green_allaxes' else 'green',modes['bubble' if 'bubble' in arm else 'gap' if arm=='sdp_gap' else 'green'])]
    axes=[1]+[j for j in range(f.n) if j!=1] if arm.endswith('allaxes') else [1]
    workspace=(np.r_[np.eye(f.n),-np.eye(f.n)],np.r_[[3]+[2]*(f.n-1),[3]+[2]*(f.n-1)]) if arm.startswith('workspace') else None
    a=recover(reference,selected,old['obstacles'],k=f.ms.k,axes=axes,workspace=workspace)
   if a['ok']:
    J=F(*map(int,a['cost_exact']));L=F(*map(int,old['lower_exact']));a['normalized_gap_upper']=float((J-L)/max(F(1),abs(L)))
   arms[arm]=a
  rows.append(dict(id=old['id'],stratum=old['stratum'],n=f.n,obstacles=old['obstacles'],family=dict(d=5,k=1,l=0,N=3,eta=2),bc0=[[-2]+[0]*(f.n-1)],bc1=[[2]+[0]*(f.n-1)],vertical_condition=old['condition_pass'],lower_exact=old['lower_exact'],setup_ms=setup,arm_order=order,arms=arms))
  if (idx+1)%10==0:print('factorial',idx+1,'/',len(inputs),flush=True)
 write_json(args.out,dict(protocol=protocol,complete=True,rows=rows));print(args.out)
if __name__=='__main__':main()
