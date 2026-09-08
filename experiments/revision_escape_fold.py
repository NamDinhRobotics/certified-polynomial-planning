"""Full first-pass replay of constructive recovery, no new SDP solve.
Pilot: first 30 rows inspected before this fixed 1,000-row protocol.
All final physical certificates use strict p>0 and zero tolerance.
"""
import sys,json,hashlib,time,argparse
from pathlib import Path
from fractions import Fraction as F
from math import comb
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from revision_certified_solver import model
from escape_fold import rational,escape_fold,physical_certificate
from exact_sdp_bounds import ExactSDPBounds


def frac(a):return [[str(x.numerator),str(x.denominator)] for x in np.asarray(a,dtype=object).flat]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=1000);ap.add_argument('--out',type=Path,default=ROOT/'artifacts/revision_escape_fold.json');args=ap.parse_args()
    source=ROOT/'artifacts/revision_certified_solver.json'
    protocol=dict(source_sha256=sha(source),population='repeat=0; seed=0..19; step=0..49',limit=args.limit,
        pilot='first 30 rows already inspected; this is retrospective replay, not held-out validation',
        directions=[-1,1],margin=1e-8,max_depth=24,max_nodes=10000,
        polish='12 exact feasibility bisections per direction, retain each feasible incumbent; no claim of connected feasible set or global fold optimality',
        gate='strict distance squared minus radius squared > 0, exact rational, zero tolerance',
        comparison='same-instance old streaming curves independently rechecked at zero tolerance; old acceptance uses +1e-9 and is shown separately',
        timing='recovery only; includes exact gate, direction checks and polish; excludes artifact/model loading and source solves; single replay',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/escape_fold.py',ROOT/'src/exact_check.py',ROOT/'src/exact_sdp_bounds.py']})
    if args.out.exists() or args.out.with_suffix('.protocol.json').exists():raise FileExistsError('Output already exists')
    args.out.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2))
    data=json.loads(source.read_text());rows=[r for r in data['rows'] if r['repeat']==0][:args.limit]
    old=json.loads((ROOT/'artifacts/revision_streaming_F2.json').read_text())
    oldrows={(r['seed'],r['step']):r for r in old['rows']}
    f=model(rows[0]['obstacles']);ex=ExactSDPBounds(f);output=[]
    for row in rows:
        start=time.perf_counter();a=row['factor'];obs=row['obstacles']
        if a['source']=='factor':Y,v,*_=f.unpack(np.asarray(a['z']))
        else:
            c=a['conic'];Y=np.asarray(c['Y']);S=np.asarray(c['W'])-Y.T@Y
            w,U=np.linalg.eigh((S+S.T)/2);v=U[:,-1]*np.sqrt(max(0,w[-1]))
        Y,v=rational(Y),rational(v)
        G=(f.G0q+Y@f.Qq.T).reshape(2,3,6).transpose(1,0,2)
        z=(v@f.Qq.T).reshape(3,6)
        J0=ex.c0+2*np.sum(ex.C*Y)+np.sum((Y@ex.K)*Y)
        plain=physical_certificate(G,obs)
        rec=dict(seed=row['seed'],step=row['step'],source=a['source'],plain_strict=plain)
        if plain:
            best=(J0,G,F(0),0);rec.update(method='plain',guaranteed_cost=float(J0))
        else:
            attempts=[];candidates=[];guaranteed=[]
            orient=1 if sum(z[0,k]*comb(5,k) for k in range(6))>0 else -1
            scale=max(abs(x) for x in z.flat)
            vn=orient*v/scale if scale else v
            A=(ex.C+Y@ex.K)@vn;alpha=vn@ex.K@vn
            for sign in (-1,1):
                res=escape_fold(G,z,obs,sign=sign)
                attempts.append({k:res[k] for k in ('ok','reason','ms','nodes','max_depth') if k in res})
                if not res['ok']:continue
                T=res['amplitude'];cost=lambda t:J0+2*sign*t*A[1]+alpha*t*t
                guaranteed.append(cost(T));inc=(cost(T),res['Gamma'],T,sign)
                lo,hi=F(0),T
                for _ in range(12):
                    t=(lo+hi)/2;candidate=G.copy();candidate[:,1,:]+=sign*t*res['z']
                    if physical_certificate(candidate,obs):
                        hi=t
                        if cost(t)<inc[0]:inc=(cost(t),candidate,t,sign)
                    else:lo=t
                candidates.append(inc)
            rec['attempts']=attempts
            best=min(candidates,key=lambda t:t[0]) if candidates else None
            rec.update(method='escape' if best else 'unknown',guaranteed_cost=float(min(guaranteed)) if guaranteed else None)
        rec['recovery_ms']=1000*(time.perf_counter()-start)
        rec['ok']=best is not None
        if best:
            J,gg,t,sign=best
            L=F(*map(int,row['exact']['factor']['lower_exact']))
            rec.update(cost=float(J),cost_exact=frac([J])[0],lower_exact=frac([L])[0],certified_relative_suboptimality=float((J-L)/max(F(1),abs(L))),
                       Gamma_shape=list(gg.shape),Gamma_exact=frac(gg),amplitude=frac([t])[0],sign=sign)
            assert J>=L
        oldr=oldrows[(row['seed'],row['step'])];rec['old']={}
        for name,b in oldr['arms'].items():
            if 'Gamma' in b and b['Gamma'] is not None:
                gg=rational(b['Gamma']).reshape(2,3,6).transpose(1,0,2)
                strict=physical_certificate(gg,obs)
            else:strict=False
            rec['old'][name]=dict(accepted=b['accepted'],strict=strict,cost=b.get('cost'))
        output.append(rec)
        if len(output)%100==0:print(len(output),'ok',sum(x['ok'] for x in output),'escape',sum(x['method']=='escape' for x in output),flush=True)
    args.out.write_text(json.dumps(dict(protocol=protocol,rows=output,complete=len(output)==args.limit),indent=2))
    print(args.out,flush=True)

if __name__=='__main__':run()
