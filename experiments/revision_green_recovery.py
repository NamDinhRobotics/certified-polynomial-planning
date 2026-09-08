"""Second, explicitly adaptive study: add a canonical Green recovery mode.
Run after observing 19 sign-check failures and a large energy tail in escape_fold.
The first protocol and all first-stage outcomes remain unchanged.
"""
import sys,json,time,hashlib
from fractions import Fraction as F
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from revision_certified_solver import model
from revision_escape_fold import frac,sha
from escape_fold import rational,escape_fold,physical_certificate,green_representer
from exact_sdp_bounds import ExactSDPBounds
from exact_green import constraint_matrix,q_gram_deriv


def decode(values,shape):return np.asarray([F(int(a),int(b)) for a,b in values],dtype=object).reshape(shape)


def run():
    first=ROOT/'artifacts/revision_escape_fold.json';source=ROOT/'artifacts/revision_certified_solver.json';out=ROOT/'artifacts/revision_green_recovery.json'
    if out.exists() or out.with_suffix('.protocol.json').exists():raise FileExistsError('Existing outcomes')
    protocol=dict(first_stage_sha256=sha(first),source_sha256=sha(source),population='all first-pass 1000 instances',
        adaptive_reason='first-stage 19 sign failures and high cost tail; no held-out claim',
        mode='Green representer at midpoint of segment 1, unit height; both vertical directions; 12 exact bisections',
        selection='retain already strict plain curves; otherwise compare gap-mode and Green-mode candidates by exact cost',
        margin=1e-8,gate='strict p>0 with zero tolerance',timing='additional Green recovery, one replay; excludes loading, setup and final independent audit',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'src/escape_fold.py',ROOT/'experiments/revision_escape_fold.py']})
    out.with_suffix('.protocol.json').write_text(json.dumps(protocol,indent=2))
    old=json.loads(first.read_text());data=json.loads(source.read_text());original={(r['seed'],r['step']):r for r in data['rows'] if r['repeat']==0}
    f=model(data['rows'][0]['obstacles']);ex=ExactSDPBounds(f)
    v,z=green_representer(f.Qq,ex.K,1,5)
    scale=max(abs(x) for x in z.flat);vn=v/scale
    Aeq=np.asarray(constraint_matrix(5,0,3,2),dtype=object).T
    Gk=np.asarray(q_gram_deriv(5,1),dtype=object)*3
    assert np.all((v@f.Qq.T)@Aeq==0)
    results=[]
    for rec in old['rows']:
        row=original[(rec['seed'],rec['step'])];a=row['factor'];obs=row['obstacles'];rec=dict(rec)
        start=time.perf_counter();rec['first_method']=rec['method'];rec['gap_mode_cost']=rec.get('cost')
        if not rec['plain_strict']:
            Y=rational(f.unpack(np.asarray(a['z']))[0] if a['source']=='factor' else a['conic']['Y'])
            G=(f.G0q+Y@f.Qq.T).reshape(2,3,6).transpose(1,0,2)
            J0=ex.c0+2*np.sum(ex.C*Y)+np.sum((Y@ex.K)*Y)
            A=(ex.C+Y@ex.K)@vn;alpha=vn@ex.K@vn;candidates=[];attempts=[]
            for sign in (-1,1):
                res=escape_fold(G,z,obs,sign=sign)
                attempts.append({k:res[k] for k in ('ok','reason','ms','nodes','max_depth') if k in res})
                if not res['ok']:continue
                cost=lambda t:J0+2*sign*t*A[1]+alpha*t*t
                T=res['amplitude'];inc=(cost(T),res['Gamma'],T,sign);lo,hi=F(0),T
                for _ in range(12):
                    t=(lo+hi)/2;gg=G.copy();gg[:,1,:]+=sign*t*res['z']
                    if physical_certificate(gg,obs):
                        hi=t
                        if cost(t)<inc[0]:inc=(cost(t),gg,t,sign)
                    else:lo=t
                candidates.append(inc)
            rec['green_attempts']=attempts
            if candidates:
                J,gg,t,sign=min(candidates,key=lambda x:x[0]);rec['green_mode_cost']=float(J)
                previous=F(*map(int,rec['cost_exact'])) if rec['ok'] else None
                if previous is None or J<previous:
                    L=F(*map(int,row['exact']['factor']['lower_exact']))
                    rec.update(ok=True,method='green',cost=float(J),cost_exact=frac([J])[0],lower_exact=frac([L])[0],
                        certified_relative_suboptimality=float((J-L)/max(F(1),abs(L))),Gamma_shape=list(gg.shape),Gamma_exact=frac(gg),amplitude=frac([t])[0],sign=sign)
        rec['green_ms']=1000*(time.perf_counter()-start);rec['combined_recovery_ms']=rec['recovery_ms']+rec['green_ms']
        if rec['ok']:
            gg=decode(rec['Gamma_exact'],rec['Gamma_shape'])
            assert physical_certificate(gg,obs)
            full=gg.transpose(1,0,2).reshape(2,18)
            assert np.all(full@Aeq==np.asarray(f.ms.B,dtype=object))
            actual=sum(np.sum((g@Gk)*g) for g in gg)
            assert actual==F(*map(int,rec['cost_exact']))
            assert actual>=F(*map(int,rec['lower_exact']))
            rec['independent_replay']=True
        results.append(rec)
        if len(results)%100==0:print(len(results),'ok',sum(r['ok'] for r in results),'green',sum(r['method']=='green' for r in results),flush=True)
    out.write_text(json.dumps(dict(protocol=protocol,rows=results,complete=True,green_v_exact=frac(v)),indent=2))
    print(out,flush=True)

if __name__=='__main__':run()
