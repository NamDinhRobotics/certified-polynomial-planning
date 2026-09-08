"""Frozen, new-geometry F2 recovery comparison, including outside-condition cases.

No outcome from this population is inspected before --freeze writes both the
full input population and protocol hashes. --run uses exactly that record.
The angular fold and geometric comparators are explicit multisegment/3-D
adaptations of the older algorithms, not reruns of their original timings.
"""
import argparse,hashlib,json,sys,time,warnings
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import cvxpy as cp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from bernstein import bd
from multisegment import MultiSegment
from certified_multisegment import CertifiedSpline
from exact_sdp_bounds import ExactSDPBounds
from constructive_recovery import (rational,encode,decode,fraction,exact_energy,coefficients,
                                   outcome,endpoint_separation,recover_modes)
from escape_fold import physical_certificate
from revision_certified_solver import witness
from revision_common import write_json

FAMILY=dict(d=5,k=1,l=0,N=3,eta=2)
STRATA=['clutter_2d','endpoint_2d','clutter_3d','endpoint_3d','outside_2d','outside_3d']
MARGIN=1e-8


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def generate(count):
    rows=[]
    for group,name in enumerate(STRATA):
        n=2 if name.endswith('2d') else 3
        for index in range(count):
            seed=910000+1000*group+index;rng=np.random.default_rng(seed)
            center=np.zeros(n);center[0]=rng.uniform(-.25,.25);center[1:]=rng.uniform(-.06,.06,n-1)
            obs=[(center,rng.uniform(.30,.42))]
            for _ in range(4):
                c=np.r_[rng.uniform(-1.2,1.2),rng.uniform(-.45,.45,n-1)]
                obs.append((c,rng.uniform(.18,.32)))
            if name.startswith('endpoint'):
                for j,side in [(1,-1),(2,1)]:
                    r=rng.uniform(.15,.30);delta=10**rng.uniform(-4,-.8)
                    c=np.zeros(n);c[0]=side*(2-r-delta);c[1:]=rng.uniform(-.2*r,.2*r,n-1)
                    obs[j]=(c,r)
            if name.startswith('outside'):
                c=np.zeros(n);c[0]=-2.;c[1]=rng.uniform(.4,.7)
                obs[1]=(c,.2) # safe endpoint, but its vertical line intersects this ball
            rows.append(dict(id=f'{name}/{index:02d}',stratum=name,n=n,seed=seed,
                             obstacles=[(c.tolist(),float(r)) for c,r in obs]))
    return rows


def model(row,cuts=None):
    n=row['n'];a=np.zeros((1,n));b=a.copy();a[0,0]=-2;b[0,0]=2
    return MultiSegment(n=n,obstacles=row['obstacles'],bc0=a,bc1=b,cuts=cuts,**FAMILY)


def root_problem(row):return CertifiedSpline(model(row))


def angular_fold(f,Y,v,obs):
    start=time.perf_counter();G=coefficients(f,Y);ans=outcome(G,obs,method='plain')
    if ans['ok']:ans['ms']=1000*(time.perf_counter()-start);return ans
    if v is None:return dict(ok=False,reason='no_gap',ms=1000*(time.perf_counter()-start))
    z=(rational(v)@f.Qq.T).reshape(3,6);scale=max(abs(x) for x in z.flat)
    if not scale:return dict(ok=False,reason='zero_gap',ms=1000*(time.perf_counter()-start))
    z=z/scale;n=f.n
    if n==2:
        theta=2*np.pi*np.arange(32)/32;directions=np.c_[np.cos(theta),np.sin(theta)]
    else:
        q=np.arange(32);zz=1-2*(q+.5)/32;theta=np.pi*(3-np.sqrt(5))*q
        directions=np.c_[np.sqrt(1-zz**2)*np.cos(theta),np.sqrt(1-zz**2)*np.sin(theta),zz]
    directions=np.vstack([directions,np.eye(n),-np.eye(n)])
    W=np.vstack([directions*t for t in np.geomspace(1e-4,16.,48)])
    B=np.array([bd(s,5) for s in np.linspace(0,1,201)]).T
    points=np.concatenate([np.asarray(g,float)@B for g in G],axis=1).T
    lift=np.concatenate([np.asarray(a,float)@B for a in z])
    P=points[None,:,:]+W[:,None,:]*lift[None,:,None]
    feasible=np.ones(len(W),dtype=bool)
    for c,r in obs:feasible&=np.min(np.sum((P-np.asarray(c))**2,axis=2),axis=1)>(r+MARGIN)**2
    valid=np.flatnonzero(feasible)
    if len(valid):
        # Exact energy coefficients would be costly for every sampled candidate;
        # float energy only orders candidates. Final exported cost is exact.
        Gflat=G.transpose(1,0,2).reshape(n,18);zf=z.ravel();H=np.kron(np.eye(3),f.ms.Gk)*f.ms.time_scale
        A=np.asarray(Gflat,float)@H@np.asarray(zf,float);alpha=float(np.asarray(zf,float)@H@np.asarray(zf,float))
        J=f.ms.true_cost(np.asarray(Gflat,float))+2*W@A+alpha*np.sum(W*W,axis=1)
        order=valid[np.argsort(J[valid])[:8]]
        candidates=[]
        for idx in order:
            gg=G+rational(W[idx])[None,:,None]*z[:,None,:]
            result=outcome(gg,obs,method='angular_gap_fold')
            if result['ok']:candidates.append(result)
        ans=min(candidates,key=lambda a:F(*map(int,a['cost_exact']))) if candidates else dict(ok=False,reason='exact_gate_rejected_all')
    else:ans=dict(ok=False,reason='finite_search_no_candidate')
    ans.update(ms=1000*(time.perf_counter()-start),sample_candidates=len(W),sample_feasible=int(len(valid)))
    return ans


def onecut(f,Y,obs,row):
    start=time.perf_counter();G=coefficients(f,Y);ans=outcome(G,obs,method='plain')
    if ans['ok']:ans.update(ms=1000*(time.perf_counter()-start),child_solves=0);return ans
    ss=np.linspace(0,1,201);B=np.array([bd(s,5) for s in ss]).T;worst=None
    for i,g in enumerate(G):
        pts=np.asarray(g,float)@B
        for j,(c,r) in enumerate(obs):
            vals=np.sum((pts-np.asarray(c)[:,None])**2,axis=0)-r*r
            q=int(np.argmin(vals));key=(float(vals[q]),i,j,q)
            if worst is None or key[0]<worst[0]:worst=key
    _,i,j,q=worst;c,r=obs[j];pts=np.asarray(G[i],float)@B
    inside=np.flatnonzero(np.sum((pts-np.asarray(c)[:,None])**2,axis=0)-r*r<0)
    arc=np.linspace(ss[inside[0]],ss[inside[-1]],5) if len(inside)>1 else [ss[q]]
    a=np.zeros(f.n);a[1]=1.;candidates=[];statuses=[]
    for side in (-1,1):
        cuts=[(i,float(t),a,float(c[1])+side*(r+MARGIN),side) for t in arc]
        child=model(row,cuts);child.Gamma0=f.G0.copy();child.Nperp=f.Q.copy()
        res=child.solve();statuses.append(res['status'])
        if res.get('converged'):
            candidate=outcome(coefficients(f,res['Y']),obs,method='onecut')
            if candidate['ok']:candidates.append(candidate)
    ans=min(candidates,key=lambda a:F(*map(int,a['cost_exact']))) if candidates else dict(ok=False,reason='no_strict_child')
    ans.update(ms=1000*(time.perf_counter()-start),child_solves=2,statuses=statuses)
    return ans


def guide_qp(f,obs):
    start=time.perf_counter();grid=np.linspace(0,1,21);transforms=[]
    def subdivide(P,t):
        P=P.copy();left=[P[:,0]];right=[P[:,-1]]
        while P.shape[1]>1:
            P=(1-t)*P[:,:-1]+t*P[:,1:];left.append(P[:,0]);right.append(P[:,-1])
        return np.column_stack(left),np.column_stack(right[::-1])
    for lo,hi in zip(grid[:-1],grid[1:]):
        _,r=subdivide(np.eye(6),lo);l,_=subdivide(r,(hi-lo)/(1-lo));transforms.append(l)
    candidates=[];statuses=[]
    for sign in (-1,1):
        Y=cp.Variable((f.n,f.r));G=f.G0+Y@f.Q.T;rows=[];bounds=[]
        for i in range(3):
            tt=(i+.5*(grid[:-1]+grid[1:]))/3;guide=np.zeros((f.n,20))
            guide[0]=-2+4*tt;guide[1]=sign*.8*4*tt*(1-tt)
            for center,radius in obs:
                normals=guide-np.asarray(center)[:,None];norms=np.linalg.norm(normals,axis=0)
                fallback=np.zeros(f.n);fallback[1]=sign
                normals[:,norms<1e-12]=fallback[:,None]
                normals/=np.maximum(np.linalg.norm(normals,axis=0),1e-12)
                for q,T in enumerate(transforms):
                    for b in T.T:
                        r=np.zeros((f.n,18));r[:,f.ms.slice_(i)]=normals[:,q,None]*b
                        rows.append(r.ravel());bounds.append(radius+MARGIN+normals[:,q]@center)
        obj=f.c0+2*cp.sum(cp.multiply(f.C,Y))+sum(cp.quad_form(Y[j],cp.psd_wrap(f.K)) for j in range(f.n))
        prob=cp.Problem(cp.Minimize(obj),[np.asarray(rows)@cp.vec(G,order='C')>=np.asarray(bounds)])
        try:
            prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=600)
            statuses.append(prob.status)
        except cp.error.SolverError:statuses.append('solver_error');continue
        if prob.status=='optimal' and Y.value is not None:
            result=outcome(coefficients(f,Y.value),obs,method='guide_qp')
            if result['ok']:candidates.append(result)
    ans=min(candidates,key=lambda a:F(*map(int,a['cost_exact']))) if candidates else dict(ok=False,reason='no_strict_guide')
    ans.update(ms=1000*(time.perf_counter()-start),statuses=statuses)
    return ans


def freeze(path,count):
    if path.exists():raise FileExistsError(path)
    sources=[Path(__file__),ROOT/'src/constructive_recovery.py',ROOT/'src/escape_fold.py',ROOT/'src/certified_multisegment.py',ROOT/'src/exact_sdp_bounds.py',ROOT/'src/exact_check.py',ROOT/'src/multisegment.py']
    record=dict(protocol=dict(per_stratum=count,strata=STRATA,family=FAMILY,margin=MARGIN,
        population='new seeds 910000+1000*stratum+index; fixed inputs before outcomes',
        recovery='gap and middle-segment-midpoint Green modes, both vertical signs, 12 exact bisections, depth cap 24, node cap 10000',
        angular_fold='adapted a46 grid mechanism; 32 planar/Fibonacci directions plus signed axes; 48 log amplitudes [1e-4,16] after lift normalization; 201 samples/segment; exact check up to eight lowest float-cost sampled candidates; no local descent',
        onecut='adapted five-point commitment on deepest sampled violated segment/ball; 201 samples; both vertical signs; two child SDPs',
        guide='adapted two mirrored parabolic guides, height .8, 20 Bernstein cells per segment, same energy and exact affine basis',
        gate='strict physical p>0, rational coefficients, zero tolerance; requested positive margin 1e-8 where construction imposes a margin',
        arm_order='rotate [constructive,angular_fold,onecut,guide_qp] by global index modulo four',
        timing='one serial pass; root and setup reported separately; arm includes construction and exact gate; dual verification and independent replay reported separately',
        no_answer='retain every case and reason; root failure uses affine minimum-energy reference for constructive/cut methods; no gap means angular fold unknown',
        bounds='exact repaired root dual if valid, otherwise exact obstacle-free lower bound 16',
        selection='no filtering by success, sign, rank, objective gap, or endpoint condition'),
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sources},rows=generate(count))
    write_json(path,record);print('FROZEN',path,digest(path),len(record['rows']),flush=True)


def run(pop_path,out,limit=None):
    if out.exists():raise FileExistsError(out)
    data=json.loads(pop_path.read_text())
    for name,h in data['source_hashes'].items():
        assert digest(ROOT/name)==h,'Frozen source changed: '+name
    selected=data['rows'] if limit is None else data['rows'][:limit]
    rows=[];family_witness=witness();assert family_witness['passed']
    for idx,row in enumerate(selected):
        t=time.perf_counter();f=root_problem(row);ex=ExactSDPBounds(f);setup_ms=1000*(time.perf_counter()-t)
        t=time.perf_counter();root=f.conic();root_ms=1000*(time.perf_counter()-t)
        if root['ok']:
            Y=np.asarray(root['Y']);gap=np.asarray(root['W'])-Y.T@Y;e,U=np.linalg.eigh((gap+gap.T)/2);v=U[:,-1]*np.sqrt(max(0,e[-1]))
        else:Y=np.zeros((f.n,f.r));v=None
        t=time.perf_counter();L=ex.c0;bound_kind='obstacle_free'
        if root.get('bound',{}).get('ok'):
            candidate=ex.lower(root['bound']['multipliers'],row['obstacles'])
            if candidate is not None and candidate>L:L=candidate;bound_kind='repaired_dual'
        bound_ms=1000*(time.perf_counter()-t)
        actions=dict(constructive=lambda:recover_modes(f,ex,Y,v,row['obstacles']),angular_fold=lambda:angular_fold(f,Y,v,row['obstacles']),
                     onecut=lambda:onecut(f,Y,row['obstacles'],row),guide_qp=lambda:guide_qp(f,row['obstacles']))
        names=list(actions);names=names[idx%4:]+names[:idx%4];arms={}
        for name in names:
            arm_start=time.perf_counter()
            try:arms[name]=actions[name]()
            except (ArithmeticError,ValueError) as exc:arms[name]=dict(ok=False,reason='verification_unknown',error=repr(exc),ms=None)
            arms[name]['ms']=1000*(time.perf_counter()-arm_start)
            if arms[name]['ok']:
                J=F(*map(int,arms[name]['cost_exact']));assert J>=L
                arms[name]['relative_gap_upper']=float((J-L)/max(F(1),abs(L)))
        G=coefficients(f,Y);separation=endpoint_separation(G,row['obstacles'])
        root_slim={k:root[k] for k in ('ok','status','cost','residual','relative_gap','Y','W','bound') if k in root}
        rec=dict(**row,root=root_slim,reference_Y=encode(Y),gap_v=None if v is None else encode(v),setup_ms=setup_ms,root_ms=root_ms,bound_ms=bound_ms,
                 lower_exact=fraction(L),lower_kind=bound_kind,separation_exact=fraction(separation),condition_pass=separation>0,arm_order=names,arms=arms)
        rows.append(rec)
        if len(rows)%10==0 or len(rows)==len(selected):
            write_json(out,dict(population_sha256=digest(pop_path),population_file=pop_path.name,protocol=data['protocol'],source_hashes=data['source_hashes'],
                                family_witness=family_witness,rows=rows,complete=len(rows)==len(selected),limited_smoke=limit is not None))
            print(len(rows),row['stratum'],{a:sum(r['arms'][a]['ok'] for r in rows) for a in actions},flush=True)
    print(out,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--freeze',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--count',type=int,default=40)
    ap.add_argument('--population',type=Path,default=ROOT/'artifacts/revision_recovery_population.json');ap.add_argument('--out',type=Path,default=ROOT/'artifacts/revision_recovery_generalization.json');ap.add_argument('--limit',type=int)
    args=ap.parse_args();warnings.filterwarnings('ignore')
    if args.freeze:freeze(args.population,args.count)
    if args.run:run(args.population,args.out,args.limit)
