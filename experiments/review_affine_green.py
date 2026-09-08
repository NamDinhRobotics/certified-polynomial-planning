"""Post-review direct-Green and feasible-seed corridor follow-up; frozen inputs unchanged.
One serial pass, no native acceleration, no SDP solve for candidate generation.
The seeded QP is a disclosed one-step safe-corridor adaptation, not a reproduction
of SIP or BMTP convergence algorithms. Every incumbent passes an exact strict gate.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse,hashlib,json,sys,time
import numpy as np
import cvxpy as cp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from revision_recovery_generalization import root_problem
from constructive_recovery import recover_modes,coefficients,decode,encode,endpoint_separation,outcome,exact_energy
from exact_sdp_bounds import ExactSDPBounds
from exact_green import constraint_matrix
from bernstein import bd
from revision_common import write_json

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def refine(f,seed,obs):
    start=time.perf_counter()
    if not seed['ok']:return dict(ok=False,reason='no_feasible_seed',ms=1000*(time.perf_counter()-start))
    Gs=decode(seed['Gamma']);Y=cp.Variable((f.n,f.r));G=f.G0+Y@f.Q.T;rows=[];bounds=[]
    def subdivide(P,t):
        P=P.copy();left=[P[:,0]];right=[P[:,-1]]
        while P.shape[1]>1:
            P=(1-t)*P[:,:-1]+t*P[:,1:];left.append(P[:,0]);right.append(P[:,-1])
        return np.column_stack(left),np.column_stack(right[::-1])
    grid=np.linspace(0,1,21)
    for i in range(3):
        for lo,hi in zip(grid[:-1],grid[1:]):
            _,right=subdivide(np.eye(6),lo);T,_=subdivide(right,(hi-lo)/(1-lo))
            point=np.asarray(Gs[i],float)@bd((lo+hi)/2,5)
            for c,r in obs:
                c=np.asarray(c);a=point-c;a/=np.linalg.norm(a)
                for b in T.T:
                    row=np.zeros((f.n,18));row[:,f.ms.slice_(i)]=a[:,None]*b
                    rows.append(row.ravel());bounds.append(float(r)+1e-8+a@c)
    obj=f.c0+2*cp.sum(cp.multiply(f.C,Y))+sum(cp.quad_form(Y[j],cp.psd_wrap(f.K)) for j in range(f.n))
    prob=cp.Problem(cp.Minimize(obj),[np.asarray(rows)@cp.vec(G,order='C')>=np.asarray(bounds)])
    chosen=dict(seed);accepted_update=False;status='not_solved'
    try:
        prob.solve(solver='CLARABEL',tol_gap_abs=1e-9,tol_gap_rel=1e-9,tol_feas=1e-9,max_iter=600)
        status=prob.status
        if prob.status=='optimal' and Y.value is not None:
            a=outcome(coefficients(f,Y.value),obs,method='seeded_corridor')
            if a['ok'] and F(*map(int,a['cost_exact']))<=F(*map(int,seed['cost_exact'])):chosen=a;accepted_update=True
    except cp.error.SolverError:status='solver_error'
    chosen.update(ms=1000*(time.perf_counter()-start),accepted_update=accepted_update,qp_status=status)
    return chosen

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'artifacts/review_affine_green.json');ap.add_argument('--verify',action='store_true');args=ap.parse_args()
    poppath=ROOT/'artifacts/revision_recovery_population.json';oldpath=ROOT/'artifacts/revision_recovery_generalization.json'
    pop=json.loads(poppath.read_text());old=json.loads(oldpath.read_text())
    if args.verify:
        data=json.loads(args.out.read_text());assert data['population_sha256']==sha(poppath)
        count=0;polys=0
        for row,src,prior in zip(data['rows'],pop['rows'],old['rows']):
            assert row['id']==src['id']==prior['id'] and row['obstacles']==src['obstacles']
            f=root_problem(src);L=F(*map(int,prior['lower_exact']));A=np.asarray(constraint_matrix(5,0,3,2),dtype=object).T
            for name in ['direct','refined']:
                a=row[name]
                if a['ok']:
                    G=decode(a['Gamma']);assert outcome(G,src['obstacles'])['ok'];J=exact_energy(G)
                    assert G.transpose(1,0,2).reshape(src['n'],18)@A is not None
                    assert np.all(G.transpose(1,0,2).reshape(src['n'],18)@A==np.asarray(f.ms.B,dtype=object))
                    assert J==F(*map(int,a['cost_exact'])) and J>=L
                    if name=='refined':assert J<=F(*map(int,row['direct']['cost_exact']))
                    count+=1;polys+=15
        assert len(data['rows'])==240
        # A deliberate constant colliding curve must fail the strict predicate.
        G=np.zeros((3,2,6),dtype=object);assert not outcome(G,[([0.,0.],.3)])['ok']
        report=dict(passed=True,instances=240,accepted_curves=count,strict_polynomials=polys,affine_and_energy_checks=count,collision_mutation_rejected=True)
        write_json(args.out.with_suffix('.verification.json'),report);print(report);return
    protocol=args.out.with_suffix('.protocol.json')
    if args.out.exists() or protocol.exists():raise FileExistsError(args.out)
    sources=[Path(__file__),ROOT/'src/constructive_recovery.py',ROOT/'src/escape_fold.py',ROOT/'src/exact_green.py',ROOT/'src/certified_multisegment.py']
    cfg=dict(scope='Exploratory post-review ablation on all original frozen inputs; no retuning; not held-out',
        population_sha256=sha(poppath),assessment_sha256=sha(oldpath),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sources},
        generation='minimum-energy affine particular, Y=0, no gap, canonical midpoint Green, vertical signs,12 bisections,margin1e-8,exact strict gate',
        refinement='one seed-oriented supporting-halfspace QP;20 cells/segment;original energy;strict gate;retain certified seed on failure or higher energy;not full SIP/BMTP',
        timing='serial pass; exact model/setup and direct recovery separately and jointly; refinement incremental; original SDP point never used for generation; archived lower bounds only for offline assessment',
        axes='test every coordinate axis for eligibility only; do not alter vertical generation')
    write_json(protocol,cfg);rows=[]
    for index,(src,prior) in enumerate(zip(pop['rows'],old['rows'])):
        t=time.perf_counter();f=root_problem(src);ex=ExactSDPBounds(f);setup=1000*(time.perf_counter()-t)
        def forbidden(*a,**kw):raise AssertionError('SDP solve forbidden for direct candidate generation')
        f.conic=forbidden
        Y=np.zeros((f.n,f.r));t=time.perf_counter();direct=recover_modes(f,ex,Y,None,src['obstacles']);elapsed=1000*(time.perf_counter()-t)
        refined=refine(f,direct,src['obstacles']);base=coefficients(f,Y)
        axes=[bool(endpoint_separation(base,src['obstacles'],axis=a)>0) for a in range(f.n)]
        L=F(*map(int,prior['lower_exact']))
        for a in [direct,refined]:
            if a['ok']:
                J=F(*map(int,a['cost_exact']));a['relative_gap_upper']=float((J-L)/max(F(1),abs(L)))
                G=decode(a['Gamma']);a['control_box_abs_bound']=float(max(abs(x) for x in G.flat))
        rows.append(dict(**src,eligible=prior['condition_pass'],axis_eligible=axes,setup_ms=setup,direct_ms=elapsed,direct_total_ms=setup+elapsed,direct=direct,refined=refined))
        if (index+1)%20==0:print(index+1,flush=True)
    write_json(args.out,dict(protocol=cfg,population_sha256=sha(poppath),rows=rows,complete=True));print(args.out)
if __name__=='__main__':main()
