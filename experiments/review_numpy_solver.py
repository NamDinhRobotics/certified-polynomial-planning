"""Same-SDP F2 comparison; protocol written before measured outcomes.

Three serial paired passes, with per-seed persistent graphs, alternating arm
order, independent warm states, all initialization/fallback/verification work
charged. Imports and object construction are measured separately. Exact SDP
bounds are replayed offline for both arms of the first complete pass.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import cvxpy as cp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from multisegment import MultiSegment
from certified_multisegment import CertifiedSpline
from exact_sdp_bounds import ExactSDPBounds
from exact_green import coefficient_blocks, elevated_min, endpoint_structure, energy_nullity
from mpc_scenarios import make_scenario, obstacles_at
from revision_common import write_json, provenance

FAMILY = dict(n=2,d=5,k=1,l=0,N=3,eta=2)


def witness():
    d,k,l,N,e = [FAMILY[x] for x in ('d','k','l','N','eta')]
    blocks,r = coefficient_blocks(d,k,l,N,e)
    rows = []
    for pair,B in blocks.items():
        minimum = elevated_min(B,d,96)
        rows.append(dict(pair=pair,minimum=str(minimum),nonnegative=minimum>=0,
                         nonzero=any(x!=0 for row in B for x in row)))
    endpoints = endpoint_structure(d,k,l,N,e)
    nullity = energy_nullity(d,k,l,N,e)
    return dict(family=FAMILY,r=r,D=96,blocks=rows,endpoint=endpoints,
                energy_nullity=nullity,
                passed=bool(nullity['definite'] and all(r['nonnegative'] and r['nonzero'] for r in rows)
                            and endpoints['closed_square_ok']))


def model(obs):
    return CertifiedSpline(MultiSegment(obstacles=obs,bc0=np.array([[-2.,0.]]),
                           bc1=np.array([[2.,0.]]),**FAMILY))


def conic_point(res):
    Y = np.asarray(res['Y'])
    gap = np.asarray(res['W'])-Y.T@Y
    e,U = np.linalg.eigh((gap+gap.T)/2)
    V = (U*np.sqrt(np.maximum(e,0.))).T
    return Y,V


def slim_fast(r):
    # Warm-state construction is timed, even though only the returned conic
    # point is used as the upper-bound candidate after a fallback.
    if r['source']=='conic':
        r.pop('z',None)
    return r


def verify_row(row, exact):
    obs = row['obstacles']
    ans = {}
    for arm in ('factor','reference'):
        a = row[arm]
        c = a.get('conic',a) if arm=='factor' else a
        if not a['ok']:
            ans[arm] = dict(ok=False,reason='online_no_answer')
            continue
        bound = c['bound']
        if arm=='factor' and a['source']=='factor':
            ans[arm] = exact.verify(a['z'],bound['multipliers'],obs)
        else:
            ans[arm] = exact.verify(None,bound['multipliers'],obs,point=conic_point(c))
    return ans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=ROOT/'artifacts/review_numpy_solver.json')
    ap.add_argument('--seeds',type=int,default=20)
    ap.add_argument('--steps',type=int,default=50)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--verify-existing',action='store_true')
    args = ap.parse_args()
    import mlukacs
    mlukacs._C_POLISH = False
    assert mlukacs._c_polish() is None
    if args.verify_existing:
        data = json.loads(args.out.read_text())
    else:
        protocol_path = args.out.with_suffix('.protocol.json')
        if args.out.exists() or protocol_path.exists():
            raise FileExistsError('Refusing to overwrite outcomes or protocol')
        protocol = dict(sos_polish_backend='NumPy; _C_POLISH=False enforced before construction',
            followup='Post-review replication on existing inputs; not held-out or replacement of frozen measurements',family=FAMILY,seeds=list(range(args.seeds)),steps=args.steps,
            repeats=args.repeats,dt=.1,arm_order='(repeat+seed+step) modulo 2',
            local_iterations=12,local_residual=1e-7,local_stationarity=1e-6,
            relative_gap=1e-5,conic_tolerance_ladder=[1e-9,1e-11],
            conic_parameter_graph_reused=True,independent_arm_state=True,
            imports_excluded=True,setup_reported_separately=True,
            first_step_and_fallback_charged=True,exact_replay='both arms, repeat 0, offline',
            scope='Same root SDP; no physical-trajectory feasibility or executed MPC claim',
            development_pilot='seeds 0 and 1, steps 0 through 19; disclosed overlap, no new population claim')
        write_json(protocol_path,protocol)
        w = witness()
        if not w['passed']:
            raise ValueError('Family witness failed')
        data = dict(protocol=protocol,protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                    family_witness=w,provenance=provenance([Path(__file__)]),rows=[],setup=[],complete=False)
        # Load the optional SOS-polish extension before the measured section.
        from mlukacs import _c_polish
        _c_polish()
        for repeat in range(args.repeats):
            for seed in range(args.seeds):
                sc = make_scenario(seed,n=2,n_obs=3)
                obs = obstacles_at(sc,0.)
                start = time.perf_counter(); f = model(obs); tf=1000*(time.perf_counter()-start)
                start = time.perf_counter(); ref = model(obs); tr=1000*(time.perf_counter()-start)
                data['setup'].append(dict(repeat=repeat,seed=seed,factor_ms=tf,reference_ms=tr))
                for step in range(args.steps):
                    obs = [(np.asarray(c,float),float(r)) for c,r in obstacles_at(sc,step*.1)]
                    arms = ('factor','reference') if (repeat+seed+step)%2==0 else ('reference','factor')
                    row = dict(repeat=repeat,seed=seed,step=step,obstacles=obs,order=arms)
                    for arm in arms:
                        if arm=='factor':
                            row[arm] = slim_fast(f.solve(obs))
                        else:
                            start = time.perf_counter();ref.update(obs);res=ref.conic()
                            res['total_ms'] = 1000*(time.perf_counter()-start)
                            row[arm] = res
                    data['rows'].append(row)
                print('measured',repeat,seed,'factor',sum(r['factor']['source']=='factor' for r in data['rows'][-args.steps:]),
                      '/',args.steps,flush=True)
        data['complete'] = True
        write_json(args.out,data)
    # Separate offline phase: no verifier disturbs the measured streams.
    first = data['rows'][0]
    exact = ExactSDPBounds(model(first['obstacles']))
    for i,row in enumerate(data['rows']):
        if row['repeat']==0:
            row['exact'] = verify_row(row,exact)
            if (i+1)%50==0:
                print('exact',i+1,'/',len(data['rows'])//data['protocol']['repeats'],flush=True)
    data['exact_complete'] = True
    write_json(args.out,data)
    print(args.out,flush=True)


if __name__=='__main__':
    main()
