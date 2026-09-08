"""Illustrative 3-D application of the paper's constructive recovery.

This is a new, disclosed demonstration, not an extension of the frozen benchmark.
The smooth reference is a rest-to-rest C4 degree-7 spline; its derivative cost
is snap energy. No actuator-feasibility theorem or hardware flight is claimed.
"""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import argparse, hashlib, json, sys, time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
from exact_green import (coefficient_blocks, elevated_min, endpoint_structure,
                         energy_nullity, constraint_matrix)
from multisegment import MultiSegment
from certified_multisegment import CertifiedSpline
from exact_sdp_bounds import ExactSDPBounds
from constructive_recovery import encode, decode, coefficients, exact_energy
from escape_fold import green_representer, escape_fold, physical_certificate, positive_lift
from exact_check import bern_to_power, positive_on_unit_interval_fast

OUT = ROOT/'paper_mpc/demo3d'
FAMILY = dict(d=7, k=4, l=4, N=5, eta=4)
BODY_RADIUS = .23
TRACKING_RESERVE = .12
EXTRA_CLEARANCE = .04


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def evaluate(G, s, order=0):
    """Derivative with respect to normalized total time, including segment scale."""
    G = np.asarray(G, dtype=float)
    nseg, _, h = G.shape
    for k in range(order):
        G = np.diff(G, axis=2)*(h-1-k)*nseg
    s = np.asarray(s, dtype=float)
    i = np.minimum((np.clip(s, 0, 1)*nseg).astype(int), nseg-1)
    u = s*nseg-i; d = G.shape[2]-1
    B = np.array([comb(d,k)*u**k*(1-u)**(d-k) for k in range(d+1)]).T
    return np.einsum('...nd,...d->...n', G[i], B)


def family_certificate():
    blocks, r = coefficient_blocks(**FAMILY)
    rows = [dict(pair=list(pair), minimum=str(elevated_min(B,7,96)),
                 nonzero=any(x for row in B for x in row)) for pair,B in blocks.items()]
    ends = endpoint_structure(**FAMILY); energy = energy_nullity(**FAMILY)
    assert all(F(x['minimum'])>=0 and x['nonzero'] for x in rows)
    assert ends['closed_square_ok'] and ends['n_live_endpoints']==8 and energy['definite']
    return dict(family=FAMILY, r=r, degree_elevation=96, blocks=rows,
                endpoint_structure=ends, energy_nullity=energy, passed=True)


def collision_witness(G, obs):
    """An exact rational negative sample proves a collision; dense clearance is separate."""
    for i,g in enumerate(G):
        d=g.shape[-1]-1
        for k in range(1,80):
            s=F(k,80); b=np.array([F(comb(d,j))*s**j*(1-s)**(d-j) for j in range(d+1)],object)
            x=g@b
            for j,(c,r) in enumerate(obs):
                delta=x-np.array([F(float(a)) for a in c],object)
                clearance=sum(a*a for a in delta)-F(float(r))**2
                if clearance<0:
                    return dict(segment=i, parameter=str(s), obstacle=j,
                                squared_clearance_exact=str(clearance), point=list(map(float,x)))
    return None


def duration_bounds(G):
    """Conservative derivative bounds from convex hulls, with exact squared norms."""
    bounds={}; dd=G.copy(); N=len(G); degree=G.shape[-1]-1
    for k in range(1,5):
        dd=np.diff(dd,axis=2)*(degree+1-k)*N
        q=max(sum(x*x for x in dd[i,:,j]) for i in range(N) for j in range(dd.shape[-1]))
        bounds[k]=q
    # Fixed limits for this illustrative generic robot, not fitted to a flight.
    T=F(5)
    while not (bounds[1] <= F(2)**2*T**2 and bounds[2] <= F('2.5')**2*T**4
               and bounds[3] <= F(5)**2*T**6):
        T+=F(1,2)
    return T,{str(k):dict(normalized_squared_bound=str(q),
                         physical_bound=float(q)**.5/float(T)**k) for k,q in bounds.items()}


def fresh_demo():
    cert=family_certificate()
    start=np.array([-3.,0.,1.35]);goal=np.array([3.,0.,1.65])
    bc0=np.zeros((5,3));bc1=bc0.copy();bc0[0]=start;bc1[0]=goal
    # A dummy obstacle is needed only by the numerical model container.
    model=MultiSegment(n=3,obstacles=[([0.,0.,1.5],.4)],bc0=bc0,bc1=bc1,**FAMILY)
    f=CertifiedSpline(model);ex=ExactSDPBounds(f)
    _,za=green_representer(f.Qq,ex.K,1,7)
    _,zb=green_representer(f.Qq,ex.K,3,7)
    _,z=green_representer(f.Qq,ex.K,2,7)
    G=coefficients(f,np.zeros((3,f.r)))
    # A disclosed guide offset supplies a genuinely spatial inspection route.
    # It does not add any pinned interior constraints to the spline space.
    G[:,1,:]+=F(7,4)*(za-zb)
    grid=np.linspace(0,1,2001)
    progress=np.interp([-1.7,-.9,0.,.9,1.7],evaluate(G,grid)[:,0],grid)
    centers=evaluate(G,progress)
    centers+=np.array([[.04,.08,.0],[-.03,-.12,.08],[0.,.12,-.05],[.02,-.12,.02],[-.02,.06,-.04]])
    physical_radii=np.array([.34,.39,.44,.38,.32])
    planning_radii=physical_radii+BODY_RADIUS+TRACKING_RESERVE+EXTRA_CLEARANCE
    obstacles=[(c.tolist(),float(r)) for c,r in zip(centers,planning_radii)]
    t=time.perf_counter()
    rec=escape_fold(G,z,obstacles,axis=2,sign=1,margin=.01)
    assert rec['ok'],rec
    best=rec['Gamma'];lo=F(0);hi=rec['amplitude'];zz=rec['z']
    # A fixed 14-step feasible-incumbent polish; not a global minimum-fold search.
    for _ in range(14):
        a=(lo+hi)/2;c=G.copy();c[:,2,:]+=a*zz
        if physical_certificate(c,obstacles):hi=a;best=c
        else:lo=a
    construct_ms=1000*(time.perf_counter()-t)
    A=np.asarray(constraint_matrix(7,4,5,4),object).T
    exact_B=np.array([[F(float(x)) for x in row] for row in model.B],object)
    assert np.array_equal(best.transpose(1,0,2).reshape(3,40)@A,exact_B)
    assert physical_certificate(best,obstacles) and positive_lift(z)
    # A visual floor is present, so check it independently, including robot radius.
    assert all(positive_on_unit_interval_fast(bern_to_power(list(g[2]-F(str(BODY_RADIUS+TRACKING_RESERVE))),7)) for g in best)
    bad=collision_witness(G,obstacles);assert bad
    T,bounds=duration_bounds(best)
    s=np.linspace(0,1,2001);path=evaluate(best,s);ref=evaluate(G,s)
    clearance=np.min(np.linalg.norm(path[:,None,:]-centers[None,:,:],axis=2)-physical_radii[None,:]-BODY_RADIUS,axis=1)
    singular=np.linalg.svd(path-path.mean(axis=0),compute_uv=False)
    assert singular[-1]>.02, 'Demo path accidentally planar'
    return dict(id='inspection_quadrotor',label='3D inspection / rest-to-rest',family=FAMILY,
        family_certificate=cert,reference_Gamma=encode(G),recovered_Gamma=encode(best),
        lift=encode(zz),amplitude=str(hi),construction_nodes=rec['nodes'],construction_ms=construct_ms,
        physical_obstacles=[dict(center=c.tolist(),radius=float(r)) for c,r in zip(centers,physical_radii)],
        planning_obstacles=obstacles,body_radius=BODY_RADIUS,tracking_reserve=TRACKING_RESERVE,
        extra_clearance=EXTRA_CLEARANCE,duration=float(T),duration_exact=str(T),derivative_bounds=bounds,
        reference_collision_witness=bad,strict_reference_certificate=True,exact_affine_certificate=True,
        snap_cost_normalized_exact=str(exact_energy(best,4)),reference_snap_cost_normalized_exact=str(exact_energy(G,4)),
        no_obstacle_lower_bound_normalized_exact=str(ex.c0),
        samples=dict(s=s.tolist(),reference=ref.tolist(),recovered=path.tolist(),body_clearance_sampled=clearance.tolist()),
        spatial_singular_values=singular.tolist(),scope='Illustrative new scene. Exact reference clearance and affine/smoothness checks; derivative hull bounds. No optimum claim, no hardware flight, no certified tracking-error bound.')


def recorded_demo():
    source=ROOT/'artifacts/revision_recovery_generalization.json'
    data=json.loads(source.read_text())
    # First eligible 3-D non-plain success in original record order, not best-case cost.
    from revision_recovery_generalization import root_problem
    for row in data['rows']:
        if not (row['n']==3 and row['condition_pass'] and row['arms']['constructive']['ok']
                and row['arms']['constructive']['method']!='plain'):continue
        f=root_problem(row);G=coefficients(f,decode(row['reference_Y']));obs=row['obstacles']
        bad=collision_witness(G,obs)
        if bad:break
    else:raise ValueError('No recorded 3D case has the requested exact collision witness')
    recovered=decode(row['arms']['constructive']['Gamma'])
    assert physical_certificate(recovered,obs)
    s=np.linspace(0,1,1601)
    # Coordinate permutation and translation are display-only rigid isometries.
    def world(p):return np.asarray(p)[:,[0,2,1]]+np.array([0.,0.,1.5])
    return dict(id=row['id'],method=row['arms']['constructive']['method'],
        source=str(source.relative_to(ROOT)),source_sha256=sha(source),
        selection='First 3D condition-eligible non-plain constructive success with a negative exact 1/80-grid collision witness, in frozen record order',
        original_obstacles=obs,obstacles=[dict(center=world([c])[0].tolist(),radius=r) for c,r in obs],
        reference_Gamma=encode(G),recovered_Gamma=encode(recovered),
        reference_collision_witness=bad,strict_certificate=True,
        relative_gap_upper=row['arms']['constructive']['relative_gap_upper'],
        recorded_recovery_ms=row['arms']['constructive']['ms'],
        samples=dict(s=s.tolist(),reference=world(evaluate(G,s)).tolist(),recovered=world(evaluate(recovered,s)).tolist()),
        scope='Frozen benchmark replay. Point-robot model; red trajectory is rejected and not executed.')


def main():
    OUT.mkdir(exist_ok=True)
    data=dict(version=1,title='From relaxation to flight',created='2026-09-08',
        frozen=recorded_demo(),inspection=fresh_demo(),source_sha256=sha(__file__))
    (OUT/'planning.json').write_text(json.dumps(data,indent=2)+'\n')
    r=data['inspection']
    print(json.dumps(dict(frozen=data['frozen']['id'],family=r['family'],duration=r['duration'],
        bounds=r['derivative_bounds'],construction_ms=r['construction_ms'],
        min_sampled_body_clearance=min(r['samples']['body_clearance_sampled']),
        spatial_singular_values=r['spatial_singular_values']),indent=2))


if __name__=='__main__':main()
