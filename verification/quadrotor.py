"""Replay exact reference predicates and independently recompute visible flight metrics.

Run from any directory. The audit does not trust stored certificate booleans.
Its negative controls must be rejected. Tracking remains numerical evidence.
"""
from pathlib import Path
from fractions import Fraction as F
from math import comb
import copy, hashlib, json, sys, time
import numpy as np
HERE = Path(__file__).resolve().parents[1]/'demo'
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'src'))
from constructive_recovery import decode
from exact_check import bern_to_power, positive_on_unit_interval_fast
from exact_green import coefficient_blocks, elevated_min, endpoint_structure, energy_nullity


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rational_float(x): return F(float(x))


def evaluate_exact(g, u):
    d = g.shape[-1]-1
    b = np.array([F(comb(d,k))*u**k*(1-u)**(d-k) for k in range(d+1)],object)
    return g@b


def check_witness(G, obstacles, witness):
    g=G[witness['segment']];u=F(witness['parameter'])
    assert 0<=u<=1
    p=evaluate_exact(g,u)
    c,r=obstacles[witness['obstacle']]
    q=sum((x-rational_float(y))**2 for x,y in zip(p,c))-rational_float(r)**2
    assert q<0 and q==F(witness['squared_clearance_exact'])
    assert np.allclose(np.asarray(p,float),witness['point'],atol=1e-14,rtol=0)


def check_clearance(G, obstacles):
    """Expand squared distances directly in the power basis, then exact p>0."""
    count=0
    for g in G:
        d=g.shape[-1]-1
        for center,radius in obstacles:
            polynomial=[F(0)]*(2*d+1)
            for axis in range(3):
                a=bern_to_power(list(g[axis]-rational_float(center[axis])),d)
                for i,x in enumerate(a):
                    for j,y in enumerate(a):polynomial[i+j]+=x*y
            polynomial[0]-=rational_float(radius)**2
            assert positive_on_unit_interval_fast(polynomial,tau=0), 'Nonpositive obstacle clearance'
            count+=1
    return count


def check_boundary(G, start, goal, rest_order, continuity):
    assert np.array_equal(G[0,:,0],np.array(list(map(rational_float,start)),object))
    assert np.array_equal(G[-1,:,-1],np.array(list(map(rational_float,goal)),object))
    dd=G.copy();degree=G.shape[-1]-1;N=len(G);checks=6
    for order in range(max(rest_order,continuity)+1):
        if order>0:dd=np.diff(dd,axis=2)*(degree+1-order)*N
        if 0<order<=rest_order:
            assert all(x==0 for x in dd[0,:,0]) and all(x==0 for x in dd[-1,:,-1])
            checks+=6
        if order<=continuity:
            assert np.array_equal(dd[:-1,:,-1],dd[1:,:,0]), 'Broken knot continuity'
            checks+=3*(N-1)
    return checks


def check_duration(G, scene):
    T=F(scene['duration_exact']);assert float(T)==scene['duration']
    assert T>0
    dd=G.copy();N=len(G);degree=G.shape[-1]-1
    limits={1:F(2),2:F('2.5'),3:F(5)}
    for order in range(1,5):
        dd=np.diff(dd,axis=2)*(degree+1-order)*N
        q=max(sum(x*x for x in dd[i,:,j]) for i in range(N) for j in range(dd.shape[-1]))
        saved=scene['derivative_bounds'][str(order)]
        assert q==F(saved['normalized_squared_bound'])
        assert abs(float(q)**.5/float(T)**order-saved['physical_bound'])<1e-12
        if order in limits:assert q<=limits[order]**2*T**(2*order), 'Time scaling violates derivative bound'


def check_inflation(scene):
    for (center,radius),physical in zip(scene['planning_obstacles'],scene['physical_obstacles']):
        assert center==physical['center']
        assert rational_float(radius)>rational_float(physical['radius'])+rational_float(scene['body_radius'])+rational_float(scene['tracking_reserve']), 'Missing body/tracking inflation'


def check_tracking(plan, track, report):
    scene=plan['inspection'];a=np.asarray(track['samples'],float)
    assert np.isfinite(a).all() and a.shape==(11001,20)
    assert np.allclose(a[:,0],np.arange(len(a))*.001,atol=2e-15,rtol=0)
    assert np.max(np.abs(np.linalg.norm(a[:,4:8],axis=1)-1))<1e-12
    # Independent de Casteljau evaluation checks the reference columns used by HUD.
    G=np.asarray(decode(scene['recovered_Gamma']),float)
    s=np.clip((a[:,0]-track['pre'])/scene['duration'],0,1)
    idx=np.minimum((s*len(G)).astype(int),len(G)-1);u=s*len(G)-idx
    work=G[idx].copy()
    while work.shape[-1]>1:work=(1-u[:,None,None])*work[:,:,:-1]+u[:,None,None]*work[:,:,1:]
    ref=work[:,:,0]
    assert np.max(np.linalg.norm(ref-a[:,8:11],axis=1))<2e-13
    err=np.linalg.norm(a[:,1:4]-ref,axis=1)
    centers=np.array([o['center'] for o in scene['physical_obstacles']])
    radii=np.array([o['radius'] for o in scene['physical_obstacles']])
    clear=np.min(np.linalg.norm(a[:,None,1:4]-centers[None,:,:],axis=2)-radii[None,:]-scene['body_radius'],axis=1)
    assert np.max(abs(err-a[:,11]))<2e-13 and np.max(abs(clear-a[:,12]))<2e-13
    assert np.all((a[:,15:19]>=0)&(a[:,15:19]<=report['max_rotor_thrust']))
    # Position differences provide an independent check of the reported speed.
    velocity=np.gradient(a[:,1:4],.001,axis=0)
    assert np.max(abs(np.linalg.norm(velocity,axis=1)-a[:,13]))<.002
    q=a[:,4:8];tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
    assert np.max(abs(tilt-a[:,14]))<.03
    expected=dict(max_error=float(max(err)),min_body_clearance=float(min(clear)),
                  min_floor_clearance=float(min(a[:,3]-scene['body_radius'])),
                  max_speed=float(max(a[:,13])),max_tilt_deg=float(max(a[:,14])),
                  max_rotor_thrust=float(np.max(a[:,15:19])),final_position_error=float(err[-1]),
                  final_speed=float(a[-1,13]))
    for key,value in expected.items():
        assert abs(value-report['primary'][key])<2e-12, 'Corrupt visible metric: '+key
        assert abs(value-track['metrics'][key])<2e-12
    assert expected['min_body_clearance']>0 and expected['max_error']<scene['tracking_reserve']
    assert track['metrics']['contact_steps']==report['primary']['contact_steps']==0
    assert report['half_step']['passed'] and report['half_step']['min_body_clearance']>0
    assert report['integration_sensitivity']['max_error_difference']<.0001
    assert report['integration_sensitivity']['min_clearance_difference']<.0001
    return expected


def controls(plan, track, report):
    passed=[];G=decode(plan['inspection']['recovered_Gamma'])
    def reject(name,call):
        try:call()
        except AssertionError:passed.append(name)
        else:raise AssertionError('Negative control unexpectedly accepted: '+name)
    reject('colliding_affine_guide',lambda:check_clearance(decode(plan['inspection']['reference_Gamma']),plan['inspection']['planning_obstacles']))
    bad=G.copy();bad[2,0,0]+=F(1,100)
    reject('broken_C4_junction',lambda:check_boundary(bad,[-3,0,1.35],[3,0,1.65],4,4))
    scene=copy.deepcopy(plan['inspection']);scene['duration']=1.;scene['duration_exact']='1'
    reject('unsafe_time_compression',lambda:check_duration(G,scene))
    scene=copy.deepcopy(plan['inspection']);scene['body_radius']=1.
    reject('unmodelled_body_size',lambda:check_inflation(scene))
    bad_report=copy.deepcopy(report);bad_report['primary']['max_error']=0.
    reject('fabricated_tracking_metric',lambda:check_tracking(plan,track,bad_report))
    witness=copy.deepcopy(plan['frozen']['reference_collision_witness']);witness['squared_clearance_exact']='-1'
    reject('fabricated_collision_witness',lambda:check_witness(decode(plan['frozen']['reference_Gamma']),plan['frozen']['original_obstacles'],witness))
    return passed


def main():
    begin=time.perf_counter();plan=json.loads((HERE/'planning.json').read_text())
    track=json.loads((HERE/'tracking.json').read_text());report=json.loads((HERE/'tracking_validation.json').read_text())
    assert report['planning_sha256']==sha(HERE/'planning.json')
    assert report['source_sha256']==sha(ROOT/'experiments/demo_quadrotor_tracking.py')
    assert plan['source_sha256']==sha(ROOT/'experiments/demo_polynomial_quadrotor.py')
    scene=plan['inspection'];G=decode(scene['recovered_Gamma']);guide=decode(scene['reference_Gamma'])
    assert G.shape==(5,3,8) and scene['family']==dict(d=7,k=4,l=4,N=5,eta=4)
    family=scene['family'];blocks,r=coefficient_blocks(**family)
    minima=[elevated_min(b,7,96) for b in blocks.values()]
    assert len(blocks)==25 and r==10 and all(x>=0 for x in minima)
    assert all(any(x for row in b for x in row) for b in blocks.values())
    ends=endpoint_structure(**family);energy=energy_nullity(**family)
    assert ends['closed_square_ok'] and ends['n_live_endpoints']==8 and energy['definite']
    check_inflation(scene);check_duration(G,scene)
    scalar_checks=check_boundary(G,[-3,0,1.35],[3,0,1.65],4,4)
    check_boundary(guide,[-3,0,1.35],[3,0,1.65],4,4)
    delta=G-guide;zz=decode(scene['lift'])
    assert all(x==0 for x in delta[:,[0,1],:].flat)
    assert np.array_equal(delta[:,2,:],F(scene['amplitude'])*zz)
    for g in G:
        floor=list(g[2]-rational_float(scene['body_radius'])-rational_float(scene['tracking_reserve']))
        assert positive_on_unit_interval_fast(bern_to_power(floor,7),tau=0)
    checks=check_clearance(G,scene['planning_obstacles'])
    check_witness(guide,scene['planning_obstacles'],scene['reference_collision_witness'])
    frozen=plan['frozen'];source=ROOT/frozen['source']
    assert sha(source)==frozen['source_sha256']
    data=json.loads(source.read_text());row=next(x for x in data['rows'] if x['id']==frozen['id'])
    assert row['condition_pass'] and row['n']==3 and row['arms']['constructive']['ok']
    for key in ['relative_gap_upper','Gamma']:
        assert row['arms']['constructive'][key]==frozen['recovered_Gamma' if key=='Gamma' else key]
    assert row['obstacles']==frozen['original_obstacles']
    FG=decode(frozen['recovered_Gamma']);checks+=check_clearance(FG,frozen['original_obstacles'])
    scalar_checks+=check_boundary(FG,[-2,0,0],[2,0,0],0,2)
    check_witness(decode(frozen['reference_Gamma']),frozen['original_obstacles'],frozen['reference_collision_witness'])
    metrics=check_tracking(plan,track,report);negative=controls(plan,track,report)
    result=dict(passed=True,exact_obstacle_polynomials=checks,exact_floor_polynomials=5,
        scalar_boundary_and_junction_checks=scalar_checks,green_kernel_blocks=25,
        green_kernel_degree_elevation=96,negative_controls_rejected=negative,
        independently_recomputed_metrics=metrics,tracking_rows_checked=len(track['samples']),
        input_sha256={f:sha(HERE/f) for f in ['planning.json','tracking.json','tracking_validation.json','quadrotor.xml']},
        seconds=time.perf_counter()-begin,scope='Exact reference verification; independently replayed sampled flight diagnostics. No formal tracking bound.')
    print(json.dumps(result,indent=2))
    return result
