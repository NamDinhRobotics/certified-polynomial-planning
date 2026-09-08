"""Recompute TinySDP comparisons from per-run records, including archived results."""
import hashlib
import numpy as np
from pathlib import Path
SCENES = {'frozen_barrier':'Frozen', 'sweeping_barrier':'Sweeping',
          'vertical_gate':'Gate', 'chord1':'Chord 1', 'chord2':'Chord 2', 'chord3':'Chord 3'}
ARMS = ['sdp','sdp_layer_off'] + [f'{arm}_margin{m:g}' for m in (0.,1.,1.5)
                               for arm in ('kkt_one','kkt_iterated','admm_one')]


def metrics(cell):
    reps=cell['repeats']
    t=[p['total_us']/1000 for r in reps for p in r['plans']]
    solve=[p['solve_us']/1000 for r in reps for p in r['plans']]
    full=[s['full_us']/1000 for r in reps for s in r['steps']]
    return dict(successes=sum(r['success'] for r in reps),
                goal_steps=[r['goal_step'] for r in reps],
                min_dense=min(min(s['dense_clear'] for s in r['steps']) for r in reps),
                min_seg=min(min(s['seg_signed_dist'] for s in r['tracking']) for r in reps),
                qp_failures=sum(r['qp_failures'] for r in reps),
                rejections=sum(not p['valid'] for r in reps for p in r['plans']),
                path_length=float(np.median([np.linalg.norm(np.diff([[s[k] for k in ['x','y','z']] for s in r['tracking']],axis=0),axis=1).sum() for r in reps])),
                control_effort=float(np.median([sum(sum(s[k]**2 for k in ['u1','u2','u3']) for s in r['tracking']) for r in reps])),
                solve_p50_ms=float(np.median(solve)),
                plan_ms=dict(zip(['p50','p90','p99'],np.quantile(t,[.5,.9,.99]))),
                system_step_ms=dict(zip(['p50','p90','p99'],np.quantile(full,[.5,.9,.99]))),
                deadline_misses_100ms=sum(v>100 for v in t))


def cells(data):
    return {(r['scene'],r['arm']):metrics(r) for r in data['revision_tinysdp']['rows']}


def comparison(data):
    c=cells(data); rows=[]
    for scene in SCENES:
        a=c[scene,'sdp'];b=c[scene,'kkt_one_margin1.5'];off=c[scene,'sdp_layer_off']
        rows.append(dict(scene=scene,baseline=a,replacement=b,layer_off=off,
                         speedup=a['plan_ms']['p50']/b['plan_ms']['p50'],
                         clear_gain=b['min_dense']-a['min_dense'],
                         effort_ratio=b['control_effort']/a['control_effort']))
    h=data['a83_tinysdp_ladder']; hist=[]
    for scene in h['scenes']:
        a=h['table'][scene]['base'];b=h['table'][scene]['commit']
        hist.append(dict(scene=scene,baseline_ms=a['ms_per_solve'],replacement_ms=b['ms_per_solve'],
                         speedup=a['ms_per_solve']/b['ms_per_solve'],
                         baseline_clear=a['min_seg'],replacement_clear=b['min_seg']))
    return dict(revised=rows,historical=hist,
                timing_scope='ratio of per-scene medians of total planning calls; initial plan included',
                clearance_scope='minimum of sampled interstage tracker diagnostics over five repeats',
                replacement_additional_margin_m=1.5,
                historical_scope='archived solve-only times and frozen-ball straight-segment diagnostic')


def validate(data):
    errors=[];rows=data['revision_tinysdp']['rows']
    from revision_tinysdp import CONFIGS
    expected_env=dict(CONFIGS)
    if {(r['scene'],r['arm']) for r in rows}!={(s,a) for s in SCENES for a in ARMS}:
        errors.append('tiny: exact scene/arm coverage')
    for r in rows:
        if r['env']!=expected_env[r['arm']]: errors.append('tiny: exact arm configuration')
        m=metrics(r)
        for k,v in m.items():
            stored=r['summary'].get(k)
            if isinstance(v,dict): ok=isinstance(stored,dict) and all(np.isclose(v[q],stored.get(q,np.nan),rtol=1e-10,atol=1e-10) for q in v)
            else: ok=stored is not None and bool(np.allclose(v,stored,rtol=1e-10,atol=2e-5 if k=='min_seg' else 1e-10))
            if not ok: errors.append('tiny: recomputed summary '+k)
        for rep in r['repeats']:
            effort=sum(sum(s[k]**2 for k in ['u1','u2','u3']) for s in rep['tracking'])
            length=np.linalg.norm(np.diff([[s[k] for k in ['x','y','z']] for s in rep['tracking']],axis=0),axis=1).sum()
            if not np.isclose(effort,rep['control_effort'],rtol=1e-10,atol=1e-10):
                errors.append('tiny: executed effort in individual repeat')
            if not np.isclose(length,rep['path_length'],rtol=1e-10,atol=1e-10):
                errors.append('tiny: executed path in individual repeat')
            if len(rep['tracking'])!=len(rep['steps'])+1 or rep['plans'][0]['step']!=0:
                errors.append('tiny: initial plan or tracking alignment')
            end=rep['tracking'][-1]
            goal=np.linalg.norm([end[k] for k in ['x','y','z']])<.15 and np.linalg.norm([end[k] for k in ['vx','vy','vz']])<.05
            if bool(goal)!=rep['success'] or (rep['success'] and rep['goal_step']!=end['k']):
                errors.append('tiny: goal from executed state')
            for s in rep['steps']:
                if not np.isfinite(s['full_us']) or s['full_us']<0: errors.append('tiny: finite step time')
    h=data['a83_tinysdp_ladder']
    if h['scenes']!=list(SCENES)[:3]: errors.append('history: scene population')
    for scene in h['scenes']:
        a=h['table'][scene]['base']
        for name in ['base','tinympc','cone','kkt','commit']:
            v=h['table'][scene][name]
            if not np.isclose(v['ms_per_solve'],np.median(v['ms_per_solve_all']),rtol=0,atol=5.1e-5):
                errors.append('history: median solve time')
            if name!='base' and not np.isclose(v['speedup'],a['ms_per_solve']/v['ms_per_solve'],rtol=1e-12):
                errors.append('history: speedup denominator')
    return errors


def spheres(scene,t):
    """Independent transcription of build_scenarios / sphere_at_time in delivered C++."""
    t=np.asarray(t); balls=[]
    if scene.startswith('chord'):
        ts={'chord1':[.5],'chord2':[1/3,2/3],'chord3':[.25,.5,.75]}[scene]
        for s in ts: balls.append((np.broadcast_to([-4.5+4.5*s,0,1-s],t.shape+(3,)),.35 if scene=='chord3' else .4))
    elif scene=='vertical_gate':
        for c in [[-2.9,.9,.15],[-2.4,-.9,.15]]:balls.append((np.broadcast_to(c,t.shape+(3,)),.6))
        balls.append((np.stack([np.full_like(t,-1.35),np.zeros_like(t),.35+.75*np.sin(.48*t-1.1)],axis=-1),.55))
    else:
        balls.append((np.broadcast_to([-3.3,0,.2],t.shape+(3,)),.65))
        for x,y,z,rad,px,py,pz in [(-1.8,1.3,.35,.56,.2,.4,.5),(-1.35,1.55,.95,.55,.8,.7,1.)]:
            if scene=='frozen_barrier':c=np.broadcast_to([x,y,z],t.shape+(3,))
            else:c=np.stack([x+.03*t+.03*np.sin(.3*t+px),y-.16*t+.05*np.cos(.3*t+py),z+.04*np.sin(.4*t+pz)],axis=-1)
            balls.append((c,rad))
    return balls


def recheck_geometry(data):
    """Cross-check rounded CSV states at the same 65 diagnostic times; not a certificate."""
    here=Path(__file__).resolve().parent
    sources=[here/'reproduction/TinySDP/examples/dynamic_3d_demo.cpp',
             here.parent/'third_party/TinySDP/examples/dynamic_3d_demo.cpp']
    source=next((p for p in sources if p.is_file()),None)
    if source is None or hashlib.sha256(source.read_bytes()).hexdigest()!=data['revision_tinysdp']['provenance']['cpp_sha256']:
        raise AssertionError('TinySDP source snapshot differs from recorded run')
    max_clear=max_dyn=0.;count=0
    for cell in data['revision_tinysdp']['rows']:
        for rep in cell['repeats']:
            for k,logged in enumerate(rep['steps']):
                a=rep['tracking'][k];b=rep['tracking'][k+1]
                p=np.array([a[n] for n in ['x','y','z']]);v=np.array([a[n] for n in ['vx','vy','vz']])
                u=np.array([b[n] for n in ['u1','u2','u3']]);tau=np.linspace(0,1,65)
                xyz=p+tau[:,None]*v+.5*tau[:,None]**2*u
                val=min(float(np.min(np.linalg.norm(xyz-c,axis=1)-r)) for c,r in spheres(cell['scene'],k+tau))
                max_clear=max(max_clear,abs(val-logged['dense_clear']))
                max_dyn=max(max_dyn,float(np.max(np.abs(p+v+.5*u-np.array([b[n] for n in ['x','y','z']])))),
                            float(np.max(np.abs(v+u-np.array([b[n] for n in ['vx','vy','vz']])))))
                count+=1
    # C++ logs use about six significant digits; this is a tolerance-aware
    # consistency check, separate from the full-precision polynomial audit.
    if max_clear>1e-4 or max_dyn>1e-4: raise AssertionError(f'TinySDP CSV geometry mismatch: {max_clear}, {max_dyn}')
    return dict(passed=True,intervals=count,samples_per_interval=65,
                max_clearance_difference=max_clear,max_dynamics_residual=max_dyn,
                tolerance=1e-4,scope='rounded CSV consistency at sampled times; no continuous certificate')
