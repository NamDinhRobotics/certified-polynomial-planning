"""Independent checks and generated reporting for the follow-up experiments."""
import hashlib
from pathlib import Path
import numpy as np
from tiny_comparison import spheres
from replanning_protocol import protocol, configuration, ARMS, SCENES
from cadence_protocol import protocol as cadence_protocol
HERE=Path(__file__).resolve().parent
LABELS=['SDP','One QP','Iter. QP']


def selected(data,arm,guide=None,stride=1):
    key='revision_replanning' if stride==1 else 'revision_cadence'
    return [r for r in data[key]['rows'] if r['arm']==arm and
            (guide is None or r['guide']==guide) and (stride==1 or r['stride']==stride)]


def stats(rows):
    rr=[r['result'] for r in rows];plans=[p for r in rr for p in r['plans']]
    t=[p['total_us']/1000 for p in plans]
    return dict(runs=len(rr),goals=sum(r['success'] for r in rr),
                collision_runs=sum(r['min_dense'] < -1e-9 for r in rr),
                goals_without_observed_collision=sum(r['success'] and r['min_dense']>=-1e-9 for r in rr),
                plans=len(plans),plan_count_range=[min(len(r['plans']) for r in rr),max(len(r['plans']) for r in rr)],
                rejected=sum(not p['valid'] for p in plans),qp_failures=sum(r['qp_failures'] for r in rr),
                min_dense=min(r['min_dense'] for r in rr),plan_p50_ms=float(np.median(t)),
                plan_p99_ms=float(np.quantile(t,.99)))


def validate(data):
    errors=[]
    def check(ok,label):
        if not ok:errors.append('replanning: '+label)
    for key,cfg in [('revision_replanning',protocol()),('revision_cadence',cadence_protocol())]:
        d=data[key];check(d['complete'] and d['config']==cfg,'protocol or completeness')
        if key=='revision_replanning':
            expected={(s,a,g['name']) for s in SCENES for a in ARMS for g in cfg['guides']}
            actual=[(r['scene'],r['arm'],r['guide']) for r in d['rows']]
        else:
            expected={(s,a,n) for s in SCENES for a in ARMS for n in cfg['strides']}
            actual=[(r['scene'],r['arm'],r['stride']) for r in d['rows']]
        check(len(actual)==len(expected) and set(actual)==expected,'unique coverage')
        for row in d['rows']:
            guide=next(g for g in protocol()['guides'] if g['name']==row['guide'])
            env=configuration(row['arm'],guide)
            env['TINYSDP_3D_REPLAN_STRIDE']=row.get('stride',1)
            check(row['env']==env,'exact arm/guide/cadence configuration')
            r=row['result'];tracks=r['tracking'];steps=r['steps'];plans=r['plans']
            check(all(np.isfinite(v) for rec in tracks+steps+plans for v in rec.values()),'finite records')
            check(len(tracks)==len(steps)+1 and len(plans)>1,'multiple planning calls and tracking alignment')
            check([t['k'] for t in tracks]==list(range(len(tracks))),'tracking step order')
            check([s['step'] for s in steps]==list(range(len(steps))),'execution step order')
            pp=[p['step'] for p in plans]
            check(pp[0]==0 and pp==sorted(set(pp)) and pp[-1]<len(steps),'plan steps')
            if row.get('stride',1)==1:
                expected_steps=[0]+[i for i,t in enumerate(tracks[1:-1],1)
                                   if np.linalg.norm([t[k] for k in ['x','y','z']])>=1]
                check(pp==expected_steps,'every-step policy from executed state')
            check(all(p['total_us']>=p['solve_us']>=0 and p['valid'] in (0,1) and p['qp_ok'] in (0,1) for p in plans),'plan timing and flags')
            check(all(s['full_us']>=0 for s in steps),'step timing')
            check(r['min_dense']==min(s['dense_clear'] for s in steps),'all-interval minimum')
            check(r['plan_rejections']==sum(not p['valid'] for p in plans),'rejection count')
            end=tracks[-1]
            goal=np.linalg.norm([end[k] for k in ['x','y','z']])<.15 and np.linalg.norm([end[k] for k in ['vx','vy','vz']])<.05
            check(bool(goal)==r['success'] and r['goal_step']==(int(end['k']) if goal else -1),'goal from executed state')
            check(len(steps)==int(end['k']) and (goal or len(steps)==(26 if row['scene']=='vertical_gate' else 28)),'execution budget')
            effort=sum(sum(t[k]**2 for k in ['u1','u2','u3']) for t in tracks)
            length=np.linalg.norm(np.diff([[t[k] for k in ['x','y','z']] for t in tracks],axis=0),axis=1).sum()
            check(np.isclose(effort,r['control_effort'],rtol=1e-12) and np.isclose(length,r['path_length'],rtol=1e-12),'executed path and effort')
    return sorted(set(errors))


def recheck(data):
    """Independent dynamics and moving-ball evaluation using unrounded CSV data."""
    paths=[HERE/'reproduction/TinySDP_replanning/examples/dynamic_3d_demo.cpp',
           HERE.parent/'third_party/TinySDP_replanning/examples/dynamic_3d_demo.cpp']
    source=next((p for p in paths if p.exists()),None)
    assert source is not None,'Missing follow-up source snapshot'
    count=0;max_clear=max_dyn=0.
    for key in ['revision_replanning','revision_cadence']:
        d=data[key]
        assert hashlib.sha256(source.read_bytes()).hexdigest()==d['provenance']['cpp_sha256'],'Follow-up source mismatch'
        pf=HERE.parent/'artifacts'/f'{key}.protocol.json'
        assert hashlib.sha256(pf.read_bytes()).hexdigest()==d['provenance']['protocol_sha256'],'Protocol hash mismatch'
        for row in d['rows']:
            r=row['result']
            for k,logged in enumerate(r['steps']):
                a=r['tracking'][k];b=r['tracking'][k+1]
                p=np.array([a[n] for n in ['x','y','z']]);v=np.array([a[n] for n in ['vx','vy','vz']])
                u=np.array([b[n] for n in ['u1','u2','u3']]);tau=np.linspace(0,1,65)
                assert np.all(np.isfinite(np.r_[p,v,u])),'Nonfinite follow-up trajectory'
                xyz=p+tau[:,None]*v+.5*tau[:,None]**2*u
                val=min(float(np.min(np.linalg.norm(xyz-c,axis=1)-rad)) for c,rad in spheres(row['scene'],k+tau))
                max_clear=max(max_clear,abs(val-logged['dense_clear']))
                residual=np.r_[p+v+.5*u-np.array([b[n] for n in ['x','y','z']]),
                               v+u-np.array([b[n] for n in ['vx','vy','vz']])]
                max_dyn=max(max_dyn,float(np.max(np.abs(residual))));count+=1
    assert max_clear<1e-10 and max_dyn<1e-10,(max_clear,max_dyn)
    return dict(passed=True,intervals=count,samples_per_interval=65,
                max_clearance_difference=max_clear,max_dynamics_residual=max_dyn,
                scope='binary-double CSV consistency at sampled times, not a continuous certificate')
