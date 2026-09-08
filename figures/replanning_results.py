"""Independent checks and generated reporting for the follow-up experiments."""
import figstyle
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


def render(data,table):
    rows=[]
    for guide,label in [('nominal','Nominal'),('straight','Straight'),('offset','Offsets')]:
        for arm,name in zip(ARMS,LABELS):
            rr=selected(data,arm,guide if guide!='offset' else None)
            if guide=='offset':rr=[r for r in rr if r['guide'].startswith('offset')]
            v=stats(rr)
            rows.append(f"{label} & {name} & {v['goals']}/{v['runs']} & {v['collision_runs']} & "
                        f"{v['rejected']}/{v['plans']} & {v['plan_p50_ms']:.3f} / {v['plan_p99_ms']:.3f}"+r' \\')
    files={'table_revision_replanning.tex':table(
        'Every-step replanning from executed states: six scenes, seven guide conditions, three arms; one execution per cell. Offsets pools five prescribed guide perturbations per scene. Goal and observed collision counts are separate. Reject counts rejected planning candidates, including repeated failures within one run. Times are pooled planning-call quantiles in ms, including failed calls and the initial call; they are not independent timing repeats. Both QPs request 1.5 m additional inflation.',
        'tab:replanning','llrrrr','Guide & Arm & Goal & Coll. & Reject/calls & p50 / p99',rows).replace('\\begin{table}[t]','\\begin{table*}[t]').replace('\\end{table}','\\end{table*}')}
    rows=[]
    for stride in (1,3,5):
        vals=[stats(selected(data,a,'nominal',stride)) for a in ARMS]
        rows.append(f"{stride} & "+' & '.join(f"{v['goals']}/6 ({v['collision_runs']})" for v in vals)+
                    ' & '+' / '.join(str(v['plans']) for v in vals)+r' \\')
    files['table_revision_cadence.tex']=table(
        'Nominal-guide cadence diagnostic, same execution budgets. Goal counts (observed collision counts) and total planning calls are ordered SDP / one QP / iterated QP. Strides 3 and 5 are post-hoc diagnostics motivated by the stride-1 failures, not confirmatory robustness evidence.',
        'tab:cadence','rrrrr','Stride & SDP & One & Iter. & Calls S/O/I',rows)
    s,o,i=[stats(selected(data,a,'nominal')) for a in ARMS]
    ss,oo,ii=[stats(selected(data,a,'nominal',5)) for a in ARMS]
    files['text_revision_replanning.tex']=(
        '% Generated from every execution, including failed runs.\n'
        f"With nominal guides and a stride of one, SDP reaches the goal in {s['goals']}/6 scenes, "
        f"and each QP variant in {o['goals']}/6; no sampled tracker collision is observed in these nominal runs. "
        f"The one-step QP makes {o['plan_count_range'][0]}--{o['plan_count_range'][1]} planning calls per run, "
        f"with pooled total-planning p50/p99 {o['plan_p50_ms']:.3f}/{o['plan_p99_ms']:.3f} ms, "
        f"versus {s['plan_p50_ms']:.3f}/{s['plan_p99_ms']:.3f} ms for SDP. "
        'Faster subproblems therefore do not establish timely goal completion. '
        'Table~\\ref{tab:replanning} retains failures with perturbed and straight guides, '
        'including observed collisions in the SDP arm and rejected QP candidates. '
        'The pooled call quantiles weight longer failed executions more heavily; '
        'they are descriptive timings of different evolving state sequences, not paired solver-speed ratios.\n\n'
        'After observing these failures, we ran a separate nominal-guide cadence diagnostic '
        'at strides 3 and 5, retaining the same execution budgets and tracker. '
        f"At stride 5, both QP variants reach {oo['goals']}/6 goals using "
        f"{oo['plan_count_range'][0]}--{oo['plan_count_range'][1]} calls per run, while SDP reaches {ss['goals']}/6. "
        'No sampled tracker collision is observed in these stride-5 runs. '
        'Table~\\ref{tab:cadence} and Fig.~\\ref{fig:replanning} report all three cadences. '
        'This post-hoc result identifies sensitivity to the replanning schedule; it does not '
        'validate stride 5 on a new scene population or establish the cause of every failed run. '
        'Resetting the guide over a fresh nine-interval horizon changes the time assigned to '
        'reaching its endpoint at each replanning call, so changing cadence also changes the '
        'closed-loop behavior. No recursive-feasibility, stability, or finite-time convergence '
        'guarantee is claimed for this inherited guide/tracker combination.\n')
    return files


def plot(data):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    figstyle.use()
    fig,(ax,bx)=plt.subplots(1,2,figsize=(figstyle.TEXT_W,2.9),gridspec_kw={'width_ratios':[2.3,1]},layout='constrained')
    guides=[g['name'] for g in protocol()['guides']];grid=np.zeros((7,18))
    for i,g in enumerate(guides):
        for j,s in enumerate(SCENES):
            for k,a in enumerate(ARMS):
                r=next(r for r in selected(data,a,g) if r['scene']==s)['result']
                grid[i,3*j+k]=2 if r['min_dense'] < -1e-9 else int(r['success'])
    colors=['#E5B567','#39866D','#C75252']
    ax.imshow(grid,cmap=ListedColormap(colors),vmin=-.5,vmax=2.5,aspect='auto',interpolation='nearest')
    for i,j in np.argwhere(grid>0):
        ax.text(j,i,'G' if grid[i,j]==1 else 'X',ha='center',va='center',color='white',fontsize=9)
    ax.set_yticks(range(7),['Nominal','Straight']+[f'Offset {i}' for i in range(5)])
    ax.set_xticks(range(18),['S','O','I']*6)
    for j,name in enumerate(['Frozen','Sweep','Gate','Ch.1','Ch.2','Ch.3']):
        ax.text(3*j+1,-1.0,name,ha='center',va='center',fontsize=7)
        if j:ax.axvline(3*j-.5,c='white',lw=1.5)
    ax.set_xlabel('S: SDP; O: one QP; I: iterated QP')
    ax.set_title('(a) Replanning at every step',pad=23,fontsize=8)
    for j,(a,name,c) in enumerate(zip(ARMS,LABELS,['#356B9A','#D07935','#765B98'])):
        bx.plot([1,3,5],[stats(selected(data,a,'nominal',n))['goals'] for n in (1,3,5)],
                marker='s' if j==2 else 'o',linestyle='--' if j==2 else '-',
                label=name,color=c,ms=7 if j==1 else 4,
                markerfacecolor='none' if j==2 else c)
    bx.set_xticks([1,3,5]);bx.set_yticks([0,2,4,6]);bx.set_ylim(-.3,6.5)
    bx.set_xlabel('Replanning stride');bx.set_ylabel('Goal completions / 6')
    bx.set_title('(b) Nominal guide',pad=23,fontsize=8);bx.legend(frameon=False,fontsize=7,loc='upper left')
    fig.legend(handles=[Patch(color=c,label=l) for c,l in zip(colors,['No goal / no collision observed','G: goal / no collision observed','X: collision observed'])],
               loc='outside lower center',ncol=3,frameon=False,fontsize=7)
    figstyle.save(fig,HERE/'fig_revision_replanning')
    plt.close(fig)
