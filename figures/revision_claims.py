"""Computed manuscript claims with explicit artifact populations and invariants."""
from pathlib import Path
import json,sys
import numpy as np
from tiny_comparison import validate as validate_tiny, comparison
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'experiments'))
from revision_common import bootstrap_mean_ci
FILES=['revision_streaming_F1','revision_streaming_F2','revision_certificates',
       'revision_frozen','revision_rank_profile','revision_serial_timing','revision_tinysdp',
       'a83_tinysdp_ladder','revision_replanning','revision_cadence']


def load():
    return {k:json.loads((ROOT/'artifacts'/f'{k}.json').read_text()) for k in FILES}


def expected_grid():
    p=[];s=[]
    for d in (3,5,7,9):
        p.append((d,1,0,1,None))
        p += [(d,1,0,n,e) for n in range(2,7) for e in range(d)]
    for d,k,l in ((5,2,1),(7,2,1),(7,3,2),(9,4,3)):
        s.append((d,k,l,1,None))
        s += [(d,k,l,n,e) for n in (2,3) for e in range(k,min(d-1,k+2)+1)]
    return p+s


def invariants(data):
    errors=[]
    def check(ok,label):
        if not ok: errors.append(label)
    for f,cfg in [('F1',dict(d=3,k=1,l=0,N=1,eta=None)),('F2',dict(d=5,k=1,l=0,N=3,eta=2))]:
        x=data['revision_streaming_'+f]; conf=x['config']; rows=x['rows']
        check(x['complete'] and conf['family']==cfg,f+': family or incomplete')
        check(conf['seeds']==list(range(20)) and conf['steps']==50,f+': population')
        check(len(rows)==1000 and {(r['seed'],r['step']) for r in rows}=={(s,t) for s in range(20) for t in range(50)},f+': unique seed/step coverage')
        check(conf['terminal']=='no_answer' and conf['clearance_tolerance']==1e-9,f+': terminal/tolerance')
        check(conf['requested_margin']==0 and conf['all_segments_and_obstacles_verified'],f+': margin/verifier scope')
        for r in rows:
            check(r['family']==f,f+': row family')
            check(len(r['obstacles'])==3,f+': obstacle count')
            for arm,a in r['arms'].items():
                check(a['online_ms']>=0 and np.isfinite(a['online_ms']),f+': nonfinite time')
                check(a['n_solves'] <= (3 if arm=='onecut' else 2 if arm=='guide_qp' else 1),f+': solve budget')
                if a['accepted']:
                    e=a['exact']; check(e is not None and e['ok'] and not e.get('unknown',False),f+': rational verification')
                    check(e is not None and e.get('n_polys')==3*cfg['N'],f+': all polynomial pairs')
                    check(e is not None and e.get('tolerance')==1e-9,f+': rational tolerance')
                    check(a['pmin']>=-1e-9 and np.asarray(a['Gamma']).shape==(2,cfg['N']*(cfg['d']+1)),f+': accepted record')
                else: check(a['source']=='no_answer' and a['Gamma'] is None,f+': stale output')
    x=data['revision_certificates']; rows=x['rows']
    check(len(rows)==152 and {tuple(r['key']) for r in rows}==set(expected_grid()),'grid: exact tuple population')
    check(x['n_primary']==124 and x['n_secondary']==28,'grid: primary/secondary')
    check(all(r.get('recheck',{}).get('D')==192 and r['base']['D']==96 for r in rows),'grid: actual rechecks')
    fields=('certified','strict','nonneg','K_definite','closed_square_ok')
    for r in rows:
        check(r.get('agrees')==all(r['base'].get(f)==r.get('recheck',{}).get(f) for f in fields),'grid: agreement')
        for name in ['base','recheck']:
            q=r.get(name,{})
            check([q.get(k) for k in ['d','k','l','N','eta']]==r['key'],'grid: result configuration')
            check(q.get('certified')==bool(q.get('K_definite') and (q.get('strict') or (q.get('nonneg') and q.get('closed_square_ok')))),'grid: endpoint predicate')
    x=data['revision_frozen']; rows=x['rows']
    check(x['complete'] and len(rows)==37,'frozen: coverage')
    check(sum(r['population']=='historical_rho1' for r in rows)==27 and sum(r['population']=='corner_rho2' for r in rows)==10,'frozen: population labels')
    for r in rows:
        check(r['rho']==(1 if r['population']=='historical_rho1' else 2),'frozen: rank label')
        b=r['bnb_first']
        if b:
            check(b['conic_solves']<=r['bnb_end']['conic_solves'],'frozen: first/end calls')
            check(b['exact']['ok'] and b['exact']['n_polys']==len(r['obs']),'frozen: incumbent recheck')
    x=data['revision_rank_profile']; rows=x['rows']
    check(len(rows)==100 and sum(len(r['blocks']) for r in rows)==900,'profile: population')
    check(x['family']['N']==3 and x['family']['d']==5 and x['family']['eta']==2,'profile: family')
    check(all(v['passed'] for v in x['gates'].values()),'profile: residual gates')
    x=data['revision_serial_timing']; rows=x['rows']
    check(x['config']['repeats']==5 and len(rows)==200,'timing: repetitions')
    check(x['config']['frames']==np.random.default_rng(20260907).integers(0,50,size=20).tolist(),'timing: selected frames')
    check(x['config']['offline_exact_excluded'] and not x['config']['warm_start'],'timing: scope')
    for family in ['F1','F2']:
        check(sum(r['family']==family for r in rows)==100,'timing: family count')
    x=data['revision_tinysdp']; rows=x['rows']
    check(x['complete'] and len(rows)==66 and x['config']['repeats']==5,'tiny: complete grid')
    check(x['config']['common_monitor'] and x['config']['dense_subdivisions']==64 and x['config']['dense_check_is_certificate'] is False,'tiny: monitor scope')
    for r in rows:
        env=r['env'];check(len(r['repeats'])==5,'tiny: repeats')
        if r['arm'].startswith(('kkt_','admm_')):
            margin=float(r['arm'].split('margin')[1])
            check(env['TINYSDP_D4_INFLATE']==margin and env['TINYSDP_D4_ALLCUTS']==1,'tiny: matched margin/all obstacles')
        for rep in r['repeats']:
            check(rep['plans'] and rep['steps'],'tiny: timing/clearance records')
            check(all(p['total_us']>=p['solve_us']>=0 for p in rep['plans']),'tiny: timing inclusion')
            check(rep['min_dense']==min(s['dense_clear'] for s in rep['steps']),'tiny: minimum all steps')
    errors.extend(validate_tiny(data))
    from replanning_results import validate as validate_replanning
    errors.extend(validate_replanning(data))
    return sorted(set(errors))


def compute(data):
    c={};pop={}
    for family,word in [('F1','Fone'),('F2','Ftwo')]:
        rows=data['revision_streaming_'+family]['rows']
        counts={a:sum(r['arms'][a]['accepted'] for r in rows) for a in ['plain','onecut','guide_qp']}
        c.update({f'Stream{word}Fresh':str(counts['plain']),f'Stream{word}Onecut':str(counts['onecut']),
                  f'Stream{word}Guide':str(counts['guide_qp']),f'Stream{word}Recovered':str(counts['onecut']-counts['plain'])})
        diffs=[];energies=[]
        joint=[r for r in rows if r['arms']['onecut']['accepted'] and r['arms']['guide_qp']['accepted']]
        ratios=[r['arms']['onecut']['cost']/r['arms']['guide_qp']['cost']-1 for r in joint]
        for seed in range(20):
            rr=[r for r in rows if r['seed']==seed]
            diffs.append(100*np.mean([int(r['arms']['onecut']['accepted'])-int(r['arms']['guide_qp']['accepted']) for r in rr]))
            common=[r for r in joint if r['seed']==seed]
            energies.append(100*np.median([r['arms']['onecut']['cost']/r['arms']['guide_qp']['cost']-1 for r in common]))
        ci=bootstrap_mean_ci(diffs)
        c[word+'AvailabilityCI']=f"$[{ci['ci95'][0]:.1f},{ci['ci95'][1]:.1f}]$"
        c[word+'EnergyReduction']=f'{-100*np.median(ratios):.1f}'
        c[word+'Joint']=str(len(joint))
        pop[family]=dict(counts=counts,availability=ci,joint_n=len(joint),median_relative_cost=np.median(ratios),
                         seed_energy_medians=energies,seed_energy_bootstrap=bootstrap_mean_ci(energies))
    c['AcceptedRecords']=str(sum(sum(v['counts'].values()) for v in pop.values()))
    grid=data['revision_certificates']['rows']
    c.update(GridPrimary='124',GridSecondary='28',GridTotal=str(len(grid)),
             GridPassBase=str(sum(r['base']['certified'] for r in grid)),
             GridPassHigh=str(sum(r['recheck']['certified'] for r in grid)))
    for name,prefix in [('historical_rho1','Frozen'),('corner_rho2','Corner')]:
        rr=[r for r in data['revision_frozen']['rows'] if r['population']==name]
        c[prefix+'Onecut']=str(sum(r['onecut']['accepted'] for r in rr))
        c[prefix+'First']=str(sum(r['bnb_first'] is not None for r in rr))
        c[prefix+'Closed']=str(sum(bool(r['bnb_end']['solver_proven']) for r in rr))
        c[prefix+'FirstCalls']=f"{np.median([r['bnb_first']['conic_solves'] for r in rr if r['bnb_first']]):g}"
        c[prefix+'EndCalls']=f"{np.median([r['bnb_end']['conic_solves'] for r in rr]):g}"
    prof=data['revision_rank_profile']['summary']
    c['ProfileBlocks']=str(prof['n_blocks'])
    val=prof['worst_feas_rel']; exp=int(np.floor(np.log10(val)))
    c['ProfileResidual']=f'${val/10**exp:.2f}\\times10^{{{exp}}}$'
    collision=next(r for r in data['revision_tinysdp']['rows']
                   if r['scene']=='chord2' and r['arm']=='admm_one_margin1')
    c['TinyCollision']=f"${min(r['min_dense'] for r in collision['repeats']):.4f}$"
    tiny=comparison(data)
    for name,key in [('TinyTotal','revised'),('TinyHistorical','historical')]:
        ratios=[r['speedup'] for r in tiny[key]]
        c[name+'Min']=f'{min(ratios):.0f}';c[name+'Max']=f'{max(ratios):.0f}'
    c['TinyComparableClear']=str(sum(r['clear_gain']>=0 for r in tiny['revised']))
    c['TinyEffortHigher']=str(sum(r['effort_ratio']>1 for r in tiny['revised']))
    return dict(macros=c,populations=pop,tiny_comparison=tiny,
                artifact_map={name:'artifacts/'+name+'.json' for name in FILES},
                scopes=dict(streaming='20 paired seeds, 50 correlated instants per seed',
                            timing='20 prespecified frames per family, five cold repeats',
                            tiny='six deterministic scenes, five timing repeats',
                            exact='p(s)+1e-9 > 0, represented binary rational data, obstacles only'))
