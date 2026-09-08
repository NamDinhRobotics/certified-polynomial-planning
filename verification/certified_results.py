"""Recorded same-SDP statistics and independent rational replay."""
from pathlib import Path
from fractions import Fraction as F
import json
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'experiments'))
from revision_common import bootstrap_mean_ci


def artifact():
    return json.loads((ROOT/'artifacts/revision_certified_solver.json').read_text())


def validate(x):
    errors=[]
    def check(p,label):
        if not p: errors.append('certified: '+label)
    p=x['protocol'];rows=x['rows']
    check(x['complete'] and x['exact_complete'],'completion')
    check(p['family']==dict(n=2,d=5,k=1,l=0,N=3,eta=2),'family')
    check(p['seeds']==list(range(20)) and p['steps']==50 and p['repeats']==3,'population')
    check(len(rows)==3000 and {(r['repeat'],r['seed'],r['step']) for r in rows}==
          {(j,s,t) for j in range(3) for s in range(20) for t in range(50)},'coverage')
    check(p['first_step_and_fallback_charged'] and p['independent_arm_state'] and
          p['conic_parameter_graph_reused'],'fair timing')
    check(p['local_iterations']==12 and p['relative_gap']==1e-5 and
          p['local_residual']==1e-7 and p['local_stationarity']==1e-6 and
          p['conic_tolerance_ladder']==[1e-9,1e-11],'tolerances')
    check(p['exact_replay']=='both arms, repeat 0, offline','exact scope')
    w=x['family_witness']
    check(w['passed'] and w['r']==10 and w['D']==96 and w['energy_nullity']['definite'],'family witness')
    check(len(w['blocks'])==9 and {tuple(b['pair']) for b in w['blocks']}=={(i,j) for i in range(3) for j in range(3)},'all ordered blocks')
    check(all(F(b['minimum'])>=0 and b['nonnegative'] and b['nonzero'] for b in w['blocks']),'block signs')
    check(w['endpoint']['closed_square_ok'] and w['endpoint']['n_live_endpoints']==4,'live endpoints')
    old=json.loads((ROOT/'artifacts/revision_streaming_F2.json').read_text())
    obstacles={(r['seed'],r['step']):r['obstacles'] for r in old['rows']}
    for r in rows:
        check(r['obstacles']==obstacles[(r['seed'],r['step'])],'same frozen input')
        order=['factor','reference'] if (r['repeat']+r['seed']+r['step'])%2==0 else ['reference','factor']
        check(r['order']==order,'arm order')
        for arm in ('factor','reference'):
            a=r[arm];c=a.get('conic',a)
            check(a['ok'],'online no answer')
            check(np.isfinite(a['total_ms']) and a['total_ms']>=0,'time')
            check(c['bound']['ok'] and np.isfinite(c['bound']['lower']),'dual diagnostic')
            check(-1e-8<=c['relative_gap']<=1e-5 and c['residual']<=1e-7,'numeric gate')
            if arm=='factor':
                check(a['source'] in ('factor','conic') and 0<=a['local_ms']<=a['total_ms'],'fallback accounting')
                check(a['iterations']<=12,'local iteration budget')
                if a['source']=='factor':
                    check(np.asarray(a['z']).shape==(129,) and np.asarray(a['lam']).shape==(99,),'factor dimensions')
                    check(a['stationarity']<=1e-6,'stationarity')
                if r['step']==0:check(a['source']=='conic' and a['local_ms']==0,'cold independence')
            if r['repeat']==0:
                e=r['exact'][arm]
                check(e['ok'] and e['reason']=='bounds','exact result')
                check(e['n_polys']==9 and e['n_pd_blocks']==19,'exact coverage')
                lo=F(int(e['lower_exact'][0]),int(e['lower_exact'][1]))
                hi=F(int(e['upper_exact'][0]),int(e['upper_exact'][1]))
                check(lo<=hi and (hi-lo)/max(F(1),abs(hi))<=F(1,100000),'exact width')
                check(F(e['lower'])<=lo and F(e['upper'])>=hi,'outward rounding')
                check(F(e['relative_gap_upper'])>=(hi-lo)/max(F(1),abs(hi)),'width rounding')
                check(e['shift'] in (0.,1e-12,1e-10,1e-8,1e-6),'primal lift correction')
                check(np.isfinite(e['ms']) and e['ms']>=0,'exact timing')
        if r['repeat']==0:
            a,b=r['exact']['factor'],r['exact']['reference']
            check(max(a['lower'],b['lower'])<=min(a['upper'],b['upper']),'paired interval overlap')
    return sorted(set(errors))


def controls(x):
    import copy
    mutations=[('family block omitted',lambda y:y['family_witness']['blocks'].pop()),
        ('independent state disabled',lambda y:y['protocol'].__setitem__('independent_arm_state',False)),
        ('exact polynomial omitted',lambda y:y['rows'][0]['exact']['factor'].__setitem__('n_polys',8)),
        ('exact width falsified',lambda y:y['rows'][0]['exact']['factor'].__setitem__('upper_exact',['1000','1'])),
        ('fallback time omitted',lambda y:y['rows'][0]['factor'].__setitem__('local_ms',1e9)),
        ('changed obstacle snapshot',lambda y:y['rows'][0]['obstacles'][0][0].__setitem__(0,100.))]
    caught=[]
    for label,mutation in mutations:
        y=copy.deepcopy(x);mutation(y)
        if not validate(y):raise AssertionError('Undetected control: '+label)
        caught.append(label)
    y=copy.deepcopy(x)
    y['rows'][0]['factor']['conic']['Y'][0][0]+=.25
    try:recheck(y,limit=1)
    except AssertionError:caught.append('altered primal witness')
    else:raise AssertionError('Altered primal witness survived')
    return caught


def compute(x):
    rows=x['rows'];s={}
    for arm in ('factor','reference'):
        a=[r[arm] for r in rows]
        e=[r['exact'][arm] for r in rows if r['repeat']==0]
        s[arm]=dict(n=len(a),ok=sum(r['ok'] for r in a),
            time=dict(zip(('p50','p90','p99','max'),map(float,np.quantile([r['total_ms'] for r in a],[.5,.9,.99,1])))),
            total_ms=sum(r['total_ms'] for r in a),exact_ok=sum(r['ok'] for r in e),
            exact_time=dict(zip(('p50','p99','max'),map(float,np.quantile([r['ms'] for r in e],[.5,.99,1])))),
            exact_gap_max=max(r['relative_gap_upper'] for r in e),
            shifted=sum(r['shift']>0 for r in e),max_shift=max(r['shift'] for r in e),
            setup_p50=float(np.median([r[arm+'_ms'] for r in x['setup']])))
    s['factor_calls']=sum(r['factor']['source']=='factor' for r in rows)
    s['fallback_calls']=len(rows)-s['factor_calls']
    s['cold_calls']=sum(r['step']==0 for r in rows)
    s['median_ratio']=s['reference']['time']['p50']/s['factor']['time']['p50']
    s['aggregate_ratio']=s['reference']['total_ms']/s['factor']['total_ms']
    s['setup_inclusive_aggregate_ratio']=(s['reference']['total_ms']+sum(r['reference_ms'] for r in x['setup']))/(s['factor']['total_ms']+sum(r['factor_ms'] for r in x['setup']))
    ratios=[];diff=[]
    for seed in range(20):
        a=[r for r in rows if r['seed']==seed]
        ratios.append(sum(r['reference']['total_ms'] for r in a)/sum(r['factor']['total_ms'] for r in a))
        diff.append(np.mean([r['reference']['total_ms']-r['factor']['total_ms'] for r in a]))
    s['seed_time_difference_ms']=bootstrap_mean_ci(diff)
    s['seed_aggregate_ratios']=ratios
    return s


def recheck(x,limit=None):
    from revision_certified_solver import model,verify_row,witness
    from exact_sdp_bounds import ExactSDPBounds
    w=witness()
    if json.loads(json.dumps(w))!=x['family_witness']:
        raise AssertionError('Exact family witness differs')
    exact=ExactSDPBounds(model(x['rows'][0]['obstacles']))
    checked=0
    for r in x['rows']:
        if r['repeat']!=0:continue
        if limit is not None and checked>=limit:break
        result=verify_row(r,exact)
        for arm in ('factor','reference'):
            old=r['exact'][arm]
            for key in ('ok','reason','lower_exact','upper_exact','shift','n_polys','n_pd_blocks'):
                if result[arm].get(key)!=old.get(key):
                    raise AssertionError(f'Exact replay differs: {r["seed"]}/{r["step"]}/{arm}/{key}')
        checked+=1
        if checked%100==0:print('SDP rational replay',checked,flush=True)
    return dict(instances=checked,intervals=2*checked,polynomials=18*checked,pd_blocks=38*checked,passed=True)
