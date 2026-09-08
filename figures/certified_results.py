"""Artifact-derived same-SDP claims, scientific figure and independent replay."""
import figstyle
from pathlib import Path
from fractions import Fraction as F
import hashlib
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
    # Legacy key: ratio of marginal medians, not median of per-pair ratios.
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


def render(write=True):
    x=artifact();errors=validate(x)
    if errors:raise ValueError(errors)
    s=compute(x);a,b=s['factor'],s['reference']
    macros=dict(CertMedianRatio=f"{s['median_ratio']:.2f}",CertAggregateRatio=f"{s['aggregate_ratio']:.2f}",
        CertFactorPercent=f"{100*s['factor_calls']/3000:.0f}",CertExactCount='2,000')
    text='% Generated by certified_results.py.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items())
    table=r'''% Generated by certified_results.py.
\begin{table}[!t]
\caption{Same F2 SDP: 3,000 calls per arm, including cold calls and fallback.
Times are ms. Exact checks cover the first 1,000 calls per arm and are offline.}
\label{tab:certified}\centering\small
\begin{tabular}{@{}lrr@{}}\toprule
Metric & Factor pipeline & Conic reference\\\midrule
'''
    for label,v,w,fmt in [('Online p50',a['time']['p50'],b['time']['p50'],'.2f'),
                          ('Online p90',a['time']['p90'],b['time']['p90'],'.2f'),
                          ('Online p99',a['time']['p99'],b['time']['p99'],'.2f'),
                          ('Online maximum',a['time']['max'],b['time']['max'],'.2f'),
                          ('Setup p50 per seed',a['setup_p50'],b['setup_p50'],'.2f'),
                          ('Numerical answers',a['ok'],b['ok'],'d'),
                          ('Rational intervals',a['exact_ok'],b['exact_ok'],'d'),
                          ('Rational check p50',a['exact_time']['p50'],b['exact_time']['p50'],'.2f'),
                          ('Rational check p99',a['exact_time']['p99'],b['exact_time']['p99'],'.2f')]:
        table+=f'{label} & {v:{fmt}} & {w:{fmt}}'+r' \\'+'\n'
    table+=r'\bottomrule\end{tabular}\end{table}'+'\n'
    ci=s['seed_time_difference_ms']['ci95']
    def sci(v):
        mantissa,exponent=f'{v:.2e}'.split('e')
        return '$'+mantissa+r'\times10^{'+str(int(exponent))+'}$'
    detail=(r'% Generated by certified_results.py.'+'\n'
        +f"The pipeline accepts {s['factor_calls']:,} factor answers and invokes the conic fallback on {s['fallback_calls']:,} calls, including {s['cold_calls']} cold calls. "
        +f"It reduces the median from {b['time']['p50']:.2f} to {a['time']['p50']:.2f} ms and the total measured online work by a factor of {s['aggregate_ratio']:.2f}; including separately recorded setup changes the aggregate ratio to {s['setup_inclusive_aggregate_ratio']:.2f}. "
        +f"The seed-bootstrap 95\\% interval for mean paired time saved is [{ci[0]:.2f},{ci[1]:.2f}] ms. "
        +f"P99 increases from {b['time']['p99']:.2f} to {a['time']['p99']:.2f} ms; there is no upper-tail improvement.\n\n"
        +f"All 1,000 paired interval intersections are nonempty. The largest rigorous relative interval widths are {sci(a['exact_gap_max'])} for the factor pipeline and {sci(b['exact_gap_max'])} for the reference. "
        +f"The primal verifier adds a positive lift correction on {a['shifted']} and {b['shifted']} outputs, respectively; the largest shift in either arm is {sci(max(a['max_shift'],b['max_shift']))}. "
        +"The correction and its exact energy cost are part of each upper-bound witness, so these intervals do not assert that the unmodified floating-point point is exactly feasible.\n")
    files={'results_certified.tex':text,'table_revision_certified.tex':table,
           'text_revision_certified.tex':detail,
           'claims_certified.json':json.dumps(s,indent=2,allow_nan=False)+'\n'}
    if write:
        for name,value in files.items():(HERE/name).write_text(value)
    return files


def plot():
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter,NullLocator
    x=artifact();s=compute(x);rows=x['rows']
    figstyle.use()
    fig,ax=plt.subplots(2,2,figsize=(figstyle.TEXT_W,4.15),layout='constrained')
    colors={'factor':'#007C83','reference':'#D55E00'}
    for source,color,label in [('factor',colors['factor'],'Factor accepted'),('conic',colors['reference'],'Conic fallback')]:
        rr=[r for r in rows if r['factor']['source']==source]
        ax[0,0].scatter([r['reference']['total_ms'] for r in rr],[r['factor']['total_ms'] for r in rr],s=5,alpha=.35,c=color,marker='o' if source=='factor' else '^',label=label,rasterized=True)
    ax[0,0].plot([1,80],[1,80],color='.4',lw=.8,ls='--')
    ax[0,0].set(xscale='log',yscale='log',xlim=(5,70),ylim=(2,75),
                xlabel='Conic reference (ms)',ylabel='Factor pipeline (ms)',title='(a) Same-instance online time')
    ax[0,0].set_xticks([5,10,20,50]);ax[0,0].set_yticks([2,5,10,20,50])
    for axis in (ax[0,0].xaxis,ax[0,0].yaxis):
        axis.set_major_formatter(ScalarFormatter());axis.set_minor_locator(NullLocator())
    ax[0,0].legend(frameon=False,fontsize=7,loc='upper left',markerscale=2)
    for i,arm in enumerate(('factor','reference')):
        heights=[s[arm]['time'][k] for k in ('p50','p99')]
        ax[0,1].bar(np.arange(2)+(i-.5)*.34,heights,.34,color=colors[arm],label=arm.capitalize())
    ax[0,1].set(xticks=[0,1],xticklabels=['Median','p99'],ylabel='Online time (ms)',title='(b) Median gain, higher p99')
    ax[0,1].legend(frameon=False,fontsize=7)
    for arm in ('factor','reference'):
        e=[r['exact'][arm] for r in rows if r['repeat']==0]
        gaps=np.sort([a['relative_gap_upper'] for a in e])
        ax[1,0].plot(gaps,np.arange(1,len(e)+1)/len(e),color=colors[arm],ls='-' if arm=='factor' else '--',label=arm.capitalize())
        online=np.median([r[arm]['total_ms'] for r in rows if r['repeat']==0])
        j=0 if arm=='factor' else 1
        ax[1,1].bar(j,online,color=colors[arm])
        ax[1,1].bar(j,s[arm]['exact_time']['p50'],bottom=online,color=colors[arm],alpha=.3,hatch='///')
    ax[1,0].axvline(1e-5,color='.4',ls='--',lw=.8)
    ax[1,0].set(xscale='log',xlabel='Relative SDP interval width',ylabel='Fraction of first-pass instances',title='(c) Exact objective certificates')
    ax[1,0].legend(frameon=False,fontsize=7,loc='upper left')
    ax[1,1].set(xticks=[0,1],xticklabels=['Factor','Reference'],ylim=(0,70),ylabel='Component medians (ms)',title='(d) Cost of exact replay')
    ax[1,1].text(.5,.97,'Hatched: offline rational check',transform=ax[1,1].transAxes,ha='center',va='top',fontsize=7)
    figstyle.save(fig,HERE/'fig_revision_certified')

    plt.close(fig)


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


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--recheck',action='store_true');args=ap.parse_args()
    render();plot()
    if args.recheck:print(recheck(artifact()))
