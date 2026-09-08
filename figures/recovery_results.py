"""Computed recovery claims, complete-population validation and exact replay."""
import figstyle
from pathlib import Path
import copy,hashlib,json,sys,time
from fractions import Fraction as F
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from constructive_recovery import decode,encode,exact_energy,physical_certificate,endpoint_separation,coefficients
from exact_green import constraint_matrix
from exact_sdp_bounds import ExactSDPBounds
from revision_recovery_generalization import root_problem,STRATA

ARMS=['constructive','angular_fold','onecut','guide_qp']
NAMES=['Constructive','Angular fold','One-cut','Guide QP']


def read():
    old=json.loads((ROOT/'artifacts/revision_green_recovery.json').read_text())
    fresh=json.loads((ROOT/'artifacts/revision_recovery_generalization.json').read_text())
    pop=json.loads((ROOT/'artifacts/revision_recovery_population.json').read_text())
    return old,fresh,pop


def validate(fresh,pop):
    errors=[]
    def require(condition,message):
        if not condition:errors.append(message)
    require(fresh.get('complete') and not fresh.get('limited_smoke'),'incomplete or smoke population')
    require(fresh['protocol']==pop['protocol'],'changed protocol')
    require(fresh['source_hashes']==pop['source_hashes'],'changed source manifest')
    pop_path=ROOT/'artifacts/revision_recovery_population.json'
    require(fresh['population_sha256']==hashlib.sha256(pop_path.read_bytes()).hexdigest(),'population hash')
    require(len(fresh['rows'])==len(pop['rows'])==240,'240-case population')
    require(pop['protocol']['per_stratum']==40 and pop['protocol']['strata']==STRATA,'stratum counts')
    require(pop['protocol']['margin']==1e-8 and 'zero tolerance' in pop['protocol']['gate'],'margin/gate')
    for name,h in pop['source_hashes'].items():
        p=ROOT/name;require(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==h,'executed source '+name)
    for index,(row,input_) in enumerate(zip(fresh['rows'],pop['rows'])):
        for key in ('id','stratum','n','seed','obstacles'):require(row[key]==input_[key],'input mismatch '+str(index)+' '+key)
        require(set(row['arms'])==set(ARMS),'arm set '+row['id'])
        require(row['arm_order']==ARMS[index%4:]+ARMS[:index%4],'arm order '+row['id'])
        for arm in ARMS:
            a=row['arms'][arm]
            require(isinstance(a.get('ms'),(int,float)) and np.isfinite(a['ms']) and a['ms']>=0,'charged runtime '+row['id']+'/'+arm)
            if a['ok']:
                require(a.get('Gamma',{}).get('shape')==[3,row['n'],6],'curve shape '+row['id']+'/'+arm)
                J=F(*map(int,a['cost_exact']));L=F(*map(int,row['lower_exact']))
                require(float(J)==a['cost'] and J>=L,'energy/lower ordering '+row['id']+'/'+arm)
                require(a['relative_gap_upper']==float((J-L)/max(F(1),abs(L))),'computed gap '+row['id']+'/'+arm)
    return errors


def quantiles(values):return dict(zip(['p50','p90','p99','max'],map(float,np.percentile(values,[50,90,99,100])))) if values else None


def compute(old,fresh):
    r=fresh['rows'];strata={};groups={}
    for group in STRATA:
        rr=[x for x in r if x['stratum']==group]
        strata[group]={a:dict(accepted=sum(x['arms'][a]['ok'] for x in rr),median_ms=float(np.median([x['arms'][a]['ms'] for x in rr]))) for a in ARMS}
    for name,rr in [('eligible',[x for x in r if x['condition_pass']]),('outside',[x for x in r if not x['condition_pass']])]:
        groups[name]=dict(n=len(rr),arms={})
        for a in ARMS:
            good=[x['arms'][a] for x in rr if x['arms'][a]['ok']]
            groups[name]['arms'][a]=dict(accepted=len(good),time_ms=quantiles([x['arms'][a]['ms'] for x in rr]),
                relative_gap_percent=quantiles([100*x['relative_gap_upper'] for x in good]))
    paired={}
    for a in ARMS[1:]:
        common=[x for x in r if x['arms']['constructive']['ok'] and x['arms'][a]['ok']]
        paired[a]=dict(n=len(common),median_relative_cost_percent=float(np.median([100*(x['arms']['constructive']['cost']/x['arms'][a]['cost']-1) for x in common])))
    oldr=old['rows']
    return dict(retrospective=dict(count=len(oldr),accepted=sum(x['ok'] for x in oldr),methods={a:sum(x['method']==a for x in oldr) for a in ('plain','escape','green')},
                gap_percent=quantiles([100*x['certified_relative_suboptimality'] for x in oldr]),repair_only_median_gap_percent=float(np.median([100*x['certified_relative_suboptimality'] for x in oldr if not x['plain_strict']])),
                accumulated_ms=quantiles([x['combined_recovery_ms'] for x in oldr])),
                fresh=dict(count=len(r),root_accepted=sum(x['root']['ok'] for x in r),dual_bounds=sum(x['lower_kind']=='repaired_dual' for x in r),
                  root_ms=quantiles([x['root_ms'] for x in r]),setup_ms=quantiles([x['setup_ms'] for x in r]),groups=groups,strata=strata,paired=paired))


def render(write=False):
    old,fresh,pop=read();errors=validate(fresh,pop)
    if errors:raise AssertionError('\n'.join(errors))
    c=compute(old,fresh);rr=c['retrospective'];ff=c['fresh'];e=ff['groups']['eligible'];o=ff['groups']['outside'];a=e['arms']['constructive']
    macros={'RecoveryCount':rr['count'],'RecoveryAccepted':rr['accepted'],'RecoveryPlain':rr['methods']['plain'],'RecoveryGap':rr['methods']['escape'],'RecoveryGreen':rr['methods']['green'],
            'RecoveryWorstGap':f"{rr['gap_percent']['max']:.2f}",'RecoveryRepairMedianGap':f"{rr['repair_only_median_gap_percent']:.2f}",
            'FreshCount':ff['count'],'FreshEligible':e['n'],'FreshEligibleAccepted':a['accepted'],'FreshOutside':o['n'],'FreshOutsideAccepted':o['arms']['constructive']['accepted'],
            'FreshWorstGap':f"{a['relative_gap_percent']['max']:.2f}",'FreshMedianGap':f"{a['relative_gap_percent']['p50']:.2f}",'FreshRecoveryMedianMs':f"{a['time_ms']['p50']:.2f}"}
    for arm,label in [('angular_fold','Angular'),('onecut','Onecut'),('guide_qp','Guide')]:
        macros['Fresh'+label+'Eligible']=e['arms'][arm]['accepted']
        macros['Fresh'+label+'Outside']=o['arms'][arm]['accepted']
        macros['Fresh'+label+'Ms']=f"{e['arms'][arm]['time_ms']['p50']:.2f}"
    macros['FreshRootMedian']=f"{ff['root_ms']['p50']:.2f}"
    macros['FreshSetupMedian']=f"{ff['setup_ms']['p50']:.2f}"
    macros['FreshPhysicalCurves']=sum(x['arms'][arm]['ok'] for x in fresh['rows'] for arm in ARMS)
    macros['FreshPhysicalPolys']=f"{15*macros['FreshPhysicalCurves']:,}"
    result={'results_recovery.tex':'% Generated by recovery_results.py; do not edit.\n'+''.join('\\newcommand{\\'+k+'}{'+str(v)+'}\n' for k,v in macros.items()),
            'claims_recovery_integrated.json':json.dumps(c,indent=2)+'\n'}
    lines=[r'\begin{table*}[t]',r'\centering',r'\caption{Frozen new-geometry experiment: strict physical certification counts out of 40 per row. Median recovery times include exact gates; the common root solve and setup are separate. Methods are the explicitly specified adaptations in the text.}',r'\label{tab:fresh-recovery}',r'\begin{tabular}{lrrrrrrrr}',r'\toprule & \multicolumn{4}{c}{Strictly certified / 40} & \multicolumn{4}{c}{Median recovery time (ms)} \\',r'\cmidrule(lr){2-5}\cmidrule(lr){6-9}',r'Geometry & Constructive & Angular fold & One-cut & Guide QP & Constructive & Angular fold & One-cut & Guide QP \\',r'\midrule']
    for group in STRATA:
        row=ff['strata'][group];label={'clutter_2d':'Clutter, 2D','endpoint_2d':'Near endpoints, 2D','clutter_3d':'Clutter, 3D','endpoint_3d':'Near endpoints, 3D','outside_2d':'Outside condition, 2D','outside_3d':'Outside condition, 3D'}[group]
        lines.append(label+' & '+' & '.join(str(row[x]['accepted']) for x in ARMS)+' & '+' & '.join(f"{row[x]['median_ms']:.1f}" for x in ARMS)+r' \\')
    lines +=[r'\bottomrule',r'\end{tabular}',r'\end{table*}']
    result['table_recovery_generalization.tex']='\n'.join(lines)+'\n'
    if write:
        for name,text in result.items():(HERE/name).write_text(text)
    return result


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old,fresh,_=read();c=compute(old,fresh);e=c['fresh']['groups']['eligible'];o=c['fresh']['groups']['outside']
    figstyle.use()
    fig,axes=plt.subplots(2,2,figsize=(figstyle.TEXT_W,4.75),layout='constrained');ax=axes.ravel();x=np.arange(4)
    ax[0].bar(x-.18,[100*e['arms'][a]['accepted']/e['n'] for a in ARMS],.36,label='Eligible (160)',color='#178570')
    ax[0].bar(x+.18,[100*o['arms'][a]['accepted']/o['n'] for a in ARMS],.36,label='Outside (80)',color='#b47541')
    ax[0].set_xticks(x,['Constr.','Angular','One-cut','Guide']);ax[0].set_ylim(0,136);ax[0].set_yticks([0,25,50,75,100]);ax[0].set_ylabel('Certified physical curves (%)');ax[0].set_title('(a) Recovery availability');ax[0].legend(fontsize=9,loc='upper right',handlelength=1.1)
    for a,color in [('constructive','#178570'),('angular_fold','#397cb5'),('guide_qp','#b47541'),('onecut','#666666')]:
        rr=[r['arms'][a]['relative_gap_upper']*100 for r in fresh['rows'] if r['condition_pass'] and r['arms'][a]['ok']]
        ax[1].plot(np.sort(rr),np.arange(1,len(rr)+1)/len(rr),label=f"{dict(constructive='Constructive',angular_fold='Angular',guide_qp='Guide QP',onecut='One-cut')[a]} ({len(rr)})",color=color,lw=1.8,ls=dict(constructive='-',angular_fold='--',guide_qp='-.',onecut=':')[a])
    ax[1].set_xlabel('Suboptimality upper bound (%)');ax[1].set_ylabel('Fraction of eligible answers');ax[1].set_title('(b) Complete energy tail');ax[1].legend(fontsize=9,loc='lower right',handlelength=1.2);ax[1].grid(alpha=.15)
    timing=[e['arms'][a]['time_ms']['p50'] for a in ARMS]
    ax[2].bar(x,timing,color=['#178570','#397cb5','#8b8b8b','#b47541']);ax[2].set_xticks(x,['Constr.','Angular','One-cut','Guide']);ax[2].set_ylabel('Median recovery time (ms)');ax[2].set_title('(c) Recovery and exact checks')
    for i,v in enumerate(timing):ax[2].text(i,v+7,f'{v:.1f}',ha='center',fontsize=8)
    ax[2].set_ylim(0,max(timing)*1.16)
    follow=json.loads((ROOT/'artifacts/review_affine_green.json').read_text())
    for arm,label,ls,col in [('direct','Direct Green','--','#705b9a'),('refined','Seed + one QP','-.','#555555')]:
        vals=sorted(100*r[arm]['relative_gap_upper'] for r in follow['rows'] if r['eligible'] and r[arm]['ok'])
        ax[3].plot(vals,np.arange(1,len(vals)+1)/len(vals),label=label+' (160)',ls=ls,color=col)
    ax[3].set_xscale('symlog',linthresh=1);ax[3].set_xlabel('Suboptimality upper bound (%)');ax[3].set_ylabel('Fraction of eligible answers');ax[3].set_title('(d) Follow-up: all new energy tails');ax[3].legend(fontsize=9,loc='lower right');ax[3].grid(alpha=.15)
    figstyle.save(fig,HERE/'fig_recovery_generalization');plt.close(fig)


def physical_row(row,arm,f,L):
    a=row['arms'][arm]
    if not a['ok']:return True,0
    G=decode(a['Gamma'])
    if not physical_certificate(G,row['obstacles']):return False,0
    A=np.asarray(constraint_matrix(5,0,3,2),dtype=object).T
    if not np.all(G.transpose(1,0,2).reshape(row['n'],18)@A==np.asarray(f.ms.B,dtype=object)):return False,0
    J=exact_energy(G)
    return J==F(*map(int,a['cost_exact'])) and J>=L,15


def recheck():
    t=time.perf_counter();old,fresh,pop=read();errors=validate(fresh,pop)
    if errors:raise AssertionError('\n'.join(errors))
    from revision_certified_solver import witness
    assert json.loads(json.dumps(witness()))==fresh['family_witness']
    n=polys=0
    for row in fresh['rows']:
        f=root_problem(row);ex=ExactSDPBounds(f)
        L=ex.c0
        if row['lower_kind']=='repaired_dual':
            candidate=ex.lower(row['root']['bound']['multipliers'],row['obstacles']);assert candidate is not None;L=max(L,candidate)
        assert L==F(*map(int,row['lower_exact']))
        G=coefficients(f,decode(row['reference_Y']));separation=endpoint_separation(G,row['obstacles'])
        assert separation==F(*map(int,row['separation_exact'])) and bool(separation>0)==row['condition_pass']
        for arm in ARMS:
            ok,p=physical_row(row,arm,f,L);assert ok,(row['id'],arm);polys+=p;n+=int(p>0)
    return dict(passed=True,instances=len(fresh['rows']),accepted_curves=n,strict_physical_polynomials=polys,exact_affine_and_energy_checks=n,exact_lower_bounds=len(fresh['rows']),seconds=time.perf_counter()-t)


def controls():
    _,fresh,pop=read();detected=[]
    for name,mut in [('missing case',lambda d:d['rows'].pop()),('geometry',lambda d:d['rows'][0]['obstacles'][0][0].__setitem__(0,99.)),
                     ('timing',lambda d:d['rows'][0]['arms']['constructive'].__setitem__('ms',-1)),('gap',lambda d:d['rows'][0]['arms']['constructive'].__setitem__('relative_gap_upper',0.5))]:
        d=copy.deepcopy(fresh);mut(d);assert validate(d,pop),name;detected.append(name)
    row=fresh['rows'][0];f=root_problem(row);L=F(*map(int,row['lower_exact']))
    for name in ('energy','boundary','collision'):
        d=copy.deepcopy(row);a=d['arms']['constructive']
        if name=='energy':a['cost_exact'][0]=str(int(a['cost_exact'][0])+1)
        else:
            G=decode(a['Gamma'])
            if name=='boundary':G[:,0,:]+=100
            else:
                G[:]=0
                for k,value in enumerate(row['obstacles'][0][0]):G[:,k,:]=F(float(value))
            a['Gamma']=encode(G)
        assert not physical_row(d,'constructive',f,L)[0],name;detected.append(name)
    return detected


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--recheck',action='store_true');ap.add_argument('--control',action='store_true');ap.add_argument('--report');args=ap.parse_args()
    if not args.recheck and not args.control:render(True);plot()
    report={}
    if args.recheck:report['exact']=recheck()
    if args.control:report['controls']=controls()
    if args.report:Path(args.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
