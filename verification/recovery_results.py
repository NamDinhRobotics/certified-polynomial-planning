"""Computed recovery claims, complete-population validation and exact replay."""
from pathlib import Path
import copy, hashlib, json, sys, time
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
