"""Frozen benchmark population and physical-clearance checks."""
import copy, time, sys
import numpy as np
from revision_claims import ROOT,invariants


def controls(data):
    mutations=[('family',lambda d:d['revision_streaming_F2']['config']['family'].__setitem__('N',1)),
      ('population',lambda d:d['revision_streaming_F1']['config'].__setitem__('seeds',list(range(19)))),
      ('missing recheck',lambda d:d['revision_certificates']['rows'][0].pop('recheck')),
      ('endpoint condition',lambda d:d['revision_certificates']['rows'][0]['base'].__setitem__('certified',True)),
      ('matched margin',lambda d:next(r for r in d['revision_tinysdp']['rows'] if r['arm']=='kkt_one_margin1')['env'].__setitem__('TINYSDP_D4_INFLATE',1.5)),
      ('all obstacles',lambda d:next(r for r in d['revision_tinysdp']['rows'] if r['arm']=='kkt_one_margin1')['env'].__setitem__('TINYSDP_D4_ALLCUTS',0)),
      ('first versus termination',lambda d:d['revision_frozen']['rows'][0]['bnb_first'].__setitem__('conic_solves',10**6)),
      ('polynomial coverage',lambda d:next(r['arms']['plain'] for r in d['revision_streaming_F1']['rows'] if r['arms']['plain']['accepted'])['exact'].__setitem__('n_polys',1)),
      ('dense is certificate',lambda d:d['revision_tinysdp']['config'].__setitem__('dense_check_is_certificate',True)),
      ('tiny duplicate scene-arm',lambda d:d['revision_tinysdp']['rows'][1].__setitem__('arm','sdp')),
      ('tiny time summary',lambda d:d['revision_tinysdp']['rows'][0]['summary']['plan_ms'].__setitem__('p50',1e6)),
      ('historical denominator',lambda d:d['a83_tinysdp_ladder']['table']['vertical_gate']['commit'].__setitem__('speedup',1)),
      ('executed effort',lambda d:d['revision_tinysdp']['rows'][0]['repeats'][0]['tracking'][1].__setitem__('u1',100)),
      ('replanning cadence',lambda d:d['revision_replanning']['rows'][0]['env'].__setitem__('TINYSDP_3D_REPLAN_STRIDE',9)),
      ('replanning goal',lambda d:d['revision_replanning']['rows'][0]['result'].__setitem__('success',True)),
      ('replanning missing call',lambda d:d['revision_replanning']['rows'][0]['result']['plans'].pop()),
      ('post-hoc status',lambda d:d['revision_cadence']['config'].__setitem__('post_hoc',False))]
    for name,mutate in mutations:
        damaged=copy.deepcopy(data);mutate(damaged)
        errors=invariants(damaged)
        if not errors:raise AssertionError('mutation survived: '+name)
        print('CONTROL detected:',name)
    return len(mutations)

def recheck(data):
    sys.path.insert(0,str(ROOT/'src'))
    from exact_check import certify_obstacle_curves
    n=0;polys=0;t=time.perf_counter()
    for f in ['F1','F2']:
        x=data['revision_streaming_'+f];cfg=x['config']['family'];d=cfg['d']
        for row in x['rows']:
            obs=[(np.asarray(c),r) for c,r in row['obstacles']]
            for a in row['arms'].values():
                if not a['accepted']:continue
                G=np.asarray(a['Gamma']);parts=[G[:,i*(d+1):(i+1)*(d+1)] for i in range(cfg['N'])]
                v=certify_obstacle_curves(parts,obs,d,1e-9)
                if not v['ok']:raise AssertionError(f'clearance failed {f}/{row["seed"]}/{row["step"]}')
                n+=1;polys+=v['n_polys']
        print('RATIONAL',f,'complete;',n,'accepted records checked',flush=True)
    return dict(accepted_records=n,polynomials=polys,seconds=time.perf_counter()-t,passed=True)
