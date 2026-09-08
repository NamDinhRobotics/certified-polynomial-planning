"""Independent exact audit of all physical curves in the recovery addendum."""
import sys, json, hashlib, copy
from pathlib import Path
from fractions import Fraction as F
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
from exact_green import constraint_matrix,q_gram_deriv
from escape_fold import physical_certificate
from revision_certified_solver import model
from revision_green_recovery import decode


def check(rec,obs,B):
    gg=decode(rec['Gamma_exact'],rec['Gamma_shape'])
    if not physical_certificate(gg,obs):return False
    A=np.asarray(constraint_matrix(5,0,3,2),dtype=object).T
    if not np.all(gg.transpose(1,0,2).reshape(2,18)@A==B):return False
    H=3*np.asarray(q_gram_deriv(5,1),dtype=object)
    J=sum(np.sum((g@H)*g) for g in gg)
    return J==F(*map(int,rec['cost_exact'])) and J>=F(*map(int,rec['lower_exact']))


def main():
    paths=[ROOT/'artifacts'/n for n in ('revision_escape_fold.json','revision_green_recovery.json')]
    documents=[json.loads(p.read_text()) for p in paths]
    hashes=[]
    source_path=ROOT/'artifacts/revision_certified_solver.json'
    source_hash=hashlib.sha256(source_path.read_bytes()).hexdigest()
    for path,d in zip(paths,documents):
        assert d['protocol']==json.loads(path.with_suffix('.protocol.json').read_text())
        assert d['protocol']['source_sha256']==source_hash
    assert documents[1]['protocol']['first_stage_sha256']==hashlib.sha256(paths[0].read_bytes()).hexdigest()
    for d in documents:
        for name,h in d['protocol']['source_hashes'].items():
            from executed_source import checked_source
            p=checked_source(ROOT,name,h)
            hashes.append(str(p.relative_to(ROOT)))
    data=json.loads((ROOT/'artifacts/revision_certified_solver.json').read_text())
    original={(r['seed'],r['step']):r for r in data['rows'] if r['repeat']==0}
    rows=documents[1]['rows'];B=np.asarray(model(rows and original[(0,0)]['obstacles']).ms.B,dtype=object)
    assert len(rows)==1000 and len({(r['seed'],r['step']) for r in rows})==1000
    for r in rows:
        source=original[(r['seed'],r['step'])]
        assert r['lower_exact']==source['exact']['factor']['lower_exact']
        assert check(r,source['obstacles'],B)
    r=rows[0];obs=original[(r['seed'],r['step'])]['obstacles'];controls=[]
    bad=copy.deepcopy(r);bad['cost_exact']=[str(int(bad['cost_exact'][0])+1),bad['cost_exact'][1]]
    controls.append(not check(bad,obs,B))
    bad=copy.deepcopy(r);gg=decode(bad['Gamma_exact'],bad['Gamma_shape']);gg[:,0,:]+=100
    bad['Gamma_exact']=[[str(x.numerator),str(x.denominator)] for x in gg.flat]
    controls.append(not check(bad,obs,B))
    bad=copy.deepcopy(r);gg=decode(bad['Gamma_exact'],bad['Gamma_shape']);gg[:]=0
    gg[:,0,:]=F(float(obs[0][0][0]));gg[:,1,:]=F(float(obs[0][0][1]))
    bad['Gamma_exact']=[[str(x.numerator),str(x.denominator)] for x in gg.flat]
    controls.append(not check(bad,obs,B));assert all(controls)
    report=dict(passed=True,curves=1000,strict_obstacle_polynomials=9000,exact_affine_checks=1000,exact_energy_checks=1000,
                independent_lower_bound_record_matches=1000,mutation_controls=controls,verified_sources=hashes,
                scope='physical curves; lower bounds replay prior independently audited SDP-bound records')
    print(json.dumps(report,indent=2))
    return report
