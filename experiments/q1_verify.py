"""Post-run QA: bind complete archives to frozen inputs and replay certificates.

Physical arithmetic is separately implemented. Mode reconstruction and exact
SDP replay reuse declared project helpers and are identified separately.
"""
from pathlib import Path
from fractions import Fraction as F
from math import comb, nextafter, inf
import argparse
import hashlib
import json
import sys
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
import q1_independent as check

FACTORIAL_ARMS = ['affine_bubble', 'affine_green', 'sdp_bubble', 'sdp_green',
                  'sdp_gap', 'sdp_selector', 'sdp_green_allaxes',
                  'iterated_cut_qp', 'workspace_sdp_green']
F2 = dict(d=5, k=1, l=0, N=3, eta=2)


def require(value, label):
    if not value:
        raise ValueError(label)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fraction(value):
    return F(*map(int, value))


def fraction_record(value):
    return [str(value.numerator), str(value.denominator)]


def frozen_inputs(path, data):
    """Return archive kind and original row mapping after identity checks."""
    p, rows = data['protocol'], data['rows']
    require(data['complete'] is True, 'incomplete')
    require(rows and len({r['id'] for r in rows}) == len(rows), 'empty/duplicate ids')
    family_archive = 'families' in p or p.get('arm') == 'affine_green'
    protocol_path = path.with_suffix('.protocol.json')
    if family_archive and not protocol_path.exists():
        protocol_path = ROOT/'artifacts/q1_families.protocol.json'
    frozen = json.loads(protocol_path.read_text())
    if family_archive:
        require(data['protocol_sha256'] == digest(protocol_path), 'family protocol hash')
        require(p == frozen['protocol'], 'family embedded protocol')
        original = frozen['rows']
        require(json_hash(original) == frozen['input_sha256'] == data['input_sha256'], 'family input hash')
        require(len(rows) == p['instances'] == len(original), 'family population count')
        require([r['id'] for r in rows] == [o['id'] for o in original], 'family ids/order')
        declared = {x['label']: x['family'] for x in frozen['families']}
        for r, o in zip(rows, original):
            require(all(r.get(k) == v for k, v in o.items()), 'frozen family input/boundaries')
            require(r['family'] == declared[r['family_label']], 'declared family')
            require(r['lower_exact'] == ['16', '1'] and r['lower_kind'] == 'obstacle_free', 'family lower bound')
        return 'families', {o['id']: o for o in original}, [p.get('arm', 'affine_green')], 'exact frozen row list'
    require(p == frozen, 'embedded protocol differs from frozen sidecar')
    if 'protocol_sha256' in data:
        require(data['protocol_sha256'] == digest(protocol_path), 'protocol hash')
    if 'arms' in p:
        require(p['arms'] == FACTORIAL_ARMS, 'declared factorial arms')
        source = ROOT/'artifacts/revision_recovery_generalization.json'
        require(digest(source) == p['input_sha256'], 'factorial input hash')
        original = json.loads(source.read_text())['rows'][:p['instances']]
        require(len(rows) == p['instances'] == len(original), 'factorial population count')
        require([r['id'] for r in rows] == [o['id'] for o in original], 'factorial ids/order')
        for r, o in zip(rows, original):
            require(r['obstacles'] == o['obstacles'] and r['lower_exact'] == o['lower_exact'], 'frozen geometry/lower bound')
            require(r['family'] == F2 and r['n'] == o['n'], 'factorial family')
            require(r['bc0'] == [[-2]+[0]*(o['n']-1)] and r['bc1'] == [[2]+[0]*(o['n']-1)], 'factorial boundaries')
            require(r['stratum'] == o['stratum'] and r['vertical_condition'] == o['condition_pass'], 'factorial strata')
        return 'factorial', {o['id']: o for o in original}, p['arms'], 'exact frozen prefix'
    require('budget' in p and 'warm' in p, 'unrecognized archive schema')
    source = ROOT/'artifacts/revision_streaming_F2.json'
    require(digest(source) == p['input_sha256'], 'integrated input hash')
    original = json.loads(source.read_text())['rows']
    lookup = {(r['seed'], r['step']): r for r in original}
    require(len(p['budget']) == len(set(p['budget'])) and len(p['warm']) == len(set(p['warm'])), 'duplicate protocol arms')
    require(all(type(b) is int and b >= 0 for b in p['budget']) and all(type(w) is bool for w in p['warm']), 'budget/warm types')
    # Older measured protocols record count but not filter arguments. Bind their
    # rectangular prefix filter to available original geometries, and disclose
    # that the selection dimensions were reconstructed, not frozen explicitly.
    seeds = p.get('seeds', 1+max(r['seed'] for r in rows))
    steps = p.get('steps', 1+max(r['step'] for r in rows))
    require(type(seeds) is int and type(steps) is int and seeds > 0 and steps > 0, 'integrated filters')
    selected = [r for r in original if r['seed'] < seeds and r['step'] < steps]
    require(len(selected) == p['instances'], 'integrated selected population')
    keys = [(b, w, r['seed'], r['step']) for b in p['budget'] for w in p['warm'] for r in selected]
    require([(r['budget'], r['warm'], r['seed'], r['step']) for r in rows] == keys, 'integrated coverage/order')
    bound = {}
    for r in rows:
        key = (r['seed'], r['step'])
        require(r['id'] == f"b{r['budget']}_w{int(r['warm'])}_s{r['seed']}_t{r['step']}", 'integrated id')
        require(r['obstacles'] == lookup[key]['obstacles'] and r['family'] == F2, 'integrated input/family')
        require(r['bc0'] == [[-2, 0]] and r['bc1'] == [[2, 0]], 'integrated boundaries')
        bound[r['id']] = lookup[key]
    selection = 'explicit frozen filters' if 'seeds' in p and 'steps' in p else 'rectangular filter dimensions reconstructed; original protocol froze only instance count'
    return 'integrated', bound, ['green'], selection


def normalized_mode(z):
    if z is None:
        return None
    z = np.asarray(z, dtype=object)
    d = z.shape[1]-1
    value = sum(z[len(z)//2, j]*F(comb(d, j), 2**d) for j in range(d+1))
    return None if value == 0 else z/value


def binding_context(row, old, kind, cache):
    """Rebuild modes/references, never calling an optimization entry point."""
    from multisegment import MultiSegment
    from certified_multisegment import CertifiedSpline
    from exact_sdp_bounds import ExactSDPBounds
    from constructive_recovery import coefficients
    from q1_recovery import modes_for
    fam = row['family']; n = row.get('n', len(row['bc0'][0]))
    key = json.dumps([fam, n, row['bc0'], row['bc1'], len(row['obstacles'])], sort_keys=True)
    if key not in cache:
        f = CertifiedSpline(MultiSegment(n=n, obstacles=row['obstacles'], bc0=row['bc0'], bc1=row['bc1'], **fam))
        exact = ExactSDPBounds(f)
        cache[key] = (f, exact, modes_for(f, exact))
    f, exact, base_modes = cache[key]
    affine = coefficients(f, np.zeros((n, f.r)))
    Y = v = None
    if kind == 'factorial':
        Y = check.decode(old['reference_Y'])
        v = None if old['gap_v'] is None else check.decode(old['gap_v'])
    elif kind == 'integrated' and row['numerical']['ok']:
        numerical = row['numerical']
        if numerical['source'] == 'factor':
            Y, v, *_ = f.unpack(numerical['z'])
        else:
            Y = numerical['conic']['Y']
    reference = None if Y is None else coefficients(f, Y)
    modes = {k: normalized_mode(vv) for k, vv in base_modes.items() if k != 'gap'}
    if v is not None:
        from escape_fold import rational
        modes['gap'] = normalized_mode((rational(v)@f.Qq.T).reshape(fam['N'], fam['d']+1))
    else:
        modes['gap'] = None
    return dict(affine=affine, sdp=reference, modes=modes, exact=exact)


def check_binding(row, name, a, context):
    """Bind successful tail and selected curve to the declared arm, exactly."""
    if name == 'iterated_cut_qp':
        # This declared legacy comparator is a QP sequence, not an amplitude arm.
        if a.get('ok') and a.get('method') == 'plain':
            require(np.array_equal(check.decode(a['Gamma']), context['sdp']), 'QP plain reference')
        return
    ref = context['affine'] if name.startswith('affine') else context['sdp']
    if ref is None:
        require(not a['ok'] and not a.get('attempts'), 'recovery without numerical reference')
        return
    allowed = ['gap', 'green'] if name == 'sdp_selector' else ['bubble' if 'bubble' in name else 'gap' if name == 'sdp_gap' else 'green']
    n = ref.shape[1]
    axes = [1]+[i for i in range(n) if i != 1] if name.endswith('allaxes') else [1]
    successful = []
    for t in a.get('attempts', []):
        require(t.get('mode') in allowed, 'attempt mode attribution')
        require(isinstance(t.get('ok'), bool) and 'status' in t, 'attempt outcome')
        if not t['ok']:
            continue
        mode = context['modes'][t['mode']]
        require(mode is not None, 'unavailable mode claimed')
        require(np.array_equal(check.decode(t['reference']), ref), 'attempt reference binding')
        require(np.array_equal(check.decode(t['z']), mode), 'attempt mode/normalization binding')
        require(t['axis'] in axes and t['sign'] in (-1, 1), 'attempt direction')
        successful.append(t)
    if not a['ok']:
        return
    G = check.decode(a['Gamma'])
    if a.get('method') == 'plain':
        require(np.array_equal(G, ref) and fraction(a['amplitude']) == 0 and not a.get('attempts'), 'plain reference binding')
        return
    method = a.get('method')
    require(method in allowed and context['modes'][method] is not None, 'selected mode attribution')
    require(a['axis'] in axes and a['sign'] in (-1, 1), 'selected direction')
    alpha = fraction(a['amplitude'])
    require(alpha >= 0, 'negative selected amplitude')
    expected = ref.copy()
    expected[:, a['axis'], :] += a['sign']*alpha*context['modes'][method]
    require(np.array_equal(G, expected), 'selected reference/mode/amplitude binding')
    require(any(t['mode'] == method and t['axis'] == a['axis'] and t['sign'] == a['sign']
                and 'candidate_amplitude' in t and fraction(t['candidate_amplitude']) == alpha
                for t in successful), 'selected candidate missing from attempt trace')


def gap_record(row, name, a, lower, field='normalized_gap_upper'):
    J, L = fraction(a['cost_exact']), fraction(lower)
    require(J >= L, 'physical lower bound')
    gap = (J-L)/max(F(1), abs(L))
    if field in a:
        require(np.isfinite(a[field]) and a[field] == float(gap), 'normalized gap arithmetic')
    return dict(id=row['id'], arm=name, lower_exact=fraction_record(L),
                gap_exact=fraction_record(J-L), normalized_gap_exact=fraction_record(gap),
                normalized_gap_outward=nextafter(float(gap), inf))


def verify(path, check_cells=True, check_sdp=True):
    path = Path(path)
    data = json.loads(path.read_text())
    kind, originals, expected_arms, selection = frozen_inputs(path, data)
    rows = data['rows']; curves = polys = cells = intervals = 0
    clearance = []; gaps = []; legacy_status = []; cache = {}
    for row in rows:
        arms = row.get('arms', {})
        require(set(arms) == set(expected_arms), 'missing/extra arms')
        context = binding_context(row, originals[row['id']], kind, cache)
        for name, a in arms.items():
            require(isinstance(a.get('ok'), bool), 'missing outcome')
            if 'status' not in a:
                require(name == 'iterated_cut_qp' and isinstance(a.get('reason'), str), 'missing outcome status')
                legacy_status.append(dict(id=row['id'], arm=name, status='CERTIFIED_PHYSICAL' if a['ok'] else 'NO_CERTIFIED_CANDIDATE', reason=a['reason']))
            check_binding(row, name, a, context)
            if check_cells:
                for t in a.get('attempts', []):
                    if t['ok']:
                        cells += check.tail(t, row['obstacles'], row['bc0'], row['bc1'], row['family']['eta'])
            if not a['ok']:
                continue
            require('Gamma' in a and 'cost_exact' in a, 'successful physical payload omitted')
            G = check.decode(a['Gamma']); fam = row['family']; n = row.get('n', len(row['bc0'][0]))
            require(G.shape == (fam['N'], n, fam['d']+1), 'declared shape mismatch')
            require(all(len(c) == n and r >= 0 for c, r in row['obstacles']), 'obstacle dimensions/radius')
            v = check.curve(a, row['obstacles'], fam['k'], row['bc0'], row['bc1'], fam['eta'])
            curves += 1; polys += v['polynomials']
            clearance.append(dict(id=row['id'], arm=name, lower=str(v['squared_clearance_lower']), upper=str(v['squared_clearance_upper'])))
            if name.startswith('workspace'):
                limits = [F(3)]+[F(2)]*(n-1)
                require(all(abs(x) <= limits[j] for g in G for j, c in enumerate(g) for x in c), 'workspace control hull')
            if 'lower_exact' in row:
                field = 'obstacle_free_normalized_gap_upper' if kind == 'families' else 'normalized_gap_upper'
                gaps.append(gap_record(row, name, a, row['lower_exact'], field))
        if kind == 'integrated':
            stages = row['stages']; total = row['objective_return_ms']; physical = row['physical_return_ms']
            require(set(stages) == {'root_ms', 'physical_recovery_ms', 'exact_sdp_ms'}, 'stage names')
            require(all(isinstance(v, (int, float)) and np.isfinite(v) and v >= 0 for v in stages.values()), 'stage times')
            require(np.isfinite(total) and np.isfinite(physical) and 0 <= physical <= total
                    and abs(total-sum(stages.values())-row['packing_overhead_ms']) < 1e-7
                    and row['packing_overhead_ms'] >= 0, 'timer accounting')
            require(row['physical_ok'] == arms['green']['ok'], 'physical status')
            numerical, e = row['numerical'], row['exact']
            require(row['objective_ok'] == bool(row['physical_ok'] and e and e['ok']), 'objective status')
            if not numerical['ok']:
                require(not row['physical_ok'] and e is None, 'result after numerical failure')
            if check_sdp and numerical['ok']:
                c = numerical.get('conic', numerical); point = None
                if numerical['source'] == 'conic':
                    require(row.get('upper_point') is not None, 'actual conic upper witness omitted')
                    point = tuple(check.decode(x) for x in row['upper_point'])
                    from escape_fold import rational
                    require(np.array_equal(point[0], rational(c['Y'])), 'conic witness reference binding')
                got = context['exact'].verify(numerical.get('z'), c['bound']['multipliers'], row['obstacles'], point=point)
                for key in ('ok', 'reason', 'lower_exact', 'upper_exact', 'shift', 'n_polys', 'n_pd_blocks'):
                    require(got.get(key) == e.get(key), 'SDP replay '+key)
                intervals += int(got['ok'])
            if row['objective_ok']:
                gaps.append(gap_record(row, 'green', arms['green'], e['lower_exact']))
    return dict(passed=True, input_sha256=digest(path), verifier_sha256=digest(Path(__file__)),
                checker_sha256=digest(ROOT/'src/q1_independent.py'), rows=len(rows), curves=curves,
                polynomials=polys, cells=cells, sdp_intervals=intervals,
                cells_replayed=check_cells, sdp_replayed=check_sdp, population_binding=selection,
                scope='Physical algebra/sign/affine/energy/cells separately implemented; mode/reference reconstruction and SDP bounds reuse project helpers, independent of optimizer execution',
                normalized_gap_audit=gaps, normalized_legacy_comparator_status=legacy_status,
                squared_clearance_brackets=clearance)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', type=Path); ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    try:
        report = verify(args.input)
    except Exception as exc:
        args.out.write_text(json.dumps(dict(passed=False, error_type=type(exc).__name__, error=str(exc)), indent=2)+'\n')
        raise
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print({k: v for k, v in report.items() if k not in ('squared_clearance_brackets', 'normalized_gap_audit', 'normalized_legacy_comparator_status')})


if __name__ == '__main__':
    main()
