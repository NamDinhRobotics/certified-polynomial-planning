"""Bounded recovery transfer across four existing certified k=1 families.

Prepare freezes the complete protocol/input population before any recovery
outcome, then rechecks exact family witnesses. Run is serial and explicit.
This is an exploratory matched-geometry study, not held-out validation or a
broad family theorem. It does not run any SDP solver or compare solver speed.
"""
from pathlib import Path
from fractions import Fraction as F
import argparse
import hashlib
import json
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments')]
FAMILIES = [
    ('F2_reference', dict(d=5, k=1, l=0, N=3, eta=2)),
    ('cubic_two', dict(d=3, k=1, l=0, N=2, eta=1)),
    ('quintic_two', dict(d=5, k=1, l=0, N=2, eta=2)),
    ('quintic_four', dict(d=5, k=1, l=0, N=4, eta=2)),
]
BC0, BC1 = [[-2, 0]], [[2, 0]]
SOURCE_NAMES = [
    'experiments/q1_families.py', 'experiments/q1_verify.py',
    'src/q1_recovery.py', 'src/q1_independent.py', 'src/exact_api.py',
    'src/escape_fold.py', 'src/constructive_recovery.py',
    'src/exact_green.py', 'src/exact_sdp_bounds.py',
    'src/certified_multisegment.py', 'src/multisegment.py',
    'src/bernstein.py', 'src/mlukacs.py', 'src/exact_check.py',
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def write(path, data):
    # Only the declared artifacts are written, with no temporary outside paths.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def geometry(index):
    """Integer-seeded, bounded geometry; serialized floats are exact binary data."""
    seed = 2026090800 + index
    rng = random.Random(seed)
    obs = [([rng.randint(-150, 150) / 1000,
             rng.randint(-30, 30) / 1000], rng.randint(300, 420) / 1000)]
    for _ in range(2):
        obs.append(([rng.randint(-1100, 1100) / 1000,
                     rng.randint(-400, 400) / 1000],
                    rng.randint(180, 280) / 1000))
    if index < 6:
        stratum = 'clutter'
    elif index < 9:
        stratum = 'near_endpoint'
        side = -1 if index % 2 == 0 else 1
        radius = (180 + 10 * (index - 6)) / 1000
        delta = [0.002, 0.01, 0.05][index - 6]
        obs[1] = ([side * (2 - radius - delta), 0.0], radius)
    else:
        stratum = 'outside_vertical_condition'
        # Both positions remain strictly outside every ball, but the vertical
        # line through the left endpoint intersects this ball. The central
        # ball also obstructs the fixed affine reference in every case.
        obs[1] = ([-2.0, 0.45 + 0.05 * (index - 9)], 0.2)
    return dict(geometry_id=f'geometry/{index:02d}', seed=seed,
                stratum=stratum, obstacles=obs)


def freeze_record():
    table_path = ROOT / 'artifacts/cert_table.json'
    table = json.loads(table_path.read_text())['entries']
    selected = []
    for label, family in FAMILIES:
        entry = next(x for x in table
                     if all(x.get(k) == v for k, v in family.items()))
        assert entry['certified'] is True and entry['K_definite'] is True
        selected.append(dict(label=label, family=family, table_entry=entry))
    rows = []
    # Interleave families for each shared scene; rotate their order so the
    # single serial pass does not always process F2 first.
    for i in range(12):
        order = FAMILIES[i % 4:] + FAMILIES[:i % 4]
        for label, family in order:
            rows.append(dict(id=f'{label}/{i:02d}', family_label=label,
                             family=family, n=2, bc0=BC0, bc1=BC1,
                             **geometry(i)))
    assert len(rows) == 48 and len({r['id'] for r in rows}) == 48
    return dict(
        scope='Exploratory transfer on four existing certified k=1 families; '
              'matched geometries, not held-out or a broad family proof',
        families=selected, rows=rows, input_sha256=json_hash(rows),
        source_hashes={n: digest(ROOT / n) for n in SOURCE_NAMES},
        cert_table_sha256=digest(table_path),
        environment=dict(python=sys.version, executable=sys.executable,
                         platform=platform.platform()),
        protocol=dict(
            instances=48, per_family=12, obstacle_count=3,
            strata_per_family=dict(clutter=6, near_endpoint=3,
                                   outside_vertical_condition=3),
            geometry='random.Random seeds2026090800..2026090811; all generated '
                     'inputs serialized before recovery; same scenes per family',
            reference='exact minimum derivative-energy affine particular curve '
                      'between(-2,0) and(2,0), fixed independently of obstacles; '
                      'verified equal to the straight line of energy16',
            arm='affine_green',
            mode='q1_recovery.modes_for canonical Green mode: '
                 'q_star=(floor(N/2)+1/2)/N; unit value at that point. '
                 'For even N this is not global time1/2.',
            directions='vertical axis1, signs(-1,+1), both attempted unless '
                       'the identical plain-reference early-return gate accepts',
            polish_steps=12, cell_max_depth=24, cell_max_nodes=10000,
            construction_margin='dyadic halving from1/2; maximum256 decreases',
            physical_gate='strict physical p>0 on entire intervals; exact '
                          'binary-rational coefficients, zero tolerance',
            failure_policy='retain all48 rows, all attempts, all unknown statuses '
                           'and exceptions; never infer original infeasibility',
            timing='one serial pass; acquisition/model/mode setup charged per '
                   'row separately from recover; independent q1_verify replay '
                   'reported separately; no solver timing, speedup, '
                   'common-parameter optimality or higher-k claim',
            lower_bound='exact obstacle-free derivative-energy16 only; '
                        'no same-instance optimized SDP bound is claimed',
            witness='allN^2 exact coefficient blocks at degree96, nonzero '
                    'blocks, endpoint rows/corners, exact reducedK LDL, '
                    'nullspace and block identities, homogeneous mode and '
                    'affine reference constraints',
        ))


def build(family, obstacles):
    from multisegment import MultiSegment
    from certified_multisegment import CertifiedSpline
    from exact_sdp_bounds import ExactSDPBounds
    f = CertifiedSpline(MultiSegment(n=2, obstacles=obstacles,
                                    bc0=BC0, bc1=BC1, **family))
    return f, ExactSDPBounds(f)


def witness(family):
    import numpy as np
    from exact_green import (coefficient_blocks, elevated_min,
                             endpoint_structure, energy_nullity,
                             constraint_matrix, solve)
    from exact_sdp_bounds import positive_definite
    from constructive_recovery import encode, coefficients
    from q1_recovery import modes_for, normalize
    import q1_independent as independent

    d, k, l, N, eta = [family[x] for x in ('d', 'k', 'l', 'N', 'eta')]
    blocks, r = coefficient_blocks(d, k, l, N, eta)
    f, exact = build(family, [([0.0, 0.0], 0.3)])
    assert f.r == r and positive_definite(exact.K)
    constraints = np.asarray(constraint_matrix(d, l, N, eta), dtype=object)
    assert np.all(constraints @ f.Qq == 0)
    inverse_q = np.asarray(solve(exact.K.tolist(), f.Qq.T.tolist()),
                           dtype=object)
    records = []
    for pair, block in sorted(blocks.items()):
        i, j = pair
        independently_formed = f.Qq[f.ms.slice_(i)] @ inverse_q[:, f.ms.slice_(j)]
        assert np.array_equal(independently_formed, np.asarray(block, dtype=object))
        minimum = elevated_min(block, d, 96)
        nonzero = any(x != 0 for row in block for x in row)
        assert minimum >= 0 and nonzero
        records.append(dict(pair=list(pair), coefficients=encode(block),
                            elevated_minimum=str(minimum), nonzero=nonzero,
                            nonnegative=True, block_identity=True))
    ends = endpoint_structure(d, k, l, N, eta)
    energy = energy_nullity(d, k, l, N, eta)
    assert ends['closed_square_ok'] and ends['n_live_endpoints'] == 2 * (N - 1)
    assert energy['definite'] and energy['nullity'] == 0
    reference = coefficients(f, np.zeros((2, r)))
    expected = np.array([[[F(-2) + 4 * (F(i, N) + F(j, N * d))
                           for j in range(d + 1)], [F(0)] * (d + 1)]
                         for i in range(N)], dtype=object)
    assert np.array_equal(reference, expected)
    assert independent.affine(reference, BC0, BC1, eta)
    assert independent.energy(reference, k) == 16
    mode = modes_for(f, exact)['green']
    assert normalize(mode) is not None
    assert independent.affine(mode[:, None, :], [[0]], [[0]], eta)
    return dict(passed=True, family=family, r=r, elevation_degree=96,
                exact_blocks=records, endpoint=ends, energy_nullity=energy,
                exact_K=encode(exact.K), exact_Q=encode(f.Qq),
                positive_definite_K=True, nullspace_identity=True,
                canonical_mode=encode(mode), reference=encode(reference),
                reference_energy='16', affine_reference_verified=True,
                positive_homogeneous_mode_verified=True)


def check_sources(frozen):
    assert digest(ROOT / 'artifacts/cert_table.json') == frozen['cert_table_sha256']
    assert json_hash(frozen['rows']) == frozen['input_sha256']
    for name, expected in frozen['source_hashes'].items():
        assert digest(ROOT / name) == expected, 'Frozen source changed: ' + name


def prepare(protocol_path, witness_path):
    if protocol_path.exists() or witness_path.exists():
        raise FileExistsError('Refusing to overwrite frozen protocol/witness')
    record = freeze_record()
    write(protocol_path, record)  # Before mode or recovery outcomes.
    checks = [dict(label=label, **witness(family)) for label, family in FAMILIES]
    write(witness_path, dict(passed=all(x['passed'] for x in checks),
                            protocol_sha256=digest(protocol_path),
                            source_hashes=record['source_hashes'], families=checks))
    print('PREPARED', len(record['rows']), 'rows; exact family witnesses passed',
          digest(protocol_path), flush=True)


def run(protocol_path, witness_path, out):
    if out.exists():
        raise FileExistsError(out)
    import numpy as np
    import mlukacs
    mlukacs._C_POLISH = False
    from q1_recovery import recover, modes_for
    from constructive_recovery import coefficients
    frozen = json.loads(protocol_path.read_text())
    family_witness = json.loads(witness_path.read_text())
    assert family_witness['passed']
    assert family_witness['protocol_sha256'] == digest(protocol_path)
    check_sources(frozen)
    rows = []
    for old in frozen['rows']:
        begin = time.perf_counter()
        setup_ms = None
        try:
            f, exact = build(old['family'], old['obstacles'])
            reference = coefficients(f, np.zeros((2, f.r)))
            mode = modes_for(f, exact)['green']
            setup_ms = 1000 * (time.perf_counter() - begin)
            result = recover(reference, [('green', mode)], old['obstacles'],
                             k=f.ms.k, axes=(1,), polish_steps=12)
            if result['ok']:
                J = F(*map(int, result['cost_exact']))
                assert J >= 16
                result['obstacle_free_normalized_gap_upper'] = float((J - 16) / 16)
        except Exception as exc:
            result = dict(ok=False, status='EXPERIMENT_EXCEPTION',
                          error_type=type(exc).__name__, error=str(exc),
                          reason='retained_exception_not_infeasibility')
        rows.append(dict(**old, setup_ms=setup_ms,
                         total_ms=1000 * (time.perf_counter() - begin),
                         lower_exact=['16', '1'], lower_kind='obstacle_free',
                         arms=dict(affine_green=result)))
        if len(rows) == 48:
            check_sources(frozen)
        # Keep partial records even when interrupted after a case.
        write(out, dict(complete=len(rows) == 48, protocol=frozen['protocol'],
                        scope=frozen['scope'], input_sha256=frozen['input_sha256'],
                        protocol_sha256=digest(protocol_path),
                        witness_sha256=digest(witness_path),
                        source_hashes=frozen['source_hashes'], rows=rows))
        print('families', len(rows), '/48', old['id'], result['status'], flush=True)


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--run', action='store_true')
    parser.add_argument('--protocol', type=Path,
                        default=ROOT / 'artifacts/q1_families.protocol.json')
    parser.add_argument('--witness', type=Path,
                        default=ROOT / 'artifacts/q1_families.witness.json')
    parser.add_argument('--out', type=Path,
                        default=ROOT / 'artifacts/q1_families.json')
    args = parser.parse_args()
    for path in (args.protocol, args.witness, args.out):
        if (path.resolve().parent != (ROOT / 'artifacts').resolve()
                or not path.name.startswith('q1_families')
                or path.suffix != '.json'):
            parser.error('Outputs must be artifacts/q1_families*.json')
    if args.prepare:
        prepare(args.protocol, args.witness)
    else:
        run(args.protocol, args.witness, args.out)


if __name__ == '__main__':
    main()
