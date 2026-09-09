#!/usr/bin/env python3
"""Verify V3 data integrity, row accounting, and reported descriptive statistics.

Use --physical to replay rational affine, clearance, energy, workspace and
derivative checks. This does not rerun optimizers, SDP dual certificates,
bootstrap intervals, or dynamics. Run from any directory with Python 3.12+.
"""
import argparse
import copy
import csv
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'artifacts' / 'v3'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fraction(value):
    if isinstance(value, str):
        value = json.loads(value)
    return F(int(value[0]), int(value[1]))


def load_data():
    manifest = json.loads((DATA / 'MANIFEST.json').read_text())
    actual = {str(p.relative_to(DATA)) for p in DATA.rglob('*')
              if p.is_file() and p.name != 'MANIFEST.json'}
    require(actual == set(manifest), 'Data file inventory mismatch')
    for name, meta in manifest.items():
        data = (DATA / name).read_bytes()
        require(len(data) == meta['bytes'] and
                hashlib.sha256(data).hexdigest() == meta['sha256'], name)
    rows = [json.loads(p.read_text()) for p in sorted((DATA / 'raw').glob('*.json'))]
    records = {r['run_id']: r for r in rows}
    require(len(rows) == len(records) == 720, 'Expected 720 unique raw records')
    with (DATA / 'V3_FINAL_AB_PAIRED.csv').open(newline='') as f:
        pairs = list(csv.DictReader(f))
    with (DATA / 'V3_FINAL_MUJOCO_RUNS.csv').open(newline='') as f:
        robot = list(csv.DictReader(f))
    return records, pairs, robot, len(manifest)


def check_rows(records, pairs, robot):
    expected = {(seed, step, rep) for seed in range(31000, 31020)
                for step in range(8) for rep in range(2)}
    keys = [(int(r['seed']), int(r['step']), int(r['repeat'])) for r in pairs]
    require(len(keys) == 320 and set(keys) == expected, 'Incomplete paired population')
    used = set()
    for pair in pairs:
        for arm in ('conic', 'factor'):
            r = records[pair[arm + '_run_id']]
            require(r['run_id'] not in used, 'Duplicate arm record')
            used.add(r['run_id'])
            require(r['arm'] == arm and r['instance_id'] == pair['instance_id'], 'Pair identity')
            require((r['seed'], r['step'], r['repeat']) ==
                    (int(pair['seed']), int(pair['step']), int(pair['repeat'])), 'Pair key')
            for field in ('input_sha256', 'protocol_sha256', 'source_sha256'):
                require(pair[field] == r[field], field)
            for field in ('physical_ok', 'complete_ok', 'recovery_attempted'):
                require(pair[arm + '_' + field] == str(r[field]), field)
            require(pair[arm + '_factor_fallback'] == str(r['factor_fallback']), 'Fallback')
            for field in ('T_physical_return', 'T_complete_return'):
                require(float(pair[arm + '_' + field]) == r[field], field)
            require(float(pair[arm + '_T_root']) == r['timings']['T_root'], 'Root timing')
            require(fraction(pair[arm + '_trajectory_energy_exact']) ==
                    fraction(r['recovery']['cost_exact']), 'Returned energy mismatch')
        for kind in ('physical', 'complete'):
            delta = float(pair[f'factor_T_{kind}_return']) - float(pair[f'conic_T_{kind}_return'])
            require(abs(delta - float(pair[f'delta_{kind}_ms'])) < 1e-9, 'Paired difference')
    expected_robot = {(seed, rep, arm) for seed in range(41000, 41020)
                      for rep in range(2) for arm in ('conic', 'factor')}
    keys = [(int(r['seed']), int(r['repeat']), r['arm']) for r in robot]
    require(len(keys) == 80 and set(keys) == expected_robot, 'Incomplete robot population')
    for row in robot:
        r = records[row['run_id']]
        require(r['run_id'] not in used, 'Duplicate robot record')
        used.add(r['run_id'])
        require(row['instance_id'] == r['instance_id'], 'Robot identity')
        index = int(row['seed']) - 41000
        group = 'easy' if index < 5 else 'cluttered' if index < 15 else 'on_axis_recovery_target'
        require(row['group'] == group, 'Robot group')
        for field in ('physical_ok', 'complete_ok', 'status', 'projected_collision_status'):
            require(row[field] == str(r[field]), field)
        tracking = r.get('tracking', {})
        executed = bool(tracking.get('log'))
        require(row['simulation_executed'] == str(executed), 'Execution flag')
        if executed:
            require(row['success'] == str(tracking['success']), 'Tracking result')
            require(row['log'] == tracking['log'] and
                    row['log_sha256'] == tracking['log_sha256'], 'Tracking binding')
            require(hashlib.sha256((DATA / row['log']).read_bytes()).hexdigest() ==
                    row['log_sha256'], 'Tracking data hash')
    require(used == set(records), 'Unaccounted raw records')


def physical_check(r):
    import numpy as np
    sys.path.insert(0, str(ROOT / 'src'))
    import q1_independent as exact
    rec = r['recovery']
    if not rec.get('ok'):
        require(not r['physical_ok'], 'Accepted missing curve')
        return 0
    s = r['input']
    G = exact.decode(rec['Gamma'])
    exact.affine(G, s['bc0'], s['bc1'], s['family']['eta'])
    require(exact.energy(G, s['family']['k']) == fraction(rec['cost_exact']), 'Exact energy')
    clear = [exact.classify(exact.obstacle_coefficients(g, c, radius))
             for g in G for c, radius in s['obstacles']]
    collision_free = all(v['status'] == 'STRICT_POSITIVE' for v in clear)
    lo, hi = s['workspace']
    workspace = all(F(float(lo[a])) <= x <= F(float(hi[a]))
                    for g in G for a, coord in enumerate(g) for x in coord)
    dd = G.copy()
    bounds = []
    for k in range(1, 4):
        dd = np.diff(dd, axis=2) * (G.shape[-1] - k) * len(G)
        bounds.append(max(sum(v*v for v in dd[i, :, j])
                          for i in range(len(G)) for j in range(dd.shape[-1])))
    duration = F(5)
    limits = [F(2), F(5, 2), F(5)]
    while duration <= 120 and any(q > limit**2 * duration**(2*k)
                                  for k, (q, limit) in enumerate(zip(bounds, limits), 1)):
        duration += F(1, 2)
    saved = r['physical_certificate']
    require(F(saved['duration_exact']) == duration, 'Selected duration')
    require(saved['workspace_ok'] == workspace and
            saved['collision_free'] == collision_free and
            saved['derivative_ok'] == (duration <= 120), 'Physical predicates')
    require(r['physical_ok'] == (workspace and collision_free and duration <= 120), 'Physical result')
    return len(clear)


def summarize(pairs, robot):
    ratios = [fraction(r['factor_trajectory_energy_exact']) /
              fraction(r['conic_trajectory_energy_exact']) for r in pairs]
    groups = {}
    for group in sorted({r['group'] for r in robot}):
        rows = [r for r in robot if r['group'] == group]
        accepted = [r for r in rows if r['physical_ok'] == 'True']
        groups[group] = dict(attempts=len(rows), accepted=len(accepted),
                            accepted_scenes=len({r['instance_id'] for r in accepted}),
                            root_failures=sum(r['status'] == 'ROOT_FAILURE' for r in rows))
    return dict(pairs=len(pairs),
                energy_ratio=dict(median=float(statistics.median(ratios)),
                                  minimum=float(min(ratios)), maximum=float(max(ratios)),
                                  factor_over_1pct=sum(x > F(101, 100) for x in ratios),
                                  conic_over_1pct=sum(x < F(100, 101) for x in ratios)),
                paired_median_delta_ms={k: statistics.median(float(r[f'delta_{k}_ms']) for r in pairs)
                                        for k in ('physical', 'complete')}, robot=groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--physical', action='store_true')
    parser.add_argument('--mutations', action='store_true')
    args = parser.parse_args()
    records, pairs, robot, files = load_data()
    check_rows(records, pairs, robot)
    output = summarize(pairs, robot)
    output['integrity_files'] = files
    output['raw_records'] = len(records)
    output['physical_polynomials_replayed'] = 0
    if args.physical:
        require(__debug__, 'Do not use Python -O for rational replay')
        for i, r in enumerate(records.values(), 1):
            output['physical_polynomials_replayed'] += physical_check(r)
            if i % 80 == 0:
                print(f'Physical records checked: {i}/720', file=sys.stderr, flush=True)
    if args.mutations:
        rejected = []
        for kind in ('missing_pair', 'energy', 'robot_result'):
            a, b = copy.deepcopy(pairs), copy.deepcopy(robot)
            if kind == 'missing_pair':
                a.pop()
            elif kind == 'energy':
                a[0]['factor_trajectory_energy_exact'] = '["0","1"]'
            else:
                b[0]['physical_ok'] = str(b[0]['physical_ok'] != 'True')
            try:
                check_rows(records, a, b)
            except (ValueError, KeyError):
                rejected.append(kind)
        require(len(rejected) == 3, 'Mutation was not rejected')
        output['mutations_rejected'] = rejected
    output['scope'] = 'Data/row/statistics verification; physical replay only when requested. No SDP dual, bootstrap or dynamics rerun.'
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
