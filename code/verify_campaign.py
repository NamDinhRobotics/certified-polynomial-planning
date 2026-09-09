"""Replay an archived campaign's mathematical and recorded tracking witnesses.

Exact spline, recovery and SDP checks use rational arithmetic. Tracking
metrics are recomputed from the full-rate recorded state; simulation and
numerical optimization are not rerun.
"""
import argparse
from collections import Counter
import copy
from fractions import Fraction as F
import gzip
import hashlib
import json
from math import comb, isclose, isqrt
from pathlib import Path
import sys
import numpy as np

import exact
from objective import rational_pair, record, verify_objective

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = 'objective_repair_20260909'
TELEMETRY_COLUMNS = ['t', 'x', 'y', 'z', 'qw', 'qx', 'qy', 'qz', 'vx', 'vy', 'vz',
                     'wx', 'wy', 'wz', 'attitude_error_rad', 'saturation_active', 'contacts',
                     'f0', 'f1', 'f2', 'f3']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pack(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_digest(scene):
    return hashlib.sha256(pack(scene).encode()).hexdigest()


def encoded_array(value, expected_shape):
    require(value['shape'] == list(expected_shape), 'Rational array dimensions mismatch')
    require(len(value['values']) == int(np.prod(expected_shape)), 'Rational array size mismatch')
    return np.asarray([rational_pair(pair) for pair in value['values']], dtype=object).reshape(expected_shape)


def boolean(saved, derived, label):
    require(type(saved) is bool and saved == bool(derived), label + ' mismatch')


def close(saved, derived, label):
    require(isclose(float(saved), float(derived), rel_tol=1e-11, abs_tol=1e-11), label + ' mismatch')


def affine(G, scene, homogeneous=False):
    n = G.shape[1]
    bc0 = [[0] * n for _ in scene['bc0']] if homogeneous else scene['bc0']
    bc1 = [[0] * n for _ in scene['bc1']] if homogeneous else scene['bc1']
    for end, boundary in ((0, bc0), (-1, bc1)):
        for order, expected in enumerate(boundary):
            actual = [exact.derivative(row, order)[end] for row in G[end]]
            require(actual == [exact.frac(v) for v in expected], 'Exact boundary equations failed')
    for left, right in zip(G[:-1], G[1:]):
        for order in range(scene['family']['eta'] + 1):
            require([exact.derivative(row, order)[-1] for row in left]
                    == [exact.derivative(row, order)[0] for row in right],
                    'Exact continuity equations failed')


def recovery_tail(tail, scene, projected):
    N, n, width = projected.shape
    reference = encoded_array(tail['reference'], projected.shape)
    require(np.array_equal(reference, projected), 'Recovery reference differs from projected curve')
    affine(reference, scene)
    mode = encoded_array(tail['z'], (N, width))
    affine(mode[:, None, :], scene, homogeneous=True)
    axis, sign = tail['axis'], tail['sign']
    require(type(axis) is int and 0 <= axis < n and type(sign) is int and sign in (-1, 1),
            'Invalid recovery axis or sign')
    margin, amplitude = rational_pair(tail['margin']), rational_pair(tail['amplitude'])
    require(margin > 0 and amplitude >= 0, 'Invalid recovery margin or amplitude')
    cover = {}
    other_axes = [a for a in range(n) if a != axis]
    for cell in tail['cells']:
        segment, obstacle, depth, index = (cell[k] for k in ('segment', 'obstacle', 'depth', 'index'))
        require(all(type(v) is int for v in (segment, obstacle, depth, index)),
                'Recovery cell indices must be integers')
        require(0 <= segment < N and 0 <= obstacle < len(scene['obstacles'])
                and 0 <= depth <= 24 and 0 <= index < 2 ** depth, 'Invalid recovery cell interval')
        cover.setdefault((segment, obstacle), []).append((F(index, 2 ** depth), F(index + 1, 2 ** depth)))
        center, radius = scene['obstacles'][obstacle]
        R = exact.frac(radius) + margin
        delta = [exact.restrict([v - exact.frac(center[a]) for v in reference[segment, a]], depth, index)
                 for a in range(n)]
        z = exact.restrict(mode[segment], depth, index)
        if cell['kind'] == 'transverse':
            bound = sum((min(delta[a]) if min(delta[a]) > 0 else
                         -max(delta[a]) if max(delta[a]) < 0 else F(0)) ** 2 for a in other_axes)
            require(bound == rational_pair(cell['bound']) and bound > R * R,
                    'Recovery transverse bound failed')
        else:
            require(cell['kind'] == 'positive_mode', 'Unknown recovery cell proof kind')
            lo, longitudinal = min(z), min(sign * v for v in delta[axis])
            require(lo > 0, 'Recovery mode is not positive on the cell')
            threshold = max(F(0), (R - longitudinal) / lo)
            require(threshold == rational_pair(cell['threshold']) and amplitude >= threshold,
                    'Recovery longitudinal threshold failed')
            require(lo == rational_pair(cell['mode_lower'])
                    and longitudinal == rational_pair(cell['longitudinal_lower']),
                    'Recovery longitudinal coefficient bound mismatch')
    for segment in range(N):
        for obstacle in range(len(scene['obstacles'])):
            spans = sorted(cover.get((segment, obstacle), []))
            require(spans and spans[0][0] == 0 and spans[-1][1] == 1,
                    'Recovery cells do not cover the unit interval')
            require(all(left[1] == right[0] for left, right in zip(spans[:-1], spans[1:])),
                    'Recovery cell coverage has a gap or overlap')
    boolean(tail['ok'], True, 'Recovery tail result')
    return mode, len(tail['cells'])


def physical_check(row):
    scene, recovered, saved = row['input'], row['recovery'], row['physical_certificate']
    family = scene['family']
    shape = (family['N'], family['n'], family['d'] + 1)
    projected = encoded_array(row['projected_Gamma'], shape)
    G = encoded_array(recovered['Gamma'], shape)
    affine(projected, scene)
    affine(G, scene)
    cells = 0
    tails = []
    for tail in recovered['attempts']:
        mode, count = recovery_tail(tail, scene, projected)
        tails.append((tail, mode))
        cells += count
    require(type(recovered['plain']) is bool, 'Invalid plain recovery flag')
    amplitude = rational_pair(recovered['amplitude'])
    if recovered['plain']:
        require(recovered['method'] == 'plain' and amplitude == 0 and np.array_equal(G, projected),
                'Plain recovery changed the projected curve')
    else:
        matches = [(tail, mode) for tail, mode in tails
                   if (tail['mode'], tail['axis'], tail['sign'])
                   == (recovered['method'], recovered['axis'], recovered['sign'])]
        require(len(matches) == 1 and amplitude >= 0, 'Selected recovery witness is missing or ambiguous')
        tail, mode = matches[0]
        expected = projected.copy()
        expected[:, recovered['axis'], :] += recovered['sign'] * amplitude * mode
        require(np.array_equal(G, expected), 'Selected recovery curve reconstruction failed')
        if 'candidate_amplitude' in tail:
            require(amplitude == rational_pair(tail['candidate_amplitude']), 'Selected recovery amplitude mismatch')
    J = exact.energy(G, family['k'])
    require(J == rational_pair(recovered['cost_exact']) == rational_pair(saved['cost_exact']),
            'Exact physical energy mismatch')
    polynomials, clearances = [], []
    for segment, g in enumerate(G):
        for obstacle, (center, radius) in enumerate(scene['obstacles']):
            check = exact.classify(exact.obstacle_coefficients(g, center, radius))
            polynomials.append(dict(segment=segment, obstacle=obstacle, **check))
            if check['status'] == 'STRICT_POSITIVE':
                squared = exact.frac(radius) ** 2 + F(check['lower'])
                scale = 2 ** 40
                root_lower = F(isqrt(squared.numerator * scale ** 2 // squared.denominator), scale)
                clearances.append(root_lower - exact.frac(radius))
    require(pack(polynomials) == pack(saved['polynomials']), 'Exact physical clearance certificate mismatch')
    collision_free = all(p['status'] == 'STRICT_POSITIVE' for p in polynomials)
    lower, upper = scene['workspace']
    require(len(lower) == len(upper) == family['n'], 'Workspace dimensions mismatch')
    workspace = all(exact.frac(lower[a]) <= value <= exact.frac(upper[a])
                    for segment in G for a, coordinate in enumerate(segment) for value in coordinate)
    bounds, derivative = {}, G.copy()
    for order in range(1, 5):
        derivative = np.diff(derivative, axis=2) * (family['d'] + 1 - order) * family['N']
        bounds[order] = max(sum(value * value for value in derivative[i, :, j])
                            for i in range(family['N']) for j in range(derivative.shape[-1]))
    duration = F(5)
    limits = (F(2), F(5, 2), F(5))
    while any(bounds[order] > limits[order - 1] ** 2 * duration ** (2 * order)
              for order in range(1, 4)):
        duration += F(1, 2)
        require(duration <= 121, 'Physical duration exceeds the supported acceptance range')
    derivative_ok = duration <= 120
    require(F(saved['duration_exact']) == duration, 'Selected physical duration mismatch')
    close(saved['duration'], float(duration), 'Physical duration')
    require(set(saved['derivative_bounds']) == {'1', '2', '3', '4'}, 'Missing derivative bounds')
    for order, squared in bounds.items():
        entry = saved['derivative_bounds'][str(order)]
        require(F(entry['normalized_squared_bound']) == squared, 'Exact derivative bound mismatch')
        close(entry['physical_bound'], float(squared) ** .5 / float(duration) ** order, 'Physical derivative bound')
    physical = collision_free and workspace and derivative_ok
    for name, result in (('ok', physical), ('collision_free', collision_free), ('workspace_ok', workspace),
                         ('derivative_ok', derivative_ok), ('affine_ok', True), ('continuity_ok', True)):
        boolean(saved[name], result, 'Physical ' + name)
    boolean(recovered['ok'], physical, 'Recovery result')
    boolean(row['physical_ok'], physical, 'Physical return result')
    if collision_free:
        clearance = min(clearances)
        require(F(saved['certified_minimum_clearance_exact']) == clearance, 'Exact minimum clearance mismatch')
        close(saved['certified_minimum_clearance'], float(clearance), 'Minimum clearance')
    return dict(physical=physical, energy=J, Gamma=G, duration=duration,
                physical_polynomials=len(polynomials), recovery_cells=cells)


def tracking_check(row, directory, G, duration, *, file_hashes=True):
    tracking = row['tracking']
    path = (directory / tracking['log']).resolve()
    require(path.parent == directory.resolve() / 'telemetry' and path.suffix == '.npz',
            'Telemetry path escapes campaign directory')
    if file_hashes:
        require(digest(path) == tracking['log_sha256'], 'Telemetry digest mismatch')
    with np.load(path, allow_pickle=False) as source:
        require(set(source.files) == {'samples', 'columns'}, 'Unexpected telemetry arrays')
        samples, columns = source['samples'], source['columns'].tolist()
    require(columns == TELEMETRY_COLUMNS, 'Unexpected telemetry column schema')
    require(samples.ndim == 2 and samples.shape[1] == len(columns) and len(samples) >= 3,
            'Invalid telemetry dimensions')
    require(samples.dtype == np.float64 and np.isfinite(samples).all(), 'Invalid telemetry precision or values')
    indices = {name: i for i, name in enumerate(columns)}

    def values(*names):
        return samples[:, [indices[name] for name in names]]

    t = samples[:, indices['t']]
    require(t[0] == 0 and isclose(t[-1], float(duration) + 3., rel_tol=0, abs_tol=1e-12),
            'Telemetry time horizon mismatch')
    require(np.allclose(np.diff(t), .001, atol=1e-12, rtol=0), 'Telemetry cadence is not 1 ms')
    require(np.allclose(np.linalg.norm(values('qw', 'qx', 'qy', 'qz'), axis=1), 1,
                        atol=1e-10, rtol=0), 'Invalid recorded quaternion')
    gamma = np.asarray(G, dtype=float)
    N, _, width = gamma.shape
    phase = np.clip((t - 1) / float(duration), 0, 1) * N
    segment = np.minimum(phase.astype(int), N - 1)
    u, degree = phase - segment, width - 1
    basis = np.asarray([comb(degree, k) * u ** k * (1 - u) ** (degree - k) for k in range(width)]).T
    reference = np.einsum('nkh,nh->nk', gamma[segment], basis)
    position = values('x', 'y', 'z')
    error = np.linalg.norm(position - reference, axis=1)
    obstacles = row['input']['physical_obstacles']
    centers = np.asarray([obstacle['center'] for obstacle in obstacles])
    radii = np.asarray([obstacle['radius'] for obstacle in obstacles])
    require(centers.shape == (3, 3) and radii.shape == (3,) and np.isfinite(centers).all()
            and np.isfinite(radii).all() and np.all(radii >= 0), 'Invalid physical ball geometry')
    clearance = np.min(np.linalg.norm(position[:, None, :] - centers[None, :, :], axis=2)
                       - radii[None, :] - .23, axis=1)
    floor = position[:, 2] - .23
    velocity = values('vx', 'vy', 'vz')
    acceleration = np.diff(velocity, axis=0) / .001
    jerk = np.diff(acceleration, axis=0) / .001
    body_rate = values('wx', 'wy', 'wz')
    rotors = values('f0', 'f1', 'f2', 'f3')
    attitude = samples[:, indices['attitude_error_rad']]
    saturation = samples[:, indices['saturation_active']]
    contacts = samples[:, indices['contacts']]
    require(np.isin(saturation, [0., 1.]).all(), 'Invalid recorded saturation flag')
    require(((contacts >= 0) & (contacts == np.floor(contacts))).all(), 'Invalid recorded contact count')
    expected = dict(rms_position_error=float(np.sqrt(np.mean(error ** 2))),
                    max_position_error=float(np.max(error)),
                    rms_attitude_error_rad=float(np.sqrt(np.mean(attitude ** 2))),
                    max_attitude_error_rad=float(np.max(attitude)),
                    min_realized_clearance=float(np.min(clearance)), min_floor_clearance=float(np.min(floor)),
                    max_velocity=float(np.max(np.linalg.norm(velocity, axis=1))),
                    max_acceleration=float(np.max(np.linalg.norm(acceleration, axis=1))),
                    max_jerk=float(np.max(np.linalg.norm(jerk, axis=1))),
                    max_thrust=float(np.max(rotors.sum(axis=1))), max_rotor_thrust=float(np.max(rotors)),
                    max_body_rate=float(np.max(np.linalg.norm(body_rate, axis=1))),
                    actuator_saturation_updates=int(np.sum(saturation[::5])),
                    actuator_saturation_duration=float(np.sum(saturation[:-1]) * .001),
                    contact_steps=int(np.sum(contacts > 0)),
                    contact_events=int(np.sum((contacts[1:] > 0) & (contacts[:-1] == 0))) + int(contacts[0] > 0),
                    body_collision_samples=int(np.sum(clearance <= 0)), final_position_error=float(error[-1]),
                    samples=len(samples), simulation_duration=float(t[-1]))
    for key, value in expected.items():
        close(tracking[key], value, 'Tracking ' + key)
    success = bool(np.max(error) < .12 and error[-1] < .025 and np.min(clearance) > 0
                   and np.min(floor) > 0 and np.max(contacts) == 0)
    boolean(tracking['success'], success, 'Tracking success')
    return dict(samples=len(samples), success=success)


def check_record(row, directory, *, file_hashes=True):
    require(row['campaign_id'] == CAMPAIGN_ID, 'Record campaign identity mismatch')
    require(row['input_sha256'] == input_digest(row['input']), 'Input digest mismatch')
    require(row['seed'] == row['input']['seed'] and row['instance_id'] == row['input']['id'],
            'Scene identity mismatch')
    require({'ok', 'reason', 'lower_exact', 'upper_exact', 'lower', 'upper', 'relative_gap_upper',
             'shift', 'n_polys', 'n_pd_blocks'} <= set(row['objective_certificate']),
            'Missing objective certificate fields')
    physical = physical_check(row)
    objective = verify_objective(row['input'], row['basis'], row['objective_witness'],
                                 row['objective_certificate'], row['objective_path'])
    lower = rational_pair(objective['lower_exact'])
    require(physical['energy'] >= lower, 'Returned trajectory energy is below the SDP lower bound')
    close(row['objective_interval_width'],
          float(rational_pair(objective['upper_exact']) - lower), 'Objective interval width')
    boolean(row['objective_certificate']['ok'], objective['ok'], 'Objective result')
    require(row['objective_certificate']['reason'] == objective['reason'], 'Objective reason mismatch')
    boolean(row['root_optimality_certified'], objective['ok'], 'Root objective certification')
    require(type(row['root_numerically_accepted']) is bool, 'Missing numerical root acceptance record')
    complete = physical['physical'] and objective['ok']
    boolean(row['complete_ok'], complete, 'Complete return result')
    require(row['status'] == ('COMPLETE_SUCCESS' if complete else 'PHYSICAL_ONLY'), 'Return status mismatch')
    tracking = tracking_check(row, directory, physical['Gamma'], physical['duration'], file_hashes=file_hashes)
    return dict(physical=physical['physical'], objective=objective['ok'], complete=complete,
                tracking=tracking['success'], physical_polynomials=physical['physical_polynomials'],
                recovery_cells=physical['recovery_cells'], objective_polynomials=objective['n_polys'],
                objective_pd_blocks=objective['n_pd_blocks'], tracking_samples=tracking['samples'],
                exact_gap=rational_pair(objective['relative_gap_exact']),
                outward_gap=objective['relative_gap_upper'])


def load_campaign(directory):
    index = json.loads((ROOT / 'data/campaigns/index.json').read_text())
    require(index['schema_version'] == 1, 'Unsupported campaign index')
    indexed = {entry['campaign_id']: entry for entry in index['campaigns']}
    require(len(indexed) == len(index['campaigns']) == 2
            and set(indexed) == {'v3_20260909', CAMPAIGN_ID}, 'Frozen campaign index identity mismatch')
    historical_entry, current_entry = indexed['v3_20260909'], indexed[CAMPAIGN_ID]
    require(historical_entry['campaign_type'] == 'historical_recorded_campaign'
            and historical_entry['immutable'] is True
            and historical_entry['files'] == {'planning': '../planning.json.gz', 'historical': '../historical.json.gz'},
            'Historical campaign index mapping mismatch')
    require(current_entry['campaign_type'] == 'post_hoc_debug_validation'
            and current_entry['directory'] == CAMPAIGN_ID
            and current_entry['metadata'] == CAMPAIGN_ID + '/campaign.json'
            and current_entry['records'] == CAMPAIGN_ID + '/records.json.gz'
            and current_entry['provenance'] == CAMPAIGN_ID + '/provenance.json',
            'Objective campaign index mapping mismatch')
    metadata = json.loads((directory / 'campaign.json').read_text())
    require(metadata['schema_version'] == 1 and metadata['campaign_id'] == CAMPAIGN_ID,
            'Unsupported campaign metadata')
    require(metadata['campaign_type'] == 'post_hoc_debug_validation', 'Campaign type mismatch')
    require(metadata['records_file'] == 'records.json.gz'
            and metadata['provenance_file'] == 'provenance.json', 'Unexpected campaign data paths')
    provenance = json.loads((directory / 'provenance.json').read_text())
    require(provenance['schema_version'] == 1 and provenance['campaign_id'] == CAMPAIGN_ID,
            'Provenance campaign identity mismatch')
    payload = json.loads(gzip.decompress((directory / 'records.json.gz').read_bytes()))
    require(payload['schema_version'] == 1 and payload['campaign_id'] == CAMPAIGN_ID,
            'Record container campaign identity mismatch')
    rows = payload['records']
    require(len(rows) == 80, 'Expected 20 scenes by 2 arms by 2 repetitions')
    keys = [(row['seed'], row['arm'], row['repeat']) for row in rows]
    expected = {(seed, arm, repeat) for seed in range(41000, 41020)
                for arm in ('conic', 'factor') for repeat in (0, 1)}
    require(set(keys) == expected and len(set(keys)) == len(keys), 'Incomplete or duplicated paired population')
    require(len({(payload['campaign_id'], row['run_id']) for row in rows}) == len(rows),
            'Duplicate campaign and run identity')
    scenes = {}
    for row in rows:
        seed, repeat, arm = row['seed'], row['repeat'], row['arm']
        require(type(seed) is int and type(repeat) is int, 'Invalid run indices')
        require(row['scene_index'] == seed - 41000 and row['instance_id'] == 'heldout_' + str(seed),
                'Held-out scene identity mismatch')
        require(row['run_id'] == f'B_s{seed - 41000}_r{repeat}_{arm}', 'Run identifier mismatch')
        require(row['input']['family'] == dict(n=3, d=7, k=4, l=4, N=5, eta=4),
                'Unexpected polynomial family in this campaign')
        require(len(row['input']['obstacles']) == len(row['input']['physical_obstacles']) == 3,
                'Expected three obstacles in each scene')
        previous = scenes.setdefault(seed, pack(row['input']))
        require(previous == pack(row['input']), 'Paired runs use different scene inputs')
        require(type(row['planning_root_calls']) is int and row['planning_root_calls'] >= 0,
                'Invalid planning solver call count')
        require(type(row['retry_solver_calls']) is int and row['retry_solver_calls']
                == (2 if row['objective_path'] == 'independent_same_SDP_objective_witness' else 0),
                'Objective retry call count mismatch')
    exported = provenance['export_files']
    expected_files = {'campaign.json', 'records.json.gz'} | {row['tracking']['log'] for row in rows}
    require(set(exported) == expected_files, 'Export manifest does not match all referenced files')
    for relative, entry in exported.items():
        path = directory / relative
        require(path.resolve().is_relative_to(directory.resolve()), 'Export path escapes campaign directory')
        require(path.is_file() and path.stat().st_size == entry['bytes']
                and digest(path) == entry['sha256'], 'Export digest mismatch: ' + relative)
    for relative, expected_hash in provenance['historical_files_unchanged'].items():
        require(relative in {'planning.json.gz', 'historical.json.gz'}, 'Unexpected historical data path')
        require(digest(ROOT / 'data' / relative) == expected_hash, 'Historical data changed: ' + relative)
    require(set(provenance['historical_files_unchanged']) == {'planning.json.gz', 'historical.json.gz'},
            'Missing historical data commitments')
    require(historical_entry['file_sha256'] == provenance['historical_files_unchanged'],
            'Historical index and provenance digests disagree')
    require(set(provenance['selected_sources']) == set(provenance['comparison_commitments'])
            == {row['run_id'] for row in rows}, 'Source commitments do not cover all records')
    comparison_changes = dict(reference_changes=0, scientific_tracking_metric_changes=0, telemetry_hash_changes=0)
    for row in rows:
        source = provenance['selected_sources'][row['run_id']]
        require(source['record'] == 'fixed/' + row['run_id'] + '.json'
                and source['telemetry'] == 'fixed/' + row['run_id'] + '.tracking.npz',
                'Selected source is outside the fixed campaign records')
        require(source['exported_telemetry'] == row['tracking']['log']
                and source['telemetry_sha256'] == row['tracking']['source_log_sha256'],
                'Source telemetry commitment mismatch')
        commitment = provenance['comparison_commitments'][row['run_id']]
        reference_hash = hashlib.sha256(pack(row['recovery']['Gamma']).encode()).hexdigest()
        require(commitment['reference_sha256'] == reference_hash, 'Reference commitment mismatch')
        metrics = {key: value for key, value in row['tracking'].items()
                   if key not in {'log', 'log_sha256', 'source_log_sha256'}}
        metrics_hash = hashlib.sha256(pack(metrics).encode()).hexdigest()
        require(commitment['scientific_tracking_sha256'] == metrics_hash, 'Tracking metric commitment mismatch')
        for flag, count_key in (('prior_reference_equal', 'reference_changes'),
                                ('prior_scientific_tracking_equal', 'scientific_tracking_metric_changes'),
                                ('prior_telemetry_sha256_equal', 'telemetry_hash_changes')):
            require(type(commitment[flag]) is bool, 'Invalid archived comparison commitment')
            comparison_changes[count_key] += not commitment[flag]
    require(metadata['comparison_to_prior'] == comparison_changes,
            'Prior comparison summary differs from archived commitments')
    comparisons = metadata['comparison_configurations']
    require(len(comparisons) == 3
            and {entry['campaign_id'] for entry in comparisons}
            == {'v3_20260909', 'candidate_recovery_20260909', CAMPAIGN_ID},
            'Comparison configuration identities mismatch')
    require(all(entry['scope'] == 'robotics_B' and entry['attempts'] == 80 for entry in comparisons),
            'Comparison configuration population mismatch')
    protocol = dict(sample_dt=.001, body_radius=.23, pre_seconds=1, post_seconds=2,
                    max_position_error=.12, max_final_position_error=.025, controller_stride_samples=5)
    require(metadata['tracking_protocol'] == protocol, 'Tracking protocol mismatch')
    configuration = metadata['selected_config']
    require(configuration['normalized_dual_margin'] == 1e-9
            and configuration['objective_retry_divisor'] == 4
            and configuration['retry_solver'] == 'CLARABEL'
            and configuration['retry_tolerances'] == [1e-9, 1e-11]
            and configuration['max_iterations'] == 600
            and configuration['exact_objective_relative_gap'] == '1/100000 (unchanged existing verifier tolerance)',
            'Selected configuration declaration mismatch')
    return rows, metadata, provenance


def mutation_checks(rows, directory):
    """Corrupt witnesses in memory, after file provenance has been checked."""
    ordinary = rows[0]
    recovery = next(row for row in rows if row['recovery']['attempts'])
    failures = {}
    for kind in ('geometry', 'reference_coefficient', 'boundary', 'recovery_cell', 'workspace',
                 'duration', 'energy', 'input_hash', 'objective_lower', 'objective_multiplier',
                 'objective_point', 'objective_unit', 'complete_flag'):
        row = copy.deepcopy(recovery if kind == 'recovery_cell' else ordinary)
        if kind == 'geometry':
            row['input']['obstacles'][0][1] = 10.0
        elif kind == 'reference_coefficient':
            value = rational_pair(row['recovery']['Gamma']['values'][3])
            row['recovery']['Gamma']['values'][3] = record(value + 1)
        elif kind == 'boundary':
            row['input']['bc0'][0][0] += 1.0
        elif kind == 'recovery_cell':
            row['recovery']['attempts'][0]['cells'].pop()
        elif kind == 'workspace':
            row['input']['workspace'][0][0] = 4.0
        elif kind == 'duration':
            row['physical_certificate']['duration_exact'] = str(F(row['physical_certificate']['duration_exact']) + F(1, 2))
        elif kind == 'energy':
            row['recovery']['cost_exact'] = record(rational_pair(row['recovery']['cost_exact']) + 1)
        elif kind == 'input_hash':
            row['input_sha256'] = '0' * 64
        elif kind == 'objective_lower':
            row['objective_certificate']['lower_exact'] = record(rational_pair(row['objective_certificate']['lower_exact']) + 1)
        elif kind == 'objective_multiplier':
            encoded = row['objective_witness']['multipliers']
            encoded['values'] = [['0', '1'] for _ in encoded['values']]
        elif kind == 'objective_point':
            encoded = row['objective_witness']['point'][0]
            encoded['values'][0] = record(rational_pair(encoded['values'][0]) + 1)
        elif kind == 'objective_unit':
            row['objective_witness']['energy_unit'] = ['1', '1']
        elif kind == 'complete_flag':
            row['complete_ok'] = not row['complete_ok']
        if kind in {'geometry', 'boundary', 'workspace'}:
            row['input_sha256'] = input_digest(row['input'])
        try:
            check_record(row, directory, file_hashes=False)
        except (ValueError, AssertionError, KeyError) as error:
            failures[kind] = str(error) or type(error).__name__
        else:
            raise ValueError('Mutation was accepted: ' + kind)
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, choices=[CAMPAIGN_ID])
    parser.add_argument('--mutations', action='store_true')
    args = parser.parse_args()
    require(__debug__, 'Do not use Python -O for certificate verification')
    directory = ROOT / 'data' / 'campaigns' / args.campaign
    rows, metadata, provenance = load_campaign(directory)
    counts = Counter()
    gaps = []
    outward_gaps = []
    unique_logs = {}
    for index, row in enumerate(rows, 1):
        result = check_record(row, directory)
        for key in ('physical', 'objective', 'complete', 'tracking', 'physical_polynomials',
                    'recovery_cells', 'objective_polynomials', 'objective_pd_blocks', 'tracking_samples'):
            counts[key] += result[key]
        gaps.append(result['exact_gap'])
        outward_gaps.append(result['outward_gap'])
        unique_logs[row['tracking']['log']] = result['tracking_samples']
        if index % 10 == 0:
            print(f'Campaign records verified: {index}/{len(rows)}', file=sys.stderr, flush=True)
    declared = metadata['counts']
    derived = dict(attempts=len(rows), scene_ids=len({row['instance_id'] for row in rows}),
                   physical=counts['physical'], tracking=counts['tracking'], complete=counts['complete'],
                   objective_certificates=counts['objective'],
                   planning_root_calls=sum(row['planning_root_calls'] for row in rows),
                   retry_solver_calls=sum(row['retry_solver_calls'] for row in rows),
                   tracking_samples=counts['tracking_samples'], unique_telemetry_files=len(unique_logs),
                   unique_tracking_samples=sum(unique_logs.values()),
                   clearance_polynomials=counts['physical_polynomials'], recovery_cells=counts['recovery_cells'],
                   original_numerical_roots_rejected=sum(not row['root_numerically_accepted'] for row in rows))
    derived['all_conic_calls'] = derived['planning_root_calls'] + derived['retry_solver_calls']
    require(declared == derived, 'Campaign count summary differs from verified records')
    paths = dict(Counter(row['objective_path'] for row in rows))
    require(paths == metadata['certificate_paths'], 'Objective witness path summary mismatch')
    maximum_gap = max(gaps)
    require(metadata['maximum_exact_relative_gap'] == max(outward_gaps), 'Maximum objective gap mismatch')
    output = dict(campaign_id=CAMPAIGN_ID, verified=True, counts=derived, certificate_paths=paths,
                  objective_polynomials_replayed=counts['objective_polynomials'],
                  objective_positive_definite_blocks_replayed=counts['objective_pd_blocks'],
                  maximum_exact_relative_gap=record(maximum_gap), maximum_relative_gap=float(maximum_gap),
                  maximum_relative_gap_upper=max(outward_gaps),
                  export_files_verified=len(provenance['export_files']),
                  source_commitments_checked=len(provenance['selected_sources']),
                  frozen_campaign_index_verified=True,
                  archived_comparison_changes=metadata['comparison_to_prior'])
    if args.mutations:
        output['mutations_rejected'] = mutation_checks(rows, directory)
    output['scope'] = ('Exact spline, recovery and root-SDP objective verification; full-rate recorded tracking '
                       'metrics replay. The recovered trajectory is not asserted globally optimal, and '
                       'numerical optimization and simulation are not rerun.')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
