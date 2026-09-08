"""Archive identity/binding mutation controls; run only outside timing windows."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import numpy as np
import pytest
import q1_verify as qa
from constructive_recovery import encode


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2)+'\n')


def families_fixture(tmp_path, monkeypatch):
    """A tiny independently checkable physical fixture, not a timing experiment."""
    monkeypatch.setattr(qa, 'ROOT', tmp_path)
    (tmp_path/'src').mkdir()
    (tmp_path/'src/q1_independent.py').write_text('fixture checker identity\n')
    fam = dict(d=2, k=1, l=0, N=1, eta=0)
    G = np.array([[[-2, 0, 2], [0, 0, 0]]], dtype=object)
    G = np.vectorize(F)(G)
    mode = np.array([[F(0), F(2), F(0)]], dtype=object)
    old = dict(id='tiny/00', family_label='tiny', family=fam, n=2,
               bc0=[[-2, 0]], bc1=[[2, 0]], obstacles=[([0, 3], .25)],
               geometry_id='geometry/00', seed=1, stratum='clear')
    p = dict(instances=1, arm='affine_green')
    frozen = dict(protocol=p, rows=[old], families=[dict(label='tiny', family=fam)],
                  input_sha256=qa.json_hash([old]))
    path = tmp_path/'artifacts/q1_families.json'
    pp = path.with_suffix('.protocol.json'); write(pp, frozen)
    result = dict(ok=True, status='CERTIFIED_PHYSICAL', method='plain',
                  Gamma=encode(G), cost_exact=['16', '1'], amplitude=['0', '1'],
                  attempts=[], obstacle_free_normalized_gap_upper=0.)
    row = dict(**old, lower_exact=['16', '1'], lower_kind='obstacle_free',
               arms=dict(affine_green=result))
    data = dict(complete=True, protocol=p, protocol_sha256=qa.digest(pp),
                input_sha256=frozen['input_sha256'], rows=[row])
    write(path, data)
    context = dict(affine=G, sdp=G, modes=dict(green=mode, bubble=mode, gap=mode))
    monkeypatch.setattr(qa, 'binding_context', lambda *a: context)
    return path, data, context


def test_complete_family_archive_and_exact_outward_gap(tmp_path, monkeypatch):
    path, data, context = families_fixture(tmp_path, monkeypatch)
    report = qa.verify(path)
    assert report['passed'] and report['curves'] == 1 and report['polynomials'] == 1
    gap = report['normalized_gap_audit'][0]
    assert gap['normalized_gap_exact'] == ['0', '1']
    assert F(gap['normalized_gap_outward']) >= 0


@pytest.mark.parametrize('mutation', [
    lambda d: d['rows'].clear(),
    lambda d: d['rows'][0]['arms'].clear(),
    lambda d: d['rows'][0]['arms']['affine_green'].pop('Gamma'),
    lambda d: d['rows'][0]['bc0'][0].__setitem__(0, -3),
    lambda d: d['rows'][0]['family'].__setitem__('d', 3),
    lambda d: d['rows'][0]['obstacles'][0][0].__setitem__(0, .1),
    lambda d: d.__setitem__('input_sha256', '0'*64),
    lambda d: d.__setitem__('protocol_sha256', '0'*64),
    lambda d: d['rows'][0].__setitem__('lower_exact', ['17', '1']),
    lambda d: d['rows'][0]['arms']['affine_green'].__setitem__('obstacle_free_normalized_gap_upper', 1.),
])
def test_family_archive_mutations_fail_closed(tmp_path, monkeypatch, mutation):
    path, data, context = families_fixture(tmp_path, monkeypatch)
    # JSON round-trip makes fixture containers match delivered JSON lists.
    changed = json.loads(json.dumps(data)); mutation(changed); write(path, changed)
    with pytest.raises((ValueError, AssertionError, KeyError)):
        qa.verify(path)


def test_changed_frozen_protocol_requires_new_hash(tmp_path, monkeypatch):
    path, data, context = families_fixture(tmp_path, monkeypatch)
    pp = path.with_suffix('.protocol.json')
    frozen = json.loads(pp.read_text()); frozen['rows'][0]['bc1'] = [[3, 0]]
    write(pp, frozen)
    with pytest.raises(ValueError, match='protocol hash'):
        qa.verify(path)


def folded_result():
    G = np.array([[[F(-2), F(0), F(2)], [F(0)]*3]], dtype=object)
    z = np.array([[F(0), F(2), F(0)]], dtype=object)
    alpha = F(3, 2); out = G.copy(); out[:, 1, :] += alpha*z
    t = dict(ok=True, status='CERTIFIED_TAIL', mode='green', reference=encode(G),
             z=encode(z), axis=1, sign=1, amplitude=['2', '1'],
             candidate_amplitude=['3', '2'])
    a = dict(ok=True, status='CERTIFIED_PHYSICAL', method='green', axis=1, sign=1,
             amplitude=['3', '2'], Gamma=encode(out), attempts=[t])
    context = dict(affine=G, sdp=G, modes=dict(green=z, bubble=z, gap=z))
    return a, context


def test_selected_curve_is_bound_to_reference_mode_and_amplitude():
    a, context = folded_result()
    qa.check_binding({}, 'sdp_green', a, context)


@pytest.mark.parametrize('change', [
    lambda a: a.__setitem__('amplitude', ['1', '1']),
    lambda a: a.__setitem__('sign', -1),
    lambda a: a.__setitem__('axis', 0),
    lambda a: a.__setitem__('method', 'bubble'),
    lambda a: a['attempts'][0].__setitem__('mode', 'bubble'),
    lambda a: a['attempts'][0]['reference']['values'][0].__setitem__(0, '-3'),
    lambda a: a['attempts'][0]['z']['values'][1].__setitem__(0, '3'),
    lambda a: a['attempts'][0].__setitem__('candidate_amplitude', ['1', '1']),
    lambda a: a['attempts'].clear(),
])
def test_binding_mutations_rejected(change):
    a, context = folded_result(); change(a)
    with pytest.raises(ValueError):
        qa.check_binding({}, 'sdp_green', a, context)


def test_zero_mode_normalization_is_unavailable():
    assert qa.normalized_mode(np.array([[F(0)]*3], dtype=object)) is None


def test_legacy_qp_status_is_normalized_in_audit_only(tmp_path, monkeypatch):
    path, data, context = families_fixture(tmp_path, monkeypatch)
    a = data['rows'][0]['arms'].pop('affine_green')
    a.pop('status'); a['reason'] = 'strict_physical_certificate'
    data['rows'][0]['arms']['iterated_cut_qp'] = a
    write(path, data)
    monkeypatch.setattr(qa, 'frozen_inputs', lambda *a: ('factorial', {data['rows'][0]['id']: {}}, ['iterated_cut_qp'], 'test fixture'))
    before = path.read_bytes()
    report = qa.verify(path)
    assert report['normalized_legacy_comparator_status'][0]['status'] == 'CERTIFIED_PHYSICAL'
    assert path.read_bytes() == before


def test_outward_gap_contains_exact_fraction():
    a = dict(cost_exact=['50', '3'], normalized_gap_upper=float(F(1,24)))
    report = qa.gap_record(dict(id='a'), 'green', a, ['16', '1'])
    assert F(*map(int, report['normalized_gap_exact'])) == F(1,24)
    assert F(report['normalized_gap_outward']) >= F(1,24)


def test_factorial_source_hash_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, 'ROOT', tmp_path)
    source = tmp_path/'artifacts/revision_recovery_generalization.json'
    write(source, dict(rows=[]))
    p = dict(arms=qa.FACTORIAL_ARMS, instances=1, input_sha256='0'*64)
    path = tmp_path/'artifacts/factorial.json'; write(path.with_suffix('.protocol.json'), p)
    data = dict(complete=True, protocol=p, rows=[dict(id='one')])
    with pytest.raises(ValueError, match='input hash'):
        qa.frozen_inputs(path, data)


def test_integrated_boundary_and_population_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(qa, 'ROOT', tmp_path)
    source = tmp_path/'artifacts/revision_streaming_F2.json'
    original = [dict(seed=0, step=0, obstacles=[[[0, 0], .3]])]
    write(source, dict(rows=original))
    p = dict(budget=[12], warm=[True], instances=1, seeds=1, steps=1,
             input_sha256=qa.digest(source))
    path = tmp_path/'artifacts/integrated.json'; write(path.with_suffix('.protocol.json'), p)
    row = dict(id='b12_w1_s0_t0', budget=12, warm=True, seed=0, step=0,
               obstacles=original[0]['obstacles'], family=qa.F2,
               bc0=[[-2,0]], bc1=[[2,0]])
    data = dict(complete=True, protocol=p, rows=[row])
    assert qa.frozen_inputs(path, data)[0] == 'integrated'
    changed=deepcopy(data); changed['rows'][0]['bc1']=[[3,0]]
    with pytest.raises(ValueError, match='boundaries'):qa.frozen_inputs(path, changed)
    changed=deepcopy(data); changed['rows'][0]['step']=1
    with pytest.raises(ValueError, match='coverage'):qa.frozen_inputs(path, changed)
