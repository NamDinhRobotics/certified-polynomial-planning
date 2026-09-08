"""Deterministic safety/dispatch regressions; no benchmark or optimizer run."""
from fractions import Fraction as F
from copy import deepcopy
from types import SimpleNamespace
import numpy as np
import pytest
import exact_check
import escape_fold as fold
import constructive_recovery as recovery
import exact_sdp_bounds as bounds
import q1_recovery as q1
import q1_independent as independent
from certified_multisegment import CertifiedSpline
from q1_solver import InstrumentedSpline


@pytest.mark.parametrize('convert', [fold.rational, bounds.rational_array])
def test_exact_inputs_and_historical_float_semantics(convert):
    values = [F(1, 3), 2**60+1, np.int64(2**60+1), 0.1, np.float32(0.1)]
    actual = convert(values)
    assert list(actual) == [F(1, 3), F(2**60+1), F(2**60+1), F(float(0.1)), F(float(np.float32(0.1)))]


def test_exact_one_third_tangency_is_rejected():
    g = fold.rational([[[F(1,3), F(1,3)], [0,0]]])
    obs = [([0,0], F(1,3))]
    assert not fold.physical_certificate(g, obs)
    assert not recovery.outcome(g, obs)['ok']
    assert not q1.recover(g, [], obs)['ok']
    g[:,0,:] += F(1,10**30)
    assert fold.physical_certificate(g, obs)
    # A caller explicitly supplying a rounded binary radius retains that meaning.
    tangent = fold.rational([[[F(1,3), F(1,3)], [0,0]]])
    assert fold.physical_certificate(tangent, [([0,0], float(F(1,3)))])


def test_exact_endpoint_and_tail_geometry():
    g = fold.rational([[[F(1,3)]*3, [0]*3]])
    obs = [([0,0], F(1,3))]
    margin = F(1, 10**30)
    assert recovery.endpoint_separation(g, obs, margin=margin) == F(1,9)-(F(1,3)+margin)**2
    assert q1.safe_tail(g, [[0,2,0]], obs)['status'] == 'UNKNOWN_GEOMETRY'
    with pytest.raises(ValueError):
        recovery.endpoint_separation(g, obs, axis=2)


def test_exact_upper_uses_rational_radius_and_shift():
    # Constant rational curve: tangency must not become a strict upper witness.
    e = object.__new__(bounds.ExactSDPBounds)
    e.ms = SimpleNamespace(N=1, slice_=lambda i: slice(0,2))
    e.factor = SimpleNamespace(d=1)
    e.G0 = fold.rational([[F(1,3)]*2,[0,0]])
    e.Q = fold.rational([[1],[1]])
    e.K = fold.rational([[1]])
    e.C = fold.rational([[0],[0]])
    e.c0 = F(0)
    e.S = np.array([[[F(1),0],[0,0]]], dtype=object)
    assert e.upper_point([[0],[0]], [[0]], [([0,0],F(1,3))], F(0)) is None
    value, count = e.upper_point([[0],[0]], [[0]], [([0,0],F(1,3))], F(1,3))
    assert value == F(1,3) and count == 1
    _, c = e.data([([0,0],F(1,3))])
    assert c[0,0] == 0


def stub(cls=CertifiedSpline, warm=True):
    f = object.__new__(cls)
    f.z = np.array([1.]) if warm else None
    f.lam = np.array([2.]) if warm else None
    f.update = lambda obs: None
    f.newton = lambda z, lam, maxiter=12: (z.copy(),lam.copy(),1,'stationary')
    f.assess = lambda z, lam: dict(ok=False,residual=2e-7,stationarity=2e-6,
                                  relative_gap=2e-5,bound=dict(ok=False))
    f.conic = lambda: dict(ok=True, Y=np.array([[3.]]), W=np.array([[10.]]), lam=np.array([4.]))
    f.seed = lambda result: (np.array([5.]),result['lam'].copy())
    return f


def throwing(kind, message='injected numerical failure'):
    def call(*args, **kwargs):raise kind(message)
    return call


@pytest.mark.parametrize('cls',[CertifiedSpline,InstrumentedSpline])
@pytest.mark.parametrize('stage',['newton','assess'])
def test_local_numerical_failure_dispatches(cls, stage):
    f=stub(cls)
    setattr(f,stage,throwing(np.linalg.LinAlgError))
    ans=f.solve([])
    assert ans['ok'] and ans['source']=='conic'
    assert ans['dispatch_reason']=='local_numerical_failure'
    assert ans['local_assessment']['exception_type']=='LinAlgError'
    assert ans['seed_status']=='ready'
    assert (ans['rejected_local_point'] is not None)==(stage=='assess')
    assert ans['total_ms']>=sum(ans[k] for k in ['update_ms','local_ms','conic_ms','initialization_ms'])


def test_all_rejected_gates_and_single_primary_cause_survive():
    f=stub();ans=f.solve([])
    assert ans['dispatch_reason']=='equality_residual'
    assert ans['failed_gates']==['equality_residual','stationarity','dual_bound_failure','objective_gap']
    assert ans['local_status']=='stationary'  # Stationary is not accepted.
    assert ans['local_gate_results']==dict(finite_candidate=True,equality_residual=False,
                                         stationarity=False,dual_bound=False,objective_gap=False)
    np.testing.assert_array_equal(ans['rejected_local_point']['z'],[1.])
    np.testing.assert_array_equal(ans['z'],[5.])


def test_nonfinite_candidate_has_explicit_unevaluated_gates():
    f=stub();f.newton=lambda *a,**kw:(np.array([np.nan]),np.array([1.]),1,'iteration_limit')
    f.assess=lambda *a:dict(ok=False,reason='nonfinite_candidate')
    ans=f.solve([])
    assert ans['failed_gates']==['nonfinite_candidate']
    assert ans['local_gate_results']['finite_candidate'] is False
    assert ans['local_gate_results']['stationarity'] is None


@pytest.mark.parametrize('error',[np.linalg.LinAlgError,ValueError,exact_check.ExactCheckError])
def test_valid_conic_survives_seed_failure(error):
    f=stub();f.seed=throwing(error)
    ans=f.solve([])
    assert ans['ok'] and ans['conic']['ok'] and ans['z'] is None
    assert ans['seed_status']=='numerical_failure'
    assert f.z is None and f.lam is None
    assert ans['conic']['Y'][0,0]==3
    next_ans=f.solve([])
    assert next_ans['dispatch_reason']=='cold_initialization'


def test_nonfinite_seed_is_discarded():
    f=stub();f.seed=lambda result:(np.array([np.nan]),np.array([1.]))
    ans=f.solve([])
    assert ans['ok'] and ans['z'] is None and f.z is None
    assert ans['seed_status']=='numerical_failure'


@pytest.mark.parametrize('error',[np.linalg.LinAlgError,ValueError,FloatingPointError])
def test_numerical_conic_failure_is_explicit_no_answer(error):
    f=stub();f.conic=throwing(error)
    ans=f.solve([])
    assert not ans['ok'] and ans['z'] is None and f.z is None
    assert ans['conic']['status']=='conic_numerical_failure'
    assert ans['seed_status']=='not_attempted'


def test_programming_errors_are_not_silently_reclassified():
    f=stub();f.newton=throwing(KeyError)
    with pytest.raises(KeyError):f.solve([])


def test_local_acceptance_skips_fallback():
    f=stub();f.assess=lambda *a:dict(ok=True,residual=0.,stationarity=0.,relative_gap=0.,bound=dict(ok=True))
    f.conic=throwing(AssertionError,'fallback should not run')
    ans=f.solve([])
    assert ans['source']=='factor' and ans['dispatch_reason']=='local_accepted'
    assert ans['failed_gates']==[] and ans['conic_ms']==0


@pytest.mark.parametrize('kwargs,reason',[({},'cold_initialization'),({'use_warm_start':False},'warm_start_disabled'),({'local_iterations':0},'local_budget_zero')])
def test_skip_reasons_are_distinct(kwargs,reason):
    f=stub(warm=bool(kwargs));f.newton=throwing(AssertionError,'local must not run')
    assert f.solve([],**kwargs)['dispatch_reason']==reason


def test_actual_exact_degree_exhaustion_is_unknown_at_service_boundaries(monkeypatch):
    monkeypatch.setattr(exact_check,'MAX_DEGREE',0)
    def exhausted(*args,**kwargs):
        return exact_check.positive_on_unit_interval([F(1),F(-1),F(1)])
    monkeypatch.setattr(fold,'positive_on_unit_interval_fast',exhausted)
    g=fold.rational([[[-1,0,1],[0,0,0]]]);obs=[([0,0],F(1,3))]
    with pytest.raises(exact_check.ExactCheckError):fold.physical_certificate(g,obs)
    calls=[lambda:fold.escape_fold(g,[[0,2,0]],obs),lambda:recovery.outcome(g,obs),
           lambda:q1.safe_tail(g,[[0,2,0]],obs),lambda:q1.recover(g,[],obs)]
    e=object.__new__(bounds.ExactSDPBounds);e.lower=exhausted
    calls.append(lambda:e.verify(None,None,obs))
    for call in calls:
        ans=call()
        assert not ans['ok'] and ans['status']=='UNKNOWN_EXACT_CHECK'
        assert ans['exception_type']=='ExactCheckError' and ans['ms']>=0


def test_independent_checker_rejects_tangency_collision_and_wrong_energy():
    g=fold.rational([[[F(1,3)]*3,[0]*3]])
    a=dict(Gamma=recovery.encode(g),cost_exact=['0','1'])
    bc=[[F(1,3),0]]
    independent.curve(a,[([0,0],F(1,4))],1,bc,bc,0)
    for radius in (F(1,3),F(1,2)):
        with pytest.raises(AssertionError):independent.curve(a,[([0,0],radius)],1,bc,bc,0)
    bad=deepcopy(a);bad['cost_exact']=['1','1']
    with pytest.raises(AssertionError):independent.curve(bad,[([0,0],F(1,4))],1,bc,bc,0)
    bad=deepcopy(a);bad['Gamma']['values'][0]=['2','3']
    with pytest.raises(AssertionError):independent.curve(bad,[([0,0],F(1,4))],1,bc,bc,0)


def test_independent_tail_rejects_tampered_threshold_and_cell_cover():
    g=fold.rational([[[-1,0,1],[0,0,0]]]);obs=[([0,0],F(1,3))]
    t=q1.serialize_tail(q1.safe_tail(g,[[0,2,0]],obs))
    assert t['ok']
    independent.tail(t,obs,[[-1,0]],[[1,0]],0)
    bad=deepcopy(t);bad['amplitude']=['0','1']
    with pytest.raises(AssertionError):independent.tail(bad,obs,[[-1,0]],[[1,0]],0)
    bad=deepcopy(t);bad['cells']=bad['cells'][:-1]
    with pytest.raises((AssertionError,KeyError)):independent.tail(bad,obs,[[-1,0]],[[1,0]],0)
