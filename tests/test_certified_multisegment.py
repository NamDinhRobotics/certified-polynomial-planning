"""Independent checks of the new same-SDP formulation and exact certificates."""
from fractions import Fraction as F
import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from multisegment import MultiSegment
from certified_multisegment import CertifiedSpline
from exact_sdp_bounds import ExactSDPBounds, positive_definite
from exact_green import constraint_matrix


def make():
    obs = [(np.array([0.,.1]),.4),(np.array([.9,-.3]),.2)]
    return CertifiedSpline(MultiSegment(2,5,1,0,3,obs,np.array([[-2.,0.]]),
                                        np.array([[2.,0.]]),eta=2))


@pytest.fixture(scope='module')
def f():
    return make()


def test_exact_affine_coordinates_and_zero_linear_cost(f):
    A = np.asarray(constraint_matrix(5,0,3,2),dtype=object).T
    assert np.all(f.Qq.T@A==0)
    assert np.all(f.G0q@A==np.asarray(f.ms.B,dtype=object))
    e = ExactSDPBounds(f)
    assert np.all(e.C==0)
    assert e.c0==16


def test_derivatives_against_finite_differences(f):
    rng = np.random.default_rng(123)
    z = rng.normal(size=f.nz)*.15
    lam = rng.normal(size=f.nr)*.1
    np.testing.assert_allclose(f.jac(z),approx_derivative(f.residual,z),atol=2e-8,rtol=1e-6)
    np.testing.assert_allclose(f.grad(z),approx_derivative(lambda x:np.array([f.cost(x)]),z).ravel(),atol=2e-8,rtol=1e-6)
    np.testing.assert_allclose(f.hess(lam),approx_derivative(lambda x:f.grad(x)+f.jac(x).T@lam,z),atol=2e-8,rtol=1e-6)


def test_cost_and_constraints_against_full_lift(f):
    z = np.random.default_rng(55).normal(size=f.nz)*.3
    Y,v,_,_ = f.unpack(z)
    G = f.ms.full_Gamma(Y)
    X = f.ms.full_X(Y,Y.T@Y+np.outer(v,v))
    cost = f.ms.time_scale*sum(np.sum(f.ms.Gk*X[f.ms.slice_(i),f.ms.slice_(i)]) for i in range(f.ms.N))
    assert abs(f.cost(z)-cost)<1e-10
    targets=[]
    for i in range(f.ms.N):
        for c,r in f.ms.obs:
            M = f.ms.M_obstacle(X,G,i,c,r)
            targets.append(np.einsum('kij,ij->k',f.ms.Sd,M))
    np.testing.assert_allclose(f.target(z),targets,atol=1e-11)


def test_parameterized_conic_matches_original_sdp():
    f=make()
    original=MultiSegment(2,5,1,0,3,f.ms.obs,np.array([[-2.,0.]]),np.array([[2.,0.]]),eta=2)
    reference=original.solve()
    result=f.conic()
    assert result['ok'] and reference['converged']
    assert abs(result['cost']-reference['cost'])<1e-6
    assert f._cvx['problem'].is_dpp()


@pytest.mark.parametrize('matrix,answer',[
    ([[F(2),F(1)],[F(1),F(2)]],True),
    ([[F(1),F(1)],[F(1),F(1)]],False),
    ([[F(1),F(2)],[F(2),F(1)]],False),
    ([[F(1),F(0)],[F(1),F(1)]],False)])
def test_exact_ldl_rejects_singular_indefinite_and_asymmetric(matrix,answer):
    assert positive_definite(matrix)==answer


def test_exact_dual_bound_and_wrong_sign_rejection(f):
    e=ExactSDPBounds(f)
    repaired=f.dual_bound(np.zeros(f.nr))
    L=e.lower(repaired['multipliers'],f.ms.obs)
    assert L is not None and L<=16
    assert e.lower(np.ones(f.nr),f.ms.obs) is None


def test_exact_upper_rejects_collision(f):
    e=ExactSDPBounds(f)
    z=np.zeros(f.nz)
    assert e.upper(z,[(np.array([0.,0.]),.4),(np.array([.9,0.]),.2)],0.) is None
    safe=[(np.array([0.,10.]),.4),(np.array([.9,10.]),.2)]
    result=e.upper(z,safe,0.)
    assert result is not None and result[0]==16 and result[1]==6


def test_factor_global_check_rejects_bad_dual(f):
    r=f.conic()
    z,lam=f.seed(r)
    z,lam,_,_=f.newton(z,lam)
    good=f.assess(z,lam)
    assert good['ok']
    bad=f.assess(z,np.ones(f.nr))
    assert not bad['ok']
    exact=ExactSDPBounds(f).verify(z,good['bound']['multipliers'],f.ms.obs)
    assert exact['ok'] and exact['relative_gap_upper']<=1e-5


def test_local_failure_uses_conic_fallback(monkeypatch):
    f=make()
    first=f.solve(f.ms.obs)
    assert first['source']=='conic' and first['ok']
    monkeypatch.setattr(f,'newton',lambda z,l:(z,l,12,'iteration_limit'))
    monkeypatch.setattr(f,'assess',lambda z,l:dict(ok=False))
    second=f.solve(f.ms.obs)
    assert second['source']=='conic' and second['ok']
    assert second['total_ms']>=second['local_ms']
