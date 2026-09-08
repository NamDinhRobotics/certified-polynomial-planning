from fractions import Fraction as F
import numpy as np
import pytest
from multisegment import MultiSegment
from certified_multisegment import CertifiedSpline
from exact_sdp_bounds import ExactSDPBounds
from exact_green import constraint_matrix
from constructive_recovery import (encode,decode,exact_energy,coefficients,recover_modes,endpoint_separation)
from escape_fold import physical_certificate


@pytest.fixture(scope='module',params=[2,3])
def example(request):
    n=request.param;a=np.zeros((1,n));b=a.copy();a[0,0]=-2;b[0,0]=2
    obs=[(np.zeros(n),.4)]
    f=CertifiedSpline(MultiSegment(n,5,1,0,3,obs,a,b,eta=2))
    return f,ExactSDPBounds(f),obs


def test_rational_serialization_does_not_round():
    a=np.array([[F(1,3),F(1,10**40)]],dtype=object)
    assert np.array_equal(a,decode(encode(a)))


def test_straight_energy_and_exact_affine(example):
    f,e,obs=example;G=coefficients(f,np.zeros((f.n,f.r)))
    assert exact_energy(G)==16
    assert endpoint_separation(G,obs)>0
    A=np.asarray(constraint_matrix(5,0,3,2),dtype=object).T
    assert np.all(G.transpose(1,0,2).reshape(f.n,18)@A==f.ms.B)


def test_recovery_without_any_sdp_gap(example):
    f,e,obs=example
    r=recover_modes(f,e,np.zeros((f.n,f.r)),None,obs,polish_steps=2)
    assert r['ok'] and r['method']=='green'
    G=decode(r['Gamma']);assert physical_certificate(G,obs)
    assert exact_energy(G)==F(*map(int,r['cost_exact']))
    A=np.asarray(constraint_matrix(5,0,3,2),dtype=object).T
    assert np.all(G.transpose(1,0,2).reshape(f.n,18)@A==f.ms.B)


def test_outside_condition_reports_unknown(example):
    f,e,obs=example;c=np.zeros(f.n);c[0]=-2;c[1]=.5
    obs=obs+[(c,.2)]
    r=recover_modes(f,e,np.zeros((f.n,f.r)),None,obs,polish_steps=2)
    assert not r['ok']
    assert all(a['reason']=='endpoint_lines_not_separated' for a in r['modes'] if a['mode']=='green')


def test_failed_line_condition_can_still_have_a_feasible_fold(example):
    from escape_fold import green_representer
    f,e,obs=example;c=np.zeros(f.n);c[0]=-2;c[1]=.5;obs=obs+[(c,.2)]
    G=coefficients(f,np.zeros((f.n,f.r)));_,z=green_representer(f.Qq,e.K,1,5)
    assert endpoint_separation(G,obs)<0
    G[:,1,:]-=z
    assert physical_certificate(G,obs) # Failure of the sufficient test is not infeasibility.
