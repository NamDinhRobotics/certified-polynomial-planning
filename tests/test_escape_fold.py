from fractions import Fraction as F
import numpy as np
from escape_fold import split,positive_lift,physical_certificate,escape_fold,rational


def scene():
    return rational([[[-2,0,2],[0,0,0]]]),rational([[0,1,0]]),[([0.,0.],.5)]


def test_de_casteljau_exact():
    l,r=split(rational([0,1,0]))
    assert list(l)==[F(0),F(1,2),F(1,2)]
    assert list(r)==[F(1,2),F(1,2),F(0)]


def test_positive_lift_deflates_pinned_zeros():
    assert positive_lift([[0,0,1,0,0]])
    assert not positive_lift([[0,-1,0]])
    assert not positive_lift([[1,-2,1]])
    assert not positive_lift([[0,0,0]])
    assert not positive_lift([[0,1,0],[0,1,0]]) # dead interior knot


def test_strict_physical_recovery_and_endpoint_preservation():
    g,z,obs=scene()
    assert not physical_certificate(g,obs)
    a=escape_fold(g,z,obs)
    assert a['ok'] and a['amplitude']>0
    assert np.array_equal(a['Gamma'][:,:,0],g[:,:,0])
    assert np.array_equal(a['Gamma'][:,:,-1],g[:,:,-1])
    assert physical_certificate(a['Gamma'],obs)


def test_lift_orientation_and_two_escape_signs():
    g,z,obs=scene()
    a=escape_fold(g,z,obs);b=escape_fold(g,-z,obs)
    c=escape_fold(g,z,obs,sign=-1)
    assert a['ok'] and b['ok'] and c['ok']
    assert np.array_equal(a['Gamma'],b['Gamma'])
    assert np.array_equal(a['Gamma'][:,1,:],-c['Gamma'][:,1,:])


def test_failed_sufficient_condition_is_unknown():
    g,z,obs=scene()
    a=escape_fold(g,z,obs,axis=0)
    assert not a['ok'] and a['reason']=='endpoint_lines_not_separated'


def test_budget_is_not_a_feasibility_claim():
    g,z,obs=scene()
    a=escape_fold(g,z,obs,max_depth=0)
    assert not a['ok'] and a['reason']=='budget_unknown'


def test_zero_tolerance_rejects_exact_tangency():
    assert not physical_certificate([[[0,0],[F(1,2),F(1,2)]]],[([0,0],.5)])
    assert physical_certificate([[[0,0],[F(1,2)+F(1,10**20)]*2]],[([0,0],.5)])


def test_three_dimensional_transverse_box():
    g,z,obs=scene()
    g=np.concatenate([g,np.zeros((1,1,3),dtype=object)],axis=1)
    assert escape_fold(g,z,[([0,0,0],.5)])['ok']


def test_green_representer_unit_height_and_energy():
    from escape_fold import green_representer
    Q=rational([[0],[1],[0]]);K=rational([[4]])
    v,z=green_representer(Q,K,0,2)
    assert v[0]==2 and list(z[0])==[0,2,0]
    assert positive_lift(z)
    # Unit midpoint value, unique solution in this one-dimensional nullspace.
    assert sum(z[0,k]*F([1,2,1][k],4) for k in range(3))==1
