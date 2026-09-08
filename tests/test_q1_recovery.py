from fractions import Fraction as F
import numpy as np
from q1_recovery import safe_tail,recover,bubble_mode,workspace_interval
from escape_fold import rational,physical_certificate

def scene():return rational([[[-2,0,2],[0,0,0]]]),rational([[0,2,0]]),[([0,0],.5)]
def test_tail_and_cell_cover_preserve_endpoints():
 g,z,o=scene();a=safe_tail(g,z,o);assert a['ok'] and a['proof']
 assert np.array_equal(a['Gamma'][:,:,0],g[:,:,0])
 for multiplier in [1,2,100]:
  x=g.copy();x[:,1,:]+=multiplier*a['amplitude']*a['z'];assert physical_certificate(x,o)
def test_unknown_geometry_sign_budget_are_distinct():
 g,z,o=scene();assert safe_tail(g,z,o,axis=0)['status']=='UNKNOWN_GEOMETRY'
 assert safe_tail(g,np.zeros_like(z),o)['status']=='UNKNOWN_SIGN'
 assert safe_tail(g,z,o,max_depth=0)['status']=='UNKNOWN_BUDGET'
def test_workspace_empty_is_unknown_and_valid_interval_is_safe():
 g,z,o=scene();normal=np.r_[np.eye(2),-np.eye(2)]
 a=recover(g,[('bubble',z)],o,workspace=(normal,[3,6,3,6]));assert a['ok']
 b=recover(g,[('bubble',z)],o,workspace=(normal,[3,.1,3,.1]));assert not b['ok'] and b['status']=='UNKNOWN_WORKSPACE'
def test_f2_bubble_junctions_and_endpoint_derivatives():
 z=bubble_mode(3,5)
 assert z[0,0]==z[-1,-1]==0
 for left,right in zip(z[:-1],z[1:]):
  for j in range(3):assert np.diff(left,n=j)[-1]==np.diff(right,n=j)[0]
def test_polishing_never_discards_guaranteed_incumbent():
 g,z,o=scene();a=recover(g,[('bubble',z)],o,polish_steps=12)
 assert a['ok'] and all(x['candidate_cost']<=x['checkpoints'][0]['cost'] for x in a['attempts'] if x['ok'])
