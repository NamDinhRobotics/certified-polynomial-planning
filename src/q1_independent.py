"""Independent certificate arithmetic: no imports from solver or project helpers.

Bernstein convolution, derivative boundary checks and energy are rebuilt here.
Strict positivity uses exact coefficient subdivision; unresolved cells use SymPy
Sturm root counting. A sampled negative value is an exact counterexample, not a
sample-based positive certificate. All values represent the recorded geometry.
"""
from fractions import Fraction as F
from math import comb,factorial
import numpy as np
if not __debug__:
 raise RuntimeError('Certificate checker requires assertions; Python -O is forbidden')


def frac(x):return x if isinstance(x,F) else F(int(x)) if isinstance(x,(int,np.integer)) else F(float(x))
def decode(x):return np.array([F(int(a),int(b)) for a,b in x['values']],dtype=object).reshape(x['shape'])
def split(c):
 rows=[list(c)];left=[c[0]];right=[c[-1]]
 while len(rows[-1])>1:
  x=rows[-1];row=[(a+b)/2 for a,b in zip(x[:-1],x[1:])];rows.append(row);left.append(row[0]);right.append(row[-1])
 return left,right[::-1]
def restrict(c,depth,index):
 out=list(c)
 for j in range(depth-1,-1,-1):out=split(out)[(index>>j)&1]
 return out

def square(c):
 d=len(c)-1
 return [sum(c[j]*c[k-j]*F(comb(d,j)*comb(d,k-j),comb(2*d,k)) for j in range(max(0,k-d),min(d,k)+1)) for k in range(2*d+1)]
def obstacle_coefficients(g,center,radius):
 d=g.shape[1]-1;ans=[F(0)]*(2*d+1)
 for coord,c in zip(g,center):
  sq=square([v-frac(c) for v in coord]);ans=[a+b for a,b in zip(ans,sq)]
 return [a-frac(radius)**2 for a in ans]

def classify(c,max_depth=18):
 c=list(c);stack=[(c,0)];lower=None;upper=min(c[0],c[-1]);leaves=0
 while stack:
  a,depth=stack.pop();lo=min(a)
  if a[0]<0 or a[-1]<0:return dict(status='NEGATIVE_WITNESS',value=str(min(a[0],a[-1])))
  if a[0]==0 or a[-1]==0:return dict(status='NOT_STRICT_ZERO_WITNESS')
  if lo>0:
   lower=lo if lower is None else min(lower,lo);upper=min(upper,a[0],a[-1]);leaves+=1;continue
  if depth==max_depth:
   import sympy as sp
   x=sp.symbols('x');d=len(a)-1
   p=sp.Poly(sum(sp.Rational(v.numerator,v.denominator)*comb(d,j)*x**j*(1-x)**(d-j) for j,v in enumerate(a)),x)
   if p.count_roots(0,1):return dict(status='NOT_STRICT_ROOT_WITNESS')
   # No root and positive endpoints prove strict positivity, but a zero
   # lower bound is all this branch exports for quantitative clearance.
   lower=F(0);upper=min(upper,a[0],a[-1]);leaves+=1;continue
  l,r=split(a);stack.extend([(r,depth+1),(l,depth+1)])
 return dict(status='STRICT_POSITIVE',lower=str(lower),upper=str(upper),leaves=leaves)

def derivative(c,order):
 c=list(c);d=len(c)-1
 for j in range(order):c=[(d-j)*(b-a) for a,b in zip(c[:-1],c[1:])]
 return c

def affine(G,bc0,bc1,eta):
 for j,values in enumerate(bc0):
  assert [derivative(c,j)[0] for c in G[0]]==[frac(x) for x in values]
 for j,values in enumerate(bc1):
  assert [derivative(c,j)[-1] for c in G[-1]]==[frac(x) for x in values]
 for left,right in zip(G[:-1],G[1:]):
  for j in range(eta+1):assert [derivative(c,j)[-1] for c in left]==[derivative(c,j)[0] for c in right]
 return True

def energy(G,k):
 N,n,h=G.shape;d=h-1-k;total=F(0)
 for g in G:
  for c in g:
   a=derivative(c,k)
   total+=sum(a[i]*a[j]*F(comb(d,i)*comb(d,j),(2*d+1)*comb(2*d,i+j)) for i in range(d+1) for j in range(d+1))
 return F(N)**(2*k-1)*total

def curve(a,obs,k,bc0,bc1,eta):
 G=decode(a['Gamma']);affine(G,bc0,bc1,eta);J=energy(G,k)
 assert J==F(*map(int,a['cost_exact']))
 reports=[classify(obstacle_coefficients(g,c,r)) for g in G for c,r in obs]
 assert all(x['status']=='STRICT_POSITIVE' for x in reports)
 return dict(polynomials=len(reports),squared_clearance_lower=min(F(x['lower']) for x in reports),squared_clearance_upper=min(F(x['upper']) for x in reports))

def tail(t,obs,bc0,bc1,eta):
 G,z=decode(t['reference']),decode(t['z']);affine(G,bc0,bc1,eta)
 affine(z[:,None,:],[[0] for _ in bc0],[[0] for _ in bc1],eta)
 axis,sign=t['axis'],t['sign'];eps=F(*map(int,t['margin']));T=F(*map(int,t['amplitude']));assert eps>0 and T>=0 and axis in range(G.shape[1]) and sign in (-1,1)
 cover={};axes=[a for a in range(G.shape[1]) if a!=axis]
 for cell in t['cells']:
  i,j,depth,index=[cell[k] for k in ('segment','obstacle','depth','index')]
  assert 0<=i<len(G) and 0<=j<len(obs) and 0<=index<2**depth
  cover.setdefault((i,j),[]).append((F(index,2**depth),F(index+1,2**depth)))
  c,r=obs[j];R=frac(r)+eps
  dd=[restrict([v-frac(c[a]) for v in G[i,a]],depth,index) for a in range(G.shape[1])]
  zz=restrict(z[i],depth,index)
  if cell['kind']=='transverse':
   bound=sum((min(dd[a]) if min(dd[a])>0 else -max(dd[a]) if max(dd[a])<0 else F(0))**2 for a in axes)
   assert bound==F(*map(int,cell['bound'])) and bound>R*R
  else:
   lo=min(zz);a=min(sign*x for x in dd[axis]);assert lo>0
   threshold=max(F(0),(R-a)/lo)
   assert threshold==F(*map(int,cell['threshold'])) and T>=threshold
   assert lo==F(*map(int,cell['mode_lower'])) and a==F(*map(int,cell['longitudinal_lower']))
 for i in range(len(G)):
  for j in range(len(obs)):
   spans=sorted(cover[(i,j)]);assert spans[0][0]==0 and spans[-1][1]==1
   assert all(a[1]==b[0] for a,b in zip(spans[:-1],spans[1:]))
 return len(t['cells'])
