from fractions import Fraction as F
from math import comb, factorial
import numpy as np
if not __debug__:
    raise RuntimeError('Certificate checker requires assertions; Python -O is forbidden')

def frac(x):
    return x if isinstance(x, F) else F(int(x)) if isinstance(x, (int, np.integer)) else F(float(x))

def decode(x):
    return np.array([F(int(a), int(b)) for a, b in x['values']], dtype=object).reshape(x['shape'])

def split(c):
    rows = [list(c)]
    left = [c[0]]
    right = [c[-1]]
    while len(rows[-1]) > 1:
        x = rows[-1]
        row = [(a + b) / 2 for a, b in zip(x[:-1], x[1:])]
        rows.append(row)
        left.append(row[0])
        right.append(row[-1])
    return (left, right[::-1])

def restrict(c, depth, index):
    out = list(c)
    for j in range(depth - 1, -1, -1):
        out = split(out)[index >> j & 1]
    return out

def square(c):
    d = len(c) - 1
    return [sum((c[j] * c[k - j] * F(comb(d, j) * comb(d, k - j), comb(2 * d, k)) for j in range(max(0, k - d), min(d, k) + 1))) for k in range(2 * d + 1)]

def obstacle_coefficients(g, center, radius):
    d = g.shape[1] - 1
    ans = [F(0)] * (2 * d + 1)
    for coord, c in zip(g, center):
        sq = square([v - frac(c) for v in coord])
        ans = [a + b for a, b in zip(ans, sq)]
    return [a - frac(radius) ** 2 for a in ans]

def classify(c, max_depth=18):
    c = list(c)
    stack = [(c, 0)]
    lower = None
    upper = min(c[0], c[-1])
    leaves = 0
    while stack:
        a, depth = stack.pop()
        lo = min(a)
        if a[0] < 0 or a[-1] < 0:
            return dict(status='NEGATIVE_WITNESS', value=str(min(a[0], a[-1])))
        if a[0] == 0 or a[-1] == 0:
            return dict(status='NOT_STRICT_ZERO_WITNESS')
        if lo > 0:
            lower = lo if lower is None else min(lower, lo)
            upper = min(upper, a[0], a[-1])
            leaves += 1
            continue
        if depth == max_depth:
            import sympy as sp
            x = sp.symbols('x')
            d = len(a) - 1
            p = sp.Poly(sum((sp.Rational(v.numerator, v.denominator) * comb(d, j) * x ** j * (1 - x) ** (d - j) for j, v in enumerate(a))), x)
            if p.count_roots(0, 1):
                return dict(status='NOT_STRICT_ROOT_WITNESS')
            lower = F(0)
            upper = min(upper, a[0], a[-1])
            leaves += 1
            continue
        l, r = split(a)
        stack.extend([(r, depth + 1), (l, depth + 1)])
    return dict(status='STRICT_POSITIVE', lower=str(lower), upper=str(upper), leaves=leaves)

def derivative(c, order):
    c = list(c)
    d = len(c) - 1
    for j in range(order):
        c = [(d - j) * (b - a) for a, b in zip(c[:-1], c[1:])]
    return c

def affine(G, bc0, bc1, eta):
    for j, values in enumerate(bc0):
        assert [derivative(c, j)[0] for c in G[0]] == [frac(x) for x in values]
    for j, values in enumerate(bc1):
        assert [derivative(c, j)[-1] for c in G[-1]] == [frac(x) for x in values]
    for left, right in zip(G[:-1], G[1:]):
        for j in range(eta + 1):
            assert [derivative(c, j)[-1] for c in left] == [derivative(c, j)[0] for c in right]
    return True

def energy(G, k):
    N, n, h = G.shape
    d = h - 1 - k
    total = F(0)
    for g in G:
        for c in g:
            a = derivative(c, k)
            total += sum((a[i] * a[j] * F(comb(d, i) * comb(d, j), (2 * d + 1) * comb(2 * d, i + j)) for i in range(d + 1) for j in range(d + 1)))
    return F(N) ** (2 * k - 1) * total
