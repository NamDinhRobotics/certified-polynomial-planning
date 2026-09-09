"""Exact reconstruction of an exported root polynomial SDP objective interval.

The checker consumes an affine Bernstein basis, a lifted primal point and
physical-energy dual multipliers. It does not call an optimizer. Floating
scene coordinates and the declared shift ladder mean their binary rational
values; encoded arrays and objective bounds carry numerator/denominator pairs.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import comb, factorial, inf, nextafter, prod
import json
import numpy as np

from exact import classify, frac, square


class ObjectiveError(ValueError):
    """An objective witness or its claimed interval failed verification."""


def require(condition, message):
    if not condition:
        raise ObjectiveError(message)


def rational_pair(value):
    require(isinstance(value, (list, tuple)) and len(value) == 2,
            'Expected a numerator/denominator pair')
    require(all(isinstance(v, (str, int)) and not isinstance(v, bool)
                for v in value), 'Rational components must be integer strings or integers')
    numerator, denominator = map(int, value)
    require(denominator > 0, 'Rational denominator must be positive')
    return F(numerator, denominator)


def record(value):
    return [str(value.numerator), str(value.denominator)]


def array(encoded, *, binary=False):
    require(isinstance(encoded, dict), 'Expected an encoded rational array')
    shape, values = encoded['shape'], encoded['values']
    require(isinstance(shape, list) and len(shape) in (1, 2),
            'Expected a vector or matrix')
    require(all(type(v) is int and v >= 0 for v in shape), 'Invalid array dimensions')
    require(len(values) == prod(shape), 'Encoded array size mismatch')
    entries = [rational_pair(v) for v in values]
    if binary:
        require(all(v.denominator & (v.denominator - 1) == 0 for v in entries),
                'Witness values must be exact binary rationals')
    return np.asarray(entries, dtype=object).reshape(shape)


def _endpoint(d, order, end):
    row = [F(0)] * (d + 1)
    scale = factorial(d) // factorial(d - order)
    for j in range(order + 1):
        row[(d - order if end else 0) + j] = F(scale * (-1) ** (order - j) * comb(order, j))
    return row


def _constraint_system(scene):
    family = scene['family']
    require(not set(family) - {'n', 'd', 'k', 'l', 'N', 'eta', 'normalise_time', 'cuts'},
            'Unsupported family fields')
    require(family.get('normalise_time', True) is True and not family.get('cuts'),
            'Only normalized-time root SDPs without point cuts are supported')
    n, d, k, l, N = (family[name] for name in ('n', 'd', 'k', 'l', 'N'))
    eta = family.get('eta', d - 1)
    require(all(type(v) is int for v in (n, d, k, l, N, eta)),
            'Family dimensions must be integers')
    require(n > 0 and N > 0 and 1 <= d and 0 <= k <= d
            and 0 <= l <= d and 0 <= eta <= d, 'Invalid polynomial family')
    M = N * (d + 1)
    bc0 = np.asarray([[frac(x) for x in row] for row in scene['bc0']], dtype=object)
    bc1 = np.asarray([[frac(x) for x in row] for row in scene['bc1']], dtype=object)
    require(bc0.shape == bc1.shape == (l + 1, n), 'Boundary data dimensions mismatch')
    rows, rhs = [], []
    for order in range(l + 1):
        for end, boundary in ((0, bc0), (1, bc1)):
            row = [F(0)] * M
            start = (N - 1) * (d + 1) if end else 0
            row[start:start + d + 1] = _endpoint(d, order, end)
            rows.append(row)
            rhs.append(list(boundary[order]))
    for segment in range(N - 1):
        for order in range(eta + 1):
            row = [F(0)] * M
            start = segment * (d + 1)
            row[start:start + d + 1] = _endpoint(d, order, 1)
            row[start + d + 1:start + 2 * (d + 1)] = [-v for v in _endpoint(d, order, 0)]
            rows.append(row)
            rhs.append([F(0)] * n)
    return np.asarray(rows, dtype=object), np.asarray(rhs, dtype=object), (n, d, k, l, N, eta)


def _rank(matrix):
    rows = [list(row) for row in matrix]
    rank = 0
    for column in range(len(rows[0])):
        pivot = next((i for i in range(rank, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        divisor = rows[rank][column]
        for i in range(rank + 1, len(rows)):
            multiplier = rows[i][column] / divisor
            if multiplier:
                for j in range(column, len(rows[0])):
                    rows[i][j] -= multiplier * rows[rank][j]
        rank += 1
        if rank == len(rows):
            break
    return rank


def _ldl(matrix):
    """Rational LDL decomposition with strictly positive pivots."""
    A = [list(row) for row in matrix]
    n = len(A)
    require(all(len(row) == n for row in A), 'PSD block is not square')
    require(all(A[i][j] == A[j][i] for i in range(n) for j in range(i)),
            'PSD block is not symmetric')
    L = [[F(i == j) for j in range(n)] for i in range(n)]
    D = []
    for k in range(n):
        pivot = A[k][k]
        require(pivot > 0, 'PSD block has a nonpositive exact LDL pivot')
        D.append(pivot)
        for i in range(k + 1, n):
            L[i][k] = A[i][k] / pivot
            for j in range(i, n):
                A[j][i] -= A[i][k] * A[j][k] / pivot
                A[i][j] = A[j][i]
    return L, D


def _solve_ldl(decomposition, rhs):
    L, D = decomposition
    result = np.asarray(rhs, dtype=object).copy()
    for i in range(len(D)):
        for j in range(i):
            result[i] -= L[i][j] * result[j]
    for i, divisor in enumerate(D):
        result[i] /= divisor
    for i in range(len(D) - 1, -1, -1):
        for j in range(i + 1, len(D)):
            result[i] -= L[j][i] * result[j]
    return result


def _derivative_gram(d, k):
    reduced = d - k
    gram = np.asarray([[F(comb(reduced, i) * comb(reduced, j),
                          (2 * reduced + 1) * comb(2 * reduced, i + j))
                        for j in range(reduced + 1)] for i in range(reduced + 1)], dtype=object)
    derivative = np.zeros((d + 1, reduced + 1), dtype=object)
    scale = factorial(d) // factorial(reduced)
    for j in range(reduced + 1):
        for t in range(k + 1):
            derivative[j + t, j] = F(scale * (-1) ** (k - t) * comb(k, t))
    return derivative @ gram @ derivative.T


def _square_norm(rows, d):
    coefficients = [F(0)] * (2 * d + 1)
    for row in rows:
        coefficients = [a + b for a, b in zip(coefficients, square(row))]
    return coefficients


class _Model:
    def __init__(self, scene, basis):
        constraints, rhs, dimensions = _constraint_system(scene)
        self.n, self.d, self.k, _, self.N, _ = dimensions
        self.M = self.N * (self.d + 1)
        self.G0, self.Q = array(basis['G0']), array(basis['Q'])
        require(self.G0.shape == (self.n, self.M), 'Affine basis dimensions mismatch')
        require(self.Q.ndim == 2 and self.Q.shape[0] == self.M,
                'Nullspace basis dimensions mismatch')
        self.r = self.Q.shape[1]
        self.constraint_rank = _rank(constraints)
        self.constraint_rows = len(constraints)
        require(self.r == self.M - self.constraint_rank and self.r > 0,
                'Nullspace basis does not have the complete exact nullity')
        require(np.array_equal(constraints @ self.G0.T, rhs),
                'Affine basis violates exact boundary or continuity equations')
        require(not np.any(constraints @ self.Q),
                'Nullspace basis violates exact homogeneous equations')
        self.obstacles = []
        for center, radius in scene['obstacles']:
            center = np.asarray([frac(v) for v in center], dtype=object)
            radius = frac(radius)
            require(center.shape == (self.n,) and radius >= 0, 'Invalid ball obstacle')
            self.obstacles.append((center, radius))
        self.m = len(self.obstacles)
        self.b = self.N * self.m
        self.h = 2 * self.d + 1
        self.K = np.zeros((self.r, self.r), dtype=object)
        self.C = np.zeros((self.n, self.r), dtype=object)
        self.c0 = F(0)
        gram = _derivative_gram(self.d, self.k)
        tau = F(self.N) ** (2 * self.k - 1)
        for i in range(self.N):
            sl = self.segment(i)
            q, g = self.Q[sl], self.G0[:, sl]
            self.K += tau * (q.T @ gram @ q)
            self.C += tau * (g @ gram @ q)
            self.c0 += tau * np.sum((g @ gram) * g)
        # Positive energy on Q proves independent columns; its column count
        # and homogeneous equations then prove that it spans the nullspace.
        _ldl(self.K)
        self.unit = F(1)
        while self.unit < max(F(1), abs(self.c0)):
            self.unit *= 2
        self.S = np.zeros((self.h, self.d + 1, self.d + 1), dtype=object)
        self.T = np.zeros((self.h, self.d, self.d), dtype=object)
        for k in range(self.h):
            for i in range(self.d + 1):
                j = k - i
                if 0 <= j <= self.d:
                    self.S[k, i, j] = F(comb(self.d, i) * comb(self.d, j), comb(2 * self.d, k))
            for i in range(self.d):
                j = k - 1 - i
                if 0 <= j < self.d:
                    self.T[k, i, j] = F(comb(self.d - 1, i) * comb(self.d - 1, j), comb(2 * self.d, k))

    def segment(self, index):
        return slice(index * (self.d + 1), (index + 1) * (self.d + 1))

    def lower(self, multipliers):
        require(multipliers.shape in ((self.b * self.h,), (self.b, self.h)),
                'Dual multiplier dimensions mismatch')
        multipliers = multipliers.reshape(self.b, self.h)
        Z, h, constant = self.K.copy(), self.C.copy(), self.c0
        for segment in range(self.N):
            q, g = self.Q[self.segment(segment)], self.G0[:, self.segment(segment)]
            for obstacle, (center, radius) in enumerate(self.obstacles):
                lam = multipliers[segment * self.m + obstacle]
                weighted_S = sum((lam[k] * self.S[k] for k in range(self.h)))
                weighted_T = sum((lam[k] * self.T[k] for k in range(self.h)))
                _ldl(-weighted_S)
                _ldl(-weighted_T)
                delta = g - center[:, None]
                Z += q.T @ weighted_S @ q
                h += delta @ weighted_S @ q
                constant += np.sum((delta @ weighted_S) * delta) - radius ** 2 * sum(lam)
        solved = _solve_ldl(_ldl(Z), h.T).T
        return constant - np.sum(h * solved)

    def upper_polynomials(self, Y, V):
        gamma, lift = self.G0 + Y @ self.Q.T, V @ self.Q.T
        polynomials = []
        for i in range(self.N):
            sl = self.segment(i)
            lifted = _square_norm(lift[:, sl], self.d)
            shifted = _square_norm(self.Q[sl].T, self.d)
            for center, radius in self.obstacles:
                distance = _square_norm(gamma[:, sl] - center[:, None], self.d)
                polynomials.append(([a + b - radius ** 2 for a, b in zip(distance, lifted)], shifted))
        cost = (self.c0 + 2 * np.sum(self.C * Y) + np.sum((Y @ self.K) * Y)
                + np.sum((V @ self.K) * V))
        return cost, polynomials


@lru_cache(maxsize=64)
def _cached_model(scene_key, basis_key):
    return _Model(json.loads(scene_key), json.loads(basis_key))


def verify_objective(scene, basis, witness, saved, objective_path):
    """Recompute and check a strict root-SDP interval, raising on any failure.

    ``basis`` contains encoded ``G0`` (n by N(d+1)) and ``Q`` (N(d+1)
    by r). ``witness`` contains encoded matrices ``point=[Y,V]``, encoded
    physical-energy ``multipliers``, and ``energy_unit=[num,den]``.
    ``saved`` supplies ``lower_exact``, ``upper_exact`` and ``shift``.
    Success flags in exported records do not enter any verification step.
    """
    require(objective_path in {'original_certificate', 'current_root_last_attempt',
                               'independent_same_SDP_objective_witness'},
            'Unknown objective witness path')
    model_scene = {key: scene[key] for key in ('family', 'bc0', 'bc1', 'obstacles')}
    model = _cached_model(json.dumps(model_scene, sort_keys=True, allow_nan=False),
                          json.dumps(basis, sort_keys=True, allow_nan=False))
    expected_unit = model.unit / (4 if objective_path == 'independent_same_SDP_objective_witness' else 1)
    require(rational_pair(witness['energy_unit']) == expected_unit,
            'Objective energy-unit provenance mismatch')
    require(isinstance(witness['point'], list) and len(witness['point']) == 2,
            'Expected primal point [Y,V]')
    Y, V = [array(value, binary=True) for value in witness['point']]
    require(Y.shape == (model.n, model.r), 'Primal Y dimensions mismatch')
    require(V.ndim == 2 and V.shape[1] == model.r, 'Primal V dimensions mismatch')
    lower = model.lower(array(witness['multipliers'], binary=True))
    primal_cost, polynomials = model.upper_polynomials(Y, V)
    polynomial_tests = 0
    for trial, shift in enumerate(map(F, (0., 1e-12, 1e-10, 1e-8, 1e-6)), 1):
        feasible = True
        for coefficients, shifted in polynomials:
            polynomial_tests += 1
            p = [a + shift * b for a, b in zip(coefficients, shifted)]
            if classify(p)['status'] != 'STRICT_POSITIVE':
                feasible = False
                break
        if feasible:
            break
    else:
        raise ObjectiveError('No strict lifted primal witness on the declared shift ladder')
    upper = primal_cost + shift * np.trace(model.K)
    gap = (upper - lower) / max(F(1), abs(upper))
    require(0 <= gap <= F(1, 100000), 'Exact root-SDP relative gap exceeds 1/100000')
    require(lower == rational_pair(saved['lower_exact']), 'Saved exact lower bound mismatch')
    require(upper == rational_pair(saved['upper_exact']), 'Saved exact upper bound mismatch')
    saved_shift = rational_pair(saved['shift']) if isinstance(saved['shift'], (list, tuple)) else frac(saved['shift'])
    require(shift == saved_shift, 'Saved shift is not the first feasible ladder shift')
    for key, expected in (('n_polys', model.b), ('n_pd_blocks', 1 + 2 * model.b)):
        if key in saved:
            require(saved[key] == expected, 'Saved ' + key + ' count mismatch')
    outward = dict(lower=nextafter(float(lower), -inf), upper=nextafter(float(upper), inf),
                   relative_gap_upper=nextafter(float(gap), inf))
    for key, expected in outward.items():
        if key in saved:
            require(saved[key] == expected, 'Saved outward-rounded ' + key + ' mismatch')
    return dict(ok=True, reason='bounds', lower_exact=record(lower), upper_exact=record(upper),
                relative_gap_exact=record(gap), **outward,
                shift=float(shift), shift_exact=record(shift), energy_unit_exact=record(expected_unit),
                n_polys=model.b, n_pd_blocks=1 + 2 * model.b,
                n_basis_pd_blocks=1, n_shift_trials=trial,
                n_upper_polynomial_tests=polynomial_tests, constraint_rank=model.constraint_rank,
                nullity=model.r, n_affine_equations=model.constraint_rows * model.n,
                n_homogeneous_equations=model.constraint_rows * model.r,
                constant_energy_exact=record(model.c0), linear_energy_zero=not bool(np.any(model.C)))
