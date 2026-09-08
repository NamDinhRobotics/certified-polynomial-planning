"""Exact input conversion and resource-failure boundary for certificate services."""
from fractions import Fraction
from numbers import Integral
from functools import wraps
import time
from exact_check import ExactCheckError


def rational_scalar(value):
    """Preserve exact inputs; retain historical binary-float interpretation."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value))
    return Fraction(float(value))


def exact_service(function):
    """A predicate exhaustion is UNKNOWN, never collision or infeasibility.

    Low-level Boolean predicates still raise ExactCheckError, so callers cannot
    silently confuse an undecided check with a mathematically negative result.
    """
    @wraps(function)
    def checked(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        except ExactCheckError as exc:
            return dict(ok=False, status='UNKNOWN_EXACT_CHECK',
                        reason='exact_check_unresolved', exception_type=type(exc).__name__,
                        detail=str(exc), ms=1000*(time.perf_counter()-start))
    return checked
