"""New-protocol name for the shared, instrumented certified dispatcher."""
from certified_multisegment import CertifiedSpline


class InstrumentedSpline(CertifiedSpline):
    """Identical gates, fallback handling, and telemetry to CertifiedSpline."""
