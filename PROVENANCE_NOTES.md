# Measured source and later documentation corrections

The new NumPy benchmark records the SHA-256 of `src/exact_green.py` as it existed at measurement. The later review corrected docstrings only: nonnegative coefficient blocks plus endpoint structure are the required witness; strict coefficient positivity is a stronger optional property. The helper is named `endpoint_structure`.

The exact measured source bytes are preserved in `provenance/exact_green_at_numpy_measurement.py`. Their hash is the one recorded by the NumPy protocol. Removing docstrings yields the same executable AST as the delivered `src/exact_green.py`. The direct-Green protocol was recorded between the documentation corrections and also refers to the same executable implementation. No measured record or protocol hash is rewritten. Historical timing records still do not identify their optional polishing backend; the new NumPy driver explicitly disables it.

In presentation helpers, the legacy key `median_ratio` means the ratio of marginal medians (reference p50 divided by factor p50), not the median of per-pair ratios. Reported values are unchanged.
