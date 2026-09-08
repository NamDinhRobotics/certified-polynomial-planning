# 2026-09-08 Q1 technical revision

This revision fixes exact rational input preservation, local-solver exception
fallback, and warm-seed failure handling. It adds complete controlled evidence:
240 reference/mode cases, 1,000 integrated calls, 480 budget/warm-use calls,
48 family-transfer cases, and diagnostics for 1,250 repair inputs.

The independent physical audit checks complete populations, exact input binding,
affine constraints, energy, cell covers and workspace inequalities. SDP replay
uses the actual serialized conic witnesses. Mutation and regression controls
reject malformed certificates and preserve UNKNOWN for exhausted checks.

Bubble often outperforms Green in recovered energy. The conservative workspace
tail adds no new recovered success in this population. Complete-return physical
and objective times include their exact checks and serialization. These results
do not imply universal runtime or full-planner superiority.

Recorded historical artifacts remain byte-identical. Exact historical source
versions live under their recorded SHA-256 in `executed_sources/`; current `src/`
contains the corrected implementation. See `EXECUTED_SOURCES.md` and
`Q1_REPRODUCTION.md`. No manuscript, LaTeX, response or review record is published.

## Earlier 2026-09-08 five-reviewer evidence export

This update synchronizes the public reproduction material with the revised
manuscript. It contains code, recorded numerical data, figure-generation
scripts and demonstration videos. It contains no manuscript, manuscript LaTeX,
submission package or internal review correspondence.

- Add the complete 240-case direct-Green / one-step feasible-seed QP follow-up,
  including failures and the large energy tail. This is an exploratory use of
  previously seen inputs, not a held-out experiment or a full SIP/BMTP benchmark.
- Add a separate 3,000-pair same-SDP experiment with NumPy polishing explicitly
  enforced, including 2,000 stored first-pass rational intervals. Preserve the
  historical timing records whose optional polishing backend was not recorded.
- Correct the Green witness documentation and preserve the source bytes from
  measurement. The executable solver implementation is unchanged by that fix.
- Clarify that `median_ratio` is the ratio of marginal median runtimes. In the
  new NumPy run this ratio is 2.50; factor/reference p99 times are 50.37/45.75 ms,
  so the factor arm has a worse tail in this measurement.
- Add regeneration scripts for the retained figures, scene stills and audited
  video metadata, and the smaller revised quadrotor video. Flight records are
  unchanged. Simulated tracking diagnostics are not a hardware safety proof.
- Add the follow-up verifier. Its temporary replay files stay outside the
  delivered datasets; optional reports belong under `results/`.

All previously published numerical artifacts in `artifacts/` and the original
planning/tracking records are retained byte for byte. Each new experiment has
its own data and protocol files. TinySDP support remains source snapshots and
recorded-result replay, not a complete executable benchmark distribution.
