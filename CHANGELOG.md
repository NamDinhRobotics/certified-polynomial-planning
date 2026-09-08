# 2026-09-08 revised evidence export

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
