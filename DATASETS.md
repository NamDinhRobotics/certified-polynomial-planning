# Data used by the verification

Only datasets supporting the reported experiments are included. Complete
populations are retained, including rejected cases and slow runs. No rows are
removed to improve success rates or timings. The original JSON bytes and
protocol hashes are preserved.

| Files in `artifacts/` | Evidence checked |
| --- | --- |
| `revision_streaming_F1.json`, `revision_streaming_F2.json` | Original streaming arms, decisions, costs, runtimes and accepted polynomial curves. |
| `revision_certificates.json` | Family certificate grid, including cases outside the sufficient condition. |
| `revision_frozen.json`, `revision_rank_profile.json`, `revision_serial_timing.json` | Frozen-scene ablation, rank profiles and serial timing measurements. |
| `revision_certified_solver.json`, `revision_certified_solver.protocol.json` | All 3,000 paired calls on the same SDP, fallback decisions and stored rational witnesses. |
| `revision_escape_fold.json`, `revision_escape_fold.protocol.json`, `revision_green_recovery.json`, `revision_green_recovery.protocol.json` | Development recovery comparison and its fixed protocols. |
| `revision_recovery_population.json`, `revision_recovery_generalization.json` | All 240 frozen inputs in six strata and every method's outcomes, including failures and outside-condition cases. |
| `revision_tinysdp.json`, `a83_tinysdp_ladder.json` | Recorded TinySDP accuracy/latency comparison and historical result used in the comparison. |
| `revision_replanning.json`, `revision_replanning.protocol.json`, `revision_cadence.json`, `revision_cadence.protocol.json` | Replanning/tracking logs and update-cadence measurements with fixed protocols. |

`demo/planning.json` contains the selected frozen case, obstacle geometry,
polynomial coefficients, guide construction and certificate records.
`demo/tracking.json` contains the full sampled simulated flight and model XML.
`demo/tracking_validation.json` contains simulation configuration, source hashes,
tracking metrics and disturbance variants. `demo/quadrotor.xml` is the flight
model. The verification recomputes geometry and metrics from these records.

The two MP4 files illustrate these recorded experiments. Captions accompany
the 3D quadrotor video. Video rendering is illustrative; the polynomial data
and flight logs are the evidence checked by the code.

Some original provenance records name source modules from the larger research
workspace. Their hashes are retained as recorded metadata; unrelated module
contents and workspace history are not distributed here. Timing records refer
to the recorded machine and software configuration, not universal performance.

`review_affine_green.json` and its protocol contain all 240 direct/refined
outcomes; `review_numpy_solver.json` and protocol contain a separate 3,000-pair
NumPy run and 2,000 rational intervals. These are post-review replications on
previously seen inputs, not additional held-out populations. Original files
are byte-preserved.
