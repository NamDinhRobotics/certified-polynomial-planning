# Certified polynomial trajectory planning

Minimal code, recorded experiment data and videos for checking the polynomial
trajectory-planning results. The manuscript and LaTeX sources are not included.

The implementation provides Bernstein polynomial models, a reusable factored
solver for the same SDP as the conic reference, rational primal/dual checks,
and physical-curve recovery. The quadrotor example separates the certified
reference trajectory from its measured MuJoCo tracking performance.

## Q1 revision evidence

The current source fixes rational-radius preservation, numerical fallback and
warm-seed failure handling. The complete follow-up contains 240 reference/mode
cases, 1,000 integrated calls, 480 budget/warm-use calls, and 48 family-transfer
cases. All outcomes, including unfavorable energy tails and failures, are retained.
See [Q1_REPRODUCTION.md](Q1_REPRODUCTION.md) for the exact commands and scope.
Historical executed source versions are preserved by their recorded SHA-256;
current src contains the corrected APIs. Use both the legacy and Q1 audits.

The controlled comparison finds that bubble often outperforms Green in recovered
energy. Workspace-tail construction adds no repaired success on the 240-case
population. These are disclosed limitations, not omitted cases. Manuscript,
LaTeX, response letters and editorial/reviewer records remain private.

## Verify the delivered results

Use Python 3.12 and run from the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py --full --mutations --output results/verification.json
python verify_peer_review.py --full --output results/followup_verification.json
python -m pytest -q tests
```

`verify.py` checks SHA-256 file integrity, complete experiment populations,
recomputed statistics, rational SDP intervals and continuous-time polynomial
constraints. Deliberately corrupted records must be rejected. Omit `--full`
for the shorter population, statistics and quadrotor audit. Do not use Python
`-O`, which disables the assertions used by the verification routines.

The full replay checks 3,000 paired same-SDP calls, 2,000 stored first-pass
rational intervals, all 152 family configurations at two elevation degrees,
1,000 development recoveries, 714 accepted fresh-population
recoveries, 5,287 accepted legacy trajectories, and both recorded TinySDP tracker
comparisons. Unit tests also compare formulations, derivatives, exact arithmetic
and rejected certificates. See [DATASETS.md](DATASETS.md) for the input map.

These checks establish the reported finite experiment results and certificate
predicates. They are not a machine-checked proof of every theorem, and the
simulation does not provide a hardware tracking-safety guarantee. Historical
clearance checks use the recorded tolerance; strict physical recovery and
quadrotor checks are reported separately.

## Regenerate the quadrotor demonstration

The delivered flight is one deterministic simulated transit motivated by inspection;
it does not demonstrate inspection coverage. New
outputs go to `results/quadrotor`; the recorded evidence in `demo/` is retained.

```sh
python examples/quadrotor.py plan
python -m pip install -r requirements-physics.txt
python examples/quadrotor.py physics
python examples/quadrotor.py check
```

The recorded physics run used MuJoCo 3.3.2 in a separate Python 3.10 environment.
The same output directory can be passed to each command with `--out`. Timing
and floating-point solver results can vary with the platform and dependencies.

## Rerun solver and recovery experiments

Use a fresh output path for each run. The following commands use the same
protocols and frozen recovery inputs as the delivered results:

```sh
mkdir -p results
python experiments/revision_certified_solver.py --out results/same_sdp.json
python experiments/revision_recovery_generalization.py --run \
  --population artifacts/revision_recovery_population.json \
  --out results/recovery.json
```

For a short executable check, add `--seeds 1 --steps 2 --repeats 1` to the
same-SDP command or `--limit 1` to the recovery command. Those subsets do not
reproduce the full reported population or timing statistics.

## Videos

- [3D quadrotor planning and simulated flight](video/polynomial_quadrotor_3d.mp4)
- [Replanning comparison](video/revision_replanning.mp4)
- [English captions](video/subtitles_english.srt) / [Vietnamese captions](video/subtitles_vietnamese.srt)

## Package contents

`src/` contains the mathematical implementation; `verification/` contains the
independent replay routines; `experiments/` contains the solver, recovery and
quadrotor runners; `tests/` contains regression and rejection tests. `artifacts/`
and `demo/` contain the evidence used by those routines. `MANIFEST.json` binds
all delivered files by hash and byte count.

`third_party/` retains the two TinySDP example-source snapshots used to identify
and recheck the recorded comparisons, together with their licenses and notices.
It is not a complete TinySDP build or benchmark runner. The minimal package
replays those stored comparisons; it does not regenerate their executable
measurements. `provenance/escape_fold.py` is the earlier implementation needed
to check the development-recovery source hash.

The delivered solver defaults to NumPy. The original timing protocol did not
record whether its optional native polishing was active. The separate
review_numpy_solver run explicitly forces NumPy and records that backend;
its results are not substituted for the archived measurements. Verification does not require a C compiler, LaTeX,
Blender or a video renderer.

## Post-review additions

All 240 original geometries were reused in an exploratory direct-Green and
one-step feasible-seed corridor follow-up. New outputs retain failures and
the full large energy tail. This corridor adaptation is not SIP or BMTP.

Run `python verify_peer_review.py --full` to replay all 320 new strict
physical outputs and 2,000 new NumPy SDP intervals. Run
`python experiments/review_affine_green.py --out results/direct_green.json`
or `python experiments/review_numpy_solver.py --out results/numpy_sdp.json`
for fresh serial runs. Never run timing experiments alongside rendering/tests.
The documented NumPy backend is enforced in the latter runner.

Install `requirements-figures.txt`, then run `python figures/generate.py` to
regenerate the ten historical scientific figures and the two new Q1 figures from the delivered data in
`results/figures`. Set writable `MPLCONFIGDIR` and `XDG_CACHE_HOME` if needed.
Run `python figures/peer_review_results.py` to recompute the added numerical
claims in `results/followup` (JSON plus generated numeric LaTeX macros).
The quadrotor figure uses delivered rendered stills and audited logs. The
complete Blender/video production pipeline is not part of this artifact.
The separate schematic in the manuscript is a LaTeX drawing, not data.

## Revision correspondence

This export matches the code and recorded evidence for the manuscript revision
of 2026-09-08. See [CHANGELOG.md](CHANGELOG.md) for the additions and limitations.
The Git commit and `MANIFEST.json` identify the exact public artifact version.
The manuscript, its LaTeX sources, submission packages and internal reviews
remain private and are not distributed in this repository.
