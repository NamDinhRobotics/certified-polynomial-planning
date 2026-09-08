# Certified polynomial trajectory planning

Minimal code, recorded experiment data and videos for checking the polynomial
trajectory-planning results. The manuscript and LaTeX sources are not included.

The implementation provides Bernstein polynomial models, a reusable factored
solver for the same SDP as the conic reference, rational primal/dual checks,
and physical-curve recovery. The quadrotor example separates the certified
reference trajectory from its measured MuJoCo tracking performance.

## Verify the delivered results

Use Python 3.12 and run from the repository root:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python verify.py --full --mutations --output results/verification.json
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

The delivered flight is one deterministic simulated inspection scenario. New
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

The solver uses the NumPy implementation. Optional native acceleration and its
build helpers are omitted. Verification does not require a C compiler, LaTeX,
Blender or a video renderer.
