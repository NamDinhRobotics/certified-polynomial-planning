# Reproducing the recorded results

[Back to README](../README.md)

This public repository is a compact export of recorded results, an independent
physical-reference checker, and telemetry visualization. It does not contain the
optimization or simulation runners used to produce the campaign.

## Environment

Use Python 3.12 or newer. The physical checker requires NumPy and may use SymPy
for exact polynomial root counting. The video renderer uses NumPy, Matplotlib,
and FFmpeg; its optional transition card uses Pillow, a Matplotlib dependency.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install numpy sympy matplotlib
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell.
Install FFmpeg separately and ensure `ffmpeg` is on `PATH` for video commands.
No MuJoCo installation is required for replay. This is an unpinned inspection
environment, not a reconstruction of the original benchmark environment.

## What the checker verifies

```sh
python code/verify.py --mutations
python code/verify.py --physical --mutations
```

The first command checks all 720 V3 records against the CSV tables, enforces
the expected populations and pair identities, checks telemetry file hashes,
recomputes summary statistics, and rejects three intentionally modified inputs.

The second command additionally recomputes the following from stored records:

- Rational endpoint and inter-segment continuity constraints and trajectory energy.
- Strict continuous-time reference clearance using rational polynomial positivity.
- Workspace containment through the Bernstein control-point hull.
- Time-scaled derivative hull bounds for speed, acceleration, and jerk.
- Reference positions, position errors, body clearances, and tracking acceptance
  from 1 ms telemetry, for records with an executed flight.

Failed planning records remain in the 720-record population. There is no
recovered curve or executed telemetry to replay for those failures.
Do not run exact replay with Python `-O`, which disables assertions.

`physical_ok` denotes acceptance of the physical reference constraints.
`complete_ok` is the archived outcome including the additional SDP objective
certificate gate. The export checks the consistency of the latter flag and
timing fields but **does not recompute SDP dual certificates**. A physically
accepted reference was eligible for simulation even if the complete gate failed:
the robot campaign has 32 recorded complete successes and 8 physical-only
successes among its 40 executed runs.

The checker does not rerun optimization, simulation dynamics, bootstrap
intervals, historical studies, or the separate 800-geometry endpoint-ray study.
It also does not reproduce the recorded timing measurements. Stored hash fields
bind records within the export; they do not by themselves reproduce omitted
source code or establish every claim in the associated research.

## Dataset map

| File | Contents |
| --- | --- |
| [`planning.json.gz`](../data/planning.json.gz) | 720 reduced V3 records, source identifier, acceptance contract, exact coefficients, and telemetry references |
| [`solver.csv`](../data/solver.csv) | 320 matched pairs: 20 seeds × 8 steps × 2 repetitions, with both arms retained |
| [`robot.csv`](../data/robot.csv) | 80 planning attempts: 20 scenes × 2 repetitions × 2 arms |
| [`tracking/`](../data/tracking/) | 15 unique NPZ telemetry files referenced by 40 executed runs; identical logs are stored once |
| [`historical.json.gz`](../data/historical.json.gz) | Earlier family, recovery, reference/mode, and same-SDP numerical datasets |

The JSON export preserves the recorded scientific values and acceptance
settings. Sharing a byte-identical telemetry file does not make repeated runs
additional distinct scenes.

To inspect either compressed dataset:

```python
import gzip
import json

with gzip.open("data/planning.json.gz", "rt") as f:
    planning = json.load(f)
print(planning["contract"])
print(len(planning["records"]))  # 720

with gzip.open("data/historical.json.gz", "rt") as f:
    historical = json.load(f)
print(list(historical))
```

## Interpreting the comparison

Energy ratios use exact recovered-trajectory energies: factor divided by conic.
The "conic more than 1% higher" count uses the reciprocal threshold, so it
means `conic_energy > 1.01 * factor_energy`.

Latency differences are computed **within each matched pair** as factor minus
conic, followed by the median over 320 pairs. The median of paired differences
need not equal the difference of the two arm medians. The positive paired
medians in this dataset do not support a factor-arm end-to-end speedup.
The factor arm includes 124 calls with conic fallback; it is not a comparison
of successful standalone factor solves against conic solves only.

All 40 executed robot runs passed the stored tracking gate: no contact, positive
sampled body and floor clearance, maximum position error below 0.12 m, and final
position error below 0.025 m. This describes 10 accepted scenes out of 20
attempted scenes. The other 40 planning attempts ended in root failure.

Exact reference feasibility applies to the encoded trajectory and obstacle
model. Tracking acceptance is sampled numerical evidence and does not certify
the robot's continuous-time tracking error or a hardware deployment.

## Videos and README media

The shipped [`v3_execution.mp4`](../video/v3_execution.mp4) is approximately
82 seconds at 1920 × 1080, 24 fps, H.264, without audio. The first 15 seconds
show V3; a 3-second card separates a 64-second earlier illustration outside V3.
The illustration's flight scene uses recovery of an affine guide without an SDP
solve; its displayed metrics must not be combined with the V3 campaign.

Rebuild the 15-second V3 visualization from recorded data into a new output path:

```sh
python code/video.py --output results/v3_replay.mp4
```

This renders `B_s13_r0_conic` (scene 41013, repetition 0). It does not execute
optimization or simulation dynamics. The displayed maximum error is 2.47 cm
and minimum body clearance is 15.17 cm for that run. The campaign minimum
clearance is 14.71 cm.

To rebuild the combined movie, provide the separate original silent illustration
clip. That source clip is not duplicated in this public repository:

```sh
python code/video.py --output results/combined_demo.mp4 \
  --append /path/to/polynomial_quadrotor_3d_silent.mp4
```

Avoid omitting `--output` when experimenting: `code/video.py` defaults to the
path of the shipped full movie and will overwrite it.

To regenerate README media from the shipped full movie:

```sh
python code/readme_media.py
```

This creates two GIF previews, a 15-second V3-only MP4, and a source/clip hash
manifest in [`assets/`](../assets/). It preserves the original video and all
scientific data. The V3 preview retains the first 15 seconds; the illustrative
preview retains seconds 49–57 of the combined movie, including its on-screen
"outside V3" label. The V3 GIF is 960 pixels wide at 10 fps; the illustrative GIF
is 640 pixels wide at 8 fps. Both retain the original playback speed.

The README uses linked GIFs for inline animation and ordinary MP4 links for
full playback. The repository-relative links work when the README, `assets/`,
and `video/` folders are committed together.
