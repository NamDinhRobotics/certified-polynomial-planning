# Reproducing the recorded results

[Back to README](../README.md)

This public repository contains recorded results, independent objective and
physical-reference verification, and telemetry visualization. It does not contain
the optimization or simulation runners. Replaying an exported result does not
rerun the experiment that generated it.

Two public campaign IDs keep the new result separate from the historical data:
`v3_20260909` and `objective_repair_20260909`. The
[`campaign index`](../data/campaigns/index.json) assigns the historical ID without
rewriting the original V3 files.

## Environment

Use Python 3.12 or newer. Campaign verification requires only NumPy and SymPy.
The video renderer additionally uses Matplotlib and FFmpeg; its optional
transition card uses Pillow, a Matplotlib dependency.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install numpy sympy
```

Activate with `.venv\Scripts\Activate.ps1` in Windows PowerShell.
For video commands, install Matplotlib with `python -m pip install matplotlib`,
install FFmpeg separately, and ensure `ffmpeg` is on `PATH`.
No MuJoCo installation is required for replay. This is an unpinned inspection
environment, not a reconstruction of the original benchmark environment.

## What the checker verifies

### Objective-repair campaign

```sh
python code/verify_campaign.py --campaign objective_repair_20260909 --mutations
```

This command replays the exported **80 attempts across 20 scene IDs**, with
two arms and two repetitions per ID. It independently checks stored rational
root-SDP objective witnesses, recovered physical references, and recorded
tracking telemetry. The physical scope includes endpoint and continuity
constraints, energy, strict continuous-time obstacle clearance, workspace
containment, and time-scaled speed, acceleration, and jerk bounds. Tracking
checks recompute reference positions, errors, sampled body clearances, and
acceptance from telemetry.

The verified replay totals are:

| Check | Verified count |
| --- | ---: |
| Exact root-SDP objective certificates | 80 |
| Physical reference curves | 80 |
| Clearance polynomials | 1,200 |
| Recovery cells | 1,800 |
| Tracking samples | 824,080 |
| Rejected mutation controls | 13 |

The **824,080 tracking samples** count the logs referenced by all **80 runs**.
Identical telemetry is stored once: **25 unique NPZ logs contain 256,025
samples**. Repeated references do not increase the number of distinct scenes
or unique recorded samples.

Mutation controls include changes to the objective lower bound, multiplier,
point, scale provenance, and complete flag, as well as physical-reference
mutations. These controls test rejection of those altered inputs; they do not
establish rejection of every possible invalid input.

`root_optimality_certified` denotes an exact objective interval for the **root
SDP relaxation**. In this campaign, `complete_ok` requires an accepted physical
reference and that objective certificate. Tracking acceptance remains a
separate recorded check. All 80 attempts satisfy all three outcomes. An
objective witness can differ from the planning candidate used for recovery;
certification does not imply that the recovered path is globally optimal.

The maximum exact relative interval gap has the outward-rounded upper bound
**8.027405685153957 × 10⁻⁶**, within the unchanged **10⁻⁵** gate.
Rational reference energy is also checked against
the certified lower bound. Numerical solver status and
`root_numerically_accepted` remain separate: the original numerical root gate
still rejects **40 of 80 attempts**, and those rejection labels are preserved.

### Historical V3

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

For these historical V3 records, `physical_ok` denotes acceptance of the physical
reference constraints. `complete_ok` is the archived outcome including the
additional SDP objective certificate gate. The export checks the consistency of
the latter flag and timing fields, but **`code/verify.py` does not recompute SDP
dual certificates**. A physically accepted reference was eligible for simulation
even if the complete gate failed:
the robot campaign has 32 recorded complete successes and 8 physical-only
successes among its 40 executed runs.

Neither checker reruns optimization or simulation dynamics. The historical
checker also does not rerun bootstrap intervals, historical studies, or the
separate 800-geometry endpoint-ray study.
It also does not reproduce the recorded timing measurements. Stored hash fields
bind records within the export; they do not by themselves reproduce omitted
source code or establish every claim in the associated research.

## Dataset map

### New campaign

| File | Contents |
| --- | --- |
| [`code/verify_campaign.py`](../code/verify_campaign.py) | Campaign population, physical-reference, tracking, provenance, and mutation checks |
| [`code/objective.py`](../code/objective.py) | Exact root-SDP model reconstruction and rational objective interval verification |
| [`campaigns/index.json`](../data/campaigns/index.json) | Campaign IDs and locations, including the preserved historical V3 data |
| [`campaign.json`](../data/campaigns/objective_repair_20260909/campaign.json) | Objective-repair campaign metadata and acceptance configuration |
| [`records.json.gz`](../data/campaigns/objective_repair_20260909/records.json.gz) | 80 records with physical references, exact objective witnesses, and acceptance outcomes |
| [`provenance.json`](../data/campaigns/objective_repair_20260909/provenance.json) | Provenance for the exported campaign |
| [`telemetry/`](../data/campaigns/objective_repair_20260909/telemetry/) | NPZ tracking logs for the new campaign |

The public export contains verification code, necessary recorded data,
documentation, and existing videos. Diagnostic experiments and optimization
runners are outside its reproduction scope.

### Preserved historical data

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

### Campaign progression and objective repair

| Stage | Physical passes | Tracking passes | Root-SDP objective certificates |
| --- | ---: | ---: | ---: |
| Original V3 | 40 / 80 | 40 / 80 | 32 / 80 |
| Intermediate candidate-recovery debugging | 80 / 80 | 80 / 80 | 32 / 80 |
| Objective repair | 80 / 80 | 80 / 80 | 80 / 80 |

The intermediate stage is necessary context: objective repair preserves rational
reference coefficients and tracking metrics **relative to that stage**. The
original V3 had only 40 accepted physical references and executed flights.
The objective repair alone did not produce that earlier recovery improvement.
The intermediate comparison is recorded through hashes and equality commitments
in the provenance file. Public replay checks the current artifacts against
those commitments; the intermediate records are not included for a fresh
comparison.

The selected certificate paths are:

| Path | Attempts | Method |
| --- | ---: | --- |
| Original certificate | 32 | Retain the already valid interval |
| Current candidate dual repair | 20 | Repair the dual witness in normalized units, then map back to physical units |
| Independent same-SDP objective witness | 28 | Retry in a separate context with `energy_unit / 4` |

Normalized dual repair uses a margin of **10⁻⁹**. This changes witness
construction without relaxing the exact acceptance gate. The fallback uses one
fixed scale divisor of **4**, the same SDP geometry, boundary conditions,
basis, and physical cost, and solver tolerances **10⁻⁹ / 10⁻¹¹** with a
**600-iteration** budget. The fallback witness is used only for the objective
interval; it does not replace the physical reference or update planning warm
state. An exact check decides whether a witness is accepted.

The 28 fallback attempts add **56 conic solver calls**, yielding **206 conic
calls** including planning roots. Original V3 latency measurements do not
measure this pipeline, and no new end-to-end speedup is claimed.

This is **post-hoc debugging and validation on the same V3 population**. The
20 scene IDs are not a fresh confirmatory set and do not necessarily represent
20 independent geometries. The selected repair and scale were informed by
debugging on this population. The finite result does not show that the fallback
will succeed on arbitrary instances or other solver environments.

### Historical solver and robot comparisons

Energy ratios use exact recovered-trajectory energies: factor divided by conic.
The "conic more than 1% higher" count uses the reciprocal threshold, so it
means `conic_energy > 1.01 * factor_energy`.

Latency differences are computed **within each matched pair** as factor minus
conic, followed by the median over 320 pairs. The median of paired differences
need not equal the difference of the two arm medians. The positive paired
medians in this dataset do not support a factor-arm end-to-end speedup.
The factor arm includes 124 calls with conic fallback; it is not a comparison
of successful standalone factor solves against conic solves only.

In original V3, all 40 executed robot runs passed the stored tracking gate: no
contact, positive sampled body and floor clearance, maximum position error below
0.12 m, and final position error below 0.025 m. This describes 10 accepted scenes
out of 20 attempted scenes. The other 40 planning attempts ended in root failure.

Exact reference feasibility applies to the encoded trajectory and obstacle
model. Tracking acceptance is sampled numerical evidence and does not certify
the robot's continuous-time tracking error or a hardware deployment.

## Videos and README media

**All shipped media retains its original scope: historical V3 plus a separate
earlier illustration. No clip represents the objective-repair campaign.**

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
