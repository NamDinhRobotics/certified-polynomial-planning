# Certified Polynomial Trajectory Planning

**From semidefinite relaxations to verified polynomial trajectories.**

A research artifact for inspecting trajectory recovery, checking continuous-time
reference constraints, and replaying recorded quadrotor flights.

✅ Exact rational checks of stored polynomial trajectories<br>
✅ 320 matched conic–factor comparisons with recorded energies and latencies<br>
✅ A 20-scene MuJoCo campaign, including unsuccessful planning attempts<br>
✅ Flight replay driven by recorded telemetry

[![V3 quadrotor telemetry replay: projected SDP curve, recovered reference, and recorded flight](assets/v3-replay.gif)](assets/v3-replay.mp4)

**[Watch the V3 replay · 15 s](assets/v3-replay.mp4)** ·
**[Watch the full demo · 82 s](video/v3_execution.mp4)** ·
[Quick start](#quick-start) · [Results](#recorded-results) ·
[Reproduction guide](docs/REPRODUCIBILITY.md)

*Scene 41013, conic arm, repetition 0. The moving sphere represents the recorded
0.23 m robot body envelope. This is a replay of a MuJoCo run.*

## Why certified polynomial planning?

A trajectory projected from an SDP relaxation can still intersect an obstacle.
The research pipeline recovers a physical polynomial curve, checks its reference
constraints, and then evaluates how a simulated robot tracks it:

**SDP relaxation → trajectory recovery → exact reference verification → tracking**

The reference checks use rational Bernstein coefficients to verify endpoint and
continuity constraints, strict obstacle clearance, workspace bounds, and
time-scaled speed, acceleration, and jerk limits. Tracking errors and body
clearances are measured separately from simulation telemetry.

This repository contains **verification and replay code with recorded data**.
Optimization runners and the MuJoCo controller/simulation runner are not included
in this public export. The commands below check the stored experiment and rebuild
its visualization.

## Quick start

Use **Python 3.12 or newer**. Run these commands from a terminal:

```sh
git clone https://github.com/NamDinhRobotics/certified-polynomial-planning.git
cd certified-polynomial-planning

python -m venv .venv
. .venv/bin/activate
python -m pip install numpy sympy matplotlib

# Check the V3 population, CSV consistency, statistics, and rejection controls.
python code/verify.py --mutations

# Also replay the exact physical-reference checks and tracking measurements.
python code/verify.py --physical --mutations
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell. Use `python`, without `-O`, for the exact checks.
FFmpeg is needed only to rebuild video or README media; MuJoCo is not needed
to inspect the recorded results.

## Recorded results

These values are recomputed from the **V3 data shipped in this repository**.
They describe a finite recorded campaign, not a universal performance guarantee.

### Matched conic–factor comparison

The solver dataset contains 20 seeds × 8 steps × 2 repetitions: **320 pairs,
640 solver calls**. Both arms returned physically accepted trajectories on all
320 paired inputs.

| Quantity | Recorded result |
| --- | ---: |
| Median recovered-energy ratio, factor / conic | 1.000 |
| Recovered-energy ratio, minimum–maximum | 0.901–1.215 |
| Pairs where factor energy is more than 1% higher | 56 / 320 |
| Pairs where conic energy is more than 1% higher | 14 / 320 |
| Median paired physical-return latency difference, factor − conic | +2.824 ms |
| Median paired complete-return latency difference, factor − conic | +3.777 ms |
| Factor-arm calls using conic fallback | 124 / 320 |

Positive paired latency differences mean the factor arm took longer. These data
do not establish an end-to-end speedup. Physical return and complete return are
distinct recorded endpoints; see the [definitions and verification scope](docs/REPRODUCIBILITY.md#what-the-checker-verifies).

### Quadrotor campaign

The robot dataset contains **20 scenes × 2 repetitions × 2 arms = 80 planning
attempts**. Counts below combine both arms and repetitions.

| Scene group | Scenes attempted | Planning attempts | Physically accepted / executed | Root failures |
| --- | ---: | ---: | ---: | ---: |
| Easy | 5 | 20 | 20 | 0 |
| Cluttered | 10 | 40 | 20 | 20 |
| On-axis recovery target | 5 | 20 | 0 | 20 |
| **Total** | **20** | **80** | **40** | **40** |

The **40 executed runs span 10 distinct scenes**; all passed the recorded
tracking acceptance rule. Across those executions, the maximum position error
was **2.47 cm** and minimum sampled body clearance was **14.71 cm**.
The remaining 40 planning attempts failed before execution.

For the specific run shown above, minimum sampled body clearance is **15.17 cm**.
Its scene-level value differs from the campaign minimum. A certified reference
and sampled tracking measurements do not constitute a continuous-time
closed-loop safety proof or a hardware flight result.

## Videos

### V3 telemetry replay

The [15-second replay](assets/v3-replay.mp4) shows the projected SDP curve,
the recovered reference, recorded positions, tracking error, and body clearance.
Rebuild it directly from the stored coefficients and telemetry:

```sh
python code/video.py --output results/v3_replay.mp4
```

### 3D flight illustration

[![Earlier 3D quadrotor illustration, outside the V3 benchmark campaign](assets/quadrotor-illustration.gif)](video/v3_execution.mp4)

*This earlier illustration uses separate scenes and metrics outside V3.
Its tracking scene recovers an affine guide without an SDP solve.*

The [full movie](video/v3_execution.mp4) is silent and contains:

| Time | Content |
| --- | --- |
| 00:00–00:15 | V3 telemetry replay |
| 00:15–00:18 | Transition identifying the separate illustration |
| 00:18–01:22 | Earlier 3D recovery and flight illustration |

To regenerate the GIF previews and short MP4 from the existing full movie:

```sh
python code/readme_media.py
```

See the [reproduction guide](docs/REPRODUCIBILITY.md#videos-and-readme-media)
for full-movie assembly and media provenance.

## Project layout

| Path | Contents |
| --- | --- |
| [`code/verify.py`](code/verify.py) | V3 data checks, physical replay, tracking checks, and mutation controls |
| [`code/exact.py`](code/exact.py) | Rational Bernstein arithmetic and polynomial positivity checks |
| [`code/video.py`](code/video.py) | Visualization of the recorded V3 flight |
| [`code/readme_media.py`](code/readme_media.py) | GIF previews and a short MP4 extracted from the existing movie |
| [`data/`](data/) | Recorded planning data, comparison CSVs, and tracking logs |
| [`video/`](video/) | Full demonstration movie |
| [`assets/`](assets/) | README media and source/clip hashes |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | Dataset map, checker scope, and reproduction details |

## Getting help

Open a [GitHub issue](https://github.com/NamDinhRobotics/certified-polynomial-planning/issues)
with the command, error output, Python version, and repository commit.
Maintained by [Nam Dinh](https://github.com/NamDinhRobotics).

## Referencing this artifact

Reference this repository URL together with the commit used in your experiments
(`git rev-parse HEAD`). The repository does not currently supply paper citation
metadata or a software license.
