# Polynomial trajectory planning

## Install

Use Python 3.12 or newer and FFmpeg for video generation.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install numpy sympy matplotlib
```

## Check the data

```sh
python code/verify.py --physical --mutations
```

This checks all 720 V3 planning records: 640 solver calls and 80 robot planning
calls, including the 40 failed robot calls. It checks paired energies and
latencies, robot scene counts, rational affine constraints, continuous reference
clearance, workspace and time-scaled derivative limits. It also recomputes reference positions,
tracking errors, body clearances and tracking acceptance from telemetry. Three deliberately
modified inputs must be rejected. This command does not rerun optimization,
SDP dual certificates, bootstrap intervals or simulation dynamics.

`data/planning.json.gz` is a reduced export of the recorded scientific fields
and acceptance settings. `solver.csv` and `robot.csv` contain the corresponding
comparison columns. Numerical values are unchanged. The 40 executed runs retain
links to full telemetry; byte-identical logs share one stored file.

`data/historical.json.gz` retains the earlier family, recovery, reference/mode
and same-SDP numerical datasets. The command above verifies V3; it does not
replay those historical datasets or the separate 800-geometry endpoint-ray
study. To read either compressed dataset:

```python
import gzip, json
with gzip.open("data/historical.json.gz", "rt") as f:
    data = json.load(f)
print(list(data))
```

## Play or rebuild the video

Open `video/v3_execution.mp4`. The 82-second silent movie contains a 15-second
V3 telemetry replay, a three-second transition, and the full 64-second earlier
3D illustration. The appended section is labelled as outside the V3 campaign;
its scenes and displayed metrics are separate from the V3 results.

To rebuild the V3 replay alone:

```sh
python code/video.py --output /tmp/v3_replay.mp4
```

The first 15 seconds replay the recorded conic-arm execution of scene 41013,
repetition 0, with projected SDP, recovered reference and recorded positions.
The displayed maximum error is 2.47 cm and minimum body clearance is 15.17 cm
for this run. These are scene-specific sampled metrics, not campaign extrema.
The moving sphere is the recorded 0.23-m body envelope. No new simulation is
performed when generating the video.

To rebuild the combined movie, supply the original silent illustration clip
as an additional input (the standalone source clip is not duplicated here):

```sh
python code/video.py --append /path/to/polynomial_quadrotor_3d_silent.mp4
```
