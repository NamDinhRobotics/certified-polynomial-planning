"""Shared randomized-scenario generator for the streaming-MPC experiments.

Used by E-NEW-1 (multi-seed MPC statistics) and by the later experiments
E2 / E5.  A scenario is a dict of moving spherical obstacles with a
closed-form motion law, deterministic given the seed.

Motion law (matches the committed demos e_mpc_streaming.py / e_mpc_multi.py
in structure — periods 4 s in x, 3 s in y — with randomized amplitudes,
phases, base-center jitter and radii):

    c_j(t) = c0_j + [ Ax * cos(2*pi*t/4 + phix),
                      Ay * sin(2*pi*t/3 + phiy) ]

Guard (rejection sampling, count stored in the scenario dict):
  * over a dense t grid of the whole run, every obstacle stays more than
    ENDPOINT_MARGIN (0.05, the src/instances.py convention) clear of both
    endpoints (-2, 0) and (2, 0);
  * obstacles are pairwise disjoint at t = 0.

RNG draw order per obstacle (fixed — determinism is part of the contract):
  center0 jitter (n draws), radius, Ay, Ax, phix, phiy.
"""
import numpy as np

# Base geometry — same endpoints and base centers as the committed demos.
BASE_CENTERS = [(-0.6, 0.3), (0.5, -0.3), (0.0, 0.5)]
START_X, GOAL_X = -2.0, 2.0
ENDPOINT_MARGIN = 0.05          # src/instances.py _clear_of_endpoints convention
CENTER_JITTER = 0.15
RADIUS_RANGE = (0.18, 0.30)
AY_RANGE = (0.2, 0.4)
AX_RANGE = (0.05, 0.15)
PERIOD_X = 4.0                  # seconds
PERIOD_Y = 3.0                  # seconds
T_HORIZON = 5.0                 # seconds covered by the guard grid
N_GRID = 501                    # dense grid resolution for the guard


def _center_at(o, t):
    """Closed-form obstacle center at time t.  Returns an ndarray (n,)."""
    c = np.array(o["center0"], dtype=float)
    c[0] += o["Ax"] * np.cos(2.0 * np.pi * t / o["Tx"] + o["phix"])
    c[1] += o["Ay"] * np.sin(2.0 * np.pi * t / o["Ty"] + o["phiy"])
    return c


def _centers_on_grid(o, tgrid):
    """Obstacle centers over a whole t grid.  Returns (len(tgrid), n)."""
    C = np.tile(np.asarray(o["center0"], dtype=float), (len(tgrid), 1))
    C[:, 0] += o["Ax"] * np.cos(2.0 * np.pi * tgrid / o["Tx"] + o["phix"])
    C[:, 1] += o["Ay"] * np.sin(2.0 * np.pi * tgrid / o["Ty"] + o["phiy"])
    return C


def _valid(obstacles, start, goal, tgrid):
    """Guard: endpoint clearance over the whole grid + disjoint at t=0."""
    for o in obstacles:
        C = _centers_on_grid(o, tgrid)
        rm = o["radius"] + ENDPOINT_MARGIN
        if np.min(np.linalg.norm(C - start, axis=1)) <= rm:
            return False
        if np.min(np.linalg.norm(C - goal, axis=1)) <= rm:
            return False
    c0s = [_center_at(o, 0.0) for o in obstacles]
    for i in range(len(obstacles)):
        for j in range(i + 1, len(obstacles)):
            if (np.linalg.norm(c0s[i] - c0s[j])
                    <= obstacles[i]["radius"] + obstacles[j]["radius"]):
                return False
    return True


def make_scenario(seed, n=2, n_obs=3, t_horizon=T_HORIZON, n_grid=N_GRID):
    """Build one randomized moving-obstacle scenario.  Deterministic in seed.

    Returns a plain-JSON-serializable dict:
      seed, n, n_obs, bc0, bc1 (endpoint POINTS, flat lists),
      t_horizon, endpoint_margin, n_rejections,
      obstacles: list of {center0, radius, Ax, Ay, phix, phiy, Tx, Ty}.
    """
    rng = np.random.default_rng(seed)
    start = np.zeros(n)
    start[0] = START_X
    goal = np.zeros(n)
    goal[0] = GOAL_X
    tgrid = np.linspace(0.0, t_horizon, n_grid)

    n_rejections = 0
    while True:
        obstacles = []
        for j in range(n_obs):
            base = np.zeros(n)
            base[0], base[1] = BASE_CENTERS[j % len(BASE_CENTERS)]
            c0 = base + rng.uniform(-CENTER_JITTER, CENTER_JITTER, size=n)
            radius = float(rng.uniform(*RADIUS_RANGE))
            Ay = float(rng.uniform(*AY_RANGE))
            Ax = float(rng.uniform(*AX_RANGE))
            phix = float(rng.uniform(0.0, 2.0 * np.pi))
            phiy = float(rng.uniform(0.0, 2.0 * np.pi))
            obstacles.append(dict(
                center0=c0.tolist(), radius=radius,
                Ax=Ax, Ay=Ay, phix=phix, phiy=phiy,
                Tx=PERIOD_X, Ty=PERIOD_Y))
        if _valid(obstacles, start, goal, tgrid):
            break
        n_rejections += 1
        if n_rejections > 10000:
            raise RuntimeError(
                f"scenario seed={seed}: >10000 rejections — guard too strict")

    return dict(
        seed=int(seed), n=int(n), n_obs=int(n_obs),
        bc0=start.tolist(), bc1=goal.tolist(),
        t_horizon=float(t_horizon),
        endpoint_margin=ENDPOINT_MARGIN,
        n_rejections=int(n_rejections),
        obstacles=obstacles)


def obstacles_at(scenario, t):
    """Obstacle set at simulated time t: list of (center ndarray, radius)."""
    return [(_center_at(o, t), float(o["radius"]))
            for o in scenario["obstacles"]]


def scenario_config():
    """The generator's constants, for embedding in artifacts."""
    return dict(
        base_centers=[list(c) for c in BASE_CENTERS],
        endpoints=[[START_X, 0.0], [GOAL_X, 0.0]],
        endpoint_margin=ENDPOINT_MARGIN,
        center_jitter=CENTER_JITTER,
        radius_range=list(RADIUS_RANGE),
        Ay_range=list(AY_RANGE),
        Ax_range=list(AX_RANGE),
        period_x=PERIOD_X, period_y=PERIOD_Y,
        t_horizon=T_HORIZON, n_grid=N_GRID,
        guard=("endpoint clearance > 0.05 over dense grid of the full run; "
               "pairwise disjoint at t=0; whole-set rejection resampling"))
