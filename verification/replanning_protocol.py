"""Prespecified repeated-replanning and guide stress study; retain every outcome.

This follows the initial-plan study without overwriting its source or records.
Each cell is one deterministic execution, not a statistical population estimate.
"""
from pathlib import Path
import numpy as np
from tiny_protocol import SCENES, CONFIGS
ARMS = ['sdp', 'kkt_one_margin1.5', 'kkt_iterated_margin1.5']


def protocol():
    offsets = np.random.default_rng(20260907).uniform(-1., 1., (5, 2))
    guides = [dict(name='nominal', env={}),
              dict(name='straight', env={'TINYSDP_REVIEW_STRAIGHT': 1})]
    guides += [dict(name=f'offset{i}', env=dict(TINYSDP_REVIEW_GUIDE_DY=float(y),
                  TINYSDP_REVIEW_GUIDE_DZ=float(z))) for i, (y, z) in enumerate(offsets)]
    return dict(scenes=SCENES, arms=ARMS, guides=guides, runs_per_cell=1,
                common_env=dict(TINYSDP_3D_REPLAN_STRIDE=1, TINYSDP_REVIEW_PRECISION=1),
                seed=20260907, perturbation='one fixed transverse offset per guide, shared by all its interior waypoints',
                offset_distribution='PCG64 uniform [-1,1] metres, five dy/dz pairs',
                terminal_policy='inherited tracker; no further replanning inside 1 m capture radius',
                dense_subdivisions=64, dense_check_is_certificate=False)


def configuration(arm, guide):
    return dict(dict(CONFIGS)[arm], **protocol()['common_env'], **guide['env'])
