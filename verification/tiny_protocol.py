"""Matched margins and all-obstacle cuts in an instrumented TinySDP build.

All arms use the same additional stage/interstage sampled acceptance monitor.
KKT one/iterated share objective, input bounds, guide, margin and all obstacles;
the latter may update its normals. The original SDP arm solves a different
relaxation, so its comparison is a formulation comparison, not a solver race.
"""


from pathlib import Path


SCENES=['frozen_barrier','sweeping_barrier','vertical_gate','chord1','chord2','chord3']


BASE=dict(TINYSDP_REVISION=1)


COMMON=dict(TINYSDP_D4=1,TINYSDP_D4_ALLCUTS=1,TINYSDP_D4_ALPHA=1.,TINYSDP_D4_TRUST=1e6)


CONFIGS=[('sdp',BASE),('sdp_layer_off',dict(BASE,TINYSDP_ABLATE_PSD=1))]


for margin in (0.,1.,1.5):
    for arm,extra in [('kkt_one',dict(TINYSDP_D4_NEWTON=1,TINYSDP_D4_OUTER=1)),
                      ('kkt_iterated',dict(TINYSDP_D4_NEWTON=1,TINYSDP_D4_OUTER=20)),
                      ('admm_one',dict(TINYSDP_D4_OUTER=1,TINYSDP_D4_MAXITER=2000))]:
        CONFIGS.append((f'{arm}_margin{margin:g}',dict(BASE,**COMMON,**extra,TINYSDP_D4_INFLATE=margin)))
