"""Post-hoc cadence diagnostic after the every-step stress study.

Keep the same binary, six scenes, nominal guides, and 26/28-step execution
budgets. Change only the common replanning stride to 3 and 5. This is not
an independent confirmatory robustness experiment.
"""
from pathlib import Path
from replanning_protocol import ARMS, SCENES


def protocol():
    return dict(scenes=SCENES,arms=ARMS,strides=[3,5],guide='nominal',runs_per_cell=1,
                post_hoc=True,execution_budget='unchanged: gate 26, other scenes 28')
