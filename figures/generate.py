"""Regenerate all ten retained scientific figures; no LaTeX or Blender needed."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'figures'),str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT/'verification')]
import revision_claims,make_revision_results,make_tiny_figures,replanning_results
import certified_results,make_recovery_addendum,recovery_results,quadrotor_results
out=ROOT/'results/figures';out.mkdir(parents=True,exist_ok=True)
modules=[make_revision_results,make_tiny_figures,replanning_results,certified_results,make_recovery_addendum,recovery_results,quadrotor_results]
for m in modules:m.HERE=out
quadrotor_results.DEMO=ROOT/'demo'
data=revision_claims.load()
make_revision_results.plot(data);make_tiny_figures.plot_all(data);replanning_results.plot(data)
certified_results.plot();make_recovery_addendum.plot();recovery_results.plot();quadrotor_results.plot()
print('All ten figures regenerated in results/figures; original datasets retained.')
