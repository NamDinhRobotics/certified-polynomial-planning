# Q1 revision: code and recorded evidence

This archive contains code, raw data and verification reports. Manuscript,
LaTeX sources, author response and private contextual prose are not included.
The primary original-SDP comparison is artifacts/review_numpy_solver.json;
revision_certified_solver.json is a retained backend-ambiguous historical run.

Run from this extracted root with Python 3.12 and requirements.txt installed:

```
python experiments/q1_verify.py artifacts/q1_factorial_final.json --out results/factorial_audit.json
python experiments/q1_verify.py artifacts/q1_integrated.json --out results/integrated_audit.json
python experiments/q1_verify.py artifacts/q1_budget.json --out results/budget_audit.json
python experiments/q1_verify.py artifacts/q1_families.json --out results/families_audit.json
python -m pytest -q tests
python verify.py --full --mutations --output results/legacy_audit.json
python verify_peer_review.py --full
python figures/generate.py
```

Command-to-manuscript mapping: q1_factorial_final controls reference/mode,
all-axis recovery, workspace and iterated-cut QP; q1_integrated measures the
sequential physical then full-SDP-objective emissions; q1_budget measures
factor-only warm-use/iteration ablation; q1_families tests four exact-certified
k=1,l=0 tuples; q1_repair reconstructs six-candidate repair sensitivity.
The figure generator writes results/figures, including all historical figures.
Input archives are immutable. Verification writes a separately named output;
never overwrite a frozen data/protocol file. Historical figures/data are retained.

To repeat measurements, set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
MKL_NUM_THREADS=1, keep other computation idle, and choose a fresh --out
filename. q1_factorial.py and q1_integrated.py reject existing output/protocol
files. q1_integrated.py defaults to all20seeds/all50steps; --ablation --seeds4
--steps20 (arguments separated by spaces) reproduces the declared budget scope.
Family prepare/run use dedicated fresh protocol/witness/output paths.

Frozen protocols identify the actually executed workspace snapshot. Their
provenance can include hashes of unrelated workspace modules not needed by
this minimal archive. The manifest identifies the delivered dependency closure.
The post-run verifier is separately versioned and checks all successful physical
payloads plus input population/arm/witness/timing consistency. Physical polynomial,
affine and energy arithmetic are separately implemented; mode reconstruction
and exact SDP bounds reuse project helpers independently of numerical optimization.
The raw normalized_gap_upper float field is a rounded diagnostic view; exact
J,L and audit-exported fractions/outward values are authoritative.

A predicate exhaustion returns UNKNOWN_EXACT_CHECK; no answer is inferred to
mean original infeasibility. Integrated time includes two serializations. Factor
warm-off does not disable conic graph or conic warm reuse. Imported historical
TinySDP logs are replayable, but their full build/patch is not shipped, so those
timings are not promised to regenerate. Quadrotor evidence is a simulation;
strict reference certificates do not establish continuous tracking safety.
