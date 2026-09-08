"""Provenance and seed-level summaries for the September manuscript revision."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def provenance(extra=()):
    paths = list((ROOT / 'src').glob('*.py')) + list(extra)
    # Source exports intentionally have no .git directory (or Git binary).
    # Hashes remain available and must not depend on repository metadata.
    try:
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                             capture_output=True, text=True, check=False)
        git_head = git.stdout.strip() if git.returncode == 0 else None
    except OSError:
        git_head = None
    return dict(python=sys.version, platform=platform.platform(),
                machine=platform.machine(), processor=platform.processor(),
                git_head=git_head,
                working_tree_note='Source hashes identify the executed source snapshot; Git metadata is optional.',
                packages={p: importlib.metadata.version(p) for p in
                          ['numpy', 'scipy', 'cvxpy', 'clarabel', 'pytest']},
                source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in paths})


def clean(obj):
    if isinstance(obj, dict): return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray): return clean(obj.tolist())
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    return obj


def write_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(obj), indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def bootstrap_mean_ci(values):
    a = np.asarray(values, float)
    rng = np.random.default_rng(20260907)
    boot = np.mean(rng.choice(a, (10000, len(a)), replace=True), axis=1)
    return dict(mean=float(np.mean(a)), ci95=np.quantile(boot, [.025, .975]).tolist(),
                n_seeds=len(a), unit='seed')
