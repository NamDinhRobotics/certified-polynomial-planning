"""Check the added curves and NumPy run without modifying delivered records."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / p) for p in ('experiments', 'src', 'verification')]


def run(full=False):
    if not __debug__:
        raise RuntimeError('Run without -O: exact checks require assertions.')
    import review_affine_green
    import certified_results
    from verify import check_manifest

    files = check_manifest()
    prior = sys.argv
    with tempfile.TemporaryDirectory(prefix='polynomial-replay-') as directory:
        record = Path(directory) / 'curves.json'
        shutil.copyfile(ROOT / 'artifacts/review_affine_green.json', record)
        try:
            sys.argv = ['review_affine_green', '--verify', '--out', str(record)]
            review_affine_green.main()
        finally:
            sys.argv = prior
        curves = json.loads(record.with_suffix('.verification.json').read_text())
    data = json.loads((ROOT / 'artifacts/review_numpy_solver.json').read_text())
    backend = 'NumPy; _C_POLISH=False enforced before construction'
    assert data['protocol']['sos_polish_backend'] == backend
    errors = certified_results.validate(data)
    if errors:
        raise AssertionError('\n'.join(errors))
    result = dict(passed=True, files_checked=files, full_exact_replay=full,
                  curves=curves, numpy_rows=len(data['rows']), backend=backend,
                  numpy_statistics=certified_results.compute(data))
    if full:
        result['numpy_exact'] = certified_results.recheck(data)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
        manifest = json.loads((ROOT / 'MANIFEST.json').read_text())
        if output == ROOT / 'MANIFEST.json' or any(output == (ROOT / p).resolve() for p in manifest):
            parser.error('The output must not overwrite a delivered file.')
    result = run(args.full)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
