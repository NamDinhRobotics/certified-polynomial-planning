"""Verify recorded numerical claims without manuscript or LaTeX dependencies."""
from pathlib import Path
import argparse
import copy
import contextlib
import hashlib
import io
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT/'verification'), str(ROOT/'src'), str(ROOT/'experiments')]


def check_manifest():
    path = ROOT/'MANIFEST.json'
    if not path.exists():
        raise FileNotFoundError('The release MANIFEST.json is required.')
    records = json.loads(path.read_text())
    for name, record in records.items():
        file = (ROOT/name).resolve()
        if not file.is_relative_to(ROOT) or not file.is_file():
            raise AssertionError('Missing or external file: '+name)
        data = file.read_bytes()
        if len(data) != record['bytes'] or hashlib.sha256(data).hexdigest() != record['sha256']:
            raise AssertionError('File hash mismatch: '+name)
    return len(records)


def quiet(call, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return call(*args)


def run(full=False, mutations=False):
    if not __debug__:
        raise RuntimeError('Run without -O: exact checks require assertions.')
    import revision_claims as legacy
    import certified_results as certified
    import recovery_results as recovery
    import development_recovery as development
    import tiny_comparison as tiny
    import replanning_results as replanning
    import legacy_checks
    import families
    import quadrotor

    begin = time.perf_counter()
    report = {'passed': False, 'full_exact_replay': full, 'files_checked': check_manifest()}
    data = legacy.load()
    errors = legacy.invariants(data)
    if errors:
        raise AssertionError('\n'.join(errors))
    sdp = certified.artifact()
    errors = certified.validate(sdp)
    if errors:
        raise AssertionError('\n'.join(errors))
    old, fresh, population = recovery.read()
    errors = recovery.validate(fresh, population)
    if errors:
        raise AssertionError('\n'.join(errors))

    report['same_sdp'] = certified.compute(sdp)
    report['recovery'] = recovery.compute(old, fresh)
    report['legacy_populations'] = {name: len(d.get('rows', [])) for name, d in data.items()}
    report['tinysdp_comparison'] = tiny.comparison(data)
    report['quadrotor'] = quiet(quadrotor.main)
    print('Population, numeric claims, reference and flight checks passed.', flush=True)

    if mutations:
        report['mutation_controls'] = {
            'legacy': quiet(legacy_checks.controls, data),
            'same_sdp': quiet(certified.controls, sdp),
            'recovery': quiet(recovery.controls),
            'quadrotor': report['quadrotor']['negative_controls_rejected'],
        }
        print('Claim and geometry mutation controls passed.', flush=True)
    if full:
        report['family_grid'] = families.recheck(data['revision_certificates'])
        report['exact_sdp'] = certified.recheck(sdp)
        report['development_recovery'] = quiet(development.main)
        report['fresh_recovery'] = recovery.recheck()
        report['legacy_clearance'] = legacy_checks.recheck(data)
        report['tiny_tracking'] = tiny.recheck_geometry(data)
        report['replanning_tracking'] = replanning.recheck(data)
        if mutations:
            detected=[]
            for name, check in [('revision_tinysdp', tiny.recheck_geometry),
                                ('revision_replanning', replanning.recheck)]:
                damaged=copy.deepcopy(data)
                if name=='revision_tinysdp':
                    damaged[name]['rows'][0]['repeats'][0]['tracking'][0]['x']+=1
                else:
                    damaged[name]['rows'][0]['result']['tracking'][2]['vx']+=.1
                try:
                    check(damaged)
                except AssertionError:
                    detected.append(name)
                else:
                    raise AssertionError('Corrupt tracker record was accepted: '+name)
            report['mutation_controls']['tracking']=detected
            report['mutation_controls']['development']=report['development_recovery']['mutation_controls']
        print('All stored SDP intervals, physical recoveries and tracker records passed.', flush=True)
    report['passed']=True
    report['seconds']=time.perf_counter()-begin
    report['scope']='Finite experiment and rational-certificate replay; not a formal proof of every theorem or a closed-loop hardware safety guarantee.'
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full', action='store_true', help='Replay every recorded rational interval and accepted physical trajectory.')
    parser.add_argument('--mutations', action='store_true', help='Require deliberate corruptions to be rejected.')
    parser.add_argument('--output', type=Path, help='Write the recomputed JSON report outside the delivered data.')
    args=parser.parse_args()
    if args.output:
        output=args.output.resolve()
        manifest=json.loads((ROOT/'MANIFEST.json').read_text())
        if output==ROOT/'MANIFEST.json' or any(output==(ROOT/p).resolve() for p in manifest):
            parser.error('The output must not overwrite a delivered file.')
    report=run(args.full, args.mutations)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed':report['passed'], 'full_exact_replay':report['full_exact_replay'],
                      'same_sdp_median_ratio':report['same_sdp']['median_ratio'],
                      'quadrotor_max_error_m':report['quadrotor']['independently_recomputed_metrics']['max_error'],
                      'quadrotor_min_body_clearance_m':report['quadrotor']['independently_recomputed_metrics']['min_body_clearance'],
                      'seconds':report['seconds']}, indent=2))


if __name__=='__main__':
    main()
