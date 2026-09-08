"""Regenerate the quadrotor example while preserving delivered data."""
from pathlib import Path
import argparse
import importlib.util
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT), str(ROOT/'src'), str(ROOT/'experiments'), str(ROOT/'verification')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['plan','physics','check'])
    parser.add_argument('--out', type=Path, default=ROOT/'results/quadrotor')
    args=parser.parse_args()
    out=args.out.resolve()
    if out==ROOT/'demo' or out.is_relative_to(ROOT/'artifacts'):
        parser.error('Use a new output directory to retain the delivered data.')
    if args.mode=='plan':
        if out.exists():
            parser.error('Planning requires a new output directory.')
        name='demo_polynomial_quadrotor'
    elif args.mode=='physics':
        if not (out/'planning.json').is_file() or (out/'tracking.json').exists():
            parser.error('Run plan first; physics must not overwrite an existing flight.')
        name='demo_quadrotor_tracking'
    else:
        from verification import quadrotor
        quadrotor.HERE=out
        quadrotor.main()
        return
    path=ROOT/'experiments'/f'{name}.py'
    spec=importlib.util.spec_from_file_location(name, path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out.parent.mkdir(parents=True, exist_ok=True)
    module.OUT=out
    module.main()


if __name__=='__main__':
    main()
