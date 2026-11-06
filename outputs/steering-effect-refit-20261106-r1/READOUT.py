"""Require the exact corrected-fit gate result before entering native readout."""
import argparse
from pathlib import Path
import runpy
import sys

import ADMISSION


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    args, _ = parser.parse_known_args()
    run_contract = Path(__file__).with_name('CONTRACT.json')
    proof = ADMISSION.require(args.fit, run_contract)
    if args.commit != proof['binding']['source_commit']:
        raise ValueError('Readout code must preserve the measured corrected-fit implementation')
    print('Candidate-bound corrected steering admission passed:', proof['binding']['candidate_sha256'], flush=True)
    # Keep the already audited robot case/draw/history protocol. Shared storage,
    # packing and the remaining GPU budget still require a separate launch check.
    original = args.code / 'outputs/robocasa-three-task-kv-velocity-meta-20261105-r1/READOUT.py'
    runpy.run_path(str(original), run_name='__main__')


if __name__ == '__main__':
    main()
