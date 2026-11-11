"""Evaluate the selected recipe and matched baselines on untouched sessions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trainer-code', type=Path, required=True)
    ap.add_argument('--grid', type=Path, required=True)
    ap.add_argument('--original-grid', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    selected = json.loads(args.selection.read_text())['selected_corrections_recipe']
    assert selected in ('both', 'base-lr', 'both-lr', 'wide')
    args.out.mkdir()
    manifests = []
    groups = [('selected-corrections', selected, 'corrections', False),
              ('original-corrections', 'corrections', 'corrections', True),
              ('attempts-both-lr', 'attempts-both-lr', 'attempts', False),
              ('original-attempts', 'attempts', 'attempts', True)]
    for label, recipe, cond, old in groups:
        for seed in range(3):
            base = args.original_grid if old else args.grid
            source = base / f'{recipe}-s{seed}'
            stem = cond if old else 'result'
            saved = json.loads((source / f'{stem}.json').read_text())['args']
            ckpt = source / f'{stem}.pt'
            assert saved['cond'] == cond and saved['seed'] == str(seed)
            identity = f'{label}-s{seed}'
            result = args.out / f'{identity}.json'
            command = [sys.executable, '-Bu', str(args.trainer_code/'exploration/learning-histories/train.py'),
                       '--cond', cond, '--resume', str(ckpt), '--out', str(result),
                       '--device', 'cuda', '--eval-n', '8192', '--eval-batch', '256', '--eval-seed', '90000']
            for key, default in dict(K='6', steps='20000', batch='256', lr='0.0003', seed=str(seed),
                                     fourier='0', d='128', layers='4', heads='4').items():
                command += ['--'+key, saved.get(key, default)]
            if saved.get('dense', 'False') == 'True':
                command += ['--dense']
            before = digest(ckpt)
            with (args.out / f'{identity}.log').open('w') as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            assert digest(ckpt) == before
            row = dict(identity=identity, checkpoint=str(ckpt), sha256=before, command=command)
            manifests.append(row)
            (args.out/'manifest.json').write_text(json.dumps(manifests, indent=2)+'\n')
            print('CONFIRMED', identity, flush=True)
    (args.out/'complete').touch()


if __name__ == '__main__':
    main()
