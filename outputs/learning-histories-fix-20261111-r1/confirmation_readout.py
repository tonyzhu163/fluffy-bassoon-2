"""Validate and summarize the frozen fresh-session confirmation (JSON only)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from readout import CURVES, aggregate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('confirmation', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = args.confirmation
    assert json.loads((root/'clock.json').read_text())['exit_code'] == 0
    assert json.loads((root/'gpu.json').read_text())['status'] == 'passed'
    assert (root/'results/complete').is_file()
    manifest = json.loads((root/'results/manifest.json').read_text())
    groups = ('selected-corrections', 'original-corrections',
              'attempts-both-lr', 'original-attempts')
    expected = {f'{group}-s{seed}' for group in groups for seed in range(3)}
    assert len(manifest) == 12 and {r['identity'] for r in manifest} == expected
    assert {p.stem for p in (root/'results').glob('*-s*.json')} == expected
    rows = {}
    hashes = {}
    for m in manifest:
        name = m['identity']
        path = root/'results'/f'{name}.json'
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        r = json.loads(path.read_text())
        a = r['args']
        for key, value in dict(seed=name[-1], K='6', steps='20000', batch='256',
                               eval_n='8192', eval_seed='90000', device='cuda',
                               resume=m['checkpoint']).items():
            assert a[key] == value, (name, key)
        assert len(m['sha256']) == 64
        int(m['sha256'], 16)
        assert not any(line.startswith('step ') for line in
                       (root/'results'/f'{name}.log').read_text().splitlines())
        assert r['rules'] == {'train': 99, 'heldout': 27}
        for split in ('train', 'heldout'):
            for curve in CURVES:
                values = r[split][curve]
                assert len(values) == 7 and all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        rows[name] = r
    for seed in range(3):
        for split in ('train', 'heldout'):
            assert rows[f'selected-corrections-s{seed}'][split]['bayes_on_learner_history'] == rows[f'original-corrections-s{seed}'][split]['bayes_on_learner_history']
    metrics = {g: aggregate([rows[f'{g}-s{s}'] for s in range(3)]) for g in groups}
    deltas = {}
    for new, old in ((groups[0], groups[1]), (groups[2], groups[3])):
        deltas[new] = {}
        for split in ('train', 'heldout'):
            values = [[a-b for a, b in zip(rows[f'{new}-s{s}'][split]['learner'],
                                           rows[f'{old}-s{s}'][split]['learner'])] for s in range(3)]
            deltas[new][split] = {'seeds': values, 'mean': [mean(v) for v in zip(*values)]}
    result = dict(panel='fresh confirmation: evaluation RNG base 90000', metrics=metrics,
                  improvement_over_original=deltas, input_sha256=hashes,
                  validation='12 complete CUDA evaluations; no training steps in logs; matched correction references; launcher verified checkpoint hashes unchanged')
    args.out.mkdir()
    (args.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# Fresh-session confirmation', '',
             'Three frozen fits per recipe; 8,192 sessions per split and fit. No additional updates.', '',
             '| Recipe | Train k=6 | Held-out k=0 | Held-out k=6 | Same-history reference k=6 | Held-out seed range |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for g, m in metrics.items():
        h = m['heldout']
        vals = [m['train']['learner']['mean'][-1], h['learner']['mean'][0],
                h['learner']['mean'][-1], h['bayes_on_learner_history']['mean'][-1]]
        seeds = [v[-1] for v in h['learner']['seeds']]
        lines.append('| '+g+' | '+' | '.join(f'{100*v:.2f}%' for v in vals)+
                     f' | {100*min(seeds):.2f}–{100*max(seeds):.2f}% |')
    lines += ['', 'Seed ranges describe three fits; they are not confidence intervals. '
              'Corrections have matched histories. Attempt histories depend on each learner’s own choices.', '']
    (args.out/'RESULT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
