"""Read the completed recipe screen; no model computation."""
import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev

RECIPES = ('both', 'base-lr', 'both-lr', 'wide', 'attempts-both-lr')
CURVES = ('learner', 'bayes_on_learner_history', 'oracle_own_attempts')


def load_runs(grid, name, cond, old=False):
    runs = []
    for seed in range(3):
        folder = grid / f'{name}-s{seed}'
        assert json.loads((folder / 'clock.json').read_text())['exit_code'] == 0
        r = json.loads((folder / (f'{cond}.json' if old else 'result.json')).read_text())
        for key, value in dict(cond=cond, seed=str(seed), K='6', steps='20000',
                               batch='256', eval_n='8192').items():
            assert r['args'][key] == value, (name, seed, key)
        assert r['args'].get('eval_seed', '10000') == '10000'
        for split in ('train', 'heldout'):
            for key in CURVES:
                v = r[split][key]
                assert len(v) == 7 and all(math.isfinite(x) and 0 <= x <= 1 for x in v)
        runs.append(r)
    return runs


def aggregate(runs):
    result = {}
    for split in ('train', 'heldout'):
        result[split] = {}
        for key in CURVES:
            values = [r[split][key] for r in runs]
            result[split][key] = dict(mean=[mean(v) for v in zip(*values)],
                                     std=[stdev(v) for v in zip(*values)], seeds=values)
        gaps = [[b-a for a, b in zip(r[split]['learner'], r[split]['bayes_on_learner_history'])]
                for r in runs]
        result[split]['reference_gap_seeds'] = gaps
        result[split]['within_5pp_all_k_each_seed'] = [max(v) <= .05 for v in gaps]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('grid', type=Path)
    ap.add_argument('original_grid', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir()  # New analysis identity; never overwrite a prior readout.
    runs = {f'original-{cond}': load_runs(args.original_grid, cond, cond, old=True)
            for cond in ('corrections', 'attempts')}
    for recipe in RECIPES:
        cond = 'attempts' if recipe.startswith('attempts-') else 'corrections'
        runs[recipe] = load_runs(args.grid, recipe, cond)
        if cond == 'corrections':
            for r, baseline in zip(runs[recipe], runs['original-corrections']):
                for split in ('train', 'heldout'):
                    assert r[split]['bayes_on_learner_history'] == baseline[split]['bayes_on_learner_history']
    metrics = {name: aggregate(rr) for name, rr in runs.items()}
    selected = max(RECIPES[:4], key=lambda name: metrics[name]['heldout']['learner']['mean'][-1])
    summary = dict(panel='development: evaluation RNG base 10000', selected_corrections_recipe=selected,
                   selection_rule='largest three-seed mean held-out k=6 accuracy; recipe order breaks ties',
                   fresh_confirmation='pending; evaluation RNG base 90000', metrics=metrics)
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Learning-history recipe screen', '',
             'Development panel only. Three seeds per recipe, 20,000 updates, batch 256; '
             '8,192 evaluation sessions per split and seed. Fresh confirmation is pending.', '',
             '| Recipe | Train k=6 | Held-out k=0 | Held-out k=6 | Reference on same histories k=6 | Seed range k=6 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, m in metrics.items():
        h = m['heldout']
        values = [m['train']['learner']['mean'][-1], h['learner']['mean'][0],
                  h['learner']['mean'][-1], h['bayes_on_learner_history']['mean'][-1]]
        seeds = [v[-1] for v in h['learner']['seeds']]
        lines.append('| '+name+' | '+' | '.join(f'{100*v:.2f}%' for v in values)
                     + f' | {100*min(seeds):.2f}–{100*max(seeds):.2f}% |')
    lines += ['', f'Selected corrections recipe for fresh confirmation: `{selected}`.', '',
              'The full-prior enumeration reader is a reference, not the split-specific optimum. '
              'Corrections use identical histories across recipes. Attempts use each learner\'s own choices, '
              'so their histories and enumeration reference can differ.', '',
              'Dense changes attention and the training objective together; wide changes size and learning rate. '
              'This screen cannot isolate a supervision-density or capacity effect. Seed ranges describe these '
              'three fits and are not confidence intervals. No longer-duration fit was tested.', '']
    (args.out / 'RESULT.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
