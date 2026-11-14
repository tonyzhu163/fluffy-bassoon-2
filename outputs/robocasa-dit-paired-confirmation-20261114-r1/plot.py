"""Render the completed, audited frozen-DIT paired confirmation on Linux CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert sys.platform == 'linux', 'Render with the allocated Linux CPU environment'
    report = json.loads(args.result.read_text())
    assert report['completed'] and report['verified_physical_episodes'] == 11520
    assert report['decision']['cases_per_task'] == 256
    assert all(row['cases_per_task'] == 256 for row in report['panels'].values())
    assert not args.output.exists()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    arms = ['DIT_full_history', 'DIT_episode_reset', 'original_SFT', 'fixed_PPO130']
    # Names are checked against the audited result, never silently reordered.
    assert all(set(row['success_percent']) == set(arms) for row in report['panels'].values())
    labels = ['DIT full history', 'DIT reset', 'SFT', 'PPO-130']
    colors = ['#225EA8', '#77AADD', '#888888', '#D58B32']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, (rates, effects) = plt.subplots(1, 2, figsize=(12.2, 5.1),
                                       gridspec_kw={'width_ratios': [1, 1.3]})
    x = np.arange(2)
    for i, (arm, label, color) in enumerate(zip(arms, labels, colors, strict=True)):
        values = [report['panels'][panel]['success_percent'][arm] for panel in ['ID', 'OOD']]
        bars = rates.bar(x + (i - 1.5) * .19, values, .18, color=color, label=label)
        rates.bar_label(bars, fmt='%.1f', padding=3, fontsize=8)
    rates.set(xticks=x, xticklabels=['ID: 3 tasks', 'OOD: 2 tasks'], ylim=(0, 100),
              ylabel='Episode-six success (%)', title='Performance on the same cases')
    rates.legend(frameon=False, loc='upper left', fontsize=8)
    rates.grid(axis='y', alpha=.18)
    rates.set_axisbelow(True)

    rows = []
    names = []
    for panel in ['ID', 'OOD']:
        for control, label in zip(arms[1:], ['History effect: full − reset',
                                           'Full − SFT', 'Full − PPO-130'], strict=True):
            row = report['panels'][panel]['contrasts'][control]
            rows.append((panel, control, row))
            names.append(f'{panel}  {label}')
    low = min(row['paired_bootstrap95_pp'][0] for _, _, row in rows)
    high = max(row['paired_bootstrap95_pp'][1] for _, _, row in rows)
    span = max(high - low, 5)
    effects.axvline(0, color='#666666', lw=1, ls='--')
    for y, (panel, control, row) in enumerate(rows):
        value = row['difference_pp']
        lo, hi = row['paired_bootstrap95_pp']
        color = '#225EA8' if control == arms[1] else '#666666'
        effects.errorbar(value, y, xerr=[[value - lo], [hi - value]], fmt='o',
                         color=color, capsize=4, markersize=5)
        effects.text(high + span * .08, y, f'{value:+.1f} [{lo:+.1f}, {hi:+.1f}]',
                     va='center', fontsize=8)
    effects.axhline(2.5, color='#DDDDDD', lw=.8)
    effects.set(yticks=np.arange(6), yticklabels=names, ylim=(5.6, -.6),
                xlim=(min(low, 0) - span * .12, high + span * .9),
                xlabel='Difference in percentage points (full DIT minus control)',
                title='Paired differences and 95% intervals')
    effects.grid(axis='x', alpha=.18)
    effects.set_axisbelow(True)
    fig.suptitle('Frozen DIT: independent paired confirmation', fontsize=15, y=.97)
    fig.text(.5, .08, '256 fresh cases/task · 1,280 paired cases · 11,520 physical episodes · fixed episode-six endpoint',
             ha='center', fontsize=9)
    fig.text(.5, .035, 'Task-stratified paired bootstrap; marginal intervals. One DIT fit. OOD is relative to meta-training tasks.',
             ha='center', fontsize=8, color='#555555')
    fig.subplots_adjust(left=.065, right=.98, top=.85, bottom=.23, wspace=.85)
    args.output.mkdir(parents=True)
    artifacts = {}
    for suffix in ['png', 'pdf']:
        path = args.output / f'frozen-dit-paired-confirmation.{suffix}'
        fig.savefig(path, dpi=180, facecolor='white')
        artifacts[path.name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    plt.close(fig)
    receipt = dict(result={'path': str(args.result), 'sha256': sha(args.result)},
                   script_sha256=sha(__file__), matplotlib_version=matplotlib.__version__,
                   execution='Linux CPU; no model computation', GPU_hours=0,
                   artifacts=artifacts, visual_review='pending')
    (args.output / 'FIGURE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
