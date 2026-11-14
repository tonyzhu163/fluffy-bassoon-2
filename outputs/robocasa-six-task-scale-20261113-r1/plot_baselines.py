"""Render the already-verified six-task source baseline estimates."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).with_name('SIX-TASK-BASELINES.json')
    data = json.loads(source.read_text())
    assert data['all_six_baselines_verified']
    args.output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    fig, (table, plot) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw={'width_ratios': [1.2, 1.6]})
    fig.subplots_adjust(left=.035, right=.98, top=.76, bottom=.24, wspace=.07)
    for ax in (table, plot):
        ax.set_ylim(5.6, -.65)
    table.axis('off')
    for x, label, align in [(0, 'Task', 'left'), (.75, 'SFT', 'center'), (.95, 'PPO30', 'center')]:
        table.text(x, -.8, label, weight='bold', ha=align)
    for i, (task, record) in enumerate(data['results'].items()):
        r = record['paired_result']
        table.text(0, i, task + ('*' if task == 'OpenDrawer' else ''), va='center', fontsize=10)
        for x, key in [(.75, 'SFT_successes'), (.95, 'PPO30_successes')]:
            table.text(x, i, f"{r[key]}/64", ha='center', va='center')
        delta = r['delta_percentage_points']
        low, high = r['paired_bootstrap95_percentage_points']
        plot.errorbar(delta, i, xerr=[[delta-low], [high-delta]], fmt='o', color='#2563a6', capsize=4, markersize=6, linewidth=2)
        plot.text(39, i, f'{delta:+.1f}', ha='right', va='center', weight='bold', fontsize=11)
    plot.axvline(0, color='#6b7280', linestyle='--', linewidth=1)
    plot.set_xlim(-22, 41)
    plot.set_xticks([-20, -10, 0, 10, 20, 30])
    plot.set_yticks([])
    plot.set_xlabel('PPO30 − SFT (percentage points)', labelpad=9)
    plot.set_title('Paired difference and 95% interval', loc='left', fontsize=11, weight='bold', pad=17)
    plot.grid(axis='x', color='#e5e7eb', linewidth=.7)
    plot.set_axisbelow(True)
    for side in ['top', 'left', 'right']:
        plot.spines[side].set_visible(False)
    fig.text(.035, .94, 'RoboCasa source baselines: fixed PPO update 30', fontsize=19, weight='bold')
    fig.text(.035, .88, '64 paired cases per task · native success · 600 controls · one attempt per policy', color='#4b5563', fontsize=11)
    fig.text(.035, .115, '95% paired bootstrap intervals (10,000 draws); marginal, not simultaneous. One trained teacher per task.', fontsize=10, color='#4b5563')
    fig.text(.035, .065, '*Drawer uses a fresh panel; original SFT 33/64 remains separate. These source baselines do not measure meta-learning or history gains.', fontsize=9, color='#4b5563')
    for suffix in ['png', 'pdf']:
        fig.savefig(args.output / f'six-task-baselines.{suffix}', dpi=160, facecolor='white')
    (args.output / 'FIGURE.json').write_text(json.dumps({'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'plot_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'matplotlib': matplotlib.__version__, 'computation': 'Displays saved verified estimates; no new statistical calculation or episode selection.', 'GPU_hours': 0}, indent=2) + '\n')


if __name__ == '__main__':
    main()
