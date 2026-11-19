"""Plot native query success against the frequency of a second practice attempt."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw(ax, value, stopping, title):
    n = value['assignments']
    eligible = value['eligible']/n
    one = value['one_success']
    full = stopping['two']
    ax.plot([0, 100], 100*np.array([one, full])/n, color='#777777', ls=':',
            lw=2, label='Random across all trials')
    ax.plot([0, 100*eligible], 100*np.array([one, stopping['stop_on_positive']])/n,
            ls='--', color='#A65816', lw=2, label='Random after failure')
    for key, label, color, marker in [('second_reach','Second-probe reachability','#7055A5','s'),
                                     ('fitted','Fitted value of practice','#14694F','o')]:
        x, gain = np.asarray(value[key]['points']).T
        ax.plot(100*x*eligible, 100*(one+gain)/n, lw=1.7, marker=marker, ms=3.5,
                label=label, color=color, markerfacecolor='white')
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
    ax.grid(axis='y', color='#e3e3e3', lw=.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlim(-2, 102)
    ax.set_xlabel('Trials receiving a second practice attempt (%)')


if __name__ == '__main__':
    d = Path(sys.argv[1])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1), sharey=True)
    for ax, suffix, title in zip(axes, ('-mixed', ''), ('Two new goals and one familiar goal', 'Three goals unseen by the reader')):
        args = [json.loads((d/f'{name}{suffix}.json').read_text()) for name in ('value','stopping')]
        draw(ax, *args, title)
    axes[0].set_ylabel('Native query success (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.08, right=.98, top=.91, bottom=.29, wspace=.15)
    fig.savefig(d/'practice-cost.png', dpi=180, bbox_inches='tight', facecolor='white')
