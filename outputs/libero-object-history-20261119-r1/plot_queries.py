"""Plot measured placements under each prompt, preserving the paired scene design."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def figure(reports, dest):
    panels = [
        ('Soup scene', ['Soup', 'Milk']),
        ('Salad dressing scene', ['Dressing', 'Ketchup']),
        ('Ketchup scene', ['Dressing', 'Ketchup']),
    ]
    cmap = LinearSegmentedColormap.from_list('success', ['#f4f4ef', '#08594e'])
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.25))
    for ax, report, (name, labels) in zip(axes, reports, panels):
        outcomes = json.loads(report.read_text())['object']['query_matrix']
        data = np.array(list(outcomes.values()))
        assert data.shape == (6, 2, 2)
        counts = data.sum(axis=0)
        ax.imshow(counts, vmin=0, vmax=6, cmap=cmap)
        ax.set_title(name, fontsize=12, pad=12)
        ax.set_xticks([0, 1], labels=labels)
        ax.set_yticks([0, 1], labels=labels)
        ax.tick_params(length=0, labelsize=10)
        for i, j in np.ndindex((2, 2)):
            value = counts[i, j]
            ax.text(j, i, f'{value}/6', ha='center', va='center', fontsize=16,
                    color='white' if value >= 4 else '#17302d')
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel('Prompt given to the motor', fontsize=10)
    fig.text(.53, .07, 'Object placed in basket (native predicate)', ha='center', fontsize=11)
    fig.subplots_adjust(top=.84, bottom=.22, left=.12, right=.99, wspace=.70)
    fig.savefig(dest, dpi=200, bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


if __name__ == '__main__':
    a, b, c, dest = map(Path, sys.argv[1:])
    figure([a, b, c], dest)
