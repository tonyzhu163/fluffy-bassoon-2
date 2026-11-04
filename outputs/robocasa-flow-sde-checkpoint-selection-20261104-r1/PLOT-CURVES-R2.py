"""Plot a frozen native DEV curve snapshot, retaining the full training axis."""
import datetime
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source = Path(sys.argv[1])
data = json.loads(source.read_text())
labels = {
    'OpenDrawer': 'Open drawer',
    'CloseToasterOvenDoor': 'Close toaster door',
    'PickPlaceDrawerToCounter': 'Drawer → counter',
    'CoffeeSetupMug': 'Coffee: set up mug',
    'TurnOffStove': 'Turn off stove',
}
colors = ['#2563eb', '#d97706', '#7c3aed', '#059669', '#dc2626']
fig, axes = plt.subplots(2, 3, figsize=(13, 7.1), sharex=True, sharey=True)
for ax, row, color in zip(axes.flat, data['rows'], colors):
    scores = sorted(row['scores'], key=lambda s: s['step'])
    x = [s['step'] for s in scores]
    y = [100*s['successes']/s['cases'] for s in scores]
    lower, upper = [], []
    for score in scores:
        n = score['cases']
        p = score['successes']/n
        z = 1.959963984540054
        center = (p+z*z/(2*n))/(1+z*z/n)
        half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        lower.append(100*(center-half))
        upper.append(100*(center+half))
    if min(x) > 0:
        ax.axvspan(-3, min(x)-2, color='#f1f3f5', zorder=0)
        ax.text((min(x)-2)/2, 48, 'pending', ha='center', va='center',
                rotation=90, color='#8a919a', fontsize=9)
    ax.fill_between(x, lower, upper, color=color, alpha=0.12, linewidth=0)
    ax.plot(x, y, marker='o', markersize=4, linewidth=1.8, color=color)
    best = min(scores, key=lambda s: (-s['successes'], s['step']))
    ax.scatter([best['step']], [100*best['successes']/64], color=color,
               edgecolor='#202936', linewidth=0.7, marker='*', s=120, zorder=5)
    for score in scores:
        if score['step'] not in {min(x), max(x), best['step']}:
            continue
        offset = -19 if score['step'] == min(x) and score['step'] != best['step'] else 10
        align = 'left' if score['step'] < 5 else 'center'
        ax.annotate(f"{score['successes']}/64", (score['step'], 100*score['successes']/64),
                    xytext=(0, offset), textcoords='offset points', ha=align,
                    fontsize=9, color=color)
    zero = next((s for s in scores if s['step'] == 0), None)
    if zero:
        ax.axhline(100*zero['successes']/64, color='#4b5563', linewidth=0.8, linestyle='--')
    ax.set_title(labels[row['task']], loc='left', fontsize=12, fontweight='bold')
    ax.set_xlim(-3, 135)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100, 130])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.tick_params(labelbottom=True, labelleft=True, labelsize=9)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#d1d5db')
    ax.set_xlabel('PPO update (training order)', fontsize=9)
    ax.set_ylabel('Native success (%)', fontsize=9)
notes = axes.flat[5]
notes.axis('off')
notes.text(0, 0.95, 'PROVISIONAL VALIDATION CURVES', transform=notes.transAxes,
           fontsize=11.5, fontweight='bold', va='top')
notes.text(0, 0.80,
    '64 fixed fresh cases per point\nSame case banks and sampler RNG\nShading: pointwise 95% Wilson intervals\nStar: best completed DEV checkpoint so far\n\n'
    'Grey: earlier checkpoints still pending\nEvaluation proceeds 130 → 0\nPlot reads left → right in training order\n\n'
    'No selected-policy improvement claim yet.\nIndependent TEST128 follows the DEV lock.',
    transform=notes.transAxes, fontsize=10, va='top', color='#374151', linespacing=1.45)
stamp = datetime.datetime.fromisoformat(data['checked_shanghai'])
completed = sum(len(row['scores']) for row in data['rows'])
fig.suptitle('PPO checkpoint curves so far', x=0.055, ha='left', fontsize=18, fontweight='bold')
fig.text(0.055, 0.925,
    f"Snapshot: {stamp:%d %b %Y, %H:%M} Shanghai · {completed}/80 complete checkpoint panels · First success within 600 controls",
    fontsize=10, color='#4b5563')
fig.subplots_adjust(left=0.065, right=0.975, top=0.84, bottom=0.09, wspace=0.30, hspace=0.48)
target = source.with_suffix('.png')
fig.savefig(target, dpi=180, facecolor='white')
print(target.resolve())
