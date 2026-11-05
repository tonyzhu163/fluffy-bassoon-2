"""Render the measured validation traces without smoothing or extrapolation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('snapshot', type=Path)
p.add_argument('output', type=Path)
args = p.parse_args()
d = json.loads(args.snapshot.read_text())
fig, axes = plt.subplots(2, 2, figsize=(12, 7.4), constrained_layout=True)
summary = []
for ax, method in zip(axes.flat, ('ad', 'dpt', 'dit', 'ic_iql')):
    for interface, label, color in [('kv_bridge', 'KV', '#2563eb'), ('velocity_bridge', 'Velocity', '#ea580c')]:
        model = d['models'][interface+'/'+method]; curve = model['curve']
        x = [row['update'] for row in curve]; y = [row['validation_flow_loss'] for row in curve]
        ax.plot(x, y, '-o', color=color, markersize=3.5, linewidth=1.7, label=label)
        best=[]
        for value in y:
            best.append(min(value, best[-1] if best else value))
        ax.plot(x, best, '--', color=color, linewidth=1, alpha=.5)
        update=model['status']['update']; start=y[0]; minimum=model['status']['best_loss']
        summary.append(dict(interface=label,method=method,update=update,
            sampled_training_targets=update*8,equivalent_epochs=update*8/524160,
            initial_logged_validation_loss=start,best_validation_loss=minimum,
            best_reduction_percent=(1-minimum/start)*100,
            plateau_stale_checks=model['status']['plateau'].get('stale_checks',0),
            converged=model['status']['converged']))
    ax.set_title(method.upper().replace('_','-'), loc='left', fontweight='bold')
    ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Validation flow loss')
    ax.grid(alpha=.18); ax.legend(frameon=False, loc='best')
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.set_xlim(0, 1550)
fig.suptitle('Frozen-expert steering: measured validation curves\n'
             'Fixed 128 held-out queries per check · lower loss is better · '+d['checked_shanghai'][:16].replace('T',' ')+' CST',
             fontsize=13)
args.output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(args.output,dpi=170)
args.output.with_suffix('.json').write_text(json.dumps(dict(training_queries=524160,
    validation_queries=74880,validation_sample_queries=128,updates_per_equivalent_epoch=65520,
    initial='first logged validation after update1; not a pre-training baseline',models=summary),indent=2)+'\n')
print(json.dumps(summary,indent=2))
