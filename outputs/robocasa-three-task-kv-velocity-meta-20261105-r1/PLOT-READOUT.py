"""Plot only complete, independently audited AD readout panels."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


p = argparse.ArgumentParser(description=__doc__)
p.add_argument('run', type=Path)
p.add_argument('output', type=Path)
args = p.parse_args()
tasks = [('OpenDrawer', 'Open drawer · ID'),
         ('CloseToasterOvenDoor', 'Close toaster oven · ID'),
         ('CoffeeSetupMug', 'Set up mug · ID'),
         ('CloseFridge', 'Close fridge · OOD'),
         ('TurnOnMicrowave', 'Turn on microwave · OOD')]
interfaces = [('kv_bridge', 'KV', '#2563eb'), ('velocity_bridge', 'Velocity', '#ea580c')]
expected_candidates = dict(kv_bridge='f7085e4a6940546e32e46710314ac4fa4319e6179e69e23320e486d48345ad74',
                           velocity_bridge='04701478c2f3f850b8ce0329855f334732212d4186ae9502e4d97a08a3bab9ca')
panels, paired, sources = {}, {}, []
for interface, label, color in interfaces:
    panels[interface], paired[interface] = {}, {}
    count = 0
    for split in ('ID', 'OOD'):
        path = args.run / 'readouts' / interface / 'ad' / split / 'SUMMARY.json'
        value = json.loads(path.read_text())
        assert value['interface'] == interface and value['objective'] == 'ad'
        assert value['selected_candidate_sha256'] == expected_candidates[interface]
        assert value['completed'] and value['native_parameters_frozen']
        assert value['deployment_optimizer_updates'] == 0 and value['cases_per_task'] == 50
        count += value['episodes']
        panels[interface].update(value['curves'])
        paired[interface].update(value['paired_full_minus_reset'])
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert count == 3000
args.output.mkdir(parents=True, exist_ok=True)
x = np.arange(6)
fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), sharey=True, constrained_layout=True)
for ax, (task, title) in zip(axes, tasks):
    for interface, label, color in interfaces:
        for mode, style in [('full', '-'), ('episode', '--')]:
            row = panels[interface][task][mode]
            assert row['denominator'] == 50
            ax.plot(x, np.asarray(row['first_success']) * 2, style, marker='o',
                    markersize=3.5, linewidth=1.7, color=color,
                    label=label + (' full history' if mode == 'full' else ' reset each episode'))
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Prior episodes')
    ax.set_xticks(x)
    ax.set_ylim(-2, 102)
    ax.grid(alpha=.18)
axes[0].set_ylabel('Success during query episode (%)')
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=4, frameon=False)
fig.suptitle('AD frozen-expert steering: complete per-query success curves\n'
             '50 paired cases per task · six episodes per case · one fitted model per interface · zero SGD at evaluation', fontsize=13)
fig.savefig(args.output / 'query-success.png', dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), sharey=True, constrained_layout=True)
for ax, (task, title) in zip(axes, tasks):
    for index, (interface, label, color) in enumerate(interfaces):
        values = [paired[interface][task][str(k)] for k in range(6)]
        mean = np.asarray([v['delta_percentage_points'] for v in values])
        bounds = np.asarray([v['paired_case_bootstrap95_percentage_points'] for v in values])
        ax.errorbar(x + (index - .5) * .06, mean,
                    yerr=np.vstack([np.maximum(0, mean - bounds[:, 0]), np.maximum(0, bounds[:, 1] - mean)]),
                    color=color, marker='o', markersize=3.5, linewidth=1.3, capsize=2, label=label)
    ax.axhline(0, color='#64748b', linewidth=1)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Prior episodes')
    ax.set_xticks(x)
    ax.grid(alpha=.18)
axes[0].set_ylabel('Full history minus episode reset (percentage points)')
fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
fig.suptitle('AD steering: paired full-history versus reset contrast\n'
             'Descriptive marginal 95% case-bootstrap intervals · one fit per interface · no multiplicity-adjusted adaptation claim', fontsize=13)
fig.savefig(args.output / 'paired-history-contrast.png', dpi=180)
plt.close(fig)
(args.output / 'SOURCE.json').write_text(json.dumps(dict(episodes=6000, cases_per_task=50,
    prior_episodes=list(range(6)), plotted_outcome='Success during each individual query episode; cumulative ever-success is separate',
    summaries=sources, interface_curves=panels, paired_full_minus_reset=paired), indent=2) + '\n')
