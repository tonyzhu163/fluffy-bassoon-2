"""Audit original endpoints, reconcile the completed allocation, and plot results."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

import torch

assert sys.platform == 'linux' and not torch.cuda.is_available()
code = Path(os.environ['TASK_CODE'])
m = runpy.run_path(str(code / 'outputs/opendrawer-memory-controls-20261102-r1/CONTROL.py'))
out = m['OUT']
b = json.loads((out / 'binding.json').read_text())
result = json.loads((out / 'RESULT.json').read_text())
assert result['complete'] and result['binding'] == m['ref'](out / 'binding.json')
assert result['verified_fresh_episodes'] == 1100 and result['evaluation_endpoints'] == 2000
assert result['new_episodes_this_allocation'] == 48 and result['recovered_episodes'] == 1052

originals = 0
reused = 0
for family, model in b['models'].items():
    full = result['models'][family]['full']
    controls = result['models'][family]['controls']
    for cohort, index in m['TASKS']:
        for attempt in range(1, 6):
            row = m['read_ref'](model['episodes'][f'{cohort}-{index:03d}'][str(attempt)])
            assert row['complete'] and row['candidate'] == model['selected']['candidate']
            assert row['native_case'] == m['native_case'](cohort, index)
            assert row['motor_seeds'] == m['motor_seeds'](cohort, index, attempt)
            assert m['sha'](row['records']['path']) == row['records']['sha256']
            truth = m['read_ref'](row['truth'])
            assert truth['case'] == row['native_case']
            assert int(truth['simulator_final_success']) == row['success']
            pair = row['initial_pairing']
            marker = m['read_ref'](pair['canonical'])
            canonical = Path(pair['canonical']['path']).parent / 'initial.pt'
            assert m['sha'](canonical) == marker['snapshot_sha256']
            assert m['sha'](pair['snapshot']['path']) == pair['snapshot']['sha256']
            paired = m['PAIRING']['compare_native_pair'](
                torch.load(canonical, map_location='cpu', weights_only=False),
                torch.load(pair['snapshot']['path'], map_location='cpu', weights_only=False))
            assert paired['matched'] and paired == pair['report']
            records = torch.load(row['records']['path'], map_location='cpu', weights_only=False)
            assert len(records) == m['horizon'](cohort) // 50
            assert records[-1]['done'] and records[-1]['reward'] == row['success']
            assert row['memory_entry']['cache_length'] == (attempt - 1) * m['horizon'](cohort) // 50
            assert full[cohort]['attempt_cases'][str(attempt)][index] == row['success']
            originals += 1
            for arm in b['arms']:
                if m['reusable'](arm, cohort, attempt):
                    assert controls[arm][cohort][str(attempt)][index] == row['success']
                    reused += 1
    print('ORIGINAL_REFERENCE_AUDIT', family, originals, reused, flush=True)
assert originals == 1000 and reused == 900

workers = [json.loads((out / 'workers' / f'rank-{rank:02d}.json').read_text()) for rank in range(8)]
assert all(w['complete'] and w['weights_unchanged'] and w['binding'] == result['binding'] for w in workers)
assert [w['new_episodes'] for w in workers] == [0, 11, 23, 14, 0, 0, 0, 0]
assert sum(w['charged_actions'] for w in workers) == 36000
clock = json.loads((out / 'logs/clock.json').read_text())
assert clock['exit_code'] == 0 and clock['allocated_gpus'] == 1
assert clock['gpu_seconds'] == clock['finished_unix'] - clock['started_unix'] == 2544
scheduler = json.loads((out / 'SCHEDULER-CLOSEOUT.json').read_text())
assert scheduler['status'] == 'Succeeded' and scheduler['all_allocations_terminal']
assert scheduler['conservative_gpu_seconds'] == 2589
assert 62967 + scheduler['conservative_gpu_seconds'] == 65556 < 86400

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(10.5, 7), sharex=True, sharey=True)
styles = {'full': ('#2869a8', 'o', '-'), 'reset': ('#d97928', 's', '--'),
          'window64': ('#35865b', '^', ':')}
for row_index, family in enumerate(('ad', 'dpt')):
    for column, cohort in enumerate(('ID', 'OOD')):
        axis = axes[row_index, column]
        panel = result['models'][family]
        for arm, (color, marker, line) in styles.items():
            values = [panel['full'][cohort]['attempt_successes'][str(k)] if arm == 'full'
                else sum(panel['controls'][arm][cohort][str(k)]) for k in range(1, 6)]
            axis.plot(range(1, 6), [v * 2 for v in values], label=arm,
                      color=color, marker=marker, linestyle=line, linewidth=2)
        axis.set_title(f'{family.upper()} · {"OpenDrawer (ID)" if cohort == "ID" else "CloseFridge (OOD)"}')
        axis.set_xticks(range(1, 6))
        axis.set_ylim(30, 90)
        axis.grid(alpha=.18)
        if column == 0:
            axis.set_ylabel('Final simulator success (%)')
        if row_index == 1:
            axis.set_xlabel('Attempt / matched motor stream')
axes[0, 0].legend(loc='lower right', frameon=False)
fig.suptitle('Matched memory controls · 50 cases per cohort', fontsize=14)
fig.text(.5, .015, 'Identical cases and p1–p5 motor streams. Repeated streams are paired within case; exploratory cohorts.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .04, 1, .95))
fig.savefig(out / 'attempt-curves.png', dpi=180)
fig.savefig(out / 'attempt-curves.pdf')
plt.close(fig)

p5 = [r for r in result['comparisons'] if r['attempt'] == 5]
audit = dict(complete=True, checked_unix=time.time(), source=b['source'],
    analysis_commit=os.environ['TASK_ANALYSIS_PIN'], analysis_script_sha256=m['sha'](__file__),
    result=m['ref'](out / 'RESULT.json'), original_unique_episodes_verified=originals,
    reused_control_endpoints_verified=reused, collected_control_endpoints_verified=1100,
    total_control_endpoints_verified=2000, new_this_allocation=48, carried=1052,
    new_policy_controls=36000, all_workers_complete_and_weights_unchanged=True,
    measured_r4_gpu_seconds=2544, conservative_r4_gpu_seconds=2589,
    cumulative_conservative_gpu_seconds=65556, original_gpu_seconds_ceiling=86400,
    unused_gpu_seconds=20844, allocation_released=True,
    attempt5_paired_comparisons=p5,
    all_per_attempt_raw_p_min=min(r['exact_p'] for r in result['comparisons']),
    all_per_attempt_holm_p_min=min(r['holm_p_40_comparisons'] for r in result['comparisons']),
    memory_effect={f: model['memory_effect'] for f, model in result['models'].items()},
    figure=m['ref'](out / 'attempt-curves.png'))
m['write'](out / 'FINAL-AUDIT.json', audit)
print(json.dumps(audit), flush=True)
