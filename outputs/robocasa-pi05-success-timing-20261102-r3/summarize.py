"""Summarize retained timing evidence after the human selected600 controls."""
import hashlib
import json
import math
from pathlib import Path

root = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-pi05-success-timing-20261102-r3')


def ref(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


rows = [row for path in sorted(root.rglob('*-timing.json')) for row in json.loads(path.read_text())]
assert len(rows) == len({(r['case']['task'], r['case']['native_seed']) for r in rows})
for row in rows:
    steps = row['success_steps']
    assert steps == sorted(set(steps))
    assert row['first_success_step'] == (steps[0] if steps else None)
    assert all(type(s) is int and 1 <= s <= row['horizon'] for s in steps)
    assert row['simulator_final_success'] is (row['horizon'] in steps)
    assert row['native_counts']['controls'] - row['native_counts']['reset_controls'] == row['horizon']

tasks = []
for task in dict.fromkeys(r['case']['task'] for r in rows):
    cohort = [r for r in rows if r['case']['task'] == task]
    metrics = {}
    for label, key in [('ever_success', 'simulator_success_once'), ('final_success', 'simulator_final_success')]:
        times = sorted(r['first_success_step'] for r in cohort if r[key])
        metrics[label] = dict(successes=len(times), first_success_steps=times,
            empirical_p99=times[math.ceil(.99*len(times))-1] if times else None,
            maximum=max(times) if times else None)
    tasks.append(dict(task=task, episodes=len(cohort), metrics=metrics,
        success_at600=sum(600 in r['success_steps'] for r in cohort),
        final_success_lost_at600=sum(r['simulator_final_success'] and 600 not in r['success_steps'] for r in cohort)))

journals = [json.loads(p.read_text()) for p in (root/'actions').rglob('*.json')]
calls = [call for journal in journals for call in journal['calls']]
clock = json.loads((root/'job/node-00/clock/worker-time.json').read_text())
receipt = root/'job/node-00/terminal-charge.json'
cost = json.loads(receipt.read_text()) if receipt.exists() else None
external_path = root/'SCHEDULER-CLOSURE.json'
external = json.loads(external_path.read_text()) if external_path.exists() else None
summary = dict(source_commit='c5d271b40ed911b1b08136f2e2d40202b78ebb33',
    job_id='job-kpdi47srqc2m', planned_panel_complete=False, planned_episodes=250,
    completed_episodes=len(rows), human_selected_cutoff=600, action_chunk=50,
    cutoff_chunks=12, population_p99_established=False,
    definition='Nearest-rank empirical p99 conditional on ever success, separately on final success; failures not imputed.',
    limitation='Partial fixed-bank prefix; with under100 successes, empirical p99 equals the observed maximum. Stove has transient successes but no terminal750 successes.',
    tasks=tasks, maximum_observed_first_success=max(r['first_success_step'] for r in rows if r['simulator_success_once']),
    policy_controls=sum(r['horizon'] for r in rows),
    charged_actions=sum(c.get('executed_or_conservative', c['requested']) for c in calls),
    unresolved_actions=sum(not c['resolved'] for c in calls), optimizer_updates=0,
    closed_workers=[ref(p) for p in sorted(root.rglob('result.json'))],
    probe=ref(root/'job/node-00/gpu-probe.json'), clock=ref(root/'job/node-00/clock/worker-time.json'),
    terminal_charge=cost, external_closure=external,
    conservative_r3_GPU_seconds=(cost['charged_gpu_seconds'] if cost else
        external['charged_GPU_seconds_upper_bound'] if external else 7200),
    prior_r1_GPU_seconds=726, r2_GPU_seconds=0,
    clock_action_ceiling_metadata=clock['contract']['robot_actions_global_ceiling'],
    enforced_action_ceiling=212000,
    clock_metadata_note='The launcher clock field236000 resulted from textual clock replacement; per-rank escrow and driver enforce212000. Observed actions remain below both.',
    episodes=[dict(id=r['id'], task=r['case']['task'], native_seed=r['case']['native_seed'],
        horizon=r['horizon'], first_success_step=r['first_success_step'],
        ever_success=r['simulator_success_once'], final_success=r['simulator_final_success'],
        success_at600=600 in r['success_steps']) for r in rows])
print(json.dumps(summary, indent=2))
