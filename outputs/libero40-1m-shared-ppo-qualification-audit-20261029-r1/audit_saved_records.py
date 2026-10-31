"""Read saved records only; no policy construction, forward pass or tests."""
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch

root = Path('/mowenjie/fluffy-bassoon-2/outputs')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

sac = root / 'libero40-1m-sac-20261029-s00-train-r1' / 'experiment'
index = json.loads((sac / 'index.json').read_text())
counts = dict(episodes=0, chunks=0, primitive_steps=0, successes=0)
for receipt in index['episode_receipts']:
    path = sac / receipt['trace']
    assert sha(path) == receipt['trace_sha256']
    with gzip.open(path, 'rb') as stream:
        episode = pickle.load(stream)
    rows = episode['rows']
    counts['episodes'] += 1
    counts['chunks'] += len(rows)
    counts['primitive_steps'] += sum(row['primitive_steps'] for row in rows)
    counts['successes'] += int(episode['summary']['success'])
    for row in rows:
        assert row['public_features'].shape == (4104,)
        assert row['executed_noise'].shape == (32,)
        assert row['native_initial_noise'].shape == (10,32)
        assert len(row['executed_actions']) == row['primitive_steps'] == len(row['raw_rewards'])
        assert row['terminated'] == row['success']
        assert row['truncated'] == (row['done'] and not row['success'])
        assert isinstance(row['motor_observation'], dict) and isinstance(row['motor_next_observation'], dict)
assert counts['chunks'] == index['chunks'] and counts['primitive_steps'] == index['primitive_steps']
for publication in index['publications']:
    for dataset in publication['datasets'].values():
        assert sha(sac / dataset['path']) == dataset['sha256']
    assert sha(sac / publication['source_checkpoint']) == publication['source_checkpoint_sha256']
assert sha(sac / 'source-recovery.pkl') == index['latest_recovery_sha256']
save(sac.parent / 'independent-artifact-audit.json', dict(read_only_artifact_inspection=True,
    source_job='job-p5we8pk45d8i', source_commit='8ada0c0a59ea562716b251dc3cd7d71dc1ba3b18',
    counts=counts, all_trace_dataset_checkpoint_hashes_verified=True,
    raw_public_observations_and_next_observations=True,
    explicit_termination_truncation_and_durations=True,
    latest_recovery_sha256=index['latest_recovery_sha256'],
    total_physical_actions=index['total_action_budget_charge'], source_updates=index['source_updates']))

ppo = root / 'libero40-1m-shared-ppo-qual-20261029-r1' / 'experiment'
index = json.loads((ppo / 'index.json').read_text())
counts = dict(episodes=0, chunks=0, primitive_steps=0, successes=0)
tasks, behaviors = set(), set()
max_tanh_delta = 0.0
for sidecar in sorted((ppo / 'raw').glob('*.pkl.json')):
    record = json.loads(sidecar.read_text())
    path = sidecar.parent / record['path']
    assert sha(path) == record['sha256']
    with gzip.open(path, 'rb') as stream:
        episode = pickle.load(stream)
    rows, behavior = episode['rows'], episode['ppo']
    counts['episodes'] += 1
    counts['chunks'] += len(rows)
    counts['primitive_steps'] += sum(row['primitive_steps'] for row in rows)
    counts['successes'] += int(episode['summary']['success'])
    assert len(rows) == len(behavior['log_probs']) == len(behavior['raw_actions'])
    samples = behavior['raw_actions'].detach().cpu().numpy().reshape(len(rows),32)
    delta = float(np.abs(np.tanh(samples) - np.stack([row['executed_noise'] for row in rows])).max())
    max_tanh_delta = max(max_tanh_delta, delta)
    assert delta <= 1e-6, (path.name, delta)
    for row in rows:
        tasks.add(row['source_task_key'])
        assert row['instruction'] and row['public_features'].shape == (4104,)
        assert len(row['executed_actions']) == row['primitive_steps'] == len(row['raw_rewards'])
        assert row['terminated'] == row['success']
        assert row['truncated'] == (row['done'] and not row['success'])
        assert isinstance(row['motor_observation'], dict) and isinstance(row['motor_next_observation'], dict)
    behaviors.add(episode['metadata']['behavior_policy_sha256'])
behavior_paths = list((ppo / 'learner/traces').glob('behavior-*.pt'))
assert {sha(path) for path in behavior_paths} == behaviors
recovery = torch.load(ppo / 'learner/latest.pt', map_location='cpu', weights_only=False)
assert {'policy','optimizer','rng','runtime','environment_rng','iteration','counters'}.issubset(recovery)
assert 'motor_rng' in recovery['runtime']
save(ppo.parent / 'independent-artifact-audit.json', dict(read_only_artifact_inspection=True,
    source_job='job-b7u5ogwy6udu', source_commit='72bd91b988fa35acec324a6ea34cd2f427823120',
    counts=counts, actual_source_tasks=sorted(tasks), behavior_checkpoints=len(behaviors),
    all_trace_and_behavior_hashes_verified=True, actual_noise_matches_sampled_tanh=True, max_numpy_tanh_delta=max_tanh_delta, existing_collector_noise_tolerance=1e-6,
    raw_public_observations_and_next_observations=True,
    explicit_instructions_termination_truncation_and_durations=True,
    full_recovery_sha256=sha(ppo / 'learner/latest.pt'),
    full_recovery_keys=sorted(recovery), runtime_keys=sorted(recovery['runtime']),
    updates=index['summary']['updates'], physical_cost=index['physical_cost']))
print(json.dumps(dict(sac='passed', shared_ppo='passed')))
