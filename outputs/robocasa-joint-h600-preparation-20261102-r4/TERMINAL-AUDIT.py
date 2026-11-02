"""Audit this completed preparation on allocated Linux CPUs, with CUDA masked."""
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import digest, sha, validate_video
from fluffy_bassoon.meta_rl.rlinf_flora_ppo_runtime import read_ref
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import (
    PPO_TASKS, OOD_TASKS, TASKS, case_bank, joint_case_bank, private_action_seed,
    protocol, require_calibration_permission,
)
from fluffy_bassoon.meta_rl.robocasa_residual_parallel import episode_partitions, native_chunk_offsets

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-joint-h600-preparation-20261102-r4'
CODE = ROOT / 'worktrees/joint-h600-preparation-20261102-r4'
SOURCE = 'b7b9c8996c5124ebb56d06a2986824474b4b471c'
assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_available()
torch.set_num_threads(2)

def read(path):
    return json.loads(Path(path).read_text())

def ref(path):
    return dict(path=str(path), sha256=sha(path))

def fingerprint(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain'], text=True).strip()
assert not list((RUN / 'job/errors').glob('*.json'))
assert not list((RUN / 'calibration/errors').glob('*.json'))
completed = read(RUN / 'COMPLETED.json')
assert completed['completed'] and completed['calibration_completed'] and completed['common_before_completed']
common_final = read_ref(completed['common_final'])
checkpoint = torch.load(common_final['checkpoint']['path'], map_location='cpu', weights_only=False)
assert sha(common_final['checkpoint']['path']) == common_final['checkpoint']['sha256']
assert checkpoint['iteration'] == 0 and checkpoint['final'] and checkpoint['stage'] == 'boundary'
assert not checkpoint['optimizer']['state'] and checkpoint['pending_records'] is None and checkpoint['pending_episodes'] is None
assert checkpoint['contract']['source_commit'] == SOURCE and checkpoint['contract']['evaluation_only']
assert checkpoint['environment']['cursor'] == 250 and checkpoint['environment']['rollout_world_size'] == 4
assert len(checkpoint['environment']['worker_recovery']) == 4
assert checkpoint['rng'] and set(checkpoint['generators']) == {'action', 'update'}
assert all(isinstance(v, torch.Tensor) and v.numel() for v in checkpoint['generators'].values())
initial = read(RUN / 'common/initial-policy.json')
assert initial['completed'] and initial['optimizer_states'] == 0
assert fingerprint(checkpoint['policy']) == initial['policy_tensor_sha256']
assert checkpoint['motor'] == np.random.default_rng(initial['motor_seed']).bit_generator.state

banks, episode_rows, videos, worker_receipts = [], [], [], []
total_records = 0

def audit_bank(directory, phase, cases, monitoring, count, *, task=None, index=None):
    global total_records
    profile = read(directory / 'profile.json') if task else read(RUN / 'common/profile.json')
    assert profile['source_commit'] == SOURCE and profile['rollout_lanes_per_gpu'] == 4
    assert len(profile['rollout_devices']) == len({d['uuid'] for d in profile['rollout_devices']}) == 4
    assert profile['initializer'] == initial['initializer']
    seed = profile['evaluation_motor_seeds'][phase]
    if task: assert seed == private_action_seed('baseline', index * 10000)
    offsets = native_chunk_offsets(cases)
    rng = np.random.default_rng(seed)
    expected_seeds = [int(rng.integers(0, 2**63 - 1)) for _ in range(offsets[-1])]
    bank = directory / (phase + '-000') / 'videos/parallel-000000'
    episodes, records, all_seeds = [], [], []
    for rank, (first, stop) in enumerate(episode_partitions(count, 4)):
        worker = bank / f'rank-{rank:02d}'
        request = torch.load(worker / 'request.pt', map_location='cpu', weights_only=False)
        result = torch.load(worker / 'result.pt', map_location='cpu', weights_only=False)
        assert request['rank'] == rank and request['world'] == 4
        assert request['profile'] == profile and request['cases'] == cases[first:stop] == result['cases']
        assert request['deterministic'] and request['noise'] is None
        assert request['seeds'] == expected_seeds[offsets[first]:offsets[stop]]
        assert fingerprint(request['policy']) == initial['policy_tensor_sha256']
        assert len(result['episodes']) == stop - first
        assert len(result['records']) == len(request['seeds'])
        assert result['environment']['cursor'] == stop - first and result['rollout_lanes_per_gpu'] == 4
        assert [v['noise_seed'] for v in result['records']] == request['seeds']
        assert all(v['reward'] == 0 and np.count_nonzero(v['control']) == 0 for v in result['records'])
        worker_offsets = native_chunk_offsets(request['cases'])
        for k in range(stop - first):
            chunk = result['records'][worker_offsets[k]:worker_offsets[k + 1]]
            assert sum(bool(v['done']) for v in chunk) == 1 and chunk[-1]['done'] and chunk[0]['episode_start']
        episodes.extend(result['episodes']); records.extend(result['records']); all_seeds.extend(request['seeds'])
        worker_receipts.append(dict(bank=str(directory.relative_to(RUN)) + '/' + phase,
            rank=rank, request=ref(worker / 'request.pt'), result=ref(worker / 'result.pt'),
            cases_sha256=digest(request['cases']), seed_plan_sha256=digest(request['seeds'])))
    assert all_seeds == expected_seeds and len(episodes) == len(cases) == count
    assert len({e['id'] for e in episodes}) == count and len(records) == offsets[-1]
    if task:
        assert episodes == read(directory / 'episodes.json')
        assert any(v['zero_residual_verified'] for v in records)
    evaluation = read(directory / (phase + '-000') / 'evaluation.json')
    successes = 0
    for episode, case in zip(episodes, cases):
        truth = read(monitoring / (episode['id'] + '.json'))
        assert truth['case'] == case and truth['id'] == episode['id']
        horizon = TASKS[case['task']][0]
        validate_video(episode, protocol(case['task'], baseline_count=50))
        counts = truth['native_counts']; clock = episode['frame_clock']
        assert counts['controls'] - counts['reset_controls'] == horizon
        assert counts['reset_controls'] == 0 and counts['clock']['control_freq'] == clock['native_fps'] == 20.
        timestamps, value = [0.], 0.
        for _ in range(horizon):
            value += 1 / clock['native_fps']; timestamps.append(value)
        assert clock['timestamp_sha256'] == digest(timestamps)
        assert abs(counts['control_time'] - value) < 1e-8
        assert truth['video_refs_sha256'] == digest(episode['videos'])
        successes += int(truth['simulator_final_success'])
        for video in episode['videos']: videos.append(dict(**video, horizon=horizon, fps=20.))
        episode_rows.append(dict(bank=str(directory.relative_to(RUN)) + '/' + phase,
            id=episode['id'], case=case, horizon=horizon, success=truth['simulator_final_success'],
            video_refs_sha256=truth['video_refs_sha256'], frame_clock=clock))
    assert successes == evaluation['successes'] and evaluation['episodes'] == count
    total_records += len(records)
    banks.append(dict(bank=str(directory.relative_to(RUN)) + '/' + phase, phase=phase, task=task,
        episodes=count, successes=successes, policy_controls=offsets[-1] * 50,
        motor_seed=seed, cases_sha256=digest(cases), seed_plan_sha256=digest(expected_seeds),
        evaluation=ref(directory / (phase + '-000') / 'evaluation.json')))
    print('audited bank ' + banks[-1]['bank'], file=sys.stderr, flush=True)

for index, task in enumerate(PPO_TASKS):
    audit_bank(RUN / 'native' / task, 'baseline', case_bank(task, 'baseline', count=50),
        RUN / 'monitoring' / task / 'baseline-000', 50, task=task, index=index)
for phase in ('initial_DEV', 'before_ID', 'before_OOD'):
    cases = joint_case_bank(phase)
    audit_bank(RUN / 'common', phase, cases, RUN / 'common-monitoring' / (phase + '-000'), len(cases))
assert len(episode_rows) == 790 and len(videos) == 2370 and total_records == 9930
assert len({(e['bank'], e['id']) for e in episode_rows}) == 790
assert len({v['path'] for v in videos}) == 2370

def audit_video(video):
    assert sha(video['path']) == video['sha256'], video['path']
    data = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames,width,height,r_frame_rate', '-of', 'json', video['path']], text=True))
    stream = data['streams'][0]
    assert int(stream['nb_frames']) == video['horizon'] + 1 and stream['width'] == stream['height'] == 256
    assert stream['r_frame_rate'] == '20/1'
    return dict(path=video['path'], sha256=video['sha256'], frames=int(stream['nb_frames']))

with ThreadPoolExecutor(max_workers=4) as pool:
    video_results = []
    for i, result in enumerate(pool.map(audit_video, videos), 1):
        video_results.append(result)
        if i % 300 == 0: print(f'audited videos {i}/2370', file=sys.stderr, flush=True)

gates = read(RUN / 'CALIBRATION-GATES.json'); assert gates['all_passed'] and not gates['failed']
calibrations = []
for task in PPO_TASKS:
    gate_ref = gates['gates'][task]; gate = read_ref(gate_ref)
    require_calibration_permission(gate, protocol(task, baseline_count=50), calibration_sha256=gate_ref['sha256'])
    native = read_ref(gate['native_receipt']); native_episodes = read_ref(gate['native_episodes'])
    truth = read_ref(gate['native_monitor']); outputs = read_ref(gate['official_outputs'])
    assert len(native_episodes) == len(truth) == len(outputs) == 50
    assert [v['id'] for v in outputs] == [v['id'] for v in native_episodes] == [v['id'] for v in truth]
    positives, negatives = [], []
    for label, row in zip(outputs, truth):
        probabilities = label['probabilities']; expected = sum(int(k) * v for k, v in probabilities.items())
        assert abs(sum(probabilities.values()) - 1) < .0006
        assert abs(label['expected_score'] - expected) < .002
        (positives if row['simulator_final_success'] else negatives).append(label['expected_score'])
    auc = sum((p > n) + .5 * (p == n) for p in positives for n in negatives) / (len(positives) * len(negatives))
    assert abs(auc - gate['AUROC']) < 1e-12 and len(positives) == native['successes']
    shards = [read_ref(v) for v in gate['shards']]; assert len(shards) == 4
    calibrations.append(dict(task=task, count=50, successes=len(positives), AUROC=auc, gate=gate_ref, permission_passed=True))

action_calls, physical_actions, unresolved = 0, 0, 0
for path in sorted((RUN / 'actions').glob('rank-*/*.json')):
    journal = read(path); assert journal['rank_action_ceiling'] is None
    for row in journal['calls']:
        action_calls += 1; unresolved += not row['resolved']; physical_actions += row['executed_or_conservative']
        assert row['resolved'] and row['ended_unix'] >= row['started_unix']
        assert row['executed_or_conservative'] == (50 if row['phase'] == 'policy_controls' else 0)
assert physical_actions == 496500 and action_calls == 10720 and unresolved == 0

nodes = []
for rank in range(4):
    node = RUN / f'job/node-{rank:02d}'
    terminal = read(node / 'node-terminal.json'); clock = read(node / 'terminal-charge.json')
    engine = read(RUN / f'calibration/engines/rank-{rank:02d}.json')
    manifest = read(node / 'manifest.json')
    assert engine['source_commit'] == manifest['source_commit'] == SOURCE
    assert engine['gpu_uuid'] == manifest['devices'][0]['uuid'] and engine['model_load_count'] == 1
    assert engine['warmup_videos'] == 8 and engine['memory_snapshot']['free_mib'] >= 21973
    assert engine['code_commit'] == '129a2ada74c526ee4692f3d1b111bea6ca19d5b7'
    assert engine['model_revision'] == '031299e46e452fc0cedb4e0975eb9fa95fc37783'
    assert clock['status'] == 'finished' and clock['exit_code'] == 0 and not clock['slice_overrun']
    assert int((node / 'exit-code.txt').read_text()) == terminal['optimizer_updates'] == 0
    assert all(c == 0 or rank == 0 and i == 0 and c == 143 for i, c in enumerate(terminal['child_exit_codes']))
    nodes.append(dict(rank=rank, manifest=ref(node / 'manifest.json'), engine=engine, terminal=terminal, clock=clock))
charges = [n['clock']['charged_gpu_seconds'] for n in nodes]
assert charges == [4049, 4043, 4044, 4043]
print(json.dumps(dict(completed=True, audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    execution='Allocated Baige Linux CPU, CUDA masked; no model loading/forward or SGD', source_commit=SOURCE,
    calibration_episodes=250, calibration_grades=250, common_before_episodes=540, total_episodes=790,
    raw_records=total_records, policy_controls=physical_actions, optimizer_updates=0,
    bank_scoped_episode_identities=790, video_files=2370, video_headers_and_hashes_verified=True,
    video_audit_digest=digest(video_results), episode_audit_digest=digest(episode_rows),
    native_frame_clocks_verified=True, camera_order=['robot0_agentview_left','robot0_eye_in_hand','robot0_agentview_right'],
    case_and_motor_seed_plans_verified=True, deterministic_residual_action_noise=None, zero_residual_actions_verified=True,
    action_calls=action_calls, unresolved_intents=unresolved, banks=banks, calibrations=calibrations,
    nodes=nodes, worker_receipts=worker_receipts, common_final=completed['common_final'],
    initial_policy=ref(RUN / 'common/initial-policy.json'), policy_tensor_sha256=initial['policy_tensor_sha256'],
    initial_DEV=ref(RUN / 'common/initial_DEV-000/evaluation.json'), checkpoint_complete_recovery_verified=True,
    costs=dict(r4_gpu_seconds=sum(charges), prior_preparation_gpu_seconds=3268,
        combined_preparation_gpu_seconds=sum(charges)+3268, combined_preparation_gpu_hours=(sum(charges)+3268)/3600,
        authorized_gpu_seconds=26400, remaining_unspent_gpu_seconds=26400-sum(charges)-3268),
    limitations=['Frozen initializer preparation only; no learning comparison or SGD admission',
        'Four-worker panels with four native lanes/GPU; no16-lane or8-lane colocation qualification',
        'Independent full Flora weight rehash not repeated; qualified unchanged weight receipts and actual engine model revision retained']), indent=2))
