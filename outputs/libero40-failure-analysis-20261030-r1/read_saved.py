"""Bounded saved-array analysis only; no source/model imports or inference."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
DSRL = ROOT / 'libero40-target-online-dsrl-six-20261030-r1/experiment'
EXPORT = Path('/tmp/libero40-failure-analysis-20261030-r1')
EXPORT.mkdir(exist_ok=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, digest):
    if sha(path) != digest:
        raise ValueError('Saved trajectory hash differs: ' + str(path))
    with gzip.open(path, 'rb') as stream:
        return pickle.load(stream)


def numbers(values):
    a = np.asarray(values, np.float64)
    return dict(shape=list(a.shape), mean=float(a.mean()), std=float(a.std()), rms=float(np.sqrt(np.mean(a*a))),
        mean_abs=float(np.abs(a).mean()), maximum_abs=float(np.abs(a).max()),
        fraction_abs_ge_099=float((np.abs(a) >= .99).mean()), fraction_abs_gt_1=float((np.abs(a) > 1).mean()))


def stats(rows):
    result = dict(chunks=len(rows), primitive_steps=sum(r['primitive_steps'] for r in rows),
        reward_counts=dict(Counter(str(r['reward']) for r in rows)),
        raw_reward_counts=dict(Counter(str(float(x)) for r in rows for x in r['raw_rewards'])),
        noise=numbers(np.stack([r['native_initial_noise'] for r in rows])))
    for key in ('proposed_noise', 'executed_noise'):
        if rows[0].get(key) is not None:
            a = np.stack([r[key] for r in rows])
            result[key] = dict(numbers(a), temporal_coordinate_std_mean=float(a.std(0).mean()),
                coordinate_mean_l2=float(np.linalg.norm(a.mean(0))))
    actions = np.concatenate([r['executed_actions'] for r in rows])
    states = np.stack([r['motor_observation']['state'] for r in rows] + [rows[-1]['motor_next_observation']['state']])
    result.update(actions=numbers(actions), action_mean_per_axis=actions.mean(0).astype(float).tolist(),
        action_std_per_axis=actions.std(0).astype(float).tolist(),
        gripper_positive_fraction=float((actions[:,-1] > 0).mean()),
        gripper_negative_fraction=float((actions[:,-1] < 0).mean()),
        state_start=states[0].astype(float).tolist(), state_end=states[-1].astype(float).tolist(),
        eef_xyz_min=states[:,:3].min(0).astype(float).tolist(), eef_xyz_max=states[:,:3].max(0).astype(float).tolist(),
        eef_sampled_path_length=float(np.linalg.norm(np.diff(states[:,:3], axis=0), axis=1).sum()),
        last10_chunks_eef_span=np.ptp(states[-11:,:3], axis=0).astype(float).tolist())
    return result


def frames(rows, task, behavior):
    names = []
    moments = [(0, rows[0]['motor_observation']), (100, rows[10]['motor_observation']),
        (200, rows[20]['motor_observation']), (300, rows[30]['motor_observation']), (400, rows[-1]['motor_next_observation'])]
    for step, observation in moments:
        for camera in ('image', 'wrist'):
            if camera == 'wrist' and step in (100, 300):
                continue
            array = np.asarray(observation[camera])
            name = f'task{task:02d}-{behavior}-step{step:03d}-{camera}.png'
            Image.fromarray(array).save(EXPORT / name)  # Exact saved pixels; no edit/resize/composite.
            names.append(dict(file=name, sha256=sha(EXPORT/name), shape=list(array.shape)))
    return names


result = dict(source='Saved immutable trajectories; no models/rollouts/tests', cases=[], full_six_case_training=[], firstwave=[])
for task in (0,10,20,30,40,50):
    index = json.loads((DSRL / f'case-{task:02d}/index.json').read_text())
    result['full_six_case_training'].append(dict(task=task, training_episodes=index['training_episodes'],
        training_successes=sum(r['success'] for r in index['training_receipts']), chunks=index['training_chunks'],
        target_updates=index['target_optimizer_updates'], warmup_cleared=index['warmup_cleared'],
        query_successes=[r['success'] for r in index['query_receipts']],
        query_behavior=[r['query_behavior'] for r in index['query_receipts']]))
    if task not in (0,40):
        continue
    root = DSRL / f'case-{task:02d}'
    train = index['training_receipts']
    all_train_rows, stage_rows = [], {}
    for receipt in train:
        saved = load(root / receipt['trace'], receipt['trace_sha256'])
        all_train_rows.extend(saved['rows'])
        stage_rows.setdefault(receipt['behavior'], []).extend(saved['rows'])
    queries = []
    for position, behavior in ((0,'native-q0'), (5,'sac-q16')):
        receipt = index['query_receipts'][position]
        saved = load(root / receipt['trace'], receipt['trace_sha256'])
        queries.append(dict(behavior=behavior, trace=receipt['trace'], trace_sha256=receipt['trace_sha256'],
            success=receipt['success'], target_updates=receipt['target_optimizer_updates'],
            initial_state_vector_sha256=receipt['initial_state_vector_sha256'],
            initial_observation_sha256=receipt['initial_observation_sha256'], episode_random_seed=receipt['episode_random_seed'],
            stats=stats(saved['rows']), keyframes=frames(saved['rows'], task, behavior)))
    result['cases'].append(dict(task=task, instruction=train[0]['instruction'], training_successes=sum(r['success'] for r in train),
        training_episode_count=len(train), training_stats=stats(all_train_rows),
        behavior_stages={stage: dict(episodes=sum(r['behavior']==stage for r in train), stats=stats(rows)) for stage, rows in stage_rows.items()},
        training_per_episode=[{k:r[k] for k in ('training_episode','success','chunks','raw_return','behavior',
            'target_optimizer_updates_before','target_optimizer_updates','target_update_metrics')} for r in train], queries=queries))
# Bounded comparison: only two failed cases and the four actual500-update
# firstwave controllers. Read K0 saved samples, not constructed models.
for worker in range(6):
    root = ROOT / f'libero40-target-firstwave-20261030-w{worker:02d}-r1/experiment'
    index = json.loads((root/'index.json').read_text())
    for saved in index['completed_groups']:
        g = saved['group']
        if g['case']['benchmark'] != 'libero90' or g['case']['task_id'] not in (0,40) or g['support_budgets'] != [0]:
            continue
        path = Path(saved['result_path'])
        if sha(path) != saved['result_sha256']:
            raise ValueError('Firstwave group checksum differs')
        receipt = json.loads(path.read_text())['episodes'][0]
        rows = load(path.parent/receipt['trace'], receipt['trace_sha256'])
        result['firstwave'].append(dict(task=g['case']['task_id'], method=g['method'], success=receipt['success'],
            result_sha256=saved['result_sha256'], trace_sha256=receipt['trace_sha256'], stats=stats(rows)))
(EXPORT/'stats.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(export=str(EXPORT), full_six_case_training=result['full_six_case_training'],
    cases=[dict(task=c['task'],instruction=c['instruction'],training_successes=c['training_successes'],
        reward_counts=c['training_stats']['reward_counts'],
        stages={k:dict(episodes=v['episodes'],executed_noise=v['stats'].get('executed_noise')) for k,v in c['behavior_stages'].items()},
        queries=[dict(behavior=q['behavior'],success=q['success'],stats=q['stats']) for q in c['queries']]) for c in result['cases']],
    firstwave=[dict(task=x['task'],method=x['method'],success=x['success'],executed_noise=x['stats'].get('executed_noise'),noise=x['stats']['noise']) for x in result['firstwave']]), indent=2))
