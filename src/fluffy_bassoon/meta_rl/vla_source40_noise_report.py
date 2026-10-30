"""Read-only terminal verification and paired scores for the all90 noise screen.

Saved metadata/NumPy arrays only: no model/framework import, inference, tests,
allocation, waiting or writes to any worker output. Refuses live/partial runs.
"""
import argparse
from contextlib import ExitStack
import fcntl
import gzip
import hashlib
from itertools import combinations
import json
from pathlib import Path
import pickle

import numpy as np

from .vla_source40_noise_plan import FORMAT as PLAN_FORMAT, LEDGER_FORMAT, METHODS, NOISE, content_hash


FORMAT = 'source40_libero90_noise_controls_verified_report_v1'
REFERENCE = dict(native_successes=31, queries=90,
    status='Earlier firstwave reference only; contemporaneous native is the primary comparison')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def member(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Artifact reference escapes its worker output')
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'Artifact reference escapes its worker output')
    return path


def arrays_signature(observation):
    digest = hashlib.sha256()
    for key, value in sorted(observation.items()):
        array = np.asarray(value)
        digest.update(key.encode())
        digest.update(str((array.shape, array.dtype)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def initial_hash(observation):
    return hashlib.sha256(b''.join(np.asarray(observation[k]).tobytes() for k in ('image', 'wrist', 'state'))).hexdigest()


def verify_noise(rows, method, seed):
    rng = np.random.default_rng(seed)
    require(method in METHODS and bool(rows), 'Unknown control or empty actual query trace')
    for position, row in enumerate(rows):
        actual = np.asarray(row['native_initial_noise'])
        require(actual.shape == (10, 32) and actual.dtype == np.float32, 'Actual native flow noise shape/dtype differs')
        if method == 'native':
            expected = rng.standard_normal((10, 32)).astype(np.float32)
            require(row['proposed_noise'] is None and row['executed_noise'] is None, 'Native has an unexpected steering coordinate')
        else:
            proposed = np.zeros(32, np.float32) if method == 'zero' else rng.standard_normal(32).astype(np.float32)
            executed = np.clip(proposed, -1., 1.).astype(np.float32) if method == 'tied_gaussian_clipped' else proposed
            expected = np.repeat(executed[None], 10, axis=0)
            require(np.array_equal(row['proposed_noise'], proposed) and np.array_equal(row['executed_noise'], executed),
                'Actual steering coordinate differs from its exact Gaussian/std/clipping recipe')
        require(np.array_equal(actual, expected), 'Actual saved motor noise differs from exact RNG reconstruction')
        actions = np.asarray(row['executed_actions'])
        steps = row['primitive_steps']
        require(type(steps) is int and 1 <= steps <= 10 and actions.shape == (steps, 7)
            and np.isfinite(actions).all() and len(row['raw_rewards']) == steps
            and np.isfinite(row['raw_rewards']).all(), 'Actual decoded-action/reward trace is incomplete')
        require(bool(row['done']) == (position == len(rows) - 1), 'Trace episode boundaries differ')


def journal_charge(path):
    total, uncertain = 0, 0
    for item in Path(path).glob('*.json'):
        row = read(item)
        require(row['role'] == 'query' and row['kind'] in ('reset', 'step'), 'Noise diagnostic contains non-query interactions')
        cost = row['actual_steps'] if row['returned'] else row['maximum_steps']
        require(type(cost) is int and 0 <= cost <= row['maximum_steps'], 'Invalid physical journal charge')
        total += cost
        uncertain += 0 if row['returned'] else cost
    return total, uncertain


def verify_group(path, saved, expected, runtime):
    result_path = Path(path) / 'index.json'
    require(sha(result_path) == saved['result_sha256'], 'Completed query group file hash differs')
    result = read(result_path)
    case, method = expected['case'], expected['method']
    require(saved['group'] == expected and Path(saved['result_path']).parts[-2:] == result_path.parts[-2:],
        'Completed group differs from the predeclared case/method')
    require(result.get('completed') is True and result.get('weights_unchanged') is True
        and result.get('case') == case and len(result.get('episodes', [])) == 1
        and len(result.get('trials', [])) == 1, 'Query group is incomplete or mismatched')
    trial, episode = result['trials'][0], result['episodes'][0]
    require((trial['method'], trial['support_budget']) == (method, 0)
        and (episode['method'], episode['support_budget'], episode['role']) == (method, 0, 'query')
        and episode['initial_state'] == 0 and episode['scene_seed'] == 123
        and type(episode['success']) is bool and trial['query_success'] == episode['success']
        and episode.get('completed') is True and episode['settling_steps'] == 10
        and 0 < episode['primitive_steps'] <= 400, 'Actual query receipt differs from the declared K0 case')
    require(episode['episode_random_seed'] == int(content_hash(dict(case=case, role='query', ordinal=0))[:8], 16),
        'Actual query stochastic seed differs from the predeclared case/role')
    require(saved['runtime_ids'] == [runtime], 'A method/case pair crossed actual motor processes')
    invocations = read(Path(path) / 'runtime-invocations.json')
    require(invocations and {r['runtime_id'] for r in invocations} == {runtime}, 'Actual invocation lineage differs')
    trace = member(path, episode['trace'])
    require(sha(trace) == episode['trace_sha256'], 'Actual query trace checksum differs')
    with gzip.open(trace, 'rb') as stream:
        rows = pickle.load(stream)
    verify_noise(rows, method, episode['episode_random_seed'])
    require(len(rows) == episode['chunks'] and sum(r['primitive_steps'] for r in rows) == episode['primitive_steps']
        and bool(rows[-1]['success']) == episode['success'], 'Actual trace cost/outcome differs from query receipt')
    public = rows[0]['motor_observation']
    require(initial_hash(public) == episode['initial_observation_sha256'], 'Actual trace initial frame hash differs')
    signature = arrays_signature(public)
    audit = saved['audit']
    require(audit.get('noise_recipe_exact') is True and audit.get('actual_chunks') == len(rows)
        and audit.get('initial_public_arrays_sha256') == signature
        and audit.get('initial_observation_sha256') == episode['initial_observation_sha256']
        and audit.get('nominal_state_sha256') == episode['initial_state_vector_sha256']
        and audit.get('public_arrays_match_first_group') is True
        and audit.get('public_hash_matches_first_group') is True, 'Recorded noise/actual-frame audit differs')
    raw = Path(path) / 'raw-rpc' / runtime
    resets = list(raw.glob('*-reset.pkl.gz'))
    require(len(resets) == 1 and len(list(raw.glob('*-inference.pkl.gz'))) == len(rows)
        and len(list(raw.glob('*-outcome.pkl.gz'))) == len(rows), 'Actual raw reset/action/outcome coverage differs')
    with gzip.open(resets[0], 'rb') as stream:
        reset = pickle.load(stream)
    initial = reset['reset_result']
    state = np.asarray(initial['initial_simulator_state'])
    physical = hashlib.sha256(state.tobytes()).hexdigest()
    require(reset['initial_state'] == 0 and reset['scene_seed'] == 123
        and physical == initial['initial_simulator_state_sha256']
        and initial['initial_state_vector_sha256'] == episode['initial_state_vector_sha256']
        and arrays_signature(initial['observation']) == signature
        and initial['initial_observation_sha256'] == episode['initial_observation_sha256'],
        'Actual post-settle physics/public observation differs between raw reset and query trace')
    contexts = Path(path) / 'contexts'
    archive_path = contexts / 'index.json'
    archive = read(archive_path)
    require(archive and sha(archive_path) == audit['episode_context_index_sha256'], 'Full RNG context index checksum differs')
    for row in archive:
        require(sha(member(contexts, row['path'])) == row['sha256'], 'Complete sampler/global/motor RNG archive checksum differs')
    recovery_path = Path(path) / 'evaluation-recovery.pkl'
    require(sha(recovery_path) == result['latest_recovery_sha256'], 'Final complete episode recovery checksum differs')
    # Unlearned control recovery contains only NumPy/scalar metadata, not a
    # learned controller. Global Torch RNG archives above are hash-checked only.
    with recovery_path.open('rb') as stream:
        recovery = pickle.load(stream)
    require(recovery['trial_index'] == 1 and recovery['next_episode'] == 0
        and recovery['controller'] is None and recovery['records'] == result['episodes']
        and recovery.get('sampler') and recovery.get('motor_rng'), 'Complete query cursor/sampler/motor RNG is absent')
    cost, uncertain = journal_charge(Path(path) / 'physical-journal')
    require(uncertain == 0 and cost == episode['primitive_steps'] + episode['settling_steps']
        and result.get('budget_matched_primary_eligible') is True, 'Partial/unknown physical interactions preclude the paired primary score')
    return dict(task_id=case['task_id'], case_id=case['case_id'], method=method, success=episode['success'],
        primitive_steps=episode['primitive_steps'], settling_steps=10, charged_actions=cost,
        query_seed=episode['episode_random_seed'], initial_observation_sha256=episode['initial_observation_sha256'],
        initial_public_arrays_sha256=signature, nominal_state_sha256=episode['initial_state_vector_sha256'],
        actual_post_settle_state_sha256=physical, runtime_id=runtime, result_path=str(result_path.resolve()),
        result_sha256=saved['result_sha256'], trace_sha256=episode['trace_sha256'],
        raw_reset_sha256=sha(resets[0]), recovery_sha256=result['latest_recovery_sha256'],
        context_index_sha256=sha(archive_path), noise_recipe_exact=True)


def summarize(groups):
    tasks = {}
    for row in groups:
        lane = tasks.setdefault(row['task_id'], {})
        require(row['method'] not in lane, 'Duplicate method/task query')
        lane[row['method']] = row
    require(set(tasks) == set(range(90)) and all(set(lane) == set(METHODS) for lane in tasks.values()),
        'Full90 paired score requires all360 actual method/task queries')
    per_task = []
    for task, lane in sorted(tasks.items()):
        for key in ('query_seed', 'initial_observation_sha256', 'initial_public_arrays_sha256',
                'nominal_state_sha256', 'actual_post_settle_state_sha256', 'runtime_id'):
            require(len({row[key] for row in lane.values()}) == 1, f'Actual paired initialization/process mismatch on task{task}: {key}')
        per_task.append(dict(task_id=task, case_id=lane['native']['case_id'], results=lane))
    scores = {method: dict(successes=sum(lane[method]['success'] for lane in tasks.values()), queries=90,
        charged_actions=sum(lane[method]['charged_actions'] for lane in tasks.values())) for method in METHODS}
    paired = []
    for left, right in combinations(METHODS, 2):
        cells = {key: [] for key in ('both_success', 'both_failure', 'left_only', 'right_only')}
        for task, lane in sorted(tasks.items()):
            a, b = lane[left]['success'], lane[right]['success']
            cells['both_success' if a and b else 'left_only' if a else 'right_only' if b else 'both_failure'].append(task)
        paired.append(dict(left=left, right=right, counts={k: len(v) for k, v in cells.items()}, task_ids=cells))
    return scores, paired, per_task


def aggregate(plan_path, plan_sha, ledger_path, ledger_sha, worker_roots):
    require(sha(plan_path) == plan_sha and sha(ledger_path) == ledger_sha, 'Plan/ledger file checksum differs')
    plan, ledger = read(plan_path), read(ledger_path)
    require(plan.get('format') == PLAN_FORMAT
        and plan.get('noise_plan_sha256') == content_hash({k: v for k, v in plan.items() if k != 'noise_plan_sha256'})
        and plan['contract']['methods'] == list(METHODS)
        and plan['counts'] == dict(cases=90, groups=360, physical_episodes=360,
            nominal_actions_including_settling=147600, action_cap=152000, gpu_seconds=7200), 'Noise plan is not the immutable four-way full90 diagnostic')
    require(ledger.get('format') == LEDGER_FORMAT and ledger['noise_plan_sha256'] == plan['noise_plan_sha256']
        and ledger['maximum_total_actions'] == 152000 and ledger['maximum_gpu_seconds'] == 7200
        and ledger['source_campaign_immutable'] is True, 'Separate target diagnostic ledger differs')
    roots = {Path(root).name: Path(root).resolve() for root in worker_roots}
    require(len(roots) == 2 and set(roots) == set(ledger['reservations']), 'Supply the two exact original worker roots')
    groups, workers = [], []
    with ExitStack() as stack:
        # Existing read handles only. A live launcher owns an exclusive lock;
        # never wait, create a lock file, poll, or read an actively written run.
        for root in roots.values():
            lock = stack.enter_context((root / 'launcher.lock').open('r'))
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('Worker is still active; terminal aggregation performs no polling') from error
        for run, reservation in sorted(ledger['reservations'].items(), key=lambda item: item[1]['worker']):
            root, worker = roots[run], reservation['worker']
            require((root / 'exit-code.txt').read_text().strip() == '0', 'Worker is failed/nonterminal; no full-suite score')
            clock, result = read(root / 'worker-time.json'), read(root / 'experiment/index.json')
            expected = [g for entry in plan['cases'][worker::2] for g in entry['groups']]
            require(len(expected) == 180 and result.get('completed') is True
                and result.get('worker') == worker and result.get('noise_plan_sha256') == plan['noise_plan_sha256']
                and result.get('source_training_actions') == 0 and result.get('target_optimizer_updates') == 0
                and result.get('motor_parameters_unchanged') is True and result.get('strict_common_runtime_primary_eligible') is True
                and result['contract']['groups'] == expected and len(result.get('completed_groups', [])) == 180,
                'Worker has not completed all180 primary query groups with a frozen common motor')
            require(reservation['source_commit'] == plan['contract']['source_commit']
                and reservation['maximum_total_actions'] == 76000 and reservation['worker_seconds'] == 3600
                and reservation['source_pretraining_data'] is False and reservation['fresh_ppo_training_data'] is False
                and result['contract']['source_commit'] == reservation['source_commit']
                and result['contract']['budget_reservation']['record'] == reservation, 'Original per-worker source/action/time contract differs')
            attempts = clock['attempts']
            require(clock['worker_seconds'] == 3600 and attempts
                and all(a['status'] == 'finished' and type(a.get('charged_seconds')) is int and a['charged_seconds'] >= 0 for a in attempts)
                and attempts[-1]['exit_code'] == 0 and sum(a['charged_seconds'] for a in attempts) <= 3600
                and clock['contract']['source_commit'] == reservation['source_commit'], 'Cumulative worker time is not terminal/within its original hold')
            runtimes = result['actual_runtime_ids']
            require(len(runtimes) == 1, 'Strict paired score requires one actual frozen-motor process per worker')
            runtime = runtimes[0]
            motor = read(root / 'runtime-attempts' / runtime / 'motor-parameters-after.json')
            require(motor['equal'] is True and motor['before'] == motor['after']
                and motor['before']['sha256'] == result['motor_parameters_before_sha256'] == result['motor_parameters_after_sha256'],
                'Frozen motor before/after parameter-state evidence differs')
            for ordinal, (saved, wanted) in enumerate(zip(result['completed_groups'], expected)):
                groups.append(verify_group(root / 'experiment' / f'group-{ordinal:03d}', saved, wanted, runtime))
            cost = sum(g['charged_actions'] for g in groups if g['runtime_id'] == runtime)
            require(cost == result['charged_total_actions'] <= 76000, 'Saved worker/global physical charge differs')
            workers.append(dict(root=str(root), worker=worker, source_commit=reservation['source_commit'], runtime_id=runtime,
                actual_queries=180, charged_actions=cost, cumulative_gpu_seconds=sum(a['charged_seconds'] for a in attempts),
                index_sha256=sha(root / 'experiment/index.json'), worker_time_sha256=sha(root / 'worker-time.json'),
                comparison_recovery_sha256=sha(root / 'experiment/comparison-recovery.pkl'),
                motor_parameter_state_sha256=motor['before']['sha256']))
    require(len({w['runtime_id'] for w in workers}) == 2, 'Two scheduler workers must have distinct actual process identities')
    require(len({w['motor_parameter_state_sha256'] for w in workers}) == 1, 'Workers loaded differing motor parameter states')
    scores, paired, per_task = summarize(groups)
    total = sum(w['charged_actions'] for w in workers)
    require(total <= 152000 and sum(w['cumulative_gpu_seconds'] for w in workers) <= 7200, 'Joint diagnostic resource ceiling exceeded')
    return dict(format=FORMAT, verified=True, plan_file_sha256=plan_sha, ledger_file_sha256=ledger_sha,
        noise_plan_sha256=plan['noise_plan_sha256'], source_commit=plan['contract']['source_commit'],
        actual_queries=360, scores=scores, paired_discordances=paired, per_task=per_task, workers=workers,
        charged_actions=total, cumulative_gpu_seconds=sum(w['cumulative_gpu_seconds'] for w in workers),
        noise_distributions=NOISE, earlier_firstwave_reference=REFERENCE,
        primary='Contemporaneous native versus contemporaneous zero/tied/clipped on the same90 actual cases',
        interpretation='One initial state/query seed per task; diagnostic of noise interface, not evidence of meta-learning or an official multi-seed benchmark score')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'ledger'):
        p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--worker-root', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args(argv)
    if args.output:
        destination = args.output.resolve()
        require(not any(destination.is_relative_to(root.resolve()) for root in args.worker_root),
            'Report must be a new artifact outside the immutable worker outputs')
        require(not destination.exists(), 'Report output already exists; never overwrite evidence')
    result = aggregate(args.plan, args.plan_sha256, args.ledger, args.ledger_sha256, args.worker_root)
    if args.output:
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_task', 'noise_distributions')}, indent=2))


if __name__ == '__main__':
    main()
