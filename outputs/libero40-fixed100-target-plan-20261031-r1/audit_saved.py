"""Read only closed fixed100 receipts, bytes and NumPy traces; no model imports."""
import collections
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import numpy as np

ROOT = Path('/mowenjie/fluffy-bassoon-2')
PREFIX = 'libero40-fixed100-target-20261031-r1'
PLAN_ROOT = ROOT / 'outputs/libero40-fixed100-target-plan-20261031-r1'
SOURCE = 'e764afdd04c9ea4dcbb7ac3e3e25d245c186dd42'
MOTOR = 'c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9'
JOBS = ['job-zt7xex96p71j', 'job-404blkw0lib5', 'job-sg46ctfm4b2j',
        'job-xcxj0gvvjfxc', 'job-1hpil6fjy4bd', 'job-mwscjuukmh9t']
UI_SECONDS = [1210, 1031, 1230, 1051, 1193, 1131]
CATALOG = {}
CHECKS = collections.Counter()


def sha(path):
    path = Path(path)
    if str(path) in CATALOG:
        return CATALOG[str(path)]['sha256']
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    CATALOG[str(path)] = dict(sha256=h.hexdigest(), bytes=path.stat().st_size)
    return h.hexdigest()


def read(path):
    sha(path)
    return json.loads(Path(path).read_text())


def check(value, message):
    if not value:
        raise AssertionError(message)
    CHECKS[message] += 1


def signature(value):
    h = hashlib.sha256()
    def visit(x):
        if isinstance(x, dict):
            h.update(b'dict')
            for key in sorted(x):
                visit(key)
                visit(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(str(type(x)).encode())
            for v in x:
                visit(v)
        elif isinstance(x, np.ndarray):
            h.update(str((x.dtype, x.shape)).encode())
            h.update(np.ascontiguousarray(x).tobytes())
        else:
            h.update(str((type(x), x)).encode())
    visit(value)
    return h.hexdigest()


def public_signature(observation):
    h = hashlib.sha256()
    for key, value in sorted(observation.items()):
        a = np.asarray(value)
        h.update(key.encode())
        h.update(str((a.shape, a.dtype)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def paired(left, right):
    return dict(wins=[k for k in sorted(left) if left[k] and not right[k]],
                losses=[k for k in sorted(left) if right[k] and not left[k]],
                both_success=[k for k in sorted(left) if left[k] and right[k]],
                both_failure=[k for k in sorted(left) if not left[k] and not right[k]])


plan = read(PLAN_ROOT / 'plan.json')
ledger = read(PLAN_ROOT / 'ledger.json')
groups, workers, episodes, identity = [], [], [], {}
query_by_method = collections.defaultdict(dict)
support_by_method = collections.defaultdict(dict)
support_trace = {}
fixed = None
for w in range(6):
    root = ROOT / 'outputs' / f'{PREFIX}-w{w:02}'
    path = root / 'experiment'
    index = read(path / 'index.json')
    clock = read(root / 'worker-time.json')
    check((root / 'exit-code.txt').read_text().strip() == '0', 'worker_exit_zero')
    sha(root / 'exit-code.txt')
    check(index['completed'] is True and index['stop_reason'] is None, 'worker_complete')
    check(index['motor_parameters_unchanged'] and index['motor_parameters_before_sha256'] == MOTOR
          and index['motor_parameters_after_sha256'] == MOTOR, 'motor_parameters_unchanged')
    check(index['deployment_optimizer_updates'] == 0 and index['no_target_optimizer'], 'no_target_optimizer')
    check(index['strict_common_runtime_primary_eligible'] and index['all_actual_initial_observations_equal']
          and len(index['actual_runtime_ids']) == 1, 'one_runtime_and_initial_parity')
    check(index['wave_sha256'] == plan['firstwave_sha256'], 'wave_bound')
    check(index['contract']['fixed_source_inputs'] == plan['contract']['fixed_source_inputs'], 'source_inputs_bound')
    check(index['contract']['source_commit'] == SOURCE, 'runtime_source_bound')
    expected = [dict(g, case=e['case'], support_budgets=[g['support_budget']])
                for e in plan['cases'][w::6] for g in e['groups']]
    check([r['group'] for r in index['completed_groups']] == expected, 'exact_predeclared_groups')
    check(index['contract'] == read(path / 'contract.json'), 'contract_matches_final')
    sha(path / 'comparison-recovery.pkl')
    if fixed is None:
        fixed = index['fixed100_models']
    check(index['fixed100_models'] == fixed, 'same_four_fixed100_models')
    for model in fixed:
        check(model['updates'] == 100 and model['seed'] == 0, 'fixed100_not_target_selected')
        check(sha(model['final_path']) == model['final_sha256'], 'model_bytes_unchanged')
        check(sha(model['full_recovery_path']) == model['full_recovery_sha256'], 'source_full_recovery_hash')
    runtimes = list((root / 'runtime-attempts').iterdir())
    check(len(runtimes) == 1, 'one_runtime_directory')
    before = read(runtimes[0] / 'motor-parameters-before.json')
    after = read(runtimes[0] / 'motor-parameters-after.json')
    check(after['equal'] is True, 'motor_parameter_receipt_equal')
    attempts = clock['attempts']
    check(len(attempts) == 1 and attempts[0]['status'] == 'finished'
          and attempts[0]['exit_code'] == 0 and not attempts[0]['slice_overrun'], 'one_terminal_attempt')
    attempt = Path(attempts[0]['path'])
    probe = read(attempt / 'gpu-probe.json')
    check((attempt / 'exit-code.txt').read_text().strip() == '0', 'attempt_exit_zero')
    sha(attempt / 'exit-code.txt')
    charged = sum(a['charged_seconds'] for a in attempts)
    check(charged <= 2700 and clock['worker_seconds'] == 2700, 'worker_time_within_cap')
    check(clock['contract']['matrix_file_sha256'] == plan['contract']['fixed_source_inputs']['matrix_sha256'],
          'clock_matrix_bound')
    worker_actions = 0
    worker_eps = []
    for ordinal, row in enumerate(index['completed_groups']):
        result_path = Path(row['result_path'])
        group_root = result_path.parent
        check(sha(result_path) == row['result_sha256'], 'group_index_hash')
        result = read(result_path)
        g = row['group']
        check(result['completed'] and result['weights_unchanged'] and result['budget_matched_primary_eligible'],
              'group_complete_frozen_untruncated')
        check(result['case'] == g['case'], 'exact_target_case')
        check(len(result['episodes']) == g['support_budget'] + 1, 'actual_support_and_separate_query')
        check([(t['method'], t['support_budget']) for t in result['trials']] == [(g['method'], g['support_budget'])],
              'method_and_support_grid')
        check(not any(v for r in result['interrupted_physical_costs'].values() for v in r.values()),
              'no_interrupted_or_uncertain_actions')
        check(sha(group_root / 'evaluation-recovery.pkl') == result['latest_recovery_sha256'], 'evaluation_recovery_hash')
        read(group_root / 'contract.json')
        invocations = read(group_root / 'runtime-invocations.json')
        check(len(invocations) == 1 and invocations[0]['runtime_id'] == index['actual_runtime_ids'][0],
              'group_same_common_motor_process')
        contexts = read(group_root / 'contexts/index.json')
        check(sha(group_root / 'contexts/index.json') == row['audit']['episode_context_index_sha256'],
              'context_index_hash')
        for context in contexts:
            check(Path(context['path']).name == context['path'], 'context_relative_path')
            check(sha(group_root / 'contexts' / context['path']) == context['sha256'], 'context_sampler_rng_hash')
        case_id, task = g['case']['case_id'], g['case']['task_id']
        for ep in result['episodes']:
            check(ep['completed'] and ep['settling_steps'] == 10 and ep['initial_state'] == 0
                  and ep['scene_seed'] == 123, 'complete_declared_physical_episode')
            check(ep['role'] in ('support', 'query') and 0 < ep['primitive_steps'] <= 400, 'episode_role_and_horizon')
            relative = Path(ep['trace'])
            check(not relative.is_absolute() and '..' not in relative.parts, 'trace_relative_path')
            trace = group_root / relative
            check(sha(trace) == ep['trace_sha256'], 'trace_hash')
            with gzip.open(trace, 'rb') as stream:
                rows = pickle.load(stream)  # Saved plain NumPy arrays only; no model/checkpoint load.
            check(len(rows) == ep['chunks'] and sum(r['primitive_steps'] for r in rows) == ep['primitive_steps'],
                  'actual_trace_lengths')
            check(bool(rows[-1]['success']) == ep['success'] and bool(rows[-1]['done']), 'terminal_trace_boundary')
            check(all(np.asarray(r['executed_actions']).shape == (r['primitive_steps'], 7)
                      and len(r['raw_rewards']) == r['primitive_steps'] for r in rows), 'raw_action_reward_lengths')
            sig = public_signature(rows[0]['motor_observation'])
            check(sig == row['audit']['initial_public_arrays_sha256'], 'actual_initial_array_digest')
            ident = (ep['initial_state_vector_sha256'], ep['initial_observation_sha256'], sig)
            if case_id not in identity:
                identity[case_id] = ident
            check(identity[case_id] == ident, 'cross_method_support_query_initial_parity')
            record = {k: ep[k] for k in ('role', 'success', 'primitive_steps', 'settling_steps', 'chunks',
                'episode_random_seed', 'initial_observation_sha256', 'initial_state_vector_sha256', 'trace_sha256')}
            record.update(case_id=case_id, task_id=task, method=g['method'], support_budget=g['support_budget'],
                          worker=w, group_id=g['group_id'], initial_public_arrays_sha256=sig)
            episodes.append(record)
            worker_eps.append(record)
            key = f"{g['method']}/k{g['support_budget']}"
            if ep['role'] == 'query':
                query_by_method[key][task] = ep['success']
                check(ep['episode_random_seed'] == row['audit']['query_episode_random_seed'], 'paired_query_seed')
            else:
                support_by_method[key][task] = ep['success']
                support_trace[(g['method'], task)] = signature(rows)
            if g['method'] != 'native':
                position = ep['support_budget'] if ep['role'] == 'query' else ep['role_episode_ordinal']
                check(any(c.get('method') == g['method'] and c.get('trial_index') == ep['trial_index']
                          and c.get('episode_position') == position
                          and c.get('episode_random_seed') == ep['episode_random_seed'] for c in contexts),
                      'episode_full_context_and_rng_boundary')
        committed = {role: dict(primitive_steps=sum(e['primitive_steps'] for e in result['episodes'] if e['role'] == role),
                               settling_steps=sum(e['settling_steps'] for e in result['episodes'] if e['role'] == role))
                     for role in ('support', 'query')}
        check(committed == result['committed_episode_costs'], 'recomputed_committed_costs')
        costs = result['physical_costs']
        for role in ('support', 'query'):
            check(costs[role]['primitive_steps'] == committed[role]['primitive_steps']
                  and costs[role]['settling_steps'] == committed[role]['settling_steps']
                  and costs[role]['uncertain_primitive_steps_ceiling'] == 0
                  and costs[role]['uncertain_settling_steps_ceiling'] == 0, 'journal_costs_equal_actual_episodes')
        worker_actions += sum(e['primitive_steps'] + e['settling_steps'] for e in result['episodes'])
        groups.append(dict(group=g, result_path=str(result_path), result_sha256=row['result_sha256'],
                           audit=row['audit'], trials=result['trials']))
    check(worker_actions == index['charged_total_actions'] <= plan['workers'][w]['action_cap'], 'worker_actions_recomputed')
    check(len(index['zero_shot_reuse']) == 30, 'explicit_zero_shot_reuse_count')
    by_id = {r['group']['group_id']: r for r in index['completed_groups']}
    for reuse in index['zero_shot_reuse']:
        reference = by_id[reuse['reference_group_id']]
        result = read(reference['result_path'])
        query = next(e for e in result['episodes'] if e['role'] == 'query')
        check(reuse['reference_result_sha256'] == reference['result_sha256']
              and reuse['query_trace_sha256'] == query['trace_sha256']
              and reuse['query_success'] == query['success'] and reuse['additional_physical_actions'] == 0,
              'zero_shot_reuse_bound_to_actual_query')
    size = sum(p.stat().st_size for directory, _, names in os.walk(root)
               for name in names if (p := Path(directory) / name).is_file())
    workers.append(dict(worker=w, job_id=JOBS[w], exit_code=0, index_path=str(path / 'index.json'),
        index_sha256=sha(path / 'index.json'), clock_sha256=sha(root / 'worker-time.json'),
        attempts=attempts, scheduler_ui_seconds=UI_SECONDS[w], charged_gpu_seconds=charged,
        gpu_probe_sha256=sha(attempt / 'gpu-probe.json'), gpu_probe=probe,
        group_count=len(index['completed_groups']), episode_count=len(worker_eps), actual_actions=worker_actions,
        output_bytes=size, full_comparison_recovery_sha256=sha(path / 'comparison-recovery.pkl'),
        motor_before=before, motor_after=after, runtime_ids=index['actual_runtime_ids']))
    print(f"verified worker {w}: {len(worker_eps)} episodes / {worker_actions} actions", file=sys.stderr, flush=True)

check(len(groups) == 486 and len(episodes) == 522 and len(identity) == 90, 'entire_486_group_522_episode_90_case_grid')
check(sum(e['role'] == 'support' for e in episodes) == 36, '36_actual_support_episodes')
check(sum(e['role'] == 'query' for e in episodes) == 486, '486_separate_query_episodes')
for field, expected in [('matrix_path', 'matrix_sha256'), ('assembly_path', 'assembly_file_sha256'),
                        ('manifest_path', 'manifest_sha256')]:
    inputs = plan['contract']['fixed_source_inputs']
    check(sha(inputs[field]) == inputs[expected], 'pinned_source_input_bytes')
support_equal = {}
for obj in ('ad', 'dpt'):
    support_equal[obj] = {str(t): support_trace[(f'{obj}_full', t)] == support_trace[(f'{obj}_episode_reset', t)]
                          for t in range(0, 90, 10)}
    check(all(support_equal[obj].values()), 'full_reset_actual_support_trajectories_equal')
metrics = {key: dict(successes=sum(rows.values()), queries=len(rows), success_task_ids=[t for t in sorted(rows) if rows[t]],
                    failure_task_ids=[t for t in sorted(rows) if not rows[t]]) for key, rows in query_by_method.items()}
paired_results = {key: paired(rows, query_by_method['native/k0']) for key, rows in query_by_method.items()
                  if key.endswith('/k0') and key != 'native/k0'}
memory = {obj: dict(full_vs_episode_reset=paired(query_by_method[f'{obj}_full/k1'],
                            query_by_method[f'{obj}_episode_reset/k1']),
                    full_k1_vs_full_k0=paired(query_by_method[f'{obj}_full/k1'],
                            {t: query_by_method[f'{obj}_full/k0'][t] for t in range(0, 90, 10)}),
                    support_outcomes={method: support_by_method[f'{obj}_{method}/k1'] for method in ('full', 'episode_reset')},
                    actual_support_trace_equality=support_equal[obj]) for obj in ('ad', 'dpt')}
costs = dict(actual_actions=sum(w['actual_actions'] for w in workers),
    support_primitives=sum(e['primitive_steps'] for e in episodes if e['role'] == 'support'),
    query_primitives=sum(e['primitive_steps'] for e in episodes if e['role'] == 'query'),
    support_settling=sum(e['settling_steps'] for e in episodes if e['role'] == 'support'),
    query_settling=sum(e['settling_steps'] for e in episodes if e['role'] == 'query'),
    charged_gpu_seconds=sum(w['charged_gpu_seconds'] for w in workers),
    scheduler_ui_seconds=sum(UI_SECONDS), output_bytes=sum(w['output_bytes'] for w in workers),
    maximum_actions=220000, maximum_gpu_seconds=16200, maximum_output_bytes=20000000000,
    free_bytes=os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize,
    source_training_actions_added=0, target_optimizer_updates=0, uncertain_actions=0)
check(costs['actual_actions'] <= 220000 and costs['charged_gpu_seconds'] <= 16200
      and costs['output_bytes'] <= 20000000000, 'joint_time_actions_storage_within_scope')
report = dict(format='libero40_fixed100_target_terminal_audit_v1', completed=True,
    checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_commit=SOURCE,
    plan_sha256=sha(PLAN_ROOT / 'plan.json'), ledger_sha256=sha(PLAN_ROOT / 'ledger.json'),
    wave_sha256=plan['firstwave_sha256'], expected=dict(groups=486, episodes=522, cases=90, supports=36, queries=486),
    checks=dict(CHECKS), costs=costs, metrics=metrics, paired_vs_contemporaneous_native=paired_results,
    memory_panel=memory, per_episode=episodes, per_group=groups, workers=workers, fixed100_models=fixed,
    verified_files=CATALOG,
    audit_scope='Read-only saved metadata, byte hashes and plain NumPy trajectory arrays; no model/checkpoint deserialization, inference, training, simulator or new tests. Physical journal totals independently reconcile to complete episode/trace costs; entire RPC intent forest was not replayed. Full recovery archives rehashed, not restored.',
    interpretation='Fixed100 interim source-only checkpoint, all90 single state/scene seed. Nine predeclared one-support cases are a limited development panel; differences do not establish memory benefit, high topline or sample efficiency. Historical native waves are separate.')
print(json.dumps(report, indent=2, sort_keys=True))
