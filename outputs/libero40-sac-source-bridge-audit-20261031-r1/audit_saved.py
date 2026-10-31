"""Terminal saved-file/array audit only; no learner/model/rollout construction."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs')
CASES = (('s00-train', 'job-zh3wvn64ebfd'), ('o00-train', 'job-xli5uonwbiph'))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


reports = []
for suffix, job in CASES:
    root = BASE / ('libero40-sac-source-bridge-20261031-r1-'+suffix)
    out = root/'experiment'
    index, contract, clock = read(out/'index.json'), read(out/'contract.json'), read(root/'worker-time.json')
    candidate = read(out/'candidate.json')
    parent = index['continuation']['parent']
    new = index['episode_receipts'][parent['episodes']:]
    latest = index['publications'][-1]
    checks = {}

    def check(name, value):
        checks[name] = bool(value)
        assert value, name

    check('terminal_exit0', (root/'exit-code.txt').read_text().strip() == '0')
    check('complete_qualified_whole_cases', index['completed'] and index['collection_qualified'] and index['episode_in_case'] == 0)
    check('exact_source', contract['source_commit'] == 'cf49bfac65a98cc7c4cef7fb35df4a6a18e20c0b')
    check('slot_action_cap', index['maximum_total_actions'] == 21800 and index['total_action_budget_charge'] <= 21800)
    check('no_quota_for_another_K4', index['production_quota_exhausted'] and index['total_action_budget_charge']+1640 > 21800)
    check('full_latest_recovery_checksum', sha(out/'source-recovery.pkl') == index['latest_recovery_sha256'] == candidate['full_latest_recovery_sha256'])
    check('final_actor_checkpoint_checksum', sha(out/latest['source_checkpoint']) == latest['source_checkpoint_sha256'] == candidate['sha256'])
    with (out/'source-recovery.pkl').open('rb') as f:
        full = pickle.load(f)
    with (out/latest['source_checkpoint']).open('rb') as f:
        actor = pickle.load(f)
    check('full_recovery_contract_and_boundary', full['contract'] == contract and full['boundary'] == 'complete physical episode')
    check('full_recovery_cursor_and_replay', full['progress']['case_index'] == index['case_index']
        and full['progress']['episode_in_case'] == 0 and len(full['replay']) == index['chunks'] and full['updates'] == index['source_updates'])
    check('all_recovery_states_present', all(k in full for k in ('agent', 'agent_rng', 'sampler', 'numpy_rng', 'python_rng'))
        and full['progress']['motor_rng']['format'] == 'jax_motor_key_v1')
    check('final_publication_learning_state_equals_latest', actor['agent'] == full['agent']
        and actor['updates'] == full['updates'] and np.array_equal(actor['agent_rng'], full['agent_rng']))
    del actor, full
    before = parent['source_updates']
    chunks, stages = 0, collections.Counter()
    for receipt in new:
        check('all_new_raw_episode_hashes', sha(out/receipt['trace']) == receipt['trace_sha256'])
        with gzip.open(out/receipt['trace'], 'rb') as f:
            rows = pickle.load(f)['rows']
        check('actual_complete_new_boundaries', bool(rows) and bool(rows[-1]['done']) and len(rows) == receipt['chunks'])
        check('actual_fixed_learned_episode_stage', all(r['source_stage'] == 'sac_online' and r['source_updates_before'] == before for r in rows))
        check('actual_postwarmup_UTD20_updates', receipt['source_updates'] == before + 20*len(rows))
        check('new_executed_noise_is_finite_bounded32D', all(np.asarray(r['executed_noise']).shape == (32,)
            and np.isfinite(r['executed_noise']).all() and np.max(np.abs(r['executed_noise'])) <= 1 for r in rows))
        check('finite_new_update_metrics', bool(receipt['source_update_metrics']) and all(math.isfinite(v) for v in receipt['source_update_metrics'].values()))
        before, chunks = receipt['source_updates'], chunks+len(rows)
        stages.update(r['source_stage'] for r in rows)
    check('final_update_clock', before == index['source_updates'])
    exports = {}
    for objective, row in latest['datasets'].items():
        path = out/row['path']
        manifest = read(path.with_suffix('.json'))
        check(objective+'_export_checksum', sha(path) == row['sha256'] == manifest['sha256'])
        check(objective+'_cumulative_cost', manifest['metadata']['source_generation_cumulative_total_actions'] == index['total_action_budget_charge'])
        check(objective+'_final_actor_binding', manifest['metadata']['source_actor_checkpoint_sha256'] == latest['source_checkpoint_sha256'])
        with np.load(path, allow_pickle=False) as data:
            check(objective+'_actual_rows', len(data['observations']) == index['chunks'] == int(data['lengths'].sum()))
            check(objective+'_finite_labels', bool(np.isfinite(data['target_noise']).all()))
            if objective == 'ad':
                check('ad_targets_equal_executed_noise', np.array_equal(data['target_noise'], data['executed_noise']))
        exports[objective] = dict(path=str(path), sha256=row['sha256'], manifest_sha256=sha(path.with_suffix('.json')),
            transitions=manifest['transitions'], teacher_updates=manifest['metadata']['source_updates'],
            cumulative_teacher_actions=manifest['metadata']['source_generation_cumulative_total_actions'])
    attempt = Path(clock['attempts'][0]['path'])
    probe = read(attempt/'gpu-probe.json')
    check('actual_allocated_GPU_probe', probe['status'] == 'passed' and probe['hostname'] == job+'-master-0')
    check('frozen_motor_bytes_unchanged', index['continuation']['motor_parameters_unchanged'])
    check('original_1800_second_worker_bound', clock['worker_seconds'] == 1800
        and all(a['status'] == 'finished' and not a['slice_overrun'] for a in clock['attempts'])
        and sum(a['charged_seconds'] for a in clock['attempts']) <= 1800)
    restore_receipts = [dict(path=str(p), sha256=sha(p), value=read(p))
        for p in sorted((root/'attempts').rglob('*.json'))
        if p.name in ('restored-learner.json', 'migration-fingerprint.json', 'motor-restore.json', 'parameters-before.json', 'parameters-after.json')]
    reports.append(dict(run_id=root.name, job_id=job, source_commit=contract['source_commit'], parent=parent,
        exit_code=0, collection_qualified=True, checks=checks,
        index_sha256=sha(out/'index.json'), contract_sha256=sha(out/'contract.json'), worker_time_sha256=sha(root/'worker-time.json'),
        full_latest_recovery_sha256=index['latest_recovery_sha256'], final_actor_checkpoint_sha256=latest['source_checkpoint_sha256'],
        candidate_sha256=sha(out/'candidate.json'), plan_sha256=index['continuation']['plan_sha256'],
        new=dict(episodes=len(new), cases=len({r['case_id'] for r in new}), chunks=chunks, successes=sum(r['success'] for r in new),
            updates=index['source_updates']-parent['source_updates'], actions=index['total_action_budget_charge']-parent['actions'],
            actual_settled_state_hashes=len({r['initial_simulator_state_sha256'] for r in new}), stages=dict(stages)),
        cumulative=dict(episodes=len(index['episode_receipts']), cases=index['case_index'], chunks=index['chunks'],
            updates=index['source_updates'], actions=index['total_action_budget_charge']),
        charged_gpu_seconds=sum(a['charged_seconds'] for a in clock['attempts']),
        unused_action_headroom=21800-index['total_action_budget_charge'], quota_filled=True,
        allocated_receipts=restore_receipts, exports=exports, focused_tests=(attempt/'focused-tests.log').read_text().strip(),
        full_recovery_verification='Saved full payload/cursor/replay/learning bytes checked; initial allocated restore passed. No new independent final model restore was run.'))
print(json.dumps(dict(format='source40_two_stream_bridge_saved_audit_v1', observed_at='2026-10-31 Asia/Shanghai',
    streams=reports, total_new={k:sum(r['new'][k] for r in reports) for k in ('episodes','cases','chunks','successes','updates','actions')},
    total_gpu_seconds=sum(r['charged_gpu_seconds'] for r in reports), maximum_gpu_seconds=3600,
    limitations='Two fixed source goals;184/184 new source successes show no success headroom. No improved teacher competence, adaptation/generalization, or full24h/all40 top-up completion claim.'), indent=2, allow_nan=False))
