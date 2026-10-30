"""Read terminal saved files only; no model construction, inference or tests."""
import collections
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs')
ROOT = BASE / 'libero40-sac-topup-migration-20261031-r2'
OUT = ROOT / 'experiment'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


index, contract = read(OUT / 'index.json'), read(OUT / 'contract.json')
clock, candidate = read(ROOT / 'worker-time.json'), read(OUT / 'candidate.json')
assert (ROOT / 'exit-code.txt').read_text().strip() == '0'
assert index['completed'] and index['collection_qualified'] and index['episode_in_case'] == 0
assert contract['source_commit'] == 'cf49bfac65a98cc7c4cef7fb35df4a6a18e20c0b'
parent = index['continuation']['parent']
new = index['episode_receipts'][parent['episodes']:]
checks = {}


def check(name, value):
    checks[name] = bool(value)
    assert value, name


check('latest_full_recovery_checksum', sha(OUT / 'source-recovery.pkl') == index['latest_recovery_sha256'])
latest = index['publications'][-1]
check('final_full_actor_checkpoint_checksum', sha(OUT / latest['source_checkpoint']) == latest['source_checkpoint_sha256'] == candidate['sha256'])
check('candidate_latest_recovery_checksum', candidate['full_latest_recovery_sha256'] == index['latest_recovery_sha256'])
check('new_K4_cases_complete', len(new) == 40 and len({r['case_id'] for r in new}) == 10)
before = parent['source_updates']
new_stage_counts, raw_chunks = collections.Counter(), 0
for r in new:
    check('new_raw_episode_checksums', sha(OUT / r['trace']) == r['trace_sha256'])
    with gzip.open(OUT / r['trace'], 'rb') as f:
        payload = pickle.load(f)
    rows = payload['rows']
    check('new_actual_complete_episode_boundaries', bool(rows) and bool(rows[-1]['done']) and len(rows) == r['chunks'])
    check('new_actual_fixed_learned_stage', all(z['source_stage'] == 'sac_online' and z['source_updates_before'] == before for z in rows))
    check('actual_UTD20_optimizer_advance', r['source_updates'] == before + 20 * len(rows))
    check('finite_new_update_metrics', bool(r['source_update_metrics']) and all(math.isfinite(v) for v in r['source_update_metrics'].values()))
    before = r['source_updates']
    raw_chunks += len(rows)
    new_stage_counts.update(z['source_stage'] for z in rows)
check('final_update_count', before == index['source_updates'] == 11380)
check('new_action_cap', index['total_action_budget_charge'] - parent['actions'] == 3452 <= 5000)
check('motor_unchanged', index['continuation']['motor_parameters_unchanged'])
check('independent_post_update_full_restore', index['continuation']['post_update_independent_full_restore_passed'])
exports = {}
for objective, row in latest['datasets'].items():
    path = OUT / row['path']
    m = read(path.with_suffix('.json'))
    check(objective + '_export_checksum', sha(path) == row['sha256'] == m['sha256'])
    check(objective + '_export_cumulative_cost', m['metadata']['source_generation_cumulative_total_actions'] == 11344)
    check(objective + '_export_final_actor_binding', m['metadata']['source_actor_checkpoint_sha256'] == latest['source_checkpoint_sha256'])
    with np.load(path, allow_pickle=False) as data:
        check(objective + '_actual_export_row_count', len(data['observations']) == 1065 and int(data['lengths'].sum()) == 1065)
        check(objective + '_actual_finite_labels', bool(np.isfinite(data['target_noise']).all()))
        if objective == 'ad':
            check('ad_targets_are_executed_noise', np.array_equal(data['target_noise'], data['executed_noise']))
    exports[objective] = dict(path=str(path), sha256=row['sha256'], manifest_sha256=sha(path.with_suffix('.json')),
        transitions=m['transitions'], cumulative_teacher_actions=m['metadata']['source_generation_cumulative_total_actions'],
        teacher_updates=m['metadata']['source_updates'])
attempt = Path(clock['attempts'][0]['path'])
probe = read(attempt / 'gpu-probe.json')
check('actual_probe_passed', probe['status'] == 'passed' and probe['hostname'] == 'job-uhn04ymqc09y-master-0')
oldroot = BASE / 'libero40-sac-topup-migration-20261031-r1'
oldclock = read(oldroot / 'worker-time.json')
oldcost = sum(r['charged_seconds'] for r in oldclock['attempts'])
newcost = sum(r['charged_seconds'] for r in clock['attempts'])
check('original_1000_seconds_ceiling', oldcost == 146 and newcost == 620 and oldcost + newcost <= 1000)
receipts = []
for p in sorted((ROOT / 'attempts').rglob('*.json')):
    if p.name in ('migration-fingerprint.json', 'restored-learner.json', 'motor-restore.json', 'parameters-before.json', 'parameters-after.json'):
        receipts.append(dict(path=str(p), sha256=sha(p), content=read(p)))
result = dict(format='source40_sac_topup_qualification_saved_audit_v1', observed_at='2026-10-31 Asia/Shanghai',
    job_id='job-uhn04ymqc09y', root=str(ROOT), source_commit=contract['source_commit'],
    parent=parent, exit_code=0, qualified=True, checks=checks,
    index_sha256=sha(OUT / 'index.json'), contract_sha256=sha(OUT / 'contract.json'),
    worker_time_sha256=sha(ROOT / 'worker-time.json'), candidate_sha256=sha(OUT / 'candidate.json'),
    full_latest_recovery_sha256=index['latest_recovery_sha256'], final_actor_checkpoint_sha256=latest['source_checkpoint_sha256'],
    plan_sha256=index['continuation']['plan_sha256'],
    new=dict(episodes=len(new), cases=len({r['case_id'] for r in new}), chunks=raw_chunks,
        optimizer_updates=index['source_updates'] - parent['source_updates'], successes=sum(r['success'] for r in new),
        actions=3452, stages=dict(new_stage_counts), actual_settled_state_hashes=len({r['initial_simulator_state_sha256'] for r in new})),
    cumulative=dict(episodes=len(index['episode_receipts']), cases=index['case_index'], chunks=index['chunks'],
        optimizer_updates=index['source_updates'], actions=index['total_action_budget_charge']),
    compute=dict(first_two_preserved_failures_seconds=oldcost, successful_attempt_seconds=newcost,
        total_qualification_seconds=oldcost+newcost, original_qualification_ceiling_seconds=1000,
        initial_failures=[dict(job='job-g8ghgrpu8334', cause='Missing CUDA_VISIBLE_DEVICES before probe', seconds=76, actions=0),
            dict(job='job-68ino91bxbua', cause='Test fixture omitted required slot', seconds=70, actions=0)]),
    exports=exports, allocated_receipts=receipts,
    focused_tests=(attempt / 'focused-tests.log').read_text().strip(),
    limitations='Runtime/source-corpus migration qualification on one source goal. Near-ceiling40/40 new source successes do not establish improved teacher competence, adaptation, generalization or a memory benefit. No full24h top-up launched.')
print(json.dumps(result, indent=2, allow_nan=False))
