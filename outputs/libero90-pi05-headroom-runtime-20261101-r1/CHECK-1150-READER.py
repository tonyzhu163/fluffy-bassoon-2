"""One owned, read-only current JSON/hash check. No models or journal scan."""
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

root = Path('/mowenjie/fluffy-bassoon-2/outputs/dsrl-pi05-headroom-20261101-s0-r1')
case = root / 'experiment/case-learn-44'
def ref(path):
    path = Path(path)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), sha256=digest, bytes=path.stat().st_size)

index_bytes = (case / 'index.json').read_bytes()
index = json.loads(index_bytes)
clock = json.loads((root / 'worker-time.json').read_text())
attempt = clock['attempts'][-1]
checked = time.time()
settled = sum(row['charged_seconds'] for row in clock['attempts'] if row['status'] == 'finished')
native = {row['case']['case_id']: row for row in index['panels'][0]['receipts']}
identity_fields = ('initial_state', 'scene_seed', 'episode_random_seed',
    'initial_state_vector_sha256', 'initial_observation_sha256')
panels = []
for panel in index['panels']:
    rows = panel['receipts']
    actual = panel['actual_optimizer_updates']
    threshold = panel['threshold_updates']
    row = dict(name=panel['name'], kind=panel['kind'], queries=len(rows),
        successes=sum(x['success'] for x in rows), actual_updates=actual,
        threshold_updates=threshold, primitive_steps=sum(x['primitive_steps'] for x in rows),
        settling_steps=sum(x['settling_steps'] for x in rows),
        policy_signature=panel['policy_signature'])
    if threshold > 360000:
        row.update(optimizer_update_overshoot=actual-threshold,
            immutable_checkpoint=ref(case / panel['recovery_path']),
            immutable_checkpoint_matches=None,
            weights_unchanged=all(x['weights_unchanged'] for x in rows),
            no_query_optimizer_updates=all(x['optimizer_updates_during_query'] == 0 for x in rows),
            replay_isolated=all(x['query_data_added_to_replay'] is False for x in rows),
            policy_signature_fixed=all(x['policy_signature'] == panel['policy_signature'] for x in rows),
            matched_native_initials=all(all(x[k] == native[x['case']['case_id']][k]
                for k in identity_fields) for x in rows))
        row['immutable_checkpoint_matches'] = row['immutable_checkpoint']['sha256'] == panel['recovery_sha256']
    panels.append(row)

stable = None
for _ in range(2):
    state_before = json.loads((case / 'state.json').read_text())
    model = state_before['model']
    actual = ref(case / model['path'])
    state_after = json.loads((case / 'state.json').read_text())
    next_model = state_after['model']
    if model == next_model and actual['sha256'] == model['sha256']:
        stable = dict(pointer=model, actual=actual, stable=True, cursor_next_threshold=state_after['next_threshold'], cursor_phase=state_after['phase'])
        break
last = index['training']['receipts'][-1]
trace = ref(case / last['trace'])
before_path = max((root / 'experiment/runtime-attempts').glob('*/motor-before.json'))
before = json.loads(before_path.read_text())
active = index['active_panel']
receipt = dict(format='pi05_headroom_due_light_check_v1', checked_unix=checked,
    job_id='job-2qn8dpw3yvlb', execution_source='d2369fb94a024315c7ef41d7021bebe171c8b515', scientific_source='b6655659d8bff2c6b5d14eaf05d63f4647176d4f',
    task_id=44, phase=index['phase'], completed=index['completed'],
    index_sha256=hashlib.sha256(index_bytes).hexdigest(), updates=index['target_optimizer_updates'],
    replay_rows=index['replay_rows'], training={k:index['training'][k]
        for k in ('episodes','chunks','primitive_steps','settling_steps')},
    train_successes=sum(x['success'] for x in index['training']['receipts']),
    train_failures=sum(not x['success'] for x in index['training']['receipts']),
    last_training_receipt=last,
    last_losses_finite=all(math.isfinite(x) for x in last['update_metrics'].values()
        if isinstance(x, (int,float))), newest_raw_trace=trace,
    newest_raw_trace_matches=trace['sha256'] == last['trace_sha256'],
    full_recovery=stable, stable_recovery_read_limit=2,
    panels=panels, active_panel=None if active is None else {k:active[k]
        for k in ('name','kind','threshold_updates','actual_optimizer_updates','query_cursor')},
    whole_phase_action_charge=index['total_action_charge'],
    learning_case_action_charge=index['learning_case_action_charge'], costs=index['costs'],
    maximum_total_actions=index['maximum_total_actions'],
    storage=index['storage'], output_bytes=index['output_bytes'],
    maximum_output_bytes=index['maximum_output_bytes'], free_PFS_bytes=shutil.disk_usage(root).free,
    attempt=attempt, settled_prior_GPU_seconds=settled,
    active_elapsed_seconds=(checked-attempt['started_unix']) if attempt['status']=='started' else None,
    observed_conservative_elapsed_GPU_seconds=(settled+math.ceil(checked-attempt['started_unix'])+60) if attempt['status']=='started' else settled,
    phase_interrupted_reservation_GPU_seconds=(settled+attempt['reserved_seconds']) if attempt['status']=='started' else settled,
    maximum_phase_GPU_seconds=57600, maximum_job_GPU_seconds=25200,
    motor_before=before, motor_before_ref=ref(before_path), final_motor_after_pending=True,
    stop_reason=index['stop_reason'], next_owned_check='2026-11-01 12:10 Asia/Shanghai',
    verification_limits='Current counters, stable latest full recovery, newly completed learned panels and newest closed raw trace bytes only. Old control/milestone hashes not repeated; no journal forest, tensor deserialization, saved-array analysis, model/test/simulator execution, or job control.')
# One directory-name listing, at most3 newest JSONs; no journal forest replay.
journal = case / 'physical-journal'
names = sorted(path.name for path in journal.glob('*.json'))[-3:]
sample = None
for name in reversed(names):
    path = journal / name
    row = json.loads(path.read_text())
    # Preserve its status truthfully, including a pending intent when latest.
    sample = dict(record=ref(path), metadata=row)
    if row.get('outcome_status') == 'complete' and row.get('returned') is True:
        break
receipt['newest_RPC_sample'] = sample
if sample is not None:
    metadata = sample['metadata']
    sample['referenced_bytes'] = [ref(journal / metadata[key]) for key in ('input_path', 'outcome_path') if metadata.get(key)]
    sample['referenced_byte_hashes_match'] = all(row['sha256'] == metadata['input_sha256' if row['path'].endswith(metadata.get('input_path', '__none__')) else 'outcome_sha256'] for row in sample['referenced_bytes'])
receipt['RPC_verification_scope'] = 'One name listing and at most3 latest JSON metadata records; raw referenced bytes may be rehashed separately, no journal history replay.'
receipt['execution_migration_SHA']='dd228544a601a4745686d49a6db6da7cb337f6cca2e3efd66dc7c2eaa987954c'
receipt['advancing_beyond_restored_prefix'] = index['target_optimizer_updates']>301700 and index['training']['episodes']>1720 and index['replay_rows']>15579
receipt['first_three_controls_retain_threshold0'] = len(index['panels'])>=18 and all(p['actual_optimizer_updates']>0 or p['threshold_updates']==0 for p in index['panels'][:3])
print(json.dumps(receipt, indent=2))
