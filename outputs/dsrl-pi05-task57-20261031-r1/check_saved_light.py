"""One saved cursor, recovery, newest panel/raw/RPC sample; no journal replay."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import sys
import time

root = Path(sys.argv[1])
case = root / 'experiment/case-57'


def file_check(path, expected):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        size = 0
        for part in iter(lambda: stream.read(8*1024**2), b''):
            h.update(part)
            size += len(part)
    return dict(path=str(path), expected_sha256=expected, actual_sha256=h.hexdigest(),
        matched=h.hexdigest() == expected, bytes=size)


rotation_retries = 0
for attempt in range(2):
    index_bytes = (case/'index.json').read_bytes()
    data = json.loads(index_bytes)
    snapshot_time = time.time()
    model = data['model']
    recovery = file_check(case/model['path'], model['sha256'])
    if recovery['matched']:
        break
    if json.loads((case/'index.json').read_text())['model'] == model or attempt:
        break
    rotation_retries += 1

clock = json.loads((root/'worker-time.json').read_text())
assert clock['contract']['source_commit'] == 'd001556780c78893cb7ad0310ec46e5d817e5698'
assert clock['contract']['grant_sha256'] == '48acb746fb1a2c087671228c5ca440f871546a43b8f70f1589863ada46a7a9ec'
assert clock['worker_seconds'] == 36000
native = data['panels'][0]
native_by_state = {row['initial_state']:row for row in native['receipts']}
panel = data['panels'][-1]
pairs = []
for row in panel['receipts']:
    before = native_by_state[row['initial_state']]
    pairs.append(dict(state=row['initial_state'], success=row['success'],
        native_primitives=before['primitive_steps'], primitives=row['primitive_steps'],
        settling_steps=row['settling_steps'], primitive_delta=row['primitive_steps']-before['primitive_steps'],
        episode_random_seed=row['episode_random_seed'],
        initial_state_vector_sha256=row['initial_state_vector_sha256'],
        initial_observation_sha256=row['initial_observation_sha256'],
        seed_equal=row['episode_random_seed'] == before['episode_random_seed'],
        nominal_state_equal=row['initial_state_vector_sha256'] == before['initial_state_vector_sha256'],
        actual_initial_observation_equal=row['initial_observation_sha256'] == before['initial_observation_sha256']))
panel_recovery = file_check(case/panel['recovery_path'], panel['recovery_sha256'])
training = data['training']
candidates = [training['receipts'][-1], panel['receipts'][-1]]
if data['active_panel'] and data['active_panel']['receipts']:
    candidates.append(data['active_panel']['receipts'][-1])
raw_row = max(candidates, key=lambda row:int(Path(row['trace']).name.rsplit('-',1)[-1].split('.')[0]))
raw_check = file_check(case/raw_row['trace'], raw_row['trace_sha256'])
episode_id = ('training-%06d' % raw_row['training_episode'] if 'training_episode' in raw_row
    else '%s-state-%02d' % (raw_row['panel_name'], raw_row['initial_state']))
rpc = None
metadata_reads = 0
for path in sorted((case/'physical-journal').glob('*.json'), reverse=True):
    if int(path.stem) > int(snapshot_time*1e9):
        continue
    if metadata_reads >= 100:
        break
    row = json.loads(path.read_text())
    metadata_reads += 1
    if row['episode_id'] != episode_id:
        continue
    rpc = dict(path=str(path), episode_id=episode_id, kind=row['kind'], role=row['role'],
        returned=row['returned'], actual_steps=row['actual_steps'], maximum_steps=row['maximum_steps'],
        outcome_status=row.get('outcome_status'), checks=[file_check(
            case/'physical-journal'/row[field+'_path'], row[field+'_sha256']) for field in ('input','outcome')])
    break

summary_panels = [dict(name=x['name'], threshold_updates=x['threshold_updates'],
    actual_updates=x['actual_optimizer_updates'], episodes=len(x['receipts']),
    successes=sum(row['success'] for row in x['receipts']),
    primitives=sum(row['primitive_steps'] for row in x['receipts']),
    settling_actions=sum(row['settling_steps'] for row in x['receipts'])) for x in data['panels']]
metrics = training['receipts'][-1]['update_metrics']
costs = data['costs']
print(json.dumps(dict(format='task57_due_light_saved_metadata_v1',
    checked_shanghai=datetime.datetime.fromtimestamp(snapshot_time, datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    read_completed_unix=time.time(), job_id='job-iudscqfntb4e',
    source_commit=clock['contract']['source_commit'], grant_sha256=clock['contract']['grant_sha256'],
    index_snapshot_sha256=hashlib.sha256(index_bytes).hexdigest(), phase=data['phase'], completed=data['completed'],
    exit_code=(root/'exit-code.txt').read_text().strip() if (root/'exit-code.txt').is_file() else None,
    training={k:training[k] for k in ('episodes','chunks','primitive_steps','settling_steps')},
    training_successes=sum(row['success'] for row in training['receipts']),
    failed_training=[dict(episode=row['training_episode'],state=row['initial_state'],
        updates=row['target_optimizer_updates'],primitives=row['primitive_steps'])
        for row in training['receipts'] if not row['success']],
    optimizer_updates=data['target_optimizer_updates'], replay_rows=data['replay_rows'],
    latest_metrics=metrics, metrics_finite=all(math.isfinite(x) for x in metrics.values()),
    latest_full_recovery=recovery, rotating_slot_retries=rotation_retries,
    panels=summary_panels, newest_complete_panel_pairs=pairs,
    newest_panel_recovery=panel_recovery,
    newest_panel_query_weights_replay_unchanged=all(row['weights_unchanged'] and row['replay_rows_unchanged']
        and row['query_data_added_to_replay'] is False and row['optimizer_updates_during_query']==0 for row in panel['receipts']),
    active_panel=None if not data['active_panel'] else {k:data['active_panel'][k]
        for k in ('name','actual_optimizer_updates','query_cursor')},
    newest_raw=raw_check, newest_rpc=rpc, rpc_metadata_rows_read=metadata_reads,
    producer_costs=costs, all_in_action_charge=data['total_action_charge'],
    producer_counter_sum_matches=sum(v['known_actions']+v['uncertain_actions_ceiling']
        for v in costs.values()) == data['total_action_charge'],
    independent_journal_replay=False, previous_partial_tail_not_reused_as_verified_prefix=True,
    storage_snapshot_bytes=data['output_bytes'], maximum_storage_bytes=data['maximum_output_bytes'],
    worker=dict(attempts=len(clock['attempts']), maximum_gpu_seconds=clock['worker_seconds'],
        reserved_seconds=sum(row['reserved_seconds'] for row in clock['attempts']),
        active_elapsed_seconds=snapshot_time-clock['attempts'][-1]['started_unix'], final_charge=False),
    next_check_shanghai='2026-10-31T15:36:00+08:00'), indent=2))
