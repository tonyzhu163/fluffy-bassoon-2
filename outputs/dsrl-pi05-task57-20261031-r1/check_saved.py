"""Read one live saved-metadata snapshot; no models, tests, RPCs or writes."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import sys
import time

root = Path(sys.argv[1])
case = root / 'experiment/case-57'
index_bytes = (case / 'index.json').read_bytes()
data = json.loads(index_bytes)
clock = json.loads((root / 'worker-time.json').read_text())
now = time.time()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(part)
    return h.hexdigest()


def file_check(path, expected):
    actual = sha(path)
    return dict(path=str(path), expected_sha256=expected, actual_sha256=actual,
        matched=actual == expected, bytes=path.stat().st_size)


native = data['panels'][0]
native_rows = {row['initial_state']: row for row in native['receipts']}
panels = []
for panel in data['panels']:
    rows = panel['receipts']
    comparisons = []
    for row in rows:
        before = native_rows[row['initial_state']]
        comparisons.append(dict(state=row['initial_state'],
            native_success=before['success'], success=row['success'],
            native_primitives=before['primitive_steps'], primitives=row['primitive_steps'],
            primitive_delta=row['primitive_steps'] - before['primitive_steps'],
            settling_steps=row['settling_steps'],
            seed_equal=row['episode_random_seed'] == before['episode_random_seed'],
            nominal_state_equal=row['initial_state_vector_sha256'] == before['initial_state_vector_sha256'],
            initial_observation_equal=row['initial_observation_sha256'] == before['initial_observation_sha256']))
    panels.append(dict(name=panel['name'], kind=panel['kind'],
        threshold_updates=panel['threshold_updates'], actual_updates=panel['actual_optimizer_updates'],
        policy_signature=panel['policy_signature'], episodes=len(rows),
        successes=sum(row['success'] for row in rows),
        primitives=sum(row['primitive_steps'] for row in rows),
        settles=sum(row['settling_steps'] for row in rows),
        queries_never_train=all(row['query_data_added_to_replay'] is False
            and row['replay_rows_unchanged'] and row['weights_unchanged']
            and row['optimizer_updates_during_query'] == 0 for row in rows),
        pairs=comparisons,
        recovery=file_check(case / panel['recovery_path'], panel['recovery_sha256'])))

completed_ids = {'qualification'}
completed_ids.update('training-%06d' % row['training_episode'] for row in data['training']['receipts'])
for panel in data['panels'] + ([data['active_panel']] if data['active_panel'] else []):
    completed_ids.update('%s-state-%02d' % (panel['name'], row['initial_state']) for row in panel['receipts'])
totals = {role: dict(known_actions=0, uncertain_actions_ceiling=0, primitive_steps=0,
    settling_steps=0, reset_attempts=0) for role in ('training', 'query', 'qualification')}
extra, unresolved, sampled_rpc, journal_rows = [], [], [], []
for path in sorted((case / 'physical-journal').glob('*.json')):
    row = json.loads(path.read_text())
    assert row['kind'] in ('reset', 'step') and row['role'] in totals
    assert 1 <= row['maximum_steps'] <= 10
    assert (case / 'physical-journal' / row['input_path']).is_file()
    if row['returned']:
        assert type(row['actual_steps']) is int and 0 <= row['actual_steps'] <= row['maximum_steps']
    else:
        unresolved.append(dict(path=str(path), role=row['role'], episode_id=row['episode_id'],
            maximum_steps=row['maximum_steps']))
    if row['episode_id'] not in completed_ids:
        extra.append(dict(path=str(path), role=row['role'], episode_id=row['episode_id'],
            returned=row['returned'], action_charge=row['actual_steps'] if row['returned'] else row['maximum_steps']))
        continue
    journal_rows.append(row)
    value = totals[row['role']]
    value['known_actions' if row['returned'] else 'uncertain_actions_ceiling'] += (
        row['actual_steps'] if row['returned'] else row['maximum_steps'])
    value['reset_attempts'] += row['kind'] == 'reset'
    if row['returned']:
        value['primitive_steps' if row['kind'] == 'step' else 'settling_steps'] += row['actual_steps']
for row in (journal_rows[0], journal_rows[-1]):
    for field in ('input', 'outcome'):
        if field + '_path' in row:
            sampled_rpc.append(file_check(case / 'physical-journal' / row[field + '_path'], row[field + '_sha256']))

receipts = [row for panel in data['panels'] for row in panel['receipts']] + data['training']['receipts']
assert all((case / row['trace']).is_file() for row in receipts)
chosen = [native['receipts'][0], data['panels'][1]['receipts'][0], data['training']['receipts'][-1]]
traces = [file_check(case / row['trace'], row['trace_sha256']) for row in chosen]
model = data['model']
metrics = data['training']['receipts'][-1]['update_metrics']
value = dict(format='task57_due_saved_metadata_check_v1', checked_unix=now,
    checked_shanghai=datetime.datetime.fromtimestamp(now, datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    job_id='job-iudscqfntb4e', source_commit='d001556780c78893cb7ad0310ec46e5d817e5698',
    grant_sha256='48acb746fb1a2c087671228c5ca440f871546a43b8f70f1589863ada46a7a9ec',
    phase=data['phase'], completed=data['completed'], exit_code=(root/'exit-code.txt').read_text().strip()
    if (root/'exit-code.txt').is_file() else None,
    index_snapshot_sha256=hashlib.sha256(index_bytes).hexdigest(),
    training={key:data['training'][key] for key in ('episodes', 'chunks', 'primitive_steps', 'settling_steps')},
    training_successes=sum(row['success'] for row in data['training']['receipts']),
    optimizer_updates=data['target_optimizer_updates'], replay_rows=data['replay_rows'],
    last_metrics=metrics, last_metrics_finite=all(math.isfinite(x) for x in metrics.values()),
    all_in_action_charge=data['total_action_charge'], cost_snapshot=data['costs'],
    completed_episode_journal_costs=totals, completed_costs_match=totals == data['costs'],
    completed_charge_matches=sum(v['known_actions'] + v['uncertain_actions_ceiling'] for v in totals.values()) == data['total_action_charge'],
    live_extra_journal_rows=extra, live_unresolved_intents=unresolved,
    referenced_completed_traces=len(receipts), sampled_trace_checks=traces, sampled_rpc_checks=sampled_rpc,
    latest_recovery=file_check(case / model['path'], model['sha256']), panels=panels,
    output_bytes=data['output_bytes'], maximum_output_bytes=data['maximum_output_bytes'],
    worker=dict(maximum_gpu_seconds=clock['worker_seconds'], attempts=len(clock['attempts']),
        reservation_seconds=sum(row['reserved_seconds'] for row in clock['attempts']),
        active_elapsed_seconds=now-clock['attempts'][-1]['started_unix'], active_elapsed_is_final_charge=False),
    next_due_shanghai='2026-10-31T14:56:00+08:00',
    limits='Native10/10 is a small-bank ceiling. Shorter query lengths are descriptive and do not establish training or deployment sample-efficiency gains. No failure/recovery examples have appeared if training successes equal episodes.')
print(json.dumps(value, indent=2))
