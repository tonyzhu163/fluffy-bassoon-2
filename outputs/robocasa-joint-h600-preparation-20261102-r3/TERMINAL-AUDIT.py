"""Read-only terminal evidence audit on allocated Linux CPUs; no model imports."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = ROOT / 'outputs/robocasa-joint-h600-preparation-20261102-r3'
CODE = ROOT / 'worktrees/joint-h600-preparation-20261102-r3'
SOURCE = '0f62915062e02cac45ed1158c860c6f4b31a1da6'
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()

def bound(ref):
    assert sha(ref['path']) == ref['sha256'], ref['path']
    return read(ref['path'])

assert subprocess.check_output(['git', '-C', str(CODE), 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
assert not subprocess.check_output(['git', '-C', str(CODE), 'status', '--porcelain'], text=True).strip()
warmup = read(RUN / 'warmup-episodes.json')
assert len(warmup) == len({e['id'] for e in warmup}) == 8
video_checks = []
for episode in warmup:
    assert episode['horizon'] == episode['frame_clock']['physical_steps'] == 600
    assert episode['frame_clock']['frame_count'] == 601
    assert len(episode['videos']) == 3
    assert [v['camera'] for v in episode['videos']] == episode['camera_order']
    for video in episode['videos']:
        assert sha(video['path']) == video['sha256']
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames,width,height,r_frame_rate', '-of', 'json', video['path']], text=True))
        stream = probe['streams'][0]
        assert int(stream['nb_frames']) == 601
        assert stream['width'] == stream['height'] == 256
        assert stream['r_frame_rate'] == '20/1'
        video_checks.append(dict(path=video['path'], sha256=video['sha256'], **stream))

headroom = read(RUN / 'HEADROOM-ALL-FOUR.json')
assert headroom['completed'] and headroom['required_free_mib'] == 21973
nodes = []
for rank in range(4):
    node = RUN / f'job/node-{rank:02d}'
    manifest = read(node / 'manifest.json')
    engine = bound(headroom['engines'][rank])
    terminal = read(node / 'node-terminal.json')
    clock = read(node / 'terminal-charge.json')
    scorer = node / f'scorer-{rank:02d}'
    receipt = read(scorer / 'warmup-scores.receipt.json')
    bound(receipt['model_receipt'])
    assert sha(receipt['output']['path']) == receipt['output']['sha256']
    jobs = [json.loads(s) for s in (scorer / 'warmup-scores.jobs.jsonl').read_text().splitlines()]
    jobs_digest = hashlib.sha256(json.dumps(jobs, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    assert jobs_digest == receipt['jobs_sha256']
    labels = [json.loads(s) for s in (scorer / 'warmup-scores.jsonl').read_text().splitlines()]
    assert len(labels) == 8 and {v['id'] for v in labels} == {e['id'] for e in warmup}
    assert receipt['completed'] and receipt['model_load_count'] == 1 and receipt['completed_batches'] == 1
    assert engine['source_commit'] == manifest['source_commit'] == SOURCE
    assert engine['gpu_uuid'] == manifest['devices'][0]['uuid']
    assert engine['completed'] and engine['warmup_videos'] == 8 and engine['robot_actions'] == 0
    assert engine['batch_size'] == 8 and engine['gpu_memory_utilization'] == .75
    assert engine['memory_snapshot']['free_mib'] >= 21973
    assert engine['code_commit'] == receipt['code_commit'] == '129a2ada74c526ee4692f3d1b111bea6ca19d5b7'
    assert engine['model_revision'] == receipt['model_revision'] == '031299e46e452fc0cedb4e0975eb9fa95fc37783'
    assert terminal['optimizer_updates'] == 0 and all(c is not None for c in terminal['child_exit_codes'])
    assert clock['status'] == 'finished' and not clock['slice_overrun'] and clock['exit_code'] == 1
    assert int((node / 'exit-code.txt').read_text()) == 1
    nodes.append(dict(manifest=manifest, engine=engine, terminal=terminal, clock=clock,
                      warmup_receipt=receipt, warmup_labels=labels))
assert len({n['engine']['gpu_uuid'] for n in nodes}) == 4
assert not list(RUN.rglob('episodes.json'))
assert not (RUN / 'actions').exists()
assert not (RUN / 'CALIBRATION-GATES.json').exists()
assert not (RUN / 'common').exists()
dispatch = list((RUN / 'job/dispatch').rglob('*.json'))
assert [p.name for p in dispatch] == ['shutdown.json']
assert read(dispatch[0])['completed'] is False
profile = read(RUN / 'job/node-00/native-OpenDrawer-profile.json')
loading = profile['loading_qualification']
assert not Path(loading['path']).exists()
tracked_loading = CODE / Path(loading['path']).relative_to(ROOT)
tracked_loading_check = dict(path=str(tracked_loading), exists=tracked_loading.exists())
if tracked_loading.exists():
    tracked_loading_check.update(sha256=sha(tracked_loading), matches_binding=sha(tracked_loading) == loading['sha256'])
charges = [n['clock']['charged_gpu_seconds'] for n in nodes]
assert charges == [531, 528, 528, 528]
print(json.dumps(dict(completed=True, audit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    audit_execution='Allocated Baige development Linux CPU; CUDA masked; standard library plus ffprobe, no model imports',
    audit_cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(), source_commit=SOURCE,
    failure='Missing pinned loading_qualification artifact at legacy PFS output path; failure before action journal or native policy load',
    missing_reference=loading, tracked_source_receipt=tracked_loading_check,
    nodes=nodes, warmup_video_checks=video_checks, warmup_unique_existing_episodes=8,
    warmup_replay_labels=32, new_independent_episodes=0, fresh_calibration_episodes=0,
    common_before_episodes=0, fresh_calibration_grades=0, physical_actions=0,
    optimizer_updates=0, journals='No action root, no native dispatch commands; none unresolved',
    common_checkpoint='Not produced', initial_policy_fingerprint='Not produced by this attempt',
    costs=dict(r3_gpu_seconds=sum(charges), r1_conservative_upper_gpu_seconds=864,
               r2_charged_gpu_seconds=289, combined_preparation_gpu_seconds=1153+sum(charges),
               combined_preparation_gpu_hours=(1153+sum(charges))/3600,
               authorized_total_gpu_seconds=26400, remaining_unspent_gpu_seconds=26400-1153-sum(charges)),
    next='Restore or rebind exact qualification reference, verify full input closure before engines; no successor job or GPU retry authorized by heartbeat'), indent=2))
