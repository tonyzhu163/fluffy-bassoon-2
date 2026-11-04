"""Independent allocated-Linux CPU audit of the completed replacement qualification."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-buttons-replacement-qualification-20261104-r1')
RUN = ROOT/'gpu-r1'
TASKS = ['TurnOnSinkFaucet', 'TurnOnElectricKettle']
INSTRUCTIONS = ['Turn on the sink faucet.',
                'Press down the lever to turn on the electric kettle.']
CAMERAS = ['robot0_agentview_left', 'robot0_eye_in_hand', 'robot0_agentview_right']


def read(p):
    return json.loads(p.read_text())


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def video_check(video):
    p = Path(video['path'])
    assert sha(p) == video['sha256']
    result = subprocess.run(['ffprobe', '-v', 'error', '-threads', '1',
        '-select_streams', 'v:0', '-count_frames', '-show_entries',
        'stream=nb_read_frames,width,height', '-of', 'json', str(p)],
        check=True, capture_output=True, text=True, timeout=120)
    assert not result.stderr.strip(), result.stderr
    stream, = json.loads(result.stdout)['streams']
    assert int(stream['nb_read_frames']) == 601
    assert stream['width'] == stream['height'] == 256
    return dict(path=str(p), sha256=video['sha256'], decoded_frames=601)


torch.set_num_threads(2)
clock = read(RUN/'clock.json')
assert clock['exit_code'] == 0 and clock['allocated_gpus'] == 2
result = read(RUN/'RESULT.json')
assert result['completed'] and result['episodes'] == 100 and result['policy_controls'] == 60000
assert result['optimizer_updates'] == 0
probe = read(RUN/'GPU-PROBE.json')
assert probe['status'] == 'passed' and probe['hostname'] == 'job-ltf38050md7y-master-0'
assert len(probe['devices']) == 2
case_rows = read(RUN/'CASE-RESULTS.json')
assert len(case_rows) == 100
videos, episodes, successes = [], [], {}
for rank, task in enumerate(TASKS):
    directory = RUN/f'rank-{rank}'
    plan = read(directory/'CASE-PLAN.json')
    assert len(plan['cases']) == 50 and len(plan['motor_seeds']) == 600
    first = final = native = 0
    for slot, case in enumerate(plan['cases']):
        assert case['task'] == task and case['slot'] == slot and case['task_index'] == rank
        assert case['native_seed'] == 730100 + 1000*rank + slot
        path = directory/'episodes'/f'{slot:03d}.pt'
        data = torch.load(path, map_location='cpu', weights_only=False)
        episode, records = data['episode'], data['records']
        assert data['case'] == case and episode['task'] == INSTRUCTIONS[rank]
        assert episode['completed'] and episode['horizon'] == 600 and len(records) == 12
        assert episode['success_termination_disabled'] and episode['raw_rgb_only']
        assert not episode['info_on_video'] and not episode['extra_info_on_video']
        assert episode['camera_order'] == CAMERAS and episode['frame_clock']['frame_count'] == 601
        assert [v['camera'] for v in episode['videos']] == CAMERAS
        for chunk, record in enumerate(records):
            seed = 172000000 + 100000*rank + 1000*slot + chunk
            assert record['noise_seed'] == seed == plan['motor_seeds'][12*slot+chunk]
            assert record['zero_residual_verified'] == (slot == 0)
            assert np.count_nonzero(record['control']) == 0
            assert np.array_equal(record['base_chunk'], record['executed_chunk'])
            assert np.isfinite(record['executed_chunk']).all()
            assert record['episode_start'] == (chunk == 0) and record['done'] == (chunk == 11)
        truth_path = directory/'monitoring'/(episode['id']+'.json')
        truth = read(truth_path)
        row, = [r for r in case_rows if r['episode_id'] == episode['id']]
        assert row['case'] == truth['case'] == case
        assert row['first_success'] == truth['simulator_success_once']
        assert row['end_success'] == truth['simulator_final_success']
        assert row['first_success_step'] == truth['first_success_step']
        first += int(row['first_success'])
        final += int(row['end_success'])
        native += int(row['first_success'] and row['first_success_step'] <= (450 if rank else 600))
        videos.extend(episode['videos'])
        episodes.append(dict(task=task, seed=case['native_seed'], path=str(path),
            sha256=sha(path), monitoring_sha256=sha(truth_path)))
    summary, = [r for r in result['tasks'] if r['task'] == task]
    assert summary['first_successes'] == first and summary['end_successes'] == final
    assert summary['first_successes_within_native_horizon'] == native
    assert summary['first_success_rate'] == first/50 and summary['end_success_rate'] == final/50
    successes[task] = first
chosen = next((task for task in TASKS if 3 <= successes[task] <= 35), None)
assert result['task_selection'] == chosen
journals = list((RUN/'actions').glob('rank-*/*.json'))
calls = [c for p in journals for c in read(p)['calls']]
assert len(journals) == 8 and calls and all(c['resolved'] for c in calls)
physical = sum(c['executed_or_conservative'] for c in calls)
assert physical == 60000
assert len(videos) == len({v['path'] for v in videos}) == 300
with ThreadPoolExecutor(max_workers=4) as pool:
    decoded = list(pool.map(video_check, videos))
video_receipt = ROOT/'VIDEO-AUDIT.json'
video_receipt.write_text(json.dumps(dict(completed=True, videos=decoded,
    decoded_frames=sum(r['decoded_frames'] for r in decoded)), indent=2)+'\n')
audit = dict(completed=True, checked_unix=time.time(), job='job-ltf38050md7y',
    source_commit=result['source_commit'], episodes=100, chunks=1200,
    policy_controls=60000, physical_controls=physical, unresolved_intents=0,
    videos=300, decoded_video_frames=180300, all_video_hashes_match=True,
    exact_case_noise_and_zero_native_action_identity=True,
    task_selection=chosen, first_successes=successes, SGD=0,
    GPU_process_seconds=clock['process_gpu_seconds'],
    video_audit_sha256=sha(video_receipt), episodes_manifest=episodes,
    raw_summary_sha256=sha(RUN/'RESULT.json'),
    journal_sha256={str(p):sha(p) for p in journals})
(ROOT/'FINAL-AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ('episodes_manifest','journal_sha256')}), flush=True)
