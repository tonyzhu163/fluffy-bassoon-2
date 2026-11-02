"""Two rubric hypotheses on the same six retained, causal-prefix videos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = 'robocasa-flora-first-success-rubric-20261103-r3'
OLD = ROOT/'outputs/robocasa-flora-first-success-rubric-20261103-r2'
OUT = ROOT/'outputs'/RUN
ARMS = ('revision1_9', 'revision1_10')
REFERENCES = ('revision1_1', 'revision1_6')
ANCHORS = (('000112', 300), ('000128', 400), ('000144', 250), ('000064', 550))
def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
def task(item, criterion):
    side = re.fullmatch(r'Open the (left|right) drawer\.', item['task']).group(1)
    return criterion['task_template'].format(side=side)
def is_anchor(item):
    return any('-case-'+case+'-' in item['id'] and item['prefix_controls'] == control for case, control in ANCHORS)
def prepare(code):
    assert platform.system() == 'Linux'
    OUT.mkdir(exist_ok=False)
    frozen = read(code/'outputs'/RUN/'EXPECTED-INPUTS.json')
    for name, digest in frozen.items():
        assert sha(OLD/name) == digest, name
    assert read(OLD/'scoring/COMPLETED.json')['labels'] == 70
    assert read(OLD/'terminal-charge.json')['exit_code'] == 0
    assert read(OLD/'CPU-CHECKS.json')['all_replays_qualified']
    public = read(OLD/'PUBLIC-JOBS.json')
    reference = read(OLD/'PREFIX-REFERENCE.json')
    assert len(public) == len(reference) == 35
    assert len({item['id'].rsplit('/', 1)[0] for item in public}) == 6
    write(OUT/'PUBLIC-JOBS.json', public)
    write(OUT/'PREFIX-REFERENCE.json', reference)
    from rewarder.batch_utils import core
    from fluffy_bassoon.meta_rl.flora_terminal_worker import _PrefixCollage
    expected = {}
    for item in public:
        assert set(item) == {'id', 'task_key', 'task', 'horizon', 'videos', 'prefix_controls', 'initial_frame_included'}
        for video in item['videos']:
            assert sha(video['path']) == video['sha256']
        indices = core.terminal_frame_indices(item['prefix_controls']+1, 20.)
        assert min(indices) >= 0 and max(indices) <= item['prefix_controls']
        expected[item['id']] = len(indices)
    probes = []
    for control in (50, 100, 750):
        reader = _PrefixCollage(core.Collage([v['path'] for v in public[0]['videos']]), control)
        try:
            indices = core.terminal_frame_indices(reader.num_frames, reader.fps)
            assert reader.source.num_frames == 751 and reader.fps == 20.
            assert len(reader.frames(indices)) == len(indices)
            probes.append(dict(controls=control, frames=len(indices), max_index=max(indices)))
        finally:
            reader.release()
    terminal = code/'src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'
    original = subprocess.check_output(['git', '-C', str(code), 'show',
        '17755d7a2afecf8eccae00e301933f9bc216a726:src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'])
    assert sha(terminal) == hashlib.sha256(original).hexdigest()
    criteria = {arm: read(code/'outputs'/RUN/(arm+'.json')) for arm in (*ARMS, 'revision1_6_repeat')}
    for criterion in criteria.values():
        for item in public:
            messages = core.terminal_messages(task(item, criterion), False)
            assert len(messages) == 2 and sum(block['type'] == 'video' for block in messages[1]['content']) == 1
            assert 'predict a score from 1 to 5' in messages[0]['content']
    rows = [pair['labels'][arm] for pair in read(OLD/'PAIRED-SAMPLE.json') for arm in REFERENCES]
    assert len(rows) == len({row['id'] for row in rows}) == 70
    byid = {item['id']: item for item in public}
    for row in rows:
        uid = row['id'].rsplit('/', 1)[0]
        assert row['sampled_frames'] == expected[uid] and row['observed_through'] == byid[uid]['prefix_controls']
        if row['id'].endswith('/revision1_6'):
            assert row['task'] == task(byid[uid], criteria['revision1_6_repeat'])
    write(OUT/'RECOVERED.json', dict(rows=rows, source_sha256=frozen['PAIRED-SAMPLE.json'], source_commit='17755d7a2afecf8eccae00e301933f9bc216a726'))
    assert sum(is_anchor(item) for item in public) == 4
    write(OUT/'CPU-CHECKS.json', dict(completed=True, videos=6, prefixes=35,
        public_jobs_sha256=sha(OUT/'PUBLIC-JOBS.json'), recovered_sha256=sha(OUT/'RECOVERED.json'),
        criteria_sha256={arm: sha(code/'outputs'/RUN/(arm+'.json')) for arm in criteria},
        expected_sampled_frames=expected, actual_frame_probes=probes, unchanged_terminal_worker_sha256=sha(terminal),
        recovered_labels=70, new_candidate_labels=70, new_anchor_labels=4,
        total_new_outputs=74, total_candidate_and_reference_labels=140,
        simulator_calls=0, motor_calls=0, optimizer_steps=0, no_truth_in_public_jobs=True))
    print(json.dumps(read(OUT/'CPU-CHECKS.json')), flush=True)
def worker(code):
    assert platform.system() == 'Linux'
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    import torch
    gpu = verify_selected_cuda(torch)
    for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'NPROC_PER_NODE'):
        os.environ.pop(key, None)
    inputs = read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
    assert sha(inputs['model_receipt']['path']) == inputs['model_receipt']['sha256']
    assert read(inputs['model_receipt']['path'])['completed']
    assert subprocess.check_output(['git', '-C', inputs['official_code'], 'rev-parse', 'HEAD'], text=True).strip() == inputs['official_code_commit']
    checks = read(OUT/'CPU-CHECKS.json')
    assert checks['completed'] and sha(OUT/'PUBLIC-JOBS.json') == checks['public_jobs_sha256']
    assert sha(OUT/'RECOVERED.json') == checks['recovered_sha256']
    items = read(OUT/'PUBLIC-JOBS.json')
    assert len(items) == 35
    for item in items:
        for video in item['videos']:
            assert sha(video['path']) == video['sha256']
    criteria = {arm: read(code/'outputs'/RUN/(arm+'.json')) for arm in (*ARMS, 'revision1_6_repeat')}
    assert all(sha(code/'outputs'/RUN/(arm+'.json')) == checks['criteria_sha256'][arm] for arm in criteria)
    dest = OUT/'scoring'
    dest.mkdir(exist_ok=False)
    write(dest/'INPUTS.json', dict(source_commit=os.environ['TASK_CODE_PIN'],
        model=inputs['model'], official_code_commit=inputs['official_code_commit'],
        public_jobs_sha256=checks['public_jobs_sha256'], criteria_sha256=checks['criteria_sha256'],
        truth_read=False, new_outputs=74, recovered_labels=70))
    from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
    start = time.time()
    engine = TerminalWorker(inputs['model'], batch_size=8, gpu_memory_utilization=.78)
    write(dest/'ENGINE-READY.json', dict(gpu_uuid=gpu, cold_seconds=time.time()-start,
        model_load_count=1, batch_size=8, temperature=0))
    jobs = {arm: [dict(id=item['id']+'/'+arm, task=task(item, criteria[arm]),
        videos=[v['path'] for v in item['videos']], prefix_controls=item['prefix_controls'],
        initial_frame_included=True) for item in items if arm != 'revision1_6_repeat' or is_anchor(item)]
        for arm in ('revision1_6_repeat', *ARMS)}
    write(dest/'PROMPTS.json', [dict(id=job['id'], messages=engine.api.core.terminal_messages(job['task'], False),
        observed_through=job['prefix_controls']) for arm in jobs for job in jobs[arm]])
    completed = 0
    for arm, todo in jobs.items():
        for offset in range(0, len(todo), 8):
            assert time.time() < int(os.environ['TASK_STARTED'])+780, 'Finite batch deadline'
            batch = todo[offset:offset+8]
            path = dest/f'{arm}-{offset:02d}.jsonl'
            begin = time.time()
            rows = engine.score_jobs(batch, path)
            assert [r['id'] for r in rows] == [job['id'] for job in batch]
            assert all(row['sampled_frames'] == checks['expected_sampled_frames'][row['id'].rsplit('/',1)[0]]
                and row['observed_through'] == job['prefix_controls'] for row, job in zip(rows, batch))
            write(path.with_suffix('.receipt.json'), dict(completed=True, labels=len(rows), sha256=sha(path), seconds=time.time()-begin))
            completed += len(rows)
            print(json.dumps(dict(new_outputs_completed=completed, total=74, arm=arm)), flush=True)
    assert completed == 74
    write(dest/'COMPLETED.json', dict(completed=True, new_outputs=completed,
        candidate_labels=70, anchor_labels=4, recovered_labels=70, videos=6, prefixes=35,
        robot_actions=0, optimizer_steps=0))
def summary():
    dest = OUT/'scoring'
    assert read(dest/'COMPLETED.json')['new_outputs'] == 74
    rows = {r['id']: r for r in read(OUT/'RECOVERED.json')['rows']}
    anchors = []
    for path in dest.glob('revision*.jsonl'):
        assert read(path.with_suffix('.receipt.json'))['sha256'] == sha(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['id'].endswith('/revision1_6_repeat'):
                old = rows[row['id'].rsplit('/',1)[0]+'/revision1_6']
                anchors.append(dict(id=row['id'], cached_score=old['score'], repeated_score=row['score'],
                    equal=row['score']==old['score'], full_score_equal=(row['score']==5)==(old['score']==5)))
            else:
                assert row['id'] not in rows
                rows[row['id']] = row
    assert len(rows) == 140 and len(anchors) == 4
    reference = read(OUT/'PREFIX-REFERENCE.json')
    paired = [dict(ref, labels={arm: rows[ref['id']+'/'+arm] for arm in (*REFERENCES,*ARMS)}) for ref in reference]
    write(OUT/'PAIRED-SAMPLE.json', paired)
    metrics = {}
    for arm in (*REFERENCES, *ARMS):
        cases = []
        for eid in sorted({r['episode_id'] for r in paired}):
            probes = sorted((r for r in paired if r['episode_id'] == eid), key=lambda r:r['control'])
            actual = probes[0]['native_first_success']
            positives = [r['control'] for r in probes if r['labels'][arm]['score'] == 5]
            first = positives[0] if positives else None
            cases.append(dict(id=eid, native_first_success=actual, first_sampled_full_score=first,
                premature_or_wrong_target=first is not None and (actual is None or first<actual),
                acceptance_delay_controls=None if first is None or actual is None else first-actual,
                positive_prefixes=positives))
        false = sum(r['labels'][arm]['score']==5 and not r['native_ever_success'] for r in paired)
        detected = sum(c['native_first_success'] is not None and c['first_sampled_full_score'] is not None
            and not c['premature_or_wrong_target'] and c['acceptance_delay_controls']<=100 for c in cases)
        metrics[arm] = dict(pre_success_or_wrong_target_full_scores=false,
            post_success_full_scores=sum(r['labels'][arm]['score']==5 and r['native_ever_success'] for r in paired),
            correctly_timed_successful_episodes=detected, cases=cases,
            passes_development_screen=false==0 and detected==5 and all(a['equal'] for a in anchors))
    write(OUT/'RESULT.json', dict(completed=True, videos=6, prefixes=35,
        labels=140, new_outputs=74, anchors=anchors, metrics=metrics,
        first_success_qualified=False, production_reward_qualified=False,
        development_only=True, independent_validation=False))
    print(json.dumps(read(OUT/'RESULT.json')), flush=True)
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare','worker','summary'])
    parser.add_argument('--code', type=Path)
    args=parser.parse_args()
    {'prepare':prepare, 'worker':worker, 'summary':lambda _:summary()}[args.mode](args.code)
