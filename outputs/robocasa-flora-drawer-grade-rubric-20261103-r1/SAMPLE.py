"""Matched task-specific visual criterion, with and without explicit grade anchors."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = 'robocasa-flora-drawer-grade-rubric-20261103-r1'
OLD = ROOT/'outputs/robocasa-flora-drawer-exact-threshold-20261103-r1'
OUT = ROOT/'outputs'/RUN
ARMS = ('visual_criterion', 'visual_grades')
COMPARISONS = ('revision1_6', *ARMS)
ANCHORS = (('000112', 300), ('000128', 400), ('000144', 250), ('000064', 550))
NEW = 94

def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')
def frozen_inputs(code):
    for path, digest in read(code/'outputs'/RUN/'EXPECTED-INPUTS.json').items():
        assert sha(path) == digest, path
    assert read(OLD/'terminal-charge.json')['exit_code'] == 0
    assert all(a['equal'] for a in read(OLD/'RESULT.json')['anchors'])
def side(task): return re.fullmatch(r'Open the (left|right) drawer\.', task).group(1)
def is_anchor(item):
    return any('-case-'+case+'-' in item['id'] and item['prefix_controls'] == control for case, control in ANCHORS)
def criteria_for(code): return {a: read(code/'outputs'/RUN/(a+'.json')) for a in COMPARISONS}
def jobs_for(items, criteria):
    jobs = {}
    for arm in ('repeat_both', *ARMS):
        chosen = [i for i in items if is_anchor(i)] if arm == 'repeat_both' else items
        jobs[arm] = [dict(id=i['id']+'/'+arm,
            task=criteria['revision1_6']['task_template'].format(side=side(i['task'])) if arm == 'repeat_both' else i['task'],
            videos=[v['path'] for v in i['videos']], prefix_controls=i['prefix_controls'],
            initial_frame_included=True) for i in chosen]
    assert [len(j) for j in jobs.values()] == [4,45,45]
    return jobs
def messages_for(original, criterion, task, binary=False):
    assert binary is False
    messages = original(task, False)
    if criterion['criteria_id'] == 'revision1_6': return messages
    system = messages[0]['content']
    assert 'terminal-success' in system and 'the complete trajectory video' in system
    messages[0]['content'] = system.replace('terminal-success', 'first-success').replace(
        'the complete trajectory video', 'the observed causal video prefix')
    first, last = messages[1]['content'][0], messages[1]['content'][-1]
    assert first['type'] == 'text' and 'Complete trajectory video:' in first['text']
    assert last['text'] == 'Evaluate the terminal success of the task.'
    first['text'] = first['text'].replace('Complete trajectory video:', 'Observed video prefix:')
    last['text'] = 'Evaluate whether the task has succeeded at any time in the observed video prefix.'
    rubric = criterion['success_rubric'].format(side=side(task))
    messages[0]['content'] += '\n\nTask-specific success rubric:\n'+rubric
    if criterion['grade_anchors']:
        messages[0]['content'] += '\n\nGrade meanings:\n'+criterion['grade_anchors']
    return messages
def prepared_for(original, jobs, criteria):
    return [dict(id=j['id'], messages=messages_for(original,
        criteria['revision1_6' if arm == 'repeat_both' else arm], j['task'])) for arm in jobs for j in jobs[arm]]

def prepare(code):
    assert platform.system() == 'Linux'
    frozen_inputs(code)
    OUT.mkdir(exist_ok=False)
    items = read(OLD/'PUBLIC-JOBS.json')
    reference = read(OLD/'PREFIX-REFERENCE.json')
    previous = read(OLD/'CPU-CHECKS.json')
    assert previous['completed'] and len(items) == len(reference) == 45
    assert len({r['episode_id'] for r in reference}) == 10
    assert {i['id'] for i in items} == {r['id'] for r in reference}
    for ref in reference:
        assert sha(ref['replay_file']) == ref['replay_sha256']
        replay = read(ref['replay_file'])
        assert replay['qualified'] and replay['first_success_control'] == ref['native_first_success']
        assert replay['states'][ref['control']]['success'] == ref['native_current_success']
        assert ref['native_ever_success'] == (ref['native_first_success'] is not None and ref['control'] >= ref['native_first_success'])
    criteria = criteria_for(code)
    assert criteria['visual_criterion']['success_rubric'] == criteria['visual_grades']['success_rubric']
    assert criteria['visual_criterion']['grade_anchors'] == ''
    assert criteria['visual_grades']['grade_anchors']
    from rewarder.batch_utils import core
    from fluffy_bassoon.meta_rl.flora_terminal_worker import _PrefixCollage
    video_checks, expected = {}, {}
    for item in items:
        assert set(item) == {'id','task_key','task','horizon','videos','prefix_controls','initial_frame_included'}
        for video in item['videos']:
            if video['path'] not in video_checks:
                assert sha(video['path']) == video['sha256']
                reader = core.Collage([video['path']])
                try:
                    assert reader.num_frames == 751 and reader.fps == 20.
                    video_checks[video['path']] = dict(sha256=video['sha256'],frames=reader.num_frames,fps=reader.fps)
                finally: reader.release()
        indices = core.terminal_frame_indices(item['prefix_controls']+1, 20.)
        assert min(indices) >= 0 and max(indices) <= item['prefix_controls']
        expected[item['id']] = len(indices)
    assert len(video_checks) == 30
    probes = []
    for case, control in (('000112',50),('000112',100),('000112',750),('000128',400),('000256',750),('000376',750)):
        item = next(i for i in items if '-case-'+case+'-' in i['id'] and i['prefix_controls'] == control)
        reader = _PrefixCollage(core.Collage([v['path'] for v in item['videos']]), control)
        try:
            indices = core.terminal_frame_indices(reader.num_frames, reader.fps)
            assert len(reader.frames(indices)) == expected[item['id']]
            probes.append(dict(id=item['id'],sampled_frames=len(indices),max_index=max(indices)))
        finally: reader.release()
    terminal = code/'src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'
    original = subprocess.check_output(['git','-C',str(code),'show','17755d7a2afecf8eccae00e301933f9bc216a726:src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'])
    assert sha(terminal) == hashlib.sha256(original).hexdigest()
    recovered = [p['labels']['revision1_6'] for p in read(OLD/'PAIRED-SAMPLE.json')]
    assert len(recovered) == len({r['id'] for r in recovered}) == 45
    by_id = {i['id']:i for i in items}
    for row in recovered:
        item = by_id[row['id'].rsplit('/',1)[0]]
        assert row['task'] == criteria['revision1_6']['task_template'].format(side=side(item['task']))
        assert row['observed_through'] == item['prefix_controls'] and row['sampled_frames'] == expected[item['id']]
    jobs = jobs_for(items, criteria)
    prepared = prepared_for(core.terminal_messages, jobs, criteria)
    assert len(prepared) == NEW
    for prompt in prepared:
        arm = prompt['id'].rsplit('/',1)[1]
        message = prompt['messages']
        assert len(message) == 2 and sum(b['type']=='video' for b in message[1]['content']) == 1
        if arm != 'repeat_both':
            item = by_id[prompt['id'].rsplit('/',1)[0]]
            assert message[1]['content'][0]['text'].startswith('Task: '+item['task']+'\n')
            assert ('Grade meanings:' in message[0]['content']) == (arm == 'visual_grades')
            assert '52.25%' in message[0]['content']
    write(OUT/'PUBLIC-JOBS.json', items)
    write(OUT/'PREFIX-REFERENCE.json', reference)
    write(OUT/'RECOVERED.json', dict(rows=recovered,source_file=str(OLD/'PAIRED-SAMPLE.json'),source_sha256=sha(OLD/'PAIRED-SAMPLE.json')))
    write(OUT/'PREPARED-PROMPTS.json', prepared)
    write(OUT/'GEOMETRY.json', read(OLD/'GEOMETRY.json'))
    write(OUT/'CPU-CHECKS.json', dict(completed=True,videos=10,prefixes=45,new_outputs=NEW,recovered_labels=45,
        public_sha256=sha(OUT/'PUBLIC-JOBS.json'),recovered_sha256=sha(OUT/'RECOVERED.json'),
        prompts_sha256=sha(OUT/'PREPARED-PROMPTS.json'),criteria_sha256={a:sha(code/'outputs'/RUN/(a+'.json')) for a in criteria},
        expected_sampled_frames=expected,actual_frame_probes=probes,video_checks=video_checks,
        unchanged_terminal_worker_sha256=sha(terminal),geometry_sha256=sha(OUT/'GEOMETRY.json'),
        custom_system_grading=True,native_task_preserved=True,official_score_grammar_unchanged=True,
        original_six_geometry_frozen=True,policy_controls=0,motor_calls=0,optimizer_steps=0))
    print(json.dumps(dict(completed=True,videos=10,prefixes=45,new_outputs=NEW,recovered_labels=45,
        prompts_sha256=sha(OUT/'PREPARED-PROMPTS.json'),actual_frame_probes=probes)),flush=True)

def worker(code):
    assert platform.system() == 'Linux'
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    import torch
    gpu = verify_selected_cuda(torch)
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE'): os.environ.pop(key,None)
    inputs = read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
    assert sha(inputs['model_receipt']['path']) == inputs['model_receipt']['sha256'] and read(inputs['model_receipt']['path'])['completed']
    assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip() == inputs['official_code_commit']
    checks = read(OUT/'CPU-CHECKS.json')
    assert checks['completed'] and sha(OUT/'GEOMETRY.json') == checks['geometry_sha256']
    for filename,key in (('PUBLIC-JOBS.json','public_sha256'),('RECOVERED.json','recovered_sha256'),('PREPARED-PROMPTS.json','prompts_sha256')):
        assert sha(OUT/filename) == checks[key]
    criteria = criteria_for(code)
    assert all(sha(code/'outputs'/RUN/(a+'.json')) == checks['criteria_sha256'][a] for a in criteria)
    for path,meta in checks['video_checks'].items(): assert sha(path) == meta['sha256']
    jobs = jobs_for(read(OUT/'PUBLIC-JOBS.json'),criteria)
    dest = OUT/'scoring'
    dest.mkdir(exist_ok=False)
    write(dest/'INPUTS.json',dict(source_commit=os.environ['TASK_CODE_PIN'],model=inputs['model'],
        official_code_commit=inputs['official_code_commit'],criteria_sha256=checks['criteria_sha256'],
        truth_read=False,custom_messages=True,native_task_preserved=True,new_outputs=NEW,recovered_labels=45))
    from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
    start=time.time()
    engine=TerminalWorker(inputs['model'],batch_size=8,gpu_memory_utilization=.78)
    write(dest/'ENGINE-READY.json',dict(gpu_uuid=gpu,cold_seconds=time.time()-start,model_load_count=1,batch_size=8,temperature=0))
    original_messages,original_render = engine.api.core.terminal_messages,engine.api.engine.render
    write(dest/'PROMPTS.json',prepared_for(original_messages,jobs,criteria))
    assert sha(dest/'PROMPTS.json') == checks['prompts_sha256']
    completed=0
    for arm,todo in jobs.items():
        criterion=criteria['revision1_6' if arm=='repeat_both' else arm]
        engine.api.core.terminal_messages=lambda task,binary=False,criterion=criterion: messages_for(original_messages,criterion,task,binary)
        for offset in range(0,len(todo),8):
            assert time.time() < int(os.environ['TASK_STARTED'])+780,'Finite batch deadline'
            batch=todo[offset:offset+8]
            path=dest/f'{arm}-{offset:02d}.jsonl'
            audit=[]
            begin=time.time()
            def audited_render(processor,messages,**kwargs):
                index=len(audit)
                assert messages == messages_for(original_messages,criterion,batch[index]['task'])
                prompt=original_render(processor,messages,**kwargs)
                assert isinstance(prompt,str)
                audit.append(dict(id=batch[index]['id'],rendered_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    messages_sha256=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()))
                return prompt
            engine.api.engine.render=audited_render
            try: rows=engine.score_jobs(batch,path)
            finally: engine.api.engine.render=original_render
            assert len(rows)==len(audit)==len(batch) and [r['id'] for r in rows]==[j['id'] for j in batch]
            assert all(r['observed_through']==j['prefix_controls'] and r['sampled_frames']==checks['expected_sampled_frames'][r['id'].rsplit('/',1)[0]] for r,j in zip(rows,batch))
            write(path.with_suffix('.prompts.json'),audit)
            write(path.with_suffix('.receipt.json'),dict(completed=True,labels=len(rows),sha256=sha(path),prompts_sha256=sha(path.with_suffix('.prompts.json')),seconds=time.time()-begin))
            completed+=len(rows)
            print(json.dumps(dict(new_outputs_completed=completed,total=NEW,arm=arm)),flush=True)
    engine.api.core.terminal_messages=original_messages
    assert completed==NEW
    write(dest/'COMPLETED.json',dict(completed=True,new_outputs=NEW,candidate_labels=90,anchor_labels=4,recovered_labels=45,robot_actions=0,optimizer_steps=0))

def summary(code):
    path=code/'outputs'/OLD.name/'SAMPLE.py'
    spec=importlib.util.spec_from_file_location('previous_exact_comparison',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.RUN,module.OUT,module.ARMS,module.COMPARISONS,module.NEW=RUN,OUT,ARMS,COMPARISONS,NEW
    module.summary()

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','worker','summary'])
    parser.add_argument('--code',type=Path,required=True)
    args=parser.parse_args()
    {'prepare':prepare,'worker':worker,'summary':summary}[args.mode](args.code)
