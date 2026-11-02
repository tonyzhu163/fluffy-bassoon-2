"""Six retained videos; three short native-task rewrites and verbatim baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

RUN='robocasa-flora-rubric-short-sample-20261103-r1'
PREVIOUS='robocasa-flora-rubric-failure-sample-20261102-r1'
ROOT=Path('/mowenjie/fluffy-bassoon-2')
OUT=ROOT/'outputs'/RUN
ARMS=('revision1_1','revision1_6','revision1_7','revision1_8')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def task_text(item,criteria):
    if 'task_template' in criteria:
        import re
        side=re.fullmatch(r'Open the (left|right) drawer\.',item['task']).group(1)
        return criteria['task_template'].format(side=side)
    return item['task']+'\n\nCompletion criteria:\n'+criteria['shared_criteria']+'\n\n'+criteria['reward_training_task_criteria'][item['task_key']]

def prepare(code):
    assert platform.system()=='Linux'
    OUT.mkdir(exist_ok=False)
    source=ROOT/'outputs'/PREVIOUS/'sample/PUBLIC-VIDEOS.json'
    items=[i for i in read(source) if i['task_key']=='OpenDrawer' and any('-case-'+n+'-' in i['id'] for n in ('000112','000064','000128','000072','000264','000144'))];assert len(items)==6
    assert all(set(i)=={'id','task_key','task','horizon','videos'} for i in items)
    assert len({i['id'] for i in items})==6
    for item in items:
        for v in item['videos']:assert sha(v['path'])==v['sha256']
    write(OUT/'PUBLIC-VIDEOS.json',items)
    from rewarder.batch_utils import core
    criteria=[read(code/'outputs'/RUN/(a+'.json')) for a in ARMS]
    assert sha(code/'outputs'/RUN/'revision1_1.json')==sha(code/'outputs'/PREVIOUS/'revision1_1.json')
    assert all('{side}' in c['task_template'] for c in criteria[1:])
    assert all('52%' not in str(c) or 'end stop' in str(c) for c in criteria[1:])
    for c in criteria:
        messages=core.terminal_messages(task_text(items[0],c),False)
        assert len(messages)==2 and sum(x['type']=='video' for x in messages[1]['content'])==1
        assert 'predict a score from 1 to 5' in messages[0]['content']
        assert 'CloseDrawer' not in str(messages)
    write(OUT/'CPU-CHECKS.json',dict(completed=True,videos=6,labels=24,public_source_sha256=sha(source),
        public_videos_sha256=sha(OUT/'PUBLIC-VIDEOS.json'),baseline_byte_identical=True,baseline_verbatim_and_candidates_direct_task_rewrites=True,
        official_graded_template=True,one_video_placeholder=True,close_drawer_reference_inactive=True,
        simulator_calls=0,robot_actions=0))
    print(json.dumps(read(OUT/'CPU-CHECKS.json')),flush=True)

def worker(code):
    assert platform.system()=='Linux' and read(OUT/'CPU-CHECKS.json')['completed']
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    import torch
    gpu=verify_selected_cuda(torch)
    for k in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE'):os.environ.pop(k,None)
    inputs=read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
    assert sha(inputs['model_receipt']['path'])==inputs['model_receipt']['sha256']
    receipt=read(inputs['model_receipt']['path']);assert receipt['completed'] and receipt['model_revision']==inputs['model_revision']
    assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip()==inputs['official_code_commit']
    items=read(OUT/'PUBLIC-VIDEOS.json');assert len(items)==6
    assert sha(OUT/'PUBLIC-VIDEOS.json')==read(OUT/'CPU-CHECKS.json')['public_videos_sha256']
    for i in items:
        for v in i['videos']:assert sha(v['path'])==v['sha256']
    criteria={a:read(code/'outputs'/RUN/(a+'.json')) for a in ARMS}
    directory=OUT/'scoring';directory.mkdir(exist_ok=False)
    write(directory/'INPUTS.json',dict(public_videos_sha256=sha(OUT/'PUBLIC-VIDEOS.json'),
        criteria_sha256={a:sha(code/'outputs'/RUN/(a+'.json')) for a in ARMS},
        videos=6,labels=24,truth_read=False,source_commit=os.environ['TASK_CODE_PIN']))
    from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
    before=time.time();engine=TerminalWorker(inputs['model'],batch_size=8,gpu_memory_utilization=.78)
    core,backend=engine.api.core,engine.api.engine
    jobs={a:[dict(id=i['id']+'/'+a,task=task_text(i,criteria[a]),videos=[v['path'] for v in i['videos']]) for i in items] for a in ARMS}
    prompts=[]
    for arm in ARMS:
        for j in jobs[arm]:
            messages=core.terminal_messages(j['task'],False)
            prompts.append(dict(id=j['id'],messages=messages,rendered_sha256=hashlib.sha256(backend.render(engine.processor,messages,thinking=False).encode()).hexdigest()))
    write(directory/'PROMPTS.json',prompts)
    write(directory/'ENGINE-READY.json',dict(gpu_uuid=gpu,model_load_count=1,cold_seconds=time.time()-before,batch_size=8,temperature=0))
    completed=0
    for arm in ARMS:
        for offset in range(0,6,8):
            assert time.time()<int(os.environ['TASK_STARTED'])+750,'Finite sample deadline'
            batch=jobs[arm][offset:offset+8];path=directory/f'{arm}-{offset:02d}.jsonl';start=time.time()
            rows=engine.score_jobs(batch,path)
            assert [r['id'] for r in rows]==[j['id'] for j in batch] and all(r['sampled_frames']==64 for r in rows)
            completed+=len(rows)
            write(path.with_suffix('.receipt.json'),dict(completed=True,labels=len(rows),sha256=sha(path),seconds=time.time()-start))
            print(json.dumps(dict(labels_completed=completed,labels_total=24,arm=arm)),flush=True)
    write(directory/'COMPLETED.json',dict(completed=True,videos=6,labels=completed,model_load_count=1,robot_actions=0,optimizer_steps=0))

def summary():
    assert platform.system()=='Linux'
    directory=OUT/'scoring';assert read(directory/'COMPLETED.json')['labels']==24
    rows={}
    for p in directory.glob('*.jsonl'):
        assert read(p.with_suffix('.receipt.json'))['sha256']==sha(p)
        for line in p.read_text().splitlines():
            r=json.loads(line);assert r['id'] not in rows;rows[r['id']]=r
    assert len(rows)==24
    previous=read(ROOT/'outputs'/PREVIOUS/'PAIRED-SAMPLE.json');pairs=[]
    for old in previous:
        if old['id']+'/'+ARMS[0] not in rows:continue
        uid=old['id'];pairs.append(dict(id=uid,truth=old['truth'],instruction=old['instruction'],
            historical_revision1_1=old['labels']['revision1_1_grade'],
            labels={a:rows[uid+'/'+a] for a in ARMS}))
    assert len(pairs)==6 and sum(p['truth'] for p in pairs)==4
    metrics={}
    for arm in ARMS:
        metrics[arm]=dict(false_successes=sum(not p['truth'] and p['labels'][arm]['score']==5 for p in pairs),
            success_controls_retained=sum(p['truth'] and p['labels'][arm]['score']==5 for p in pairs),
            baseline_false_successes_corrected=sum(not p['truth'] and p['labels'][ARMS[0]]['score']==5 and p['labels'][arm]['score']<5 for p in pairs),
            new_false_successes=sum(not p['truth'] and p['labels'][ARMS[0]]['score']<5 and p['labels'][arm]['score']==5 for p in pairs),
            baseline_success_controls_lost=sum(p['truth'] and p['labels'][ARMS[0]]['score']==5 and p['labels'][arm]['score']<5 for p in pairs),
            baseline_success_controls_gained=sum(p['truth'] and p['labels'][ARMS[0]]['score']<5 and p['labels'][arm]['score']==5 for p in pairs))
    write(OUT/'PAIRED-SAMPLE.json',pairs)
    write(OUT/'RESULT.json',dict(completed=True,videos=6,failures=2,success_controls=4,labels=24,metrics=metrics,
        baseline_hard_grade_drift=sum(p['historical_revision1_1']['score']!=p['labels'][ARMS[0]]['score'] for p in pairs),
        development_only=True,representative_accuracy=False,first_success_qualified=False,production_reward_qualified=False))
    print(json.dumps(read(OUT/'RESULT.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','summary']);p.add_argument('--code',type=Path);a=p.parse_args()
    {'prepare':prepare,'worker':worker,'summary':lambda _:summary()}[a.mode](a.code)
