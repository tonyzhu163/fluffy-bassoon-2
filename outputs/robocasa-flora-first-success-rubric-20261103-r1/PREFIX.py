"""Small causal-prefix comparison on six replay-qualified retained traces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT=Path('/mowenjie/fluffy-bassoon-2')
RUN='robocasa-flora-first-success-rubric-20261103-r1'
OUT=ROOT/'outputs'/RUN
ARMS=('revision1_1','revision1_6')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def task_text(item,c):
    if 'task_template' in c:
        import re
        return c['task_template'].format(side=re.fullmatch(r'Open the (left|right) drawer\.',item['task']).group(1))
    return item['task']+'\n\nCompletion criteria:\n'+c['shared_criteria']+'\n\n'+c['reward_training_task_criteria']['OpenDrawer']
def prepare(code):
    assert platform.system()=='Linux'
    items=read(ROOT/'outputs/robocasa-flora-rubric-short-sample-20261103-r1/PUBLIC-VIDEOS.json')
    assert len(items)==6
    old=ROOT/'outputs/robocasa-opendrawer-rubric-diagnosis-20261102-r1/physics-replay'
    public=[];private=[]
    for item in items:
        eid=item['id'].split('/')[-1]
        p=(OUT/'physics-replay' if any('-case-'+n+'-' in eid for n in ('000128','000072','000264')) else old)/(eid+'.json')
        trace=read(p);assert trace['qualified'] and trace['max_state_error']<1e-4 and trace['controls']==750
        first=trace['first_success_control']
        if first is None:
            assert '-case-000112-' in eid
            controls=list(range(50,751,50))
        else:
            before=max(1,((first-1)//50)*50)
            after=((first+49)//50)*50
            controls=sorted(set((before,after,min(750,after+50),750)))
        for control in controls:
            uid=item['id']+'/prefix-'+str(control)
            j=dict(item,id=uid,prefix_controls=control,initial_frame_included=True);public.append(j)
            s=trace['states'][control]
            private.append(dict(id=uid,episode_id=item['id'],control=control,native_first_success=first,
                native_current_success=s['success'],native_ever_success=first is not None and control>=first,
                normalized_opening=s['normalized_opening'][s['target']]['door'],replay_file=str(p),replay_sha256=sha(p)))
        for v in item['videos']:assert sha(v['path'])==v['sha256']
    assert len(public)<=35 and len(public)==len(private)
    write(OUT/'PUBLIC-JOBS.json',public);write(OUT/'PREFIX-REFERENCE.json',private)
    from rewarder.batch_utils import core
    for arm in ARMS:
        c=read(code/'outputs'/RUN/(arm+'.json'))
        for item in items:
            text=task_text(item,c);messages=core.terminal_messages(text,False)
            assert len(messages)==2 and sum(b['type']=='video' for b in messages[1]['content'])==1
            assert 'predict a score from 1 to 5' in messages[0]['content']
    write(OUT/'CPU-CHECKS.json',dict(completed=True,videos=6,prefixes=len(public),labels=len(public)*2,
        public_jobs_sha256=sha(OUT/'PUBLIC-JOBS.json'),no_truth_in_public_jobs=True,all_replays_qualified=True,
        official_graded_template=True,simulator_calls=0,robot_actions=0))
    print(json.dumps(read(OUT/'CPU-CHECKS.json')),flush=True)
def worker(code):
    assert platform.system()=='Linux'
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    import torch
    gpu=verify_selected_cuda(torch)
    for k in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE'):os.environ.pop(k,None)
    inputs=read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
    assert sha(inputs['model_receipt']['path'])==inputs['model_receipt']['sha256']
    assert read(inputs['model_receipt']['path'])['completed']
    assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip()==inputs['official_code_commit']
    checks=read(OUT/'CPU-CHECKS.json');assert checks['completed'] and sha(OUT/'PUBLIC-JOBS.json')==checks['public_jobs_sha256']
    items=read(OUT/'PUBLIC-JOBS.json');assert len(items)<=35
    for item in items:
        assert set(item)=={'id','task_key','task','horizon','videos','prefix_controls','initial_frame_included'}
        for v in item['videos']:assert sha(v['path'])==v['sha256']
    criteria={a:read(code/'outputs'/RUN/(a+'.json')) for a in ARMS}
    d=OUT/'scoring';d.mkdir(exist_ok=False)
    write(d/'INPUTS.json',dict(source_commit=os.environ['TASK_CODE_PIN'],public_jobs_sha256=sha(OUT/'PUBLIC-JOBS.json'),
        criteria_sha256={a:sha(code/'outputs'/RUN/(a+'.json')) for a in ARMS},labels=len(items)*2,truth_read=False))
    from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
    start=time.time();engine=TerminalWorker(inputs['model'],batch_size=8,gpu_memory_utilization=.78)
    write(d/'ENGINE-READY.json',dict(gpu_uuid=gpu,cold_seconds=time.time()-start,model_load_count=1,batch_size=8,temperature=0))
    jobs={a:[dict(id=i['id']+'/'+a,task=task_text(i,criteria[a]),videos=[v['path'] for v in i['videos']],
        prefix_controls=i['prefix_controls'],initial_frame_included=True) for i in items] for a in ARMS}
    core,backend=engine.api.core,engine.api.engine
    write(d/'PROMPTS.json',[dict(id=j['id'],messages=core.terminal_messages(j['task'],False),
        observed_through=j['prefix_controls']) for a in ARMS for j in jobs[a]])
    completed=0
    for a in ARMS:
        for offset in range(0,len(items),8):
            assert time.time()<int(os.environ['TASK_STARTED'])+780,'Finite prefix batch deadline'
            batch=jobs[a][offset:offset+8];p=d/f'{a}-{offset:02d}.jsonl';begin=time.time();rows=engine.score_jobs(batch,p)
            assert [r['id'] for r in rows]==[j['id'] for j in batch]
            assert all(r['sampled_frames']==64 and r['observed_through']==j['prefix_controls'] for r,j in zip(rows,batch))
            write(p.with_suffix('.receipt.json'),dict(completed=True,labels=len(rows),sha256=sha(p),seconds=time.time()-begin))
            completed+=len(rows);print(json.dumps(dict(labels_completed=completed,labels_total=len(items)*2,arm=a)),flush=True)
    write(d/'COMPLETED.json',dict(completed=True,labels=completed,videos=6,prefixes=len(items),robot_actions=0,optimizer_steps=0))
def summary():
    checks=read(OUT/'CPU-CHECKS.json');d=OUT/'scoring';assert read(d/'COMPLETED.json')['labels']==checks['labels']
    rows={}
    for p in d.glob('revision*.jsonl'):
        assert read(p.with_suffix('.receipt.json'))['sha256']==sha(p)
        for line in p.read_text().splitlines():
            x=json.loads(line);assert x['id'] not in rows;rows[x['id']]=x
    ref=read(OUT/'PREFIX-REFERENCE.json');assert len(rows)==len(ref)*2
    paired=[dict(r,labels={a:rows[r['id']+'/'+a] for a in ARMS}) for r in ref];write(OUT/'PAIRED-SAMPLE.json',paired)
    metrics={}
    for a in ARMS:
        cases=[]
        for eid in sorted({r['episode_id'] for r in paired}):
            rr=sorted([r for r in paired if r['episode_id']==eid],key=lambda r:r['control'])
            actual=rr[0]['native_first_success'];positives=[r['control'] for r in rr if r['labels'][a]['score']==5]
            first=positives[0] if positives else None
            cases.append(dict(id=eid,native_first_success=actual,first_sampled_full_score=first,
                premature_or_wrong_target=first is not None and (actual is None or first<actual),
                acceptance_delay_controls=None if first is None or actual is None else first-actual,
                positive_prefixes=positives,probes=[r['control'] for r in rr]))
        metrics[a]=dict(pre_success_or_wrong_target_full_scores=sum(r['labels'][a]['score']==5 and not r['native_ever_success'] for r in paired),
            current_state_false_full_scores=sum(r['labels'][a]['score']==5 and not r['native_current_success'] for r in paired),
            post_success_full_scores=sum(r['labels'][a]['score']==5 and r['native_ever_success'] for r in paired),cases=cases)
    write(OUT/'RESULT.json',dict(completed=True,videos=6,prefixes=len(ref),labels=len(rows),metrics=metrics,
        development_only=True,representative_accuracy=False,first_success_qualified=False,production_reward_qualified=False))
    print(json.dumps(read(OUT/'RESULT.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','summary']);p.add_argument('--code',type=Path);a=p.parse_args()
    {'prepare':prepare,'worker':worker,'summary':lambda _:summary()}[a.mode](a.code)
