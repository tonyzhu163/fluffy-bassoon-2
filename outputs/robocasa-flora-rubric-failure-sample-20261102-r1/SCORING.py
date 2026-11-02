"""Finite twenty-video prompt pilot; private labels are read only in summary."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

RUN='robocasa-flora-rubric-failure-sample-20261102-r1'
ROOT=Path('/mowenjie/fluffy-bassoon-2')
OUT=ROOT/'outputs'/RUN
ARMS=('revision1_grade','revision1_1_grade','revision1_1_binary','revision1_2_grade','revision1_2_binary')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def task_text(item,criteria):
    return item['task']+'\n\nCompletion criteria:\n'+criteria['shared_criteria']+'\n\n'+criteria['reward_training_task_criteria'][item['task_key']]

def binary_messages(core,task):
    system=('You are a terminal-success reward model for robot trajectories.\n'
        'Given the task instruction, completion criteria and complete trajectory video, '
        'predict whether the task is completely successful at the end.\n'
        'Success 1 means every requested completion condition is satisfied. '
        'Success 0 means the task is not completely successful, including partial progress.\n'
        'Judge only what is visible in the trajectory and follow the requested output format exactly.\n'
        'Do not provide a rationale. Return exactly:\nSuccess: <0-1>')
    return core._messages(system,'Task: '+core._compact(task)+'\n\nComplete trajectory video:\n<video>\n\nEvaluate the terminal success of the task.')

def binary_batch(engine,jobs,path):
    core,backend=engine.api.core,engine.api.engine
    params=engine.api.SamplingParams(temperature=0.,max_tokens=16,logprobs=20,
        structured_outputs=engine.api.StructuredOutputsParams(regex=r'Success: [01]'))
    prepared=[];rows=[]
    try:
        for job in jobs:
            collage=core.Collage(job['videos']);prepared.append((job,collage,None))
            indices=core.terminal_frame_indices(collage.num_frames,collage.fps)
            prepared[-1]=(job,collage,indices)
        requests=[dict(prompt=backend.render(engine.processor,binary_messages(core,j['task']),thinking=False),
            multi_modal_data={'video':(c.frames(i),core.video_metadata(i,c.fps,c.num_frames))},
            mm_processor_kwargs={'do_sample_frames':False}) for j,c,i in prepared]
        outputs=engine.llm.generate(requests,params,use_tqdm=False)
        assert len(outputs)==len(jobs) and all(len(x.outputs)==1 for x in outputs)
        with path.open('x') as f:
            for (job,collage,indices),output in zip(prepared,outputs,strict=True):
                text=output.outputs[0].text;match=re.fullmatch(r'Success: ([01])',text.strip())
                if match is None:raise ValueError('Malformed binary label: '+repr(text))
                row=dict(id=job['id'],task=job['task'],success=int(match[1]),output=text,
                    sampled_frames=len(indices),duration_s=round(collage.duration_s,3),
                    probability=None,protocol='direct_binary_v1')
                backend.write_record(f,row);rows.append(row)
    finally:
        for _,collage,_ in prepared:collage.release()
    return rows

def worker(code):
    assert platform.system()=='Linux'
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    import torch
    gpu=verify_selected_cuda(torch)
    for k in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE'):os.environ.pop(k,None)
    inputs=read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
    assert sha(inputs['model_receipt']['path'])==inputs['model_receipt']['sha256']
    receipt=read(inputs['model_receipt']['path']);assert receipt['completed'] and receipt['model_revision']==inputs['model_revision']
    assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip()==inputs['official_code_commit']
    items=read(OUT/'sample/PUBLIC-VIDEOS.json');assert len(items)==20
    assert all(set(i)=={'id','task_key','task','horizon','videos'} for i in items)
    for item in items:
        for v in item['videos']:assert sha(v['path'])==v['sha256']
    criteria={name:read(code/'outputs'/RUN/(name+'.json')) for name in ('revision1','revision1_1','revision1_2')}
    assert sha(code/'outputs'/RUN/'revision1.json')==sha(code/'outputs/robocasa-flora-rubric-replay-20261102-r1/candidate.json')
    jobs={arm:[dict(id=i['id']+'/'+arm,task=task_text(i,criteria[arm.rsplit('_',1)[0]]),
        videos=[v['path'] for v in i['videos']]) for i in items] for arm in ARMS}
    directory=OUT/'scoring';directory.mkdir(exist_ok=False)
    write(directory/'INPUTS.json',dict(public_videos_sha256=sha(OUT/'sample/PUBLIC-VIDEOS.json'),
        criteria_sha256={k:sha(code/'outputs'/RUN/(k+'.json')) for k in criteria},
        videos=20,labels=100,truth_read=False,source_commit=os.environ['TASK_CODE_PIN']))
    from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
    before=time.time();engine=TerminalWorker(inputs['model'],batch_size=8,gpu_memory_utilization=.78)
    core,backend=engine.api.core,engine.api.engine
    prompts=[]
    for arm in ARMS:
        for j in jobs[arm]:
            messages=binary_messages(core,j['task']) if arm.endswith('binary') else core.terminal_messages(j['task'],False)
            prompts.append(dict(id=j['id'],messages=messages,rendered_sha256=hashlib.sha256(backend.render(engine.processor,messages,thinking=False).encode()).hexdigest()))
    write(directory/'PROMPTS.json',prompts)
    write(directory/'ENGINE-READY.json',dict(gpu_uuid=gpu,model_load_count=1,cold_seconds=time.time()-before,batch_size=8,temperature=0))
    completed=0
    for arm in ARMS:
        for offset in range(0,20,8):
            assert time.time()<int(os.environ['TASK_STARTED'])+1620,'Finite sample deadline'
            batch=jobs[arm][offset:offset+8];path=directory/f'{arm}-{offset:02d}.jsonl';start=time.time()
            rows=binary_batch(engine,batch,path) if arm.endswith('binary') else engine.score_jobs(batch,path)
            assert [r['id'] for r in rows]==[j['id'] for j in batch] and all(r['sampled_frames']==64 for r in rows)
            completed+=len(rows)
            write(path.with_suffix('.receipt.json'),dict(completed=True,labels=len(rows),sha256=sha(path),seconds=time.time()-start))
            print(json.dumps(dict(labels_completed=completed,labels_total=100,arm=arm)),flush=True)
    write(directory/'COMPLETED.json',dict(completed=True,videos=20,labels=completed,model_load_count=1,robot_actions=0,optimizer_steps=0))

def summary():
    assert platform.system()=='Linux'
    directory=OUT/'scoring';assert read(directory/'COMPLETED.json')['labels']==100
    rows={}
    for p in sorted(directory.glob('*.jsonl')):
        assert read(p.with_suffix('.receipt.json'))['sha256']==sha(p)
        for line in p.read_text().splitlines():
            r=json.loads(line);assert r['id'] not in rows;rows[r['id']]=r
    assert len(rows)==100
    selection=read(OUT/'sample/SELECTION.json');pairs=[];metrics={}
    for selected in selection['selected']:
        p=selected['pair'];uid=p['id'];truth=p['truth']['success'];labels={a:rows[uid+'/'+a] for a in ARMS}
        positive={a:(bool(r['success']) if a.endswith('binary') else r['score']==5) for a,r in labels.items()}
        pairs.append(dict(id=uid,group=selected['group'],truth=truth,instruction=p['native_instruction'],
            historical_revision1=p['candidate'],labels=labels,predicted_success=positive))
    for arm in ARMS:
        metrics[arm]=dict(false_successes=sum(not p['truth'] and p['predicted_success'][arm] for p in pairs),
            success_controls_retained=sum(p['truth'] and p['predicted_success'][arm] for p in pairs),
            baseline_false_successes_corrected=sum(not p['truth'] and p['predicted_success'][ARMS[0]] and not p['predicted_success'][arm] for p in pairs),
            new_false_successes=sum(not p['truth'] and not p['predicted_success'][ARMS[0]] and p['predicted_success'][arm] for p in pairs),
            baseline_success_controls_lost=sum(p['truth'] and p['predicted_success'][ARMS[0]] and not p['predicted_success'][arm] for p in pairs))
    write(OUT/'PAIRED-SAMPLE.json',pairs)
    write(OUT/'RESULT.json',dict(completed=True,videos=20,failures=14,success_controls=6,labels=100,
        metrics=metrics,historical_revision1_hard_grade_drift=sum(p['historical_revision1']['score']!=p['labels'][ARMS[0]]['score'] for p in pairs),
        development_only=True,representative_accuracy=False,production_reward_qualified=False))
    print(json.dumps(read(OUT/'RESULT.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','summary']);p.add_argument('--code',type=Path);a=p.parse_args()
    worker(a.code) if a.mode=='worker' else summary()
