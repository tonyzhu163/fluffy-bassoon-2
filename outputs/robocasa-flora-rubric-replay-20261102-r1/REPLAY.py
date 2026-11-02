"""One frozen prompt comparison. No simulator, optimizer or truth in inference."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

RUN = 'robocasa-flora-rubric-replay-20261102-r1'
ROOT = Path('/mowenjie/fluffy-bassoon-2')
OUT = ROOT/'outputs'/RUN
TASKS = ['OpenDrawer','PickPlaceCounterToCabinet','PickPlaceDrawerToCounter','TurnOffStove','CoffeeSetupMug']
def read(path): return json.loads(Path(path).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
def ref(path): return {'path':str(path),'sha256':sha(path)}
def checked(r):
    assert sha(r['path'])==r['sha256'],r['path']
    return read(r['path'])
def candidate_task(instruction,task,candidate):
    return instruction+'\n\nCompletion criteria:\n'+candidate['shared_criteria']+'\n\n'+candidate['reward_training_task_criteria'][task]

def build(code):
    assert platform.system()=='Linux'
    OUT.mkdir(exist_ok=False)
    candidate_path=code/'outputs/robocasa-flora-rubric-audit-20261102-r1/CANDIDATE.json'
    candidate=read(candidate_path)
    assert not candidate['qualified'] and not candidate['physical_measurements_in_prompt']
    source=[];items=[];truth={};historical={}
    def remember(p):
        r=ref(p);source.append(r);return read(p)
    def add(dataset,task,episodes,labels,monitor):
        assert set(labels)=={e['id'] for e in episodes}==set(monitor)
        for e in episodes:
            eid=e['id'];uid=dataset+'/'+task+'/'+eid
            assert uid not in truth and e['completed'] is True and e['raw_rgb_only'] is True and e['info_on_video'] is False
            assert [v['camera'] for v in e['videos']]==e['camera_order']==['robot0_agentview_left','robot0_eye_in_hand','robot0_agentview_right']
            assert type(monitor[eid]['simulator_final_success']) is bool
            for v in e['videos']:assert sha(v['path'])==v['sha256']
            items.append(dict(id=uid,dataset=dataset,task_key=task,native_id=eid,native_instruction=e['task'],videos=e['videos'],horizon=e['horizon']))
            truth[uid]=dict(success=monitor[eid]['simulator_final_success'],case=monitor[eid]['case'])
            historical[uid]=labels[eid]
    for i in range(1,7):
        run='robocasa-opendrawer-flora-first-20261102-r1' if i<=4 else 'robocasa-opendrawer-flora-resume-20261102-r1'
        p=ROOT/'outputs'/run/'policy';request=remember(p/f'labels-{i:03d}/request.json')
        ready=remember(p/f'labels-{i:03d}/ready.json');labels=checked(ready['outputs']);source.append(ready['outputs'])
        profile=remember(p/'profile.json');monitor={}
        for e in request['episodes']:monitor[e['id']]=remember(Path(profile['monitoring_root'])/'train'/(e['id']+'.json'))
        assert len(request['episodes'])==64
        add('TRAIN', 'OpenDrawer',request['episodes'],{x['id']:x for x in labels},monitor)
    for task in TASKS:
        q=remember(ROOT/'outputs/robocasa-five-task-native-qualification-20261102-r1/cohorts'/task/'NATIVE-QUALIFICATION.json')
        episodes=checked(q['episodes']);monitor=checked(q['monitor']);source.extend([q['episodes'],q['monitor']])
        labels=remember(ROOT/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/exchange/cohorts'/task/'official-outputs.json')
        assert len(episodes)==len(labels)==len(monitor)==50
        add('CALIBRATION',task,episodes,{x['id']:x for x in labels},{x['id']:x for x in monitor})
    focus={uid for uid,y in truth.items() if uid.startswith('TRAIN/') and not y['success'] and historical[uid]['score']==5}
    assert len(items)==634 and len(focus)==58
    items.sort(key=lambda x:(x['id'] not in focus,x['dataset'],x['task_key'],x['id']))
    write(OUT/'videos.json',items);write(OUT/'truth.json',truth);write(OUT/'historical.json',historical)
    write(OUT/'candidate.json',candidate)
    write(OUT/'MANIFEST.json',dict(completed=True,run_id=RUN,videos=ref(OUT/'videos.json'),truth=ref(OUT/'truth.json'),
        historical=ref(OUT/'historical.json'),candidate=ref(OUT/'candidate.json'),candidate_source=ref(candidate_path),
        sources=source,video_count=634,label_count=1268,focus_ids=sorted(focus),new_physical_episodes=0,optimizer_steps=0,
        camera_order=items[0]['videos'] and [v['camera'] for v in items[0]['videos']],
        datasets=dict(Counter(x['dataset'] for x in items)),prompt_change='Append frozen qualitative criteria to unchanged native instruction; official system/output unchanged'))
    print(json.dumps({'videos':634,'labels':1268,'focus_failures':58,'manifest':ref(OUT/'MANIFEST.json')}))

def worker(profile):
    assert platform.system()=='Linux'
    p=read(profile);code=Path(p['code']);rank=p['rank'];directory=OUT/f'rank-{rank:02d}'
    directory.mkdir(exist_ok=False)
    assert subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()==p['source_commit']
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    import torch
    assert verify_selected_cuda(torch)==p['gpu_uuid']
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE'):
        os.environ.pop(key,None)
    inputs=read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
    model=checked(inputs['model_receipt'])
    assert model['completed'] and model['model_revision']==inputs['model_revision']
    assert Path(model['model_directory']).resolve()==Path(inputs['model']).resolve()
    assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip()==inputs['official_code_commit']
    m=checked(p['manifest']);items=checked(m['videos']);candidate=checked(m['candidate'])
    from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
    started=time.time();engine=TerminalWorker(inputs['model'],batch_size=8,gpu_memory_utilization=.78)
    core,backend=engine.api.core,engine.api.engine
    selected=items[rank::4];jobs=[]
    for item in selected:
        for v in item['videos']:assert sha(v['path'])==v['sha256']
        for variant in ['original','candidate']:
            task=item['native_instruction'] if variant=='original' else candidate_task(item['native_instruction'],item['task_key'],candidate)
            jobs.append(dict(id=item['id']+'/'+variant,task=task,videos=[v['path'] for v in item['videos']]))
    # Public jobs contain only model inputs. Never load the truth or historical grade files here.
    write(directory/'jobs.json',jobs)
    prompts=[dict(id=j['id'],messages=core.terminal_messages(j['task'],False),
        rendered_sha256=hashlib.sha256(backend.render(engine.processor,core.terminal_messages(j['task'],False),thinking=False).encode()).hexdigest()) for j in jobs]
    write(directory/'prompts.json',prompts)
    write(directory/'ENGINE-READY.json',dict(completed=True,gpu_uuid=p['gpu_uuid'],source_commit=p['source_commit'],
        model_receipt=inputs['model_receipt'],official_code_commit=inputs['official_code_commit'],
        batch_size=8,engine_seed=0,temperature=0,model_load_count=1,manifest=p['manifest'],
        cold_seconds=time.time()-started,truth_read=False,prompts=ref(directory/'prompts.json')))
    for offset in range(0,len(jobs),8):
        if time.time()>=p['deadline_unix']-60:raise TimeoutError('Finite replay clock reached; preserve rows')
        target=directory/f'batch-{offset//8:04d}.jsonl';before=time.time()
        rows=engine.score_jobs(jobs[offset:offset+8],target)
        assert [r['id'] for r in rows]==[j['id'] for j in jobs[offset:offset+8]]
        assert all(r['sampled_frames']==64 for r in rows)
        write(directory/f'batch-{offset//8:04d}.receipt.json',dict(completed=True,labels=len(rows),output=ref(target),
            jobs_sha256=digest(jobs[offset:offset+8]),seconds=time.time()-before,manifest_sha256=p['manifest']['sha256']))
        print(json.dumps(dict(rank=rank,labels_complete=offset+len(rows),labels_total=len(jobs))),flush=True)
    write(directory/'COMPLETED.json',dict(completed=True,rank=rank,labels=len(jobs),videos=len(selected),
        batches=(len(jobs)+7)//8,model_load_count=1,robot_actions=0,optimizer_steps=0,elapsed_seconds=time.time()-started))

def auc(y,s):
    pos=[s[i] for i in range(len(y)) if y[i]];neg=[s[i] for i in range(len(y)) if not y[i]]
    return None if not pos or not neg else sum((a>b)+.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
def metrics(rows,y):
    n=len(rows);tp=sum(t and r['score']==5 for r,t in zip(rows,y));fp=sum(not t and r['score']==5 for r,t in zip(rows,y))
    positives=sum(y);negative=n-positives
    return dict(count=n,successes=positives,AUROC=auc(y,[r['expected_score'] for r in rows]),
        Score5_true_positives=tp,Score5_false_positives=fp,Score5_false_positive_rate=fp/negative if negative else None,
        Score5_precision=tp/(tp+fp) if tp+fp else None,Score5_recall=tp/positives if positives else None,
        true_success_scores_below3=sum(t and r['score']<3 for r,t in zip(rows,y)),
        mean_reward=sum((r['expected_score']-1)/4 for r in rows)/n)
def paired_test(wins,losses):
    from math import comb
    n=wins+losses
    return 1. if not n else min(1.,2*sum(comb(n,k) for k in range(min(wins,losses)+1))/2**n)
def summarize():
    assert platform.system()=='Linux'
    m=read(OUT/'MANIFEST.json');items=checked(m['videos']);truth=checked(m['truth']);old=checked(m['historical'])
    rows={};receipts=[]
    for rank in range(4):
        d=OUT/f'rank-{rank:02d}';assert read(d/'COMPLETED.json')['completed']
        for receipt in sorted(d.glob('batch-*.receipt.json')):
            r=read(receipt);assert r['completed'] and r['manifest_sha256']==sha(OUT/'MANIFEST.json')
            assert sha(r['output']['path'])==r['output']['sha256'];receipts.append(ref(receipt))
            for line in Path(r['output']['path']).read_text().splitlines():
                row=json.loads(line);assert row['id'] not in rows;rows[row['id']]=row
    assert len(rows)==1268 and len(items)==634
    for item in items:
        for v in item['videos']:assert sha(v['path'])==v['sha256']
        for variant in ['original','candidate']:
            r=rows[item['id']+'/'+variant];assert r['sampled_frames']==64 and r['score'] in range(1,6)
    cohorts={};comparisons=[]
    for dataset,task in [('TRAIN','OpenDrawer')]+[('CALIBRATION',t) for t in TASKS]:
        subset=[i for i in items if i['dataset']==dataset and i['task_key']==task];ids=[i['id'] for i in subset]
        y=[truth[i]['success'] for i in ids];a=[rows[i+'/original'] for i in ids];b=[rows[i+'/candidate'] for i in ids]
        wins=sum(not t and x['score']==5 and z['score']<5 for t,x,z in zip(y,a,b));losses=sum(not t and x['score']<5 and z['score']==5 for t,x,z in zip(y,a,b))
        cohorts[dataset+'/'+task]=dict(saved_original=metrics([old[i] for i in ids],y),rerun_original=metrics(a,y),candidate=metrics(b,y),
            paired_false_success_corrections=wins,paired_new_false_successes=losses,paired_false_success_p=paired_test(wins,losses),
            true_full_scores_lost=sum(t and x['score']==5 and z['score']<5 for t,x,z in zip(y,a,b)),
            true_full_scores_gained=sum(t and x['score']<5 and z['score']==5 for t,x,z in zip(y,a,b)),
            original_replay_hard_grade_changes=sum(old[i]['score']!=r['score'] for i,r in zip(ids,a)),
            original_replay_max_expected_grade_difference=max(abs(old[i]['expected_score']-r['expected_score']) for i,r in zip(ids,a)))
    for item in items:
        uid=item['id'];a=rows[uid+'/original'];b=rows[uid+'/candidate']
        comparisons.append(dict(**item,truth=truth[uid],saved_original=old[uid],rerun_original=a,candidate=b,
            focus=uid in m['focus_ids'],reward_change=(b['expected_score']-a['expected_score'])/4))
    focus=[r for r in comparisons if r['focus']];assert len(focus)==58
    write(OUT/'PAIRED-EPISODES.json',comparisons);write(OUT/'FOCUS-58.json',focus)
    write(OUT/'RESULT.json',dict(completed=True,development_only=True,independent_qualification=False,
        videos=634,labels=1268,cohorts=cohorts,focus_original_failed_Score5=dict(count=58,
            saved_original_histogram=dict(Counter(str(r['saved_original']['score']) for r in focus)),
            rerun_original_histogram=dict(Counter(str(r['rerun_original']['score']) for r in focus)),
            candidate_histogram=dict(Counter(str(r['candidate']['score']) for r in focus)),
            candidate_no_longer_full_success=sum(r['candidate']['score']<5 for r in focus),
            candidate_still_full_success=sum(r['candidate']['score']==5 for r in focus)),
        manifest=ref(OUT/'MANIFEST.json'),paired_episodes=ref(OUT/'PAIRED-EPISODES.json'),focus=ref(OUT/'FOCUS-58.json'),
        inference_receipts=receipts,robot_actions=0,optimizer_steps=0))
    print(json.dumps(read(OUT/'RESULT.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','worker','summarize']);p.add_argument('--code');p.add_argument('--profile');a=p.parse_args()
    if a.mode=='build':build(Path(a.code))
    elif a.mode=='worker':worker(a.profile)
    else:summarize()
