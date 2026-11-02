"""Separate positive location from negative exclusion wording."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

ROOT=Path('/mowenjie/fluffy-bassoon-2')
RUN='robocasa-flora-drawer-location-exclusion-20261103-r1'
OLD=ROOT/'outputs/robocasa-flora-drawer-target-wording-20261103-r1'
CLAUSE=ROOT/'outputs/robocasa-flora-drawer-clause-ablation-20261103-r1'
OUT=ROOT/'outputs'/RUN
ARMS=('location_only','exclusion_only')
CACHED=('depth_only','compact_target')
COMPARISONS=('revision1_6',*CACHED,*ARMS)
ANCHORS=(('000112',300),('000128',400),('000144',250),('000064',550))
NEW=74

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def task(item,criterion):return criterion['task_template'].format(side=re.fullmatch(r'Open the (left|right) drawer\.',item['task']).group(1))
def anchor(item):return any('-case-'+case+'-' in item['id'] and item['prefix_controls']==control for case,control in ANCHORS)
def jobs_for(items,criteria):
    return {arm:[dict(id=i['id']+'/'+arm,task=task(i,criteria[arm]),videos=[v['path'] for v in i['videos']],prefix_controls=i['prefix_controls'],initial_frame_included=True) for i in items if arm!='repeat_both' or anchor(i)] for arm in ('repeat_both',*ARMS)}
def prompts(jobs,core):return [dict(id=j['id'],messages=core.terminal_messages(j['task'],False)) for batch in jobs.values() for j in batch]
def prepare(code):
    assert platform.system()=='Linux';OUT.mkdir(exist_ok=False)
    frozen=read(code/'outputs'/RUN/'EXPECTED-INPUTS.json')
    for name,digest in frozen.items():assert sha(OLD/name)==digest,name
    assert read(OLD/'terminal-charge.json')['exit_code']==0 and read(OLD/'scoring/COMPLETED.json')['new_outputs']==74
    assert all(a['equal'] for a in read(OLD/'RESULT.json')['anchors'])
    extra=read(code/'outputs'/RUN/'EXTRA-INPUTS.json')
    for path,digest in extra.items():assert sha(path)==digest,path
    assert read(CLAUSE/'terminal-charge.json')['exit_code']==0 and all(a['equal'] for a in read(CLAUSE/'RESULT.json')['anchors'])
    items=read(OLD/'PUBLIC-JOBS.json');reference=read(OLD/'PREFIX-REFERENCE.json');assert len(items)==len(reference)==35
    write(OUT/'PUBLIC-JOBS.json',items);write(OUT/'PREFIX-REFERENCE.json',reference)
    criteria={a:read(code/'outputs'/RUN/(a+'.json')) for a in (*ARMS,*CACHED,'repeat_both')}
    from rewarder.batch_utils import core
    from fluffy_bassoon.meta_rl.flora_terminal_worker import _PrefixCollage
    expected={}
    for i in items:
        assert set(i)=={'id','task_key','task','horizon','videos','prefix_controls','initial_frame_included'}
        for v in i['videos']:assert sha(v['path'])==v['sha256']
        indices=core.terminal_frame_indices(i['prefix_controls']+1,20.);assert min(indices)>=0 and max(indices)<=i['prefix_controls'];expected[i['id']]=len(indices)
    probes=[]
    for control in (50,100,750):
        i=next(i for i in items if '-case-000112-' in i['id'] and i['prefix_controls']==control)
        reader=_PrefixCollage(core.Collage([v['path'] for v in i['videos']]),control)
        try:
            assert reader.source.num_frames==751 and reader.fps==20.
            indices=core.terminal_frame_indices(reader.num_frames,reader.fps);assert len(reader.frames(indices))==len(indices)
            probes.append(dict(controls=control,sampled_frames=len(indices),max_index=max(indices)))
        finally:reader.release()
    terminal=code/'src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'
    original=subprocess.check_output(['git','-C',str(code),'show','17755d7a2afecf8eccae00e301933f9bc216a726:src/fluffy_bassoon/meta_rl/flora_terminal_worker.py']);assert sha(terminal)==hashlib.sha256(original).hexdigest()
    recovered=[]
    for label,folder,criterion in (('revision1_6',OLD,'repeat_both'),('depth_only',CLAUSE,'depth_only'),('compact_target',OLD,'compact_target')):
        assert read(folder/'PUBLIC-JOBS.json')==items
        assert read(folder/'PREFIX-REFERENCE.json')==reference
        cached=[pair['labels'][label] for pair in read(folder/'PAIRED-SAMPLE.json')];assert len(cached)==35
        for row,item in zip(cached,items):
            assert row['id']==item['id']+'/'+label and row['task']==task(item,criteria[criterion])
            assert row['sampled_frames']==expected[item['id']] and row['observed_through']==item['prefix_controls']
        recovered.extend(cached)
    assert len(recovered)==105 and len({r['id'] for r in recovered})==105
    jobs=jobs_for(items,criteria);prepared=prompts(jobs,core);assert len(prepared)==NEW
    assert all(len(p['messages'])==2 and sum(b['type']=='video' for b in p['messages'][1]['content'])==1 for p in prepared)
    write(OUT/'RECOVERED.json',dict(rows=recovered,source_sha256=frozen['PAIRED-SAMPLE.json'],extra_sources=extra,source_commits={'target_wording':'7da6f7fe863a08f2065bbcfa1ec3fafb76ae36be','clause_ablation':'db1fc5c8ec32eea8550b768f9213629175ebf17f'},baseline_provenance='revision1_6 inherited through target-wording RECOVERED.json'))
    write(OUT/'PREPARED-PROMPTS.json',prepared)
    write(OUT/'CPU-CHECKS.json',dict(completed=True,videos=6,prefixes=35,new_outputs=NEW,recovered_labels=105,
        public_sha256=sha(OUT/'PUBLIC-JOBS.json'),recovered_sha256=sha(OUT/'RECOVERED.json'),prompts_sha256=sha(OUT/'PREPARED-PROMPTS.json'),
        criteria_sha256={a:sha(code/'outputs'/RUN/(a+'.json')) for a in criteria},expected_sampled_frames=expected,actual_frame_probes=probes,
        unchanged_terminal_worker_sha256=sha(terminal),official_messages_unchanged=True,simulator_calls=0,motor_calls=0,optimizer_steps=0))
    print(json.dumps(read(OUT/'CPU-CHECKS.json')),flush=True)
def worker(code):
    assert platform.system()=='Linux'
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    import torch
    gpu=verify_selected_cuda(torch)
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE'):os.environ.pop(key,None)
    inputs=read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
    assert sha(inputs['model_receipt']['path'])==inputs['model_receipt']['sha256'] and read(inputs['model_receipt']['path'])['completed']
    assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip()==inputs['official_code_commit']
    checks=read(OUT/'CPU-CHECKS.json');assert checks['completed']
    for filename,key in (('PUBLIC-JOBS.json','public_sha256'),('RECOVERED.json','recovered_sha256'),('PREPARED-PROMPTS.json','prompts_sha256')):assert sha(OUT/filename)==checks[key]
    criteria={a:read(code/'outputs'/RUN/(a+'.json')) for a in (*ARMS,*CACHED,'repeat_both')}
    assert all(sha(code/'outputs'/RUN/(a+'.json'))==checks['criteria_sha256'][a] for a in criteria)
    items=read(OUT/'PUBLIC-JOBS.json')
    for item in items:
        for v in item['videos']:assert sha(v['path'])==v['sha256']
    dest=OUT/'scoring';dest.mkdir(exist_ok=False)
    write(dest/'INPUTS.json',dict(source_commit=os.environ['TASK_CODE_PIN'],model=inputs['model'],official_code_commit=inputs['official_code_commit'],criteria_sha256=checks['criteria_sha256'],truth_read=False,custom_messages=False,new_outputs=NEW,recovered_labels=105))
    from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
    start=time.time();engine=TerminalWorker(inputs['model'],batch_size=8,gpu_memory_utilization=.78)
    write(dest/'ENGINE-READY.json',dict(gpu_uuid=gpu,cold_seconds=time.time()-start,model_load_count=1,batch_size=8,temperature=0))
    original_render=engine.api.engine.render;jobs=jobs_for(items,criteria)
    write(dest/'PROMPTS.json',prompts(jobs,engine.api.core));assert sha(dest/'PROMPTS.json')==checks['prompts_sha256']
    completed=0
    for arm,todo in jobs.items():
        for offset in range(0,len(todo),8):
            assert time.time()<int(os.environ['TASK_STARTED'])+780,'Finite batch deadline'
            batch=todo[offset:offset+8];path=dest/f'{arm}-{offset:02d}.jsonl';audit=[];begin=time.time()
            def audited_render(processor,messages,**kwargs):
                index=len(audit);job=batch[index];assert messages==engine.api.core.terminal_messages(job['task'],False)
                prompt=original_render(processor,messages,**kwargs);assert isinstance(prompt,str)
                audit.append(dict(id=job['id'],rendered_sha256=hashlib.sha256(prompt.encode()).hexdigest(),messages_sha256=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()))
                return prompt
            engine.api.engine.render=audited_render
            try:rows=engine.score_jobs(batch,path)
            finally:engine.api.engine.render=original_render
            assert len(rows)==len(audit)==len(batch) and [r['id'] for r in rows]==[j['id'] for j in batch]
            assert all(r['observed_through']==j['prefix_controls'] and r['sampled_frames']==checks['expected_sampled_frames'][r['id'].rsplit('/',1)[0]] for r,j in zip(rows,batch))
            write(path.with_suffix('.prompts.json'),audit)
            write(path.with_suffix('.receipt.json'),dict(completed=True,labels=len(rows),sha256=sha(path),prompts_sha256=sha(path.with_suffix('.prompts.json')),seconds=time.time()-begin))
            completed+=len(rows);print(json.dumps(dict(new_outputs_completed=completed,total=NEW,arm=arm)),flush=True)
    assert completed==NEW
    write(dest/'COMPLETED.json',dict(completed=True,new_outputs=NEW,candidate_labels=70,anchor_labels=4,recovered_labels=105,robot_actions=0,optimizer_steps=0))
def summary():
    dest=OUT/'scoring';assert read(dest/'COMPLETED.json')['new_outputs']==NEW
    cached={r['id']:r for r in read(OUT/'RECOVERED.json')['rows']};rows=dict(cached);anchors=[]
    for path in dest.glob('*.jsonl'):
        assert sha(path)==read(path.with_suffix('.receipt.json'))['sha256']
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['id'].endswith('/repeat_both'):
                old=cached[r['id'].rsplit('/',1)[0]+'/revision1_6'];anchors.append(dict(id=r['id'],cached_score=old['score'],repeated_score=r['score'],equal=old['score']==r['score']))
            else:assert r['id'] not in rows;rows[r['id']]=r
    assert len(rows)==175 and len(anchors)==4
    paired=[dict(ref,labels={a:rows[ref['id']+'/'+a] for a in COMPARISONS}) for ref in read(OUT/'PREFIX-REFERENCE.json')]
    write(OUT/'PAIRED-SAMPLE.json',paired);metrics={}
    for arm in COMPARISONS:
        raw=[];confirmed={4:[],5:[]}
        for eid in sorted({r['episode_id'] for r in paired}):
            seq=sorted((r for r in paired if r['episode_id']==eid),key=lambda r:r['control']);actual=seq[0]['native_first_success']
            def record(points):
                first=points[0] if points else None
                return dict(episode_id=eid,native_first_success=actual,first_accepted=first,positive_probes=points,false_accept=first is not None and (actual is None or first<actual),correctly_timed=actual is not None and first is not None and 0<=first-actual<=100)
            raw.append(record([r['control'] for r in seq if r['labels'][arm]['score']==5]))
            for threshold in (4,5):confirmed[threshold].append(record([b['control'] for a,b in zip(seq,seq[1:]) if b['control']-a['control']==50 and min(a['labels'][arm]['score'],b['labels'][arm]['score'])>=threshold]))
        def count(cases):return dict(cases=cases,false_accepted_episodes=sum(c['false_accept'] for c in cases),correctly_timed_episodes=sum(c['correctly_timed'] for c in cases))
        c4=count(confirmed[4]);c5=count(confirmed[5]);metrics[arm]=dict(false_full_scores=sum(not r['native_ever_success'] and r['labels'][arm]['score']==5 for r in paired),post_success_full_scores=sum(r['native_ever_success'] and r['labels'][arm]['score']==5 for r in paired),raw=count(raw),two_probe4=c4,two_probe5=c5,passes_development_screen=c4['false_accepted_episodes']==0 and c4['correctly_timed_episodes']==5 and all(a['equal'] for a in anchors))
    write(OUT/'RESULT.json',dict(completed=True,new_outputs=NEW,paired_labels=175,anchors=anchors,metrics=metrics,development_only=True,independent_validation=False,production_reward_qualified=False,first_success_qualified=False))
    print(json.dumps(read(OUT/'RESULT.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','summary']);p.add_argument('--code',type=Path);a=p.parse_args();{'prepare':prepare,'worker':worker,'summary':lambda _:summary()}[a.mode](a.code)
