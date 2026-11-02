"""Recent causal chunks with a fixed drawer rubric; isolated experiment adapter."""
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
RUN='robocasa-flora-drawer-chunk-rubric-20261103-r1'
OLD=ROOT/'outputs/robocasa-flora-system-rubric-20261103-r1'
OUT=ROOT/'outputs'/RUN
ARMS={'window50':50,'window100':100}
ANCHORS=(('000112',300),('000128',400),('000144',250),('000064',550))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def task(item,criterion):return criterion['task_template'].format(side=re.fullmatch(r'Open the (left|right) drawer\.',item['task']).group(1))
def anchor(item):return any('-case-'+case+'-' in item['id'] and item['prefix_controls']==end for case,end in ANCHORS)
def all_jobs(items,criterion):
    return {arm:[dict(id=i['id']+'/'+arm,task=task(i,criterion),videos=[v['path'] for v in i['videos']],prefix_controls=i['prefix_controls'],initial_frame_included=True) for i in items if arm!='repeat_full' or anchor(i)] for arm in ('repeat_full',*ARMS)}
def bounds(job,arm,core):
    end=job['prefix_controls'];start=0 if arm=='repeat_full' else max(0,end-ARMS[arm])
    indices=core.terminal_frame_indices(end-start+1,20.)
    return dict(id=job['id'],observed_from=start,observed_through=end,frame_count=end-start+1,relative_indices=indices,absolute_indices=[start+i for i in indices],sampled_frames=len(indices))
def prompts(jobs,core):return [dict(id=j['id'],messages=core.terminal_messages(j['task'],False)) for batch in jobs.values() for j in batch]
class Window:
    def __init__(self,source,controls,width):
        assert type(controls) is int and 1<=controls<source.num_frames
        self.source=source;self.start=max(0,controls-width);self.end=controls
        self.num_frames=controls-self.start+1;self.fps=source.fps;self.duration_s=self.num_frames/self.fps
    def frames(self,indices):
        assert all(0<=i<self.num_frames for i in indices)
        absolute=[self.start+i for i in indices]
        assert all(self.start<=i<=self.end for i in absolute)
        return self.source.frames(absolute)
    def release(self):self.source.release()
def prepare(code):
    assert platform.system()=='Linux';OUT.mkdir(exist_ok=False)
    frozen=read(code/'outputs'/RUN/'EXPECTED-INPUTS.json')
    for name,digest in frozen.items():assert sha(OLD/name)==digest,name
    assert read(OLD/'terminal-charge.json')['exit_code']==0
    assert read(OLD/'scoring/COMPLETED.json')['new_outputs']==74
    assert all(a['equal'] for a in read(OLD/'RESULT.json')['anchors'])
    items=read(OLD/'PUBLIC-JOBS.json');reference=read(OLD/'PREFIX-REFERENCE.json')
    assert len(items)==len(reference)==35
    write(OUT/'PUBLIC-JOBS.json',items);write(OUT/'PREFIX-REFERENCE.json',reference)
    criterion=read(code/'outputs'/RUN/'criterion1_6.json')
    from rewarder.batch_utils import core
    from fluffy_bassoon.meta_rl.flora_terminal_worker import _PrefixCollage
    for i in items:
        assert set(i)=={'id','task_key','task','horizon','videos','prefix_controls','initial_frame_included'}
        for v in i['videos']:assert sha(v['path'])==v['sha256']
    jobs=all_jobs(items,criterion);prepared=prompts(jobs,core)
    assert len(prepared)==74 and len({j['id'] for batch in jobs.values() for j in batch})==74
    expected={j['id']:bounds(j,arm,core) for arm,batch in jobs.items() for j in batch}
    assert all(min(v['absolute_indices'])>=v['observed_from'] and max(v['absolute_indices'])<=v['observed_through'] for v in expected.values())
    import numpy as np
    probes=[]
    for end in (50,100,400,750):
        item=next(i for i in items if ('-case-000128-' in i['id'] if end>=400 else '-case-000112-' in i['id']) and i['prefix_controls']==end)
        for width in (50,100):
            source=core.Collage([v['path'] for v in item['videos']]);reader=Window(source,end,width)
            try:
                assert source.num_frames==751 and source.fps==20.
                indices=core.terminal_frame_indices(reader.num_frames,reader.fps)
                actual=reader.frames(indices);direct=source.frames([reader.start+i for i in indices])
                assert np.array_equal(actual,direct)
                assert len(actual)==len(indices)
                probes.append(dict(endpoint=end,width=width,start=reader.start,sampled_frames=len(indices),max_absolute_index=max(reader.start+i for i in indices),pixel_equal=True))
            finally:reader.release()
    terminal=code/'src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'
    original=subprocess.check_output(['git','-C',str(code),'show','17755d7a2afecf8eccae00e301933f9bc216a726:src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'])
    assert sha(terminal)==hashlib.sha256(original).hexdigest()
    recovered=[p['labels']['revision1_6'] for p in read(OLD/'PAIRED-SAMPLE.json')]
    assert len(recovered)==35 and len({r['id'] for r in recovered})==35
    for row,item in zip(recovered,items):assert row['id']==item['id']+'/revision1_6' and row['task']==task(item,criterion)
    write(OUT/'RECOVERED.json',dict(rows=recovered,source_sha256=frozen['PAIRED-SAMPLE.json'],source_commit='b20f3672f6fbec3457341ea7d2b394f3cc958500'))
    write(OUT/'PREPARED-PROMPTS.json',prepared);write(OUT/'EXPECTED-FRAMES.json',expected)
    write(OUT/'CPU-CHECKS.json',dict(completed=True,prefixes=35,videos=6,new_outputs=74,recovered_labels=35,
        public_sha256=sha(OUT/'PUBLIC-JOBS.json'),recovered_sha256=sha(OUT/'RECOVERED.json'),criterion_sha256=sha(code/'outputs'/RUN/'criterion1_6.json'),
        prompts_sha256=sha(OUT/'PREPARED-PROMPTS.json'),frames_sha256=sha(OUT/'EXPECTED-FRAMES.json'),pixel_probes=probes,
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
    for filename,key in (('PUBLIC-JOBS.json','public_sha256'),('RECOVERED.json','recovered_sha256'),('PREPARED-PROMPTS.json','prompts_sha256'),('EXPECTED-FRAMES.json','frames_sha256')):assert sha(OUT/filename)==checks[key]
    assert sha(code/'outputs'/RUN/'criterion1_6.json')==checks['criterion_sha256']
    items=read(OUT/'PUBLIC-JOBS.json');criterion=read(code/'outputs'/RUN/'criterion1_6.json');expected=read(OUT/'EXPECTED-FRAMES.json')
    for item in items:
        for v in item['videos']:assert sha(v['path'])==v['sha256']
    dest=OUT/'scoring';dest.mkdir(exist_ok=False)
    write(dest/'INPUTS.json',dict(source_commit=os.environ['TASK_CODE_PIN'],model=inputs['model'],official_code_commit=inputs['official_code_commit'],criterion_sha256=checks['criterion_sha256'],truth_read=False,custom_messages=False,new_outputs=74,recovered_labels=35))
    import fluffy_bassoon.meta_rl.flora_terminal_worker as module
    start=time.time();engine=module.TerminalWorker(inputs['model'],batch_size=8,gpu_memory_utilization=.78)
    write(dest/'ENGINE-READY.json',dict(gpu_uuid=gpu,cold_seconds=time.time()-start,model_load_count=1,batch_size=8,temperature=0))
    original_reader=module._PrefixCollage;original_render=engine.api.engine.render
    jobs=all_jobs(items,criterion);write(dest/'PROMPTS.json',prompts(jobs,engine.api.core));assert sha(dest/'PROMPTS.json')==checks['prompts_sha256']
    completed=0
    for arm,todo in jobs.items():
        for offset in range(0,len(todo),8):
            assert time.time()<int(os.environ['TASK_STARTED'])+780,'Finite batch deadline'
            batch=todo[offset:offset+8];path=dest/f'{arm}-{offset:02d}.jsonl';raw=path.with_suffix('.raw.jsonl');renders=[];videos=[];begin=time.time()
            def make_reader(source,controls):
                index=len(videos);job=batch[index];assert controls==job['prefix_controls'] and source.num_frames==751 and source.fps==20.
                info=expected[job['id']];reader=original_reader(source,controls) if arm=='repeat_full' else Window(source,controls,ARMS[arm])
                assert reader.num_frames==info['frame_count'];original_frames=reader.frames
                def audited_frames(indices):
                    assert list(indices)==info['relative_indices']
                    frames=original_frames(indices);assert len(frames)==info['sampled_frames']
                    info['actual_frame_read_verified']=True
                    return frames
                reader.frames=audited_frames;videos.append(dict(info));return reader
            def audited_render(processor,messages,**kwargs):
                index=len(renders);job=batch[index];assert messages==engine.api.core.terminal_messages(job['task'],False)
                prompt=original_render(processor,messages,**kwargs);assert isinstance(prompt,str)
                renders.append(dict(id=job['id'],rendered_sha256=hashlib.sha256(prompt.encode()).hexdigest(),messages_sha256=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()))
                return prompt
            module._PrefixCollage=make_reader;engine.api.engine.render=audited_render
            try:rows=engine.score_jobs(batch,raw)
            finally:module._PrefixCollage=original_reader;engine.api.engine.render=original_render
            assert len(rows)==len(videos)==len(renders)==len(batch)
            with path.open('x') as f:
                for row,job,video in zip(rows,batch,videos):
                    info=expected[job['id']];assert row['id']==job['id'] and row['observed_through']==job['prefix_controls'] and row['sampled_frames']==info['sampled_frames']
                    row.update(observed_from=info['observed_from'],window_controls=None if arm=='repeat_full' else ARMS[arm],score_protocol='causal_prefix_terminal_v1' if arm=='repeat_full' else 'causal_window_terminal_v1')
                    f.write(json.dumps(row)+'\n')
            # Reader creation precedes frame decoding; re-bind verified expected records after the reads.
            assert all(expected[j['id']].get('actual_frame_read_verified') for j in batch)
            videos=[dict(expected[j['id']]) for j in batch]
            write(path.with_suffix('.prompts.json'),renders);write(path.with_suffix('.frames.json'),videos)
            write(path.with_suffix('.receipt.json'),dict(completed=True,labels=len(rows),sha256=sha(path),raw_sha256=sha(raw),prompt_sha256=sha(path.with_suffix('.prompts.json')),frames_sha256=sha(path.with_suffix('.frames.json')),seconds=time.time()-begin))
            completed+=len(rows);print(json.dumps(dict(new_outputs_completed=completed,total=74,arm=arm)),flush=True)
    assert completed==74
    write(dest/'COMPLETED.json',dict(completed=True,new_outputs=74,candidate_labels=70,anchor_labels=4,recovered_labels=35,robot_actions=0,optimizer_steps=0))
def summary():
    dest=OUT/'scoring';assert read(dest/'COMPLETED.json')['new_outputs']==74
    cached={r['id']:r for r in read(OUT/'RECOVERED.json')['rows']};rows=dict(cached);anchors=[]
    for path in dest.glob('*.receipt.json'):
        data=path.with_suffix('').with_suffix('.jsonl');receipt=read(path);assert sha(data)==receipt['sha256']
        for line in data.read_text().splitlines():
            row=json.loads(line)
            if row['id'].endswith('/repeat_full'):
                old=cached[row['id'].rsplit('/',1)[0]+'/revision1_6'];anchors.append(dict(id=row['id'],cached_score=old['score'],repeated_score=row['score'],equal=old['score']==row['score']))
            else:assert row['id'] not in rows;rows[row['id']]=row
    assert len(rows)==105 and len(anchors)==4
    paired=[dict(ref,labels={arm:rows[ref['id']+'/'+arm] for arm in ('revision1_6',*ARMS)}) for ref in read(OUT/'PREFIX-REFERENCE.json')]
    write(OUT/'PAIRED-SAMPLE.json',paired);metrics={}
    for arm in ('revision1_6',*ARMS):
        cases=[];confirmed=[]
        for eid in sorted({r['episode_id'] for r in paired}):
            seq=sorted((r for r in paired if r['episode_id']==eid),key=lambda r:r['control']);actual=seq[0]['native_first_success']
            raw=[r['control'] for r in seq if r['labels'][arm]['score']==5]
            two=[b['control'] for a,b in zip(seq,seq[1:]) if b['control']-a['control']==50 and a['labels'][arm]['score']==b['labels'][arm]['score']==5]
            def record(positives):
                first=positives[0] if positives else None
                return dict(episode_id=eid,native_first_success=actual,first_accepted=first,positive_probes=positives,false_accept=first is not None and (actual is None or first<actual),correctly_timed=actual is not None and first is not None and 0<=first-actual<=100)
            cases.append(record(raw));confirmed.append(record(two))
        def count(records):return dict(cases=records,false_accepted_episodes=sum(c['false_accept'] for c in records),correctly_timed_episodes=sum(c['correctly_timed'] for c in records))
        raw=count(cases);two=count(confirmed)
        metrics[arm]=dict(false_full_scores=sum(not r['native_ever_success'] and r['labels'][arm]['score']==5 for r in paired),post_success_full_scores=sum(r['native_ever_success'] and r['labels'][arm]['score']==5 for r in paired),raw=raw,two_probe_confirmation=two,passes_development_screen=two['false_accepted_episodes']==0 and two['correctly_timed_episodes']==5 and all(a['equal'] for a in anchors))
    write(OUT/'RESULT.json',dict(completed=True,new_outputs=74,paired_labels=105,anchors=anchors,metrics=metrics,development_only=True,independent_validation=False,production_reward_qualified=False,first_success_qualified=False))
    print(json.dumps(read(OUT/'RESULT.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','summary']);p.add_argument('--code',type=Path);a=p.parse_args()
    {'prepare':prepare,'worker':worker,'summary':lambda _:summary()}[a.mode](a.code)
