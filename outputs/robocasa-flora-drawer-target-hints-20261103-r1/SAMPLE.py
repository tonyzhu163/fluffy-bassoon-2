"""Three minimal task-text variants on26 retained OpenDrawer causal prefixes."""
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
RUN='robocasa-flora-drawer-target-hints-20261103-r1'
OLD=ROOT/'outputs/robocasa-flora-drawer-grade-rubric-20261103-r1'
AUDIT=ROOT/'outputs/robocasa-flora-drawer-sampled-frames-20261103-r1'
OUT=ROOT/'outputs'/RUN
ARMS=('native_boundary','initial_target_lock','shallow_target_type')
COMPARISONS=('revision1_6',*ARMS)
ANCHORS=(('000112',300),('000128',400),('000144',250),('000064',550))
SELECTED={'000112':(300,350),'000256':(700,750),'000336':(700,750),'000376':(700,750),
    '000064':(500,550,600),'000072':(200,250,300),'000128':(350,400,450),
    '000144':(200,250,300),'000216':(200,250,300),'000264':(200,250,300)}
NEW=82
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def case(i):return i['id'].split('-case-')[1][:6]
def keep(i):return i['prefix_controls'] in SELECTED[case(i)]
def anchor(i):return any(case(i)==c and i['prefix_controls']==n for c,n in ANCHORS)
def task(i,c):return c['task_template'].format(side=re.fullmatch(r'Open the (left|right) drawer\.',i['task']).group(1))
def frozen_inputs(code):
    for path,digest in read(code/'outputs'/RUN/'EXPECTED-INPUTS.json').items():assert sha(path)==digest,path
    assert read(OLD/'terminal-charge.json')['exit_code']==0
    assert all(a['equal'] for a in read(OLD/'RESULT.json')['anchors'])
def criteria_for(code):return {a:read(code/'outputs'/RUN/(a+'.json')) for a in COMPARISONS}
def jobs_for(items,criteria):
    jobs={a:[dict(id=i['id']+'/'+a,task=task(i,criteria['revision1_6' if a=='repeat_both' else a]),
        videos=[v['path'] for v in i['videos']],prefix_controls=i['prefix_controls'],initial_frame_included=True)
        for i in items if a!='repeat_both' or anchor(i)] for a in ('repeat_both',*ARMS)}
    assert [len(j) for j in jobs.values()]==[4,26,26,26]
    return jobs
def prompts(jobs,core):return [dict(id=j['id'],messages=core.terminal_messages(j['task'],False)) for batch in jobs.values() for j in batch]
def signature(frames):
    import numpy as np
    hashes=[]
    for frame in frames:
        rgb=np.asarray(frame)
        assert rgb.dtype==np.uint8 and rgb.shape==(320,960,3)
        hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
    return dict(frame_count=len(hashes),shape=[320,960,3],rgb_sha256=hashes,
        ordered_rgb_hashes_sha256=hashlib.sha256(''.join(hashes).encode()).hexdigest())

def prepare(code):
    assert platform.system()=='Linux';frozen_inputs(code);OUT.mkdir(exist_ok=False)
    previous=read(OLD/'CPU-CHECKS.json');assert previous['completed']
    items=[i for i in read(OLD/'PUBLIC-JOBS.json') if keep(i)]
    ids={i['id'] for i in items}
    reference=[r for r in read(OLD/'PREFIX-REFERENCE.json') if r['id'] in ids]
    assert len(items)==len(ids)==len(reference)==26 and len({r['episode_id'] for r in reference})==10
    assert sum(not r['native_ever_success'] for r in reference)==14
    for ref in reference:
        assert sha(ref['replay_file'])==ref['replay_sha256']
        replay=read(ref['replay_file']);assert replay['qualified']
        assert replay['first_success_control']==ref['native_first_success']
        assert replay['states'][ref['control']]['success']==ref['native_current_success']
        assert ref['native_ever_success']==(ref['native_first_success'] is not None and ref['control']>=ref['native_first_success'])
    criteria=criteria_for(code)
    draft=read(code/'outputs'/AUDIT.name/'NEXT-RUBRICS.json')
    assert all(criteria[a]==next(d for d in draft['arms'] if d['criteria_id']==a) for a in ARMS)
    old=criteria['revision1_6']['task_template'];exact=criteria['native_boundary']['task_template']
    assert old.replace('roughly half its front-to-back depth','at least 52.25% of its front-to-back depth, just over half')==exact
    assert criteria['initial_target_lock']['task_template'].replace('Identify this drawer front in the initial frames and keep tracking that same front throughout the video. ','')==exact
    assert criteria['shallow_target_type']['task_template'].replace('The target can be a shallow drawer directly above cabinet doors. ','')==exact
    from rewarder.batch_utils import core
    from fluffy_bassoon.meta_rl.flora_terminal_worker import _PrefixCollage
    video_checks={};expected={};frame_reference={}
    audited={c['id']:c for c in read(AUDIT/'RECEIPT.json')['cases']}
    audited_matches=[]
    for item in items:
        assert set(item)=={'id','task_key','task','horizon','videos','prefix_controls','initial_frame_included'}
        assert item['initial_frame_included'] and item['horizon']==750
        for v in item['videos']:
            if v['path'] not in video_checks:
                assert sha(v['path'])==v['sha256']
                reader=core.Collage([v['path']])
                try:
                    assert reader.num_frames==751 and reader.fps==20.
                    video_checks[v['path']]=dict(sha256=v['sha256'],frames=reader.num_frames,fps=reader.fps)
                finally:reader.release()
        reader=_PrefixCollage(core.Collage([v['path'] for v in item['videos']]),item['prefix_controls'])
        try:
            indices=core.terminal_frame_indices(reader.num_frames,reader.fps)
            assert len(indices)==64 and min(indices)==0 and max(indices)==item['prefix_controls']
            frame_reference[item['id']]=dict(indices=list(indices),**signature(reader.frames(indices)))
        finally:reader.release()
        expected[item['id']]=64
        if item['id'] in audited:
            assert frame_reference[item['id']]['rgb_sha256']==[f['rgb_sha256'] for f in audited[item['id']]['sampled_frames']]
            audited_matches.append(item['id'])
    assert len(video_checks)==30 and len(audited_matches)==4
    terminal=code/'src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'
    original=subprocess.check_output(['git','-C',str(code),'show','17755d7a2afecf8eccae00e301933f9bc216a726:src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'])
    assert sha(terminal)==hashlib.sha256(original).hexdigest()
    recovered=[p['labels']['revision1_6'] for p in read(OLD/'PAIRED-SAMPLE.json') if p['id'] in ids]
    assert len(recovered)==len({r['id'] for r in recovered})==26
    by_id={i['id']:i for i in items}
    for row in recovered:
        item=by_id[row['id'].rsplit('/',1)[0]]
        assert row['task']==task(item,criteria['revision1_6']) and row['observed_through']==item['prefix_controls'] and row['sampled_frames']==64
    prepared=prompts(jobs_for(items,criteria),core);assert len(prepared)==NEW
    write(OUT/'PUBLIC-JOBS.json',items);write(OUT/'PREFIX-REFERENCE.json',reference)
    write(OUT/'RECOVERED.json',dict(rows=recovered,source_file=str(OLD/'PAIRED-SAMPLE.json'),source_sha256=sha(OLD/'PAIRED-SAMPLE.json')))
    write(OUT/'PREPARED-PROMPTS.json',prepared);write(OUT/'FRAME-REFERENCE.json',frame_reference)
    write(OUT/'CPU-CHECKS.json',dict(completed=True,videos=10,prefixes=26,new_outputs=NEW,recovered_labels=26,
        public_sha256=sha(OUT/'PUBLIC-JOBS.json'),recovered_sha256=sha(OUT/'RECOVERED.json'),prompts_sha256=sha(OUT/'PREPARED-PROMPTS.json'),
        frame_reference_sha256=sha(OUT/'FRAME-REFERENCE.json'),criteria_sha256={a:sha(code/'outputs'/RUN/(a+'.json')) for a in criteria},
        expected_sampled_frames=expected,video_checks=video_checks,actual_cpu_prefix_decodes=26,prior_visual_audit_matches=audited_matches,
        unchanged_terminal_worker_sha256=sha(terminal),official_messages_unchanged=True,policy_controls=0,motor_calls=0,optimizer_steps=0))
    print(json.dumps(dict(completed=True,videos=10,prefixes=26,new_outputs=NEW,recovered_labels=26,
        actual_cpu_prefix_decodes=26,prior_visual_audit_matches=len(audited_matches),prompts_sha256=sha(OUT/'PREPARED-PROMPTS.json'))),flush=True)

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
    for name,key in (('PUBLIC-JOBS.json','public_sha256'),('RECOVERED.json','recovered_sha256'),('PREPARED-PROMPTS.json','prompts_sha256'),('FRAME-REFERENCE.json','frame_reference_sha256')):
        assert sha(OUT/name)==checks[key]
    criteria=criteria_for(code)
    assert all(sha(code/'outputs'/RUN/(a+'.json'))==checks['criteria_sha256'][a] for a in criteria)
    for path,meta in checks['video_checks'].items():assert sha(path)==meta['sha256']
    frame_reference=read(OUT/'FRAME-REFERENCE.json');items=read(OUT/'PUBLIC-JOBS.json')
    dest=OUT/'scoring';dest.mkdir(exist_ok=False)
    write(dest/'INPUTS.json',dict(source_commit=os.environ['TASK_CODE_PIN'],model=inputs['model'],official_code_commit=inputs['official_code_commit'],
        criteria_sha256=checks['criteria_sha256'],truth_read=False,custom_messages=False,new_outputs=NEW,recovered_labels=26,raw_rgb_read_audit=True))
    from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
    start=time.time();engine=TerminalWorker(inputs['model'],batch_size=8,gpu_memory_utilization=.78)
    write(dest/'ENGINE-READY.json',dict(gpu_uuid=gpu,cold_seconds=time.time()-start,model_load_count=1,batch_size=8,temperature=0))
    jobs=jobs_for(items,criteria);write(dest/'PROMPTS.json',prompts(jobs,engine.api.core))
    assert sha(dest/'PROMPTS.json')==checks['prompts_sha256']
    original_render,original_collage=engine.api.engine.render,engine.api.core.Collage
    completed=0
    for arm,todo in jobs.items():
        for offset in range(0,len(todo),8):
            assert time.time()<int(os.environ['TASK_STARTED'])+780,'Finite batch deadline'
            batch=todo[offset:offset+8];path=dest/f'{arm}-{offset:02d}.jsonl'
            render_audit=[];frame_audit=[];created=[];begin=time.time()
            class AuditedReader:
                def __init__(self,source,job):
                    self.source,self.job=source,job
                    self.num_frames,self.fps,self.duration_s=source.num_frames,source.fps,source.duration_s
                def frames(self,indices):
                    frames=self.source.frames(indices);uid=self.job['id'].rsplit('/',1)[0]
                    actual=dict(indices=list(indices),**signature(frames))
                    assert actual==frame_reference[uid],'Actual raw RGB input differs from CPU reference'
                    frame_audit.append(dict(id=self.job['id'],**actual))
                    return frames
                def release(self):self.source.release()
            def audited_collage(paths):
                job=batch[len(created)];assert paths==job['videos']
                reader=AuditedReader(original_collage(paths),job);created.append(reader);return reader
            def audited_render(processor,messages,**kwargs):
                job=batch[len(render_audit)];assert messages==engine.api.core.terminal_messages(job['task'],False)
                prompt=original_render(processor,messages,**kwargs);assert isinstance(prompt,str)
                render_audit.append(dict(id=job['id'],rendered_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    messages_sha256=hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()))
                return prompt
            engine.api.engine.render=audited_render;engine.api.core.Collage=audited_collage
            try:rows=engine.score_jobs(batch,path)
            finally:engine.api.engine.render=original_render;engine.api.core.Collage=original_collage
            assert len(rows)==len(render_audit)==len(frame_audit)==len(created)==len(batch)
            assert [r['id'] for r in rows]==[j['id'] for j in batch]==[r['id'] for r in frame_audit]
            assert all(r['observed_through']==j['prefix_controls'] and r['sampled_frames']==64 for r,j in zip(rows,batch))
            write(path.with_suffix('.prompts.json'),render_audit);write(path.with_suffix('.frames.json'),frame_audit)
            write(path.with_suffix('.receipt.json'),dict(completed=True,labels=len(rows),sha256=sha(path),prompts_sha256=sha(path.with_suffix('.prompts.json')),
                frames_sha256=sha(path.with_suffix('.frames.json')),raw_rgb_inputs_match_cpu=True,seconds=time.time()-begin))
            completed+=len(rows);print(json.dumps(dict(new_outputs_completed=completed,total=NEW,arm=arm)),flush=True)
    assert completed==NEW
    write(dest/'COMPLETED.json',dict(completed=True,new_outputs=NEW,candidate_labels=78,anchor_labels=4,recovered_labels=26,actual_rgb_inputs_audited=NEW,robot_actions=0,optimizer_steps=0))

def summary(code):
    dest=OUT/'scoring';assert read(dest/'COMPLETED.json')['new_outputs']==NEW
    cached={r['id']:r for r in read(OUT/'RECOVERED.json')['rows']};rows=dict(cached);anchors=[]
    for path in dest.glob('*.jsonl'):
        assert sha(path)==read(path.with_suffix('.receipt.json'))['sha256']
        for line in path.read_text().splitlines():
            r=json.loads(line)
            if r['id'].endswith('/repeat_both'):
                old=cached[r['id'].rsplit('/',1)[0]+'/revision1_6']
                anchors.append(dict(id=r['id'],cached_score=old['score'],repeated_score=r['score'],equal=old['score']==r['score']))
            else:assert r['id'] not in rows;rows[r['id']]=r
    assert len(rows)==104 and len(anchors)==4
    paired=[dict(ref,labels={a:rows[ref['id']+'/'+a] for a in COMPARISONS}) for ref in read(OUT/'PREFIX-REFERENCE.json')]
    assert len(paired)==26 and sum(not r['native_ever_success'] for r in paired)==14
    write(OUT/'PAIRED-SAMPLE.json',paired);metrics={}
    for arm in COMPARISONS:
        raw=[];confirmed={4:[],5:[]}
        for eid in sorted({r['episode_id'] for r in paired}):
            seq=sorted((r for r in paired if r['episode_id']==eid),key=lambda r:r['control']);actual=seq[0]['native_first_success']
            def record(points):
                first=points[0] if points else None
                return dict(episode_id=eid,native_first_success=actual,first_accepted=first,positive_probes=points,
                    false_accept=first is not None and (actual is None or first<actual),correctly_timed=actual is not None and first is not None and 0<=first-actual<=100)
            raw.append(record([r['control'] for r in seq if r['labels'][arm]['score']==5]))
            for threshold in (4,5):confirmed[threshold].append(record([b['control'] for a,b in zip(seq,seq[1:]) if b['control']-a['control']==50 and min(a['labels'][arm]['score'],b['labels'][arm]['score'])>=threshold]))
        def count(cases):return dict(cases=cases,false_accepted_episodes=sum(c['false_accept'] for c in cases),correctly_timed_episodes=sum(c['correctly_timed'] for c in cases))
        c4=count(confirmed[4]);c5=count(confirmed[5]);false5=sum(not r['native_ever_success'] and r['labels'][arm]['score']==5 for r in paired)
        metrics[arm]=dict(false_full_scores=false5,post_success_full_scores=sum(r['native_ever_success'] and r['labels'][arm]['score']==5 for r in paired),
            raw=count(raw),two_probe4=c4,two_probe5=c5,passes_development_screen=false5==0 and c5['false_accepted_episodes']==0 and c5['correctly_timed_episodes']==6 and all(a['equal'] for a in anchors))
    write(OUT/'RESULT.json',dict(completed=True,new_outputs=NEW,paired_labels=104,anchors=anchors,metrics=metrics,negative_prefixes=14,post_success_prefixes=12,
        primary_rule='Two Score5 probes exactly50 controls apart; zero false Score5 endpoints; all6 successes detected after native crossing within100 controls',
        development_only=True,independent_validation=False,production_reward_qualified=False,first_success_qualified=False))
    print(json.dumps(read(OUT/'RESULT.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','summary']);p.add_argument('--code',type=Path,required=True)
    a=p.parse_args();{'prepare':prepare,'worker':worker,'summary':summary}[a.mode](a.code)
