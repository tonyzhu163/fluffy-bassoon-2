"""Small exact-native-threshold comparison on ten retained trajectories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT=Path('/mowenjie/fluffy-bassoon-2')
RUN='robocasa-flora-drawer-exact-threshold-20261103-r1'
OLD=ROOT/'outputs/robocasa-flora-drawer-location-exclusion-20261103-r1'
EXTRA=ROOT/'outputs/robocasa-flora-drawer-extra-failures-20261103-r1'
OUT=ROOT/'outputs'/RUN
ARMS=('direct_fraction','explicit_boundary')
COMPARISONS=('revision1_6',*ARMS)
ANCHORS=(('000112',300),('000128',400),('000144',250),('000064',550))
NEW=104

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
    with Path(p).open('x') as f:json.dump(d,f,indent=2);f.write('\n')
def frozen_inputs(code):
    for path,digest in read(code/'outputs'/RUN/'EXPECTED-INPUTS.json').items():assert sha(path)==digest,path
    assert read(OLD/'terminal-charge.json')['exit_code']==0
    assert all(a['equal'] for a in read(OLD/'RESULT.json')['anchors'])
    assert read(EXTRA/'CPU-RECEIPT.json')['qualified']==4

def geometry(code):
    assert platform.system()=='Linux';frozen_inputs(code);OUT.mkdir(exist_ok=False)
    profile=read(code/'outputs/robocasa-opendrawer-rubric-diagnosis-20261102-r1/REPLAY-SOURCE.json')['native_profile']
    os.environ.update(profile);os.environ['CUDA_VISIBLE_DEVICES']='';sys.path[:0]=profile['PYTHONPATH'].split(':')
    import numpy as np
    import torch
    assert not torch.cuda.is_available()
    from hydra import compose,initialize_config_dir
    import robosuite
    import robocasa
    from robocasa.utils.env_utils import create_env
    from robocasa.models.fixtures.cabinets import Drawer
    with initialize_config_dir(config_dir=str(code/'configs/native_ppo'),version_base=None):cfg=compose(config_name='robocasa365_pi05_eval')
    c=cfg.env.eval;native_make=robosuite.make
    def no_render(**kwargs):
        kwargs.update(has_renderer=False,has_offscreen_renderer=False,use_camera_obs=False)
        return native_make(**kwargs)
    robosuite.make=no_render
    def canonical(v,key=None):
        if isinstance(v,dict):return {k:canonical(x,k) for k,x in v.items()}
        if isinstance(v,list):return sorted(v) if key=='groups_containing_sampled_obj' else [canonical(x) for x in v]
        return v
    rows=[]
    for item in read(EXTRA/'SOURCE-SURVEY.json')['episodes']:
        saved=item['monitor']['native_ep_meta']
        env=create_env('OpenDrawer',robots=c.robot_name,camera_names=list(c.camera_names),camera_widths=c.init_params.camera_widths,camera_heights=c.init_params.camera_heights,seed=item['monitor']['case']['native_seed'],split='target',render_onscreen=False,camera_depths=False,translucent_robot=bool(c.get('translucent_robot',False)),randomize_cameras=bool(c.get('randomize_cameras',False)),generative_textures=c.get('generative_textures',None))
        try:
            env.reset();meta=json.loads(json.dumps(env.get_ep_meta()));equal={k:canonical(meta.get(k))==canonical(v) for k,v in saved.items()};assert all(equal.values()),item['id']
            target=env.drawer;drawers=[f for f in env.fixtures.values() if isinstance(f,Drawer)]
            stack=[f for f in drawers if np.linalg.norm(np.asarray(f.pos)[:2]-np.asarray(target.pos)[:2])<1e-4]
            above=[f.name for f in stack if f.pos[2]>target.pos[2]+1e-5]
            assert .7<=target.pos[2]<=.9 and not above,item['id']
            row=dict(id=item['id'],target=target.name,target_pos=np.asarray(target.pos).tolist(),metadata_equal=equal,drawers_above_target=above)
            rows.append(row);print(json.dumps(row),flush=True)
        finally:env.close()
    assert len(rows)==4
    write(OUT/'GEOMETRY.json',dict(completed=True,episodes=rows,policy_controls=0,motor_calls=0,gpu_calls=0,cuda_visible_devices='',limitations='New four-scene reset audit; the original six-scene audit is frozen separately.'))

def task(item,criterion):return criterion['task_template'].format(side=re.fullmatch(r'Open the (left|right) drawer\.',item['task']).group(1))
def anchor(item):return any('-case-'+case+'-' in item['id'] and item['prefix_controls']==control for case,control in ANCHORS)
def jobs_for(items,criteria):
    old_ids={i['id'] for i in read(OLD/'PUBLIC-JOBS.json')}
    jobs={}
    for arm in ('repeat_both','revision1_6',*ARMS):
        chosen=[i for i in items if anchor(i)] if arm=='repeat_both' else [i for i in items if i['id'] not in old_ids] if arm=='revision1_6' else items
        criterion=criteria['revision1_6' if arm=='repeat_both' else arm]
        jobs[arm]=[dict(id=i['id']+'/'+arm,task=task(i,criterion),videos=[v['path'] for v in i['videos']],prefix_controls=i['prefix_controls'],initial_frame_included=True) for i in chosen]
    assert [len(jobs[a]) for a in jobs]==[4,10,45,45]
    return jobs

def prompts(jobs,core):return [dict(id=j['id'],messages=core.terminal_messages(j['task'],False)) for batch in jobs.values() for j in batch]
def prepare(code):
    assert platform.system()=='Linux';frozen_inputs(code);assert read(OUT/'GEOMETRY.json')['completed']
    extras=code/'outputs'/EXTRA.name
    items=read(OLD/'PUBLIC-JOBS.json')+read(extras/'PUBLIC-ADDITIONS.json')
    reference=read(OLD/'PREFIX-REFERENCE.json')+read(extras/'PREFIX-ADDITIONS.json')
    assert len(items)==len(reference)==45 and len({i['id'] for i in items})==45
    assert {i['id'] for i in items}=={r['id'] for r in reference}
    assert len({r['episode_id'] for r in reference})==10
    for ref in reference:
        assert sha(ref['replay_file'])==ref['replay_sha256']
        replay=read(ref['replay_file']);assert replay['qualified']
        assert replay['first_success_control']==ref['native_first_success']
        state=replay['states'][ref['control']]
        assert state['control']==ref['control'] and state['success']==ref['native_current_success']
        assert ref['native_ever_success']==(ref['native_first_success'] is not None and ref['control']>=ref['native_first_success'])
    write(OUT/'PUBLIC-JOBS.json',items);write(OUT/'PREFIX-REFERENCE.json',reference)
    criteria={a:read(code/'outputs'/RUN/(a+'.json')) for a in COMPARISONS}
    from rewarder.batch_utils import core
    from fluffy_bassoon.meta_rl.flora_terminal_worker import _PrefixCollage
    expected={};video_checks={}
    for i in items:
        assert set(i)=={'id','task_key','task','horizon','videos','prefix_controls','initial_frame_included'}
        assert i['horizon']==750 and i['initial_frame_included']
        for v in i['videos']:
            if v['path'] not in video_checks:
                assert sha(v['path'])==v['sha256'],v['path']
                reader=core.Collage([v['path']])
                try:assert reader.num_frames==751 and reader.fps==20.;video_checks[v['path']]=dict(sha256=v['sha256'],frames=reader.num_frames,fps=reader.fps)
                finally:reader.release()
        indices=core.terminal_frame_indices(i['prefix_controls']+1,20.);assert min(indices)>=0 and max(indices)<=i['prefix_controls'];expected[i['id']]=len(indices)
    assert len(video_checks)==30
    probes=[]
    for case,control in (('000112',50),('000112',100),('000112',750),('000216',250),('000256',700),('000336',750),('000376',750)):
        i=next(i for i in items if '-case-'+case+'-' in i['id'] and i['prefix_controls']==control)
        reader=_PrefixCollage(core.Collage([v['path'] for v in i['videos']]),control)
        try:
            indices=core.terminal_frame_indices(reader.num_frames,reader.fps);assert len(reader.frames(indices))==expected[i['id']]
            probes.append(dict(id=i['id'],controls=control,sampled_frames=len(indices),max_index=max(indices)))
        finally:reader.release()
    terminal=code/'src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'
    original=subprocess.check_output(['git','-C',str(code),'show','17755d7a2afecf8eccae00e301933f9bc216a726:src/fluffy_bassoon/meta_rl/flora_terminal_worker.py']);assert sha(terminal)==hashlib.sha256(original).hexdigest()
    recovered=[p['labels']['revision1_6'] for p in read(OLD/'PAIRED-SAMPLE.json')];assert len(recovered)==35
    for row,item in zip(recovered,read(OLD/'PUBLIC-JOBS.json')):
        assert row['id']==item['id']+'/revision1_6' and row['task']==task(item,criteria['revision1_6'])
        assert row['sampled_frames']==expected[item['id']] and row['observed_through']==item['prefix_controls']
    jobs=jobs_for(items,criteria);prepared=prompts(jobs,core);assert len(prepared)==NEW
    write(OUT/'RECOVERED.json',dict(rows=recovered,source_file=str(OLD/'PAIRED-SAMPLE.json'),source_sha256=sha(OLD/'PAIRED-SAMPLE.json')))
    write(OUT/'PREPARED-PROMPTS.json',prepared)
    write(OUT/'CPU-CHECKS.json',dict(completed=True,videos=10,prefixes=45,new_outputs=NEW,recovered_labels=35,public_sha256=sha(OUT/'PUBLIC-JOBS.json'),recovered_sha256=sha(OUT/'RECOVERED.json'),prompts_sha256=sha(OUT/'PREPARED-PROMPTS.json'),criteria_sha256={a:sha(code/'outputs'/RUN/(a+'.json')) for a in criteria},expected_sampled_frames=expected,actual_frame_probes=probes,video_checks=video_checks,unchanged_terminal_worker_sha256=sha(terminal),geometry_sha256=sha(OUT/'GEOMETRY.json'),official_messages_unchanged=True,policy_controls=0,motor_calls=0,optimizer_steps=0))
    print(json.dumps(dict(completed=True,videos=10,prefixes=45,new_outputs=NEW,recovered_labels=35,prompts_sha256=sha(OUT/'PREPARED-PROMPTS.json'),frame_probes=probes)),flush=True)

def worker(code):
    assert platform.system()=='Linux'
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    import torch
    gpu=verify_selected_cuda(torch)
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE'):os.environ.pop(key,None)
    inputs=read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
    assert sha(inputs['model_receipt']['path'])==inputs['model_receipt']['sha256'] and read(inputs['model_receipt']['path'])['completed']
    assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip()==inputs['official_code_commit']
    checks=read(OUT/'CPU-CHECKS.json');assert checks['completed'] and sha(OUT/'GEOMETRY.json')==checks['geometry_sha256']
    for filename,key in (('PUBLIC-JOBS.json','public_sha256'),('RECOVERED.json','recovered_sha256'),('PREPARED-PROMPTS.json','prompts_sha256')):assert sha(OUT/filename)==checks[key]
    criteria={a:read(code/'outputs'/RUN/(a+'.json')) for a in COMPARISONS}
    assert all(sha(code/'outputs'/RUN/(a+'.json'))==checks['criteria_sha256'][a] for a in criteria)
    items=read(OUT/'PUBLIC-JOBS.json')
    for path,meta in checks['video_checks'].items():assert sha(path)==meta['sha256']
    dest=OUT/'scoring';dest.mkdir(exist_ok=False)
    write(dest/'INPUTS.json',dict(source_commit=os.environ['TASK_CODE_PIN'],model=inputs['model'],official_code_commit=inputs['official_code_commit'],criteria_sha256=checks['criteria_sha256'],truth_read=False,custom_messages=False,new_outputs=NEW,recovered_labels=35))
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
    write(dest/'COMPLETED.json',dict(completed=True,new_outputs=NEW,candidate_labels=90,new_reference_labels=10,anchor_labels=4,recovered_labels=35,robot_actions=0,optimizer_steps=0))

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
    assert len(rows)==135 and len(anchors)==4
    paired=[dict(ref,labels={a:rows[ref['id']+'/'+a] for a in COMPARISONS}) for ref in read(OUT/'PREFIX-REFERENCE.json')]
    assert sum(not r['native_ever_success'] for r in paired)==27 and sum(r['native_ever_success'] for r in paired)==18
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
        c4=count(confirmed[4]);c5=count(confirmed[5]);false5=sum(not r['native_ever_success'] and r['labels'][arm]['score']==5 for r in paired)
        metrics[arm]=dict(false_full_scores=false5,post_success_full_scores=sum(r['native_ever_success'] and r['labels'][arm]['score']==5 for r in paired),raw=count(raw),two_probe4=c4,two_probe5=c5,passes_development_screen=false5==0 and c5['false_accepted_episodes']==0 and c5['correctly_timed_episodes']==6 and all(a['equal'] for a in anchors))
    write(OUT/'RESULT.json',dict(completed=True,new_outputs=NEW,paired_labels=135,anchors=anchors,metrics=metrics,negative_prefixes=27,post_success_prefixes=18,primary_rule='Two Score5 probes exactly50 controls apart; zero false Score5 endpoints; all6 successes detected after native crossing within100 controls',development_only=True,independent_validation=False,production_reward_qualified=False,first_success_qualified=False))
    print(json.dumps(read(OUT/'RESULT.json')),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['geometry','prepare','worker','summary']);p.add_argument('--code',type=Path);a=p.parse_args();{'geometry':geometry,'prepare':prepare,'worker':worker,'summary':lambda _:summary()}[a.mode](a.code)
