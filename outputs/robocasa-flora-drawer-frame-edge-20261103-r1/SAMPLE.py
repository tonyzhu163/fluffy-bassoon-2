"""One frozen framing contrast, with raw and live-processor input audits."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = 'robocasa-flora-drawer-frame-edge-20261103-r1'
OLD = ROOT/'outputs/robocasa-flora-drawer-grade-rubric-20261103-r1'
RECENT = ROOT/'outputs/robocasa-flora-drawer-target-hints-20261103-r1'
AUDIT = ROOT/'outputs/robocasa-flora-drawer-processor-audit-20261103-r3'
OUT = ROOT/'outputs'/RUN
ARMS = ('first_success_fixture_depth','first_success_frame_edge')
ANCHORS = (('000112',300),('000128',400),('000144',250),('000064',550))
NEW = 40
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest = lambda b: hashlib.sha256(b).hexdigest()


def write(path,data):
    with Path(path).open('x') as stream:
        json.dump(data,stream,indent=2)
        stream.write('\n')


def case(item):
    return item['id'].split('-case-')[1][:6]


def anchor(item):
    return (case(item),item['prefix_controls']) in ANCHORS


def candidate(item,spec):
    return item['prefix_controls'] in spec['selection'].get(case(item),[])


def criteria_for(code):
    return {arm:read(code/'outputs'/RUN/(arm+'.json')) for arm in ('revision1_6',*ARMS)}


def jobs_for(items,criteria,spec):
    jobs = {}
    for arm in ('repeat_both',*ARMS):
        selected = [i for i in items if anchor(i)] if arm=='repeat_both' else [i for i in items if candidate(i,spec)]
        rubric = criteria['revision1_6' if arm=='repeat_both' else arm]
        jobs[arm] = [dict(id=i['id']+'/'+arm,
            task=rubric['task_template'].format(side=re.fullmatch(r'Open the (left|right) drawer\.',i['task']).group(1)),
            videos=[v['path'] for v in i['videos']],prefix_controls=i['prefix_controls'],initial_frame_included=True)
            for i in selected]
    assert [len(j) for j in jobs.values()]==[4,18,18]
    return jobs


def prompts(jobs,core):
    return [dict(id=j['id'],messages=core.terminal_messages(j['task'],False)) for batch in jobs.values() for j in batch]


def signature(frames):
    import numpy as np
    array = np.asarray(frames)
    if array.shape==(64,3,320,960):
        array = array.transpose(0,2,3,1)
    assert array.dtype==np.uint8 and array.shape==(64,320,960,3)
    hashes = [digest(frame.tobytes()) for frame in array]
    return dict(frame_count=64,shape=[320,960,3],rgb_sha256=hashes,
        ordered_rgb_hashes_sha256=digest(''.join(hashes).encode()))


def features_signature(features,grid):
    import torch
    tensor = torch.as_tensor(features).detach().cpu().contiguous()
    assert tensor.dtype in (torch.float32,torch.bfloat16)
    assert tuple(tensor.shape)==(38400,1536)
    dimensions = torch.as_tensor(grid).cpu().tolist()
    assert dimensions==[[32,20,60]]
    return dict(shape=list(tensor.shape),grid=dimensions,dtype=str(tensor.dtype),
        fp32_sha256=digest(tensor.float().numpy().tobytes()),
        bf16_sha256=digest(tensor.to(torch.bfloat16).view(torch.uint8).numpy().tobytes()))


def prepare(code):
    assert platform.system()=='Linux' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert os.environ.get('PYTHONNOUSERSITE')=='1'
    for path,expected in read(code/'outputs'/RUN/'EXPECTED-INPUTS.json').items():
        assert sha(path)==expected,path
    for source in read(code/'outputs'/RUN/'SOURCE-SURVEY.json').values():
        assert sha(source['path'])==source['sha256']
    spec = read(code/'outputs'/RUN/'PRESPEC.json')
    criteria = criteria_for(code)
    assert criteria[ARMS[0]]==spec['base'] and criteria[ARMS[1]]==spec['candidate']
    assert criteria[ARMS[1]]['task_template']==criteria[ARMS[0]]['task_template']+' '+spec['append_only_clause']
    assert read(OLD/'terminal-charge.json')['exit_code']==0 and read(AUDIT/'RECEIPT.json')['completed']
    terminal = code/'src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'
    original = subprocess.check_output(['git','-C',str(code),'show','17755d7a2afecf8eccae00e301933f9bc216a726:src/fluffy_bassoon/meta_rl/flora_terminal_worker.py'])
    assert sha(terminal)==digest(original)
    items = [i for i in read(OLD/'PUBLIC-JOBS.json') if candidate(i,spec) or anchor(i)]
    ids = {i['id'] for i in items}
    reference = [r for r in read(OLD/'PREFIX-REFERENCE.json') if r['id'] in ids and candidate(dict(r,prefix_controls=r['control']),spec)]
    assert len(items)==len(ids)==19 and len({i['id'].split('/prefix-')[0] for i in items})==8
    assert len(reference)==18 and sum(not r['native_ever_success'] for r in reference)==10
    for ref in reference:
        assert sha(ref['replay_file'])==ref['replay_sha256']
        replay = read(ref['replay_file'])
        assert replay['qualified'] and replay['first_success_control']==ref['native_first_success']
        assert replay['states'][ref['control']]['success']==ref['native_current_success']
        assert ref['native_ever_success']==(ref['native_first_success'] is not None and ref['control']>=ref['native_first_success'])
    OUT.mkdir(exist_ok=False)
    print('FROZEN_INPUTS_VERIFIED',time.time(),flush=True)
    import numpy as np
    import torch
    from transformers.video_utils import VideoMetadata
    from rewarder.batch_utils import core,engine
    from fluffy_bassoon.meta_rl.flora_terminal_worker import _PrefixCollage
    torch.set_num_threads(4)
    inputs = read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
    assert sha(inputs['model_receipt']['path'])==inputs['model_receipt']['sha256']
    assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip()==inputs['official_code_commit']
    processor = engine.load_processor(inputs['model'])
    print('PROCESSOR_READY',time.time(),flush=True)
    previous = read(RECENT/'FRAME-REFERENCE.json')
    audited = {c['id']:c for c in read(AUDIT/'RECEIPT.json')['cases']}
    videos,frames_ref,processor_ref = {},{},{}
    for item in items:
        assert set(item)=={'id','task_key','task','horizon','videos','prefix_controls','initial_frame_included'}
        for video in item['videos']:
            if video['path'] not in videos:
                assert sha(video['path'])==video['sha256']
                reader = core.Collage([video['path']])
                try:
                    assert reader.num_frames==751 and reader.fps==20.
                    videos[video['path']]=dict(sha256=video['sha256'],frames=751,fps=20.)
                finally:reader.release()
        reader = _PrefixCollage(core.Collage([v['path'] for v in item['videos']]),item['prefix_controls'])
        try:
            indices = core.terminal_frame_indices(reader.num_frames,reader.fps)
            frames = np.asarray(reader.frames(indices))
            metadata = core.video_metadata(indices,reader.fps,reader.num_frames)
        finally:reader.release()
        uid = item['id']
        frames_ref[uid]=dict(indices=list(indices),**signature(frames))
        if uid in previous:assert frames_ref[uid]==previous[uid]
        messages = core.terminal_messages('Open the left drawer.',False)
        prompt = engine.render(processor,messages,thinking=False)
        result = processor(text=[prompt],videos=[[frames]],
            video_metadata=[[VideoMetadata(**{k:v for k,v in metadata.items() if k!='do_sample_frames'})]],
            do_sample_frames=False,size=engine.COLLAGE_VIDEO_SIZE,return_tensors='pt')
        processor_ref[uid]=features_signature(result['pixel_values_videos'],result['video_grid_thw'])
        if uid in audited:
            assert frames_ref[uid]['rgb_sha256']==audited[uid]['raw_rgb_sha256']
            assert processor_ref[uid]['fp32_sha256']==audited[uid]['feature_sha256']
            assert processor_ref[uid]['bf16_sha256']==audited[uid]['bf16_feature_sha256']
        del result,frames
        gc.collect()
        print(json.dumps(dict(cpu_prefixes_completed=len(frames_ref),total=19,id=uid)),flush=True)
    assert len(videos)==24 and len(frames_ref)==len(processor_ref)==19
    assert len({r['ordered_rgb_hashes_sha256'] for r in frames_ref.values()})==19
    recovered = [p['labels']['revision1_6'] for p in read(OLD/'PAIRED-SAMPLE.json') if p['id'] in ids and anchor(dict(p,prefix_controls=p['control']))]
    assert len(recovered)==4
    prepared = prompts(jobs_for(items,criteria,spec),core)
    assert len(prepared)==NEW
    assert not torch.cuda.is_initialized()
    write(OUT/'PUBLIC-JOBS.json',items)
    write(OUT/'PREFIX-REFERENCE.json',reference)
    write(OUT/'RECOVERED.json',dict(rows=recovered,source_file=str(OLD/'PAIRED-SAMPLE.json'),source_sha256=sha(OLD/'PAIRED-SAMPLE.json')))
    write(OUT/'PREPARED-PROMPTS.json',prepared)
    write(OUT/'FRAME-REFERENCE.json',frames_ref)
    write(OUT/'PROCESSOR-REFERENCE.json',processor_ref)
    write(OUT/'CPU-CHECKS.json',dict(completed=True,videos=8,candidate_videos=7,prefixes=19,candidate_prefixes=18,
        negative_prefixes=10,post_success_prefixes=8,new_outputs=NEW,anchor_repeats=4,
        public_sha256=sha(OUT/'PUBLIC-JOBS.json'),recovered_sha256=sha(OUT/'RECOVERED.json'),
        prompts_sha256=sha(OUT/'PREPARED-PROMPTS.json'),frame_reference_sha256=sha(OUT/'FRAME-REFERENCE.json'),
        processor_reference_sha256=sha(OUT/'PROCESSOR-REFERENCE.json'),criteria_sha256={a:sha(code/'outputs'/RUN/(a+'.json')) for a in criteria},
        video_checks=videos,actual_cpu_prefix_decodes=19,actual_cpu_processor_calls=19,
        previous_processor_audit_matches=4,unchanged_terminal_worker_sha256=sha(terminal),
        official_messages_unchanged=True,policy_controls=0,motor_calls=0,optimizer_steps=0,model_weight_loads=0))
    print('ALL_CPU_CHECKS_PASS',flush=True)


def worker(code):
    assert platform.system()=='Linux'
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    import torch
    gpu = verify_selected_cuda(torch)
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE'):
        os.environ.pop(key,None)
    inputs = read(code/'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json')
    checks = read(OUT/'CPU-CHECKS.json')
    assert checks['completed']
    for name,key in (('PUBLIC-JOBS.json','public_sha256'),('RECOVERED.json','recovered_sha256'),
        ('PREPARED-PROMPTS.json','prompts_sha256'),('FRAME-REFERENCE.json','frame_reference_sha256'),
        ('PROCESSOR-REFERENCE.json','processor_reference_sha256')):
        assert sha(OUT/name)==checks[key]
    criteria = criteria_for(code)
    assert all(sha(code/'outputs'/RUN/(a+'.json'))==checks['criteria_sha256'][a] for a in criteria)
    for path,meta in checks['video_checks'].items():assert sha(path)==meta['sha256']
    assert sha(inputs['model_receipt']['path'])==inputs['model_receipt']['sha256']
    assert read(inputs['model_receipt']['path'])['completed']
    assert subprocess.check_output(['git','-C',inputs['official_code'],'rev-parse','HEAD'],text=True).strip()==inputs['official_code_commit']
    spec = read(code/'outputs'/RUN/'PRESPEC.json')
    frames_ref,processor_ref = read(OUT/'FRAME-REFERENCE.json'),read(OUT/'PROCESSOR-REFERENCE.json')
    by_rgb = {r['ordered_rgb_hashes_sha256']:uid for uid,r in frames_ref.items()}
    items = read(OUT/'PUBLIC-JOBS.json')
    dest = OUT/'scoring'
    dest.mkdir(exist_ok=False)
    write(dest/'INPUTS.json',dict(source_commit=os.environ['TASK_CODE_PIN'],model=inputs['model'],
        official_code_commit=inputs['official_code_commit'],criteria_sha256=checks['criteria_sha256'],
        truth_read=False,custom_messages=False,new_outputs=NEW,live_processor_audit=True))
    from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
    started = time.time()
    engine = TerminalWorker(inputs['model'],batch_size=8,gpu_memory_utilization=.78)
    write(dest/'ENGINE-READY.json',dict(gpu_uuid=gpu,cold_seconds=time.time()-started,model_load_count=1,batch_size=8,temperature=0))
    jobs = jobs_for(items,criteria,spec)
    write(dest/'PROMPTS.json',prompts(jobs,engine.api.core))
    assert sha(dest/'PROMPTS.json')==checks['prompts_sha256']
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
    original_processor = Qwen3VLMultiModalProcessor._call_hf_processor
    original_render,original_collage = engine.api.engine.render,engine.api.core.Collage
    processor_events = []

    def audited_processor(self,prompt,mm_data,mm_kwargs,tok_kwargs):
        result = original_processor(self,prompt,mm_data,mm_kwargs,tok_kwargs)
        videos = mm_data.get('videos',[])
        if videos:
            assert len(videos)==1
            raw = signature(videos[0][0])
            uid = by_rgb[raw['ordered_rgb_hashes_sha256']]
            actual = features_signature(result['pixel_values_videos'],result['video_grid_thw'])
            assert actual['bf16_sha256']==processor_ref[uid]['bf16_sha256'],'Live processor input differs from CPU reference'
            processor_events.append(dict(id=uid,**actual,
                fp32_equal=actual['fp32_sha256']==processor_ref[uid]['fp32_sha256'],bf16_equal=True))
        return result

    completed = 0
    Qwen3VLMultiModalProcessor._call_hf_processor = audited_processor
    try:
        for arm,todo in jobs.items():
            for offset in range(0,len(todo),8):
                assert time.time()<int(os.environ['TASK_STARTED'])+780,'Finite batch deadline'
                batch = todo[offset:offset+8]
                path = dest/f'{arm}-{offset:02d}.jsonl'
                render_audit,frame_audit,created = [],[],[]
                begin = time.time()

                class AuditedReader:
                    def __init__(self,source,job):
                        self.source,self.job=source,job
                        self.num_frames,self.fps,self.duration_s=source.num_frames,source.fps,source.duration_s
                    def frames(self,indices):
                        frames = self.source.frames(indices)
                        uid = self.job['id'].rsplit('/',1)[0]
                        actual = dict(indices=list(indices),**signature(frames))
                        assert actual==frames_ref[uid]
                        frame_audit.append(dict(id=self.job['id'],**actual))
                        return frames
                    def release(self):self.source.release()

                def audited_collage(paths):
                    job = batch[len(created)]
                    assert paths==job['videos']
                    reader = AuditedReader(original_collage(paths),job)
                    created.append(reader)
                    return reader

                def audited_render(processor,messages,**kwargs):
                    job = batch[len(render_audit)]
                    assert messages==engine.api.core.terminal_messages(job['task'],False)
                    text = original_render(processor,messages,**kwargs)
                    render_audit.append(dict(id=job['id'],rendered_sha256=digest(text.encode()),
                        messages_sha256=digest(json.dumps(messages,sort_keys=True).encode())))
                    return text

                engine.api.engine.render,engine.api.core.Collage = audited_render,audited_collage
                first_event = len(processor_events)
                try:rows = engine.score_jobs(batch,path)
                finally:engine.api.engine.render,engine.api.core.Collage=original_render,original_collage
                assert len(rows)==len(render_audit)==len(frame_audit)==len(created)==len(batch)
                assert [r['id'] for r in rows]==[j['id'] for j in batch]==[r['id'] for r in frame_audit]
                write(path.with_suffix('.prompts.json'),render_audit)
                write(path.with_suffix('.frames.json'),frame_audit)
                write(path.with_suffix('.processor.json'),processor_events[first_event:])
                write(path.with_suffix('.receipt.json'),dict(completed=True,labels=len(rows),sha256=sha(path),
                    prompts_sha256=sha(path.with_suffix('.prompts.json')),frames_sha256=sha(path.with_suffix('.frames.json')),
                    processor_sha256=sha(path.with_suffix('.processor.json')),seconds=time.time()-begin))
                completed += len(rows)
                print(json.dumps(dict(new_outputs_completed=completed,total=NEW,arm=arm)),flush=True)
    finally:Qwen3VLMultiModalProcessor._call_hf_processor=original_processor
    seen = {event['id'] for event in processor_events}
    assert completed==NEW and seen==set(frames_ref),'Every distinct input must be observed in live preprocessing'
    write(dest/'PROCESSOR-AUDIT.json',dict(completed=True,events=processor_events,unique_inputs=len(seen),
        all_bf16_equal=True,all_fp32_equal=all(e['fp32_equal'] for e in processor_events),
        limitation='Actual vLLM HF processor outputs during these scoring calls; no learned encoder/LLM internal states saved'))
    write(dest/'COMPLETED.json',dict(completed=True,new_outputs=NEW,candidate_labels=36,anchor_labels=4,
        actual_rgb_inputs_audited=NEW,unique_live_processor_inputs=len(seen),robot_actions=0,optimizer_steps=0))


def summary(code):
    dest = OUT/'scoring'
    assert read(dest/'COMPLETED.json')['new_outputs']==NEW
    old = {r['id']:r for r in read(OUT/'RECOVERED.json')['rows']}
    rows,anchors = {},[]
    for path in dest.glob('*.jsonl'):
        assert sha(path)==read(path.with_suffix('.receipt.json'))['sha256']
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['id'].endswith('/repeat_both'):
                prior = old[row['id'].rsplit('/',1)[0]+'/revision1_6']
                anchors.append(dict(id=row['id'],cached_score=prior['score'],repeated_score=row['score'],equal=prior['score']==row['score']))
            else:
                assert row['id'] not in rows
                rows[row['id']]=row
    assert len(rows)==36 and len(anchors)==4
    paired = [dict(ref,labels={a:rows[ref['id']+'/'+a] for a in ARMS}) for ref in read(OUT/'PREFIX-REFERENCE.json')]
    write(OUT/'PAIRED-SAMPLE.json',paired)
    metrics = {}
    for arm in ARMS:
        accepted = []
        for eid in sorted({r['episode_id'] for r in paired}):
            seq = sorted((r for r in paired if r['episode_id']==eid),key=lambda r:r['control'])
            actual = seq[0]['native_first_success']
            points = [b['control'] for a,b in zip(seq,seq[1:]) if b['control']-a['control']==50 and min(a['labels'][arm]['score'],b['labels'][arm]['score'])==5]
            first = points[0] if points else None
            accepted.append(dict(episode_id=eid,native_first_success=actual,first_accepted=first,
                false_accept=first is not None and (actual is None or first<actual),
                correctly_timed=actual is not None and first is not None and 0<=first-actual<=100))
        false5 = sum(not r['native_ever_success'] and r['labels'][arm]['score']==5 for r in paired)
        closed = [dict(id=r['id'],score=r['labels'][arm]['score']) for r in paired if r['native_ever_success'] and not r['native_current_success'] and r['control']==750]
        assert len(closed)==2
        timely = sum(r['correctly_timed'] for r in accepted)
        false_episodes = sum(r['false_accept'] for r in accepted)
        metrics[arm]=dict(false_full_scores=false5,
            post_success_full_scores=sum(r['native_ever_success'] and r['labels'][arm]['score']==5 for r in paired),
            correctly_timed_confirmed_episodes=timely,false_accepted_episodes=false_episodes,cases=accepted,
            closed_again=closed,closed_again_full_scores=sum(r['score']==5 for r in closed),
            passes_development_screen=false5==0 and false_episodes==0 and timely==3 and all(r['score']==5 for r in closed) and all(a['equal'] for a in anchors))
    write(OUT/'RESULT.json',dict(completed=True,new_outputs=NEW,paired_labels=36,anchors=anchors,metrics=metrics,
        negative_prefixes=10,post_success_prefixes=8,unique_inputs=19,live_processor_verified=True,
        development_only=True,independent_validation=False,production_reward_qualified=False,first_success_qualified=False))
    print(json.dumps(read(OUT/'RESULT.json')),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','worker','summary'))
    parser.add_argument('--code',type=Path,required=True)
    args = parser.parse_args()
    {'prepare':prepare,'worker':worker,'summary':summary}[args.mode](args.code)
