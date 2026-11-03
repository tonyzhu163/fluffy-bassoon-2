"""Run-local 2x2 evidence/reporting diagnostic; no production scorer changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUN = 'robocasa-flora-current-state-20261103-r2'
OUT = ROOT / 'outputs' / RUN
ARMS = ('prefix_grade', 'anchors_grade', 'prefix_conditions', 'anchors_conditions')
ANCHORS = (('000112', 300), ('000128', 400), ('000144', 250), ('000064', 550))
CONDITION_REGEX = r'Target: (identified|uncertain)\nOpening: (meets|short|uncertain)'
CONDITION_GRAMMAR = r'(Target: identified\nOpening: (meets|short|uncertain)|Target: uncertain\nOpening: uncertain)'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def readref(reference):
    assert sha(reference['path']) == reference['sha256'], reference['path']
    return read(reference['path'])


def case(item):
    return item['id'].split('-case-')[1][:6]


def mode(arm):
    return 'anchors' if arm.startswith('anchors_') else 'prefix'


def conditions(text):
    match = re.fullmatch(CONDITION_REGEX, text.strip())
    if not match:
        raise ValueError('Malformed condition report: ' + repr(text))
    target, opening = match.groups()
    if target == 'uncertain' and opening != 'uncertain':
        raise ValueError('Unidentified target cannot have an opening judgment')
    return dict(target=target, opening=opening,
                accepted=target == 'identified' and opening == 'meets',
                uncertain=target == 'uncertain' or opening == 'uncertain')


def messages(job, spec, core):
    if job['arm'] == 'reference':
        return core.terminal_messages(job['task'], False)
    side = re.fullmatch(r'Open the (left|right) drawer\.', job['task']).group(1)
    system = spec['criterion'].format(side=side)
    if job['arm'].endswith('_conditions'):
        system += '\n\n' + spec['condition_output']
    else:
        system += '\n\n' + spec['graded_output']
    return [dict(role='system', content=system), dict(role='user', content=[
        dict(type='text', text='Task: ' + job['task'] + '\n\nObserved causal video evidence:'),
        dict(type='video', video=core.VIDEO_PLACEHOLDER),
        dict(type='text', text='Judge the requested drawer at the LAST visible time only.')])]


def jobs_for(items, spec):
    jobs = []
    for item in items:
        key = (case(item), item['prefix_controls'])
        if key in ANCHORS:
            side = re.fullmatch(r'Open the (left|right) drawer\.', item['task']).group(1)
            jobs.append(dict(id=item['id'] + '/reference', source_id=item['id'], arm='reference',
                task=spec['reference_task'].format(side=side), videos=item['videos'],
                control=item['prefix_controls'], evidence='prefix'))
        if item['id'] in spec['candidate_ids']:
            for arm in ARMS:
                jobs.append(dict(id=item['id'] + '/' + arm, source_id=item['id'], arm=arm,
                    task=item['task'], videos=item['videos'], control=item['prefix_controls'], evidence=mode(arm)))
    assert len(jobs) == 76 and sum(j['arm'] == 'reference' for j in jobs) == 4
    return jobs


def frame_indices(job, core):
    control = job['control']
    indices = [0, 0, control, control] if job['evidence'] == 'anchors' else core.terminal_frame_indices(control + 1, 20.)
    assert indices[0] == 0 and indices[-1] == control and max(indices) <= control
    assert len(indices) == (4 if job['evidence'] == 'anchors' else 64)
    return list(indices)


def rgb_signature(frames):
    import numpy as np
    array = np.asarray(frames)
    if array.ndim == 4 and array.shape[1] == 3:
        array = array.transpose(0, 2, 3, 1)
    assert array.dtype == np.uint8 and array.shape[1:] == (320, 960, 3)
    hashes = [digest(frame.tobytes()) for frame in array]
    return dict(shape=list(array.shape), frame_sha256=hashes,
                rgb_sha256=digest(''.join(hashes).encode()))


def features_signature(features, grid):
    import torch
    tensor = torch.as_tensor(features).detach().cpu().contiguous()
    assert tensor.dtype in (torch.float32, torch.bfloat16)
    dimensions = torch.as_tensor(grid).cpu().tolist()
    assert len(dimensions) == 1 and dimensions[0][1:] == [20, 60]
    return dict(shape=list(tensor.shape), grid=dimensions, dtype=str(tensor.dtype),
                bf16_sha256=digest(tensor.to(torch.bfloat16).view(torch.uint8).numpy().tobytes()))


def prepare(code):
    assert platform.system() == 'Linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert os.environ.get('PYTHONNOUSERSITE') == '1'
    spec = read(code / 'outputs' / RUN / 'SPEC.json')
    inputs = read(code / 'outputs' / RUN / 'INPUTS.json')
    refs = readref(inputs['reference'])
    items = readref(inputs['public_jobs'])
    old_pairs = readref(inputs['old_pairs'])
    assert len(refs) == 18 and {r['id'] for r in refs} == set(spec['candidate_ids'])
    assert sum(r['native_current_success'] for r in refs) == 6
    assert sum(not r['native_current_success'] for r in refs) == 12
    for ref in refs:
        replay = readref(dict(path=ref['replay_file'], sha256=ref['replay_sha256']))
        assert replay['qualified'] and replay['first_success_control'] == ref['native_first_success']
        assert replay['states'][ref['control']]['success'] == ref['native_current_success']
    official = readref(dict(path=str(code / 'outputs/robocasa-flora-five-task-fifty-score-20261102-r1/INPUTS.json'), sha256=inputs['official_inputs']['sha256']))
    assert readref(official['model_receipt'])['completed']
    assert subprocess.check_output(['git', '-C', official['official_code'], 'rev-parse', 'HEAD'], text=True).strip() == official['official_code_commit']
    for path, expected in inputs['scorer_sources'].items():
        assert sha(path) == expected, path
    OUT.mkdir(exist_ok=False)
    import numpy as np
    import torch
    from transformers.video_utils import VideoMetadata
    from rewarder.batch_utils import core, engine
    torch.set_num_threads(4)
    jobs = jobs_for(items, spec)
    processor = engine.load_processor(official['model'])
    print('PROCESSOR_READY', flush=True)
    frames_ref, processor_ref, video_checks = {}, {}, {}
    prepared = [dict(id=j['id'], messages=messages(j, spec, core)) for j in jobs]
    for j in jobs:
        key = j['source_id'] + '/' + j['evidence']
        if key in frames_ref:
            continue
        for v in j['videos']:
            if v['path'] not in video_checks:
                assert sha(v['path']) == v['sha256'], v['path']
                info = core.probe_video(v['path'])
                assert info.num_frames == 751 and info.fps == 20.
                video_checks[v['path']] = v['sha256']
        collage = core.Collage([v['path'] for v in j['videos']])
        try:
            indices = frame_indices(j, core)
            frames = collage.frames(indices)
            metadata = core.video_metadata(indices, 20., j['control'] + 1)
            signature = rgb_signature(frames)
            if j['evidence'] == 'anchors':
                assert np.array_equal(frames[0], frames[1]) and np.array_equal(frames[2], frames[3])
            frames_ref[key] = dict(indices=indices, metadata=metadata, **signature)
            prompt = engine.render(processor, messages(j, spec, core), thinking=False)
            result = processor(text=[prompt], videos=[[frames]],
                video_metadata=[[VideoMetadata(**{k:v for k,v in metadata.items() if k != 'do_sample_frames'})]],
                do_sample_frames=False, size=engine.COLLAGE_VIDEO_SIZE, return_tensors='pt')
            processor_ref[key] = features_signature(result['pixel_values_videos'], result['video_grid_thw'])
            assert processor_ref[key]['grid'][0][0] == len(indices) // 2
        finally:
            collage.release()
        print(json.dumps(dict(cpu_inputs=len(frames_ref), total=37, key=key)), flush=True)
    assert len(frames_ref) == len(processor_ref) == 37
    recovered = {}
    for j in jobs:
        if j['arm'] == 'reference':
            old = next(p for p in old_pairs if p['id'] == j['source_id'])
            recovered[j['source_id']] = old['labels']['revision1_6']['score']
    assert len(recovered) == 4
    assert conditions('Target: identified\nOpening: meets')['accepted']
    assert not conditions('Target: identified\nOpening: short')['accepted']
    assert conditions('Target: uncertain\nOpening: uncertain')['uncertain']
    for bad in ('Target: uncertain\nOpening: meets', 'Target: identified\nOpening: maybe'):
        try:
            conditions(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed report accepted')
    assert not torch.cuda.is_initialized()
    write(OUT / 'PUBLIC-JOBS.json', jobs)
    write(OUT / 'PRIVATE-TRUTH.json', refs)
    write(OUT / 'REFERENCE-GRADES.json', recovered)
    write(OUT / 'PREPARED-PROMPTS.json', prepared)
    write(OUT / 'FRAME-REFERENCE.json', frames_ref)
    write(OUT / 'PROCESSOR-REFERENCE.json', processor_ref)
    write(OUT / 'CPU-CHECKS.json', dict(completed=True, new_outputs=76, candidate_endpoints=18,
        current_positive=6, current_negative=12, distinct_inputs=37,
        files={p.name:sha(p) for p in OUT.glob('*.json')},
        spec_sha256=sha(code / 'outputs' / RUN / 'SPEC.json'), input_sha256=sha(code / 'outputs' / RUN / 'INPUTS.json'),
        video_checks=video_checks, official_runtime=official, model_weight_loads=0,
        policy_controls=0, optimizer_steps=0, cuda_initialized=False))
    print('CPU_PREPARATION_PASSED', flush=True)


def worker(code):
    assert platform.system() == 'Linux'
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    import torch
    gpu = verify_selected_cuda(torch)
    for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','NPROC_PER_NODE'):
        os.environ.pop(key, None)
    checks = read(OUT / 'CPU-CHECKS.json')
    assert checks['completed']
    # The worker deliberately does not open PRIVATE-TRUTH.json or replay files.
    for name in ('PUBLIC-JOBS.json','PREPARED-PROMPTS.json','FRAME-REFERENCE.json','PROCESSOR-REFERENCE.json'):
        assert sha(OUT / name) == checks['files'][name]
    spec = read(code / 'outputs' / RUN / 'SPEC.json')
    assert sha(code / 'outputs' / RUN / 'SPEC.json') == checks['spec_sha256']
    official = checks['official_runtime']
    assert readref(official['model_receipt'])['completed']
    for path, expected in checks['video_checks'].items():
        assert sha(path) == expected
    frames_ref, processor_ref = read(OUT / 'FRAME-REFERENCE.json'), read(OUT / 'PROCESSOR-REFERENCE.json')
    by_rgb = {}
    for key, ref in frames_ref.items():
        rgb = ref['rgb_sha256']
        if rgb in by_rgb:
            assert processor_ref[by_rgb[rgb]]['bf16_sha256'] == processor_ref[key]['bf16_sha256']
        by_rgb[rgb] = key
    jobs = read(OUT / 'PUBLIC-JOBS.json')
    dest = OUT / 'scoring'
    dest.mkdir(exist_ok=False)
    from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
    started = time.time()
    model = TerminalWorker(official['model'], batch_size=8, gpu_memory_utilization=.78)
    write(dest / 'ENGINE-READY.json', dict(gpu_uuid=gpu, cold_seconds=time.time()-started, model_load_count=1))
    core, api = model.api.core, model.api
    actual_prompts = [dict(id=j['id'], messages=messages(j, spec, core)) for j in jobs]
    write(dest / 'PROMPTS.json', actual_prompts)
    assert sha(dest / 'PROMPTS.json') == checks['files']['PREPARED-PROMPTS.json']
    structured = api.SamplingParams(temperature=0., max_tokens=64, logprobs=core.TERMINAL_TOP_LOGPROBS,
        structured_outputs=api.StructuredOutputsParams(regex=CONDITION_GRAMMAR))
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
    original_processor = Qwen3VLMultiModalProcessor._call_hf_processor
    events = []
    def audited_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        result = original_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs)
        videos = mm_data.get('videos', [])
        if videos:
            assert len(videos) == 1
            raw = rgb_signature(videos[0][0])
            key = by_rgb[raw['rgb_sha256']]
            actual = features_signature(result['pixel_values_videos'], result['video_grid_thw'])
            assert actual['bf16_sha256'] == processor_ref[key]['bf16_sha256']
            events.append(dict(key=key, rgb_sha256=raw['rgb_sha256'], **actual))
        return result
    Qwen3VLMultiModalProcessor._call_hf_processor = audited_processor
    completed = 0
    try:
        for arm in ('reference', *ARMS):
            todo = [j for j in jobs if j['arm'] == arm]
            for offset in range(0, len(todo), 8):
                assert time.time() < int(os.environ['TASK_STARTED']) + 1080, 'Finite batch deadline'
                batch = todo[offset:offset+8]
                path = dest / f'{arm}-{offset:02d}.jsonl'
                requests, audits = [], []
                tick = time.time()
                for job in batch:
                    key = job['source_id'] + '/' + job['evidence']
                    collage = core.Collage([v['path'] for v in job['videos']])
                    try:
                        indices = frame_indices(job, core)
                        frames = collage.frames(indices)
                        assert rgb_signature(frames)['rgb_sha256'] == frames_ref[key]['rgb_sha256']
                    finally:
                        collage.release()
                    message = messages(job, spec, core)
                    prompt = api.engine.render(model.processor, message, thinking=False)
                    requests.append(dict(prompt=prompt, multi_modal_data={'video':(frames, frames_ref[key]['metadata'])},
                        mm_processor_kwargs={'do_sample_frames':False}))
                    audits.append(dict(id=job['id'], input_key=key, indices=indices,
                        rgb_sha256=frames_ref[key]['rgb_sha256'], rendered_sha256=digest(prompt.encode()),
                        messages_sha256=digest(json.dumps(message, sort_keys=True).encode())))
                outputs = model.llm.generate(requests, structured if arm.endswith('_conditions') else model.params, use_tqdm=False)
                assert len(outputs) == len(batch) and all(len(o.outputs) == 1 for o in outputs)
                with path.open('x') as stream:
                    for job, result in zip(batch, outputs):
                        completion = result.outputs[0]
                        row = dict(id=job['id'], source_id=job['source_id'], arm=arm,
                            output=completion.text, token_ids=list(completion.token_ids))
                        if arm.endswith('_conditions'):
                            row.update(conditions(completion.text))
                        else:
                            score, _ = core.parse_terminal(completion.text, False)
                            positions = [(model.tokenizer.decode([token]), float(candidates[token].logprob),
                                [(model.tokenizer.decode([i]), float(candidate.logprob)) for i,candidate in candidates.items()])
                                for token,candidates in zip(completion.token_ids, completion.logprobs)]
                            probabilities, expected = core.score_probabilities(positions, score)
                            row.update(score=score, expected_score=expected, progress=(expected-1)/4,
                                probabilities=probabilities, accepted=score==5, uncertain=False)
                        stream.write(json.dumps(row)+'\n')
                        stream.flush()
                        completed += 1
                write(path.with_suffix('.audit.json'), audits)
                write(path.with_suffix('.receipt.json'), dict(completed=True, labels=len(batch), sha256=sha(path),
                    audit_sha256=sha(path.with_suffix('.audit.json')), seconds=time.time()-tick))
                print(json.dumps(dict(completed=completed,total=76,arm=arm)), flush=True)
    finally:
        Qwen3VLMultiModalProcessor._call_hf_processor = original_processor
    assert completed == 76 and {e['rgb_sha256'] for e in events} == set(by_rgb)
    write(dest / 'PROCESSOR-AUDIT.json', dict(completed=True,events=events,unique_rgb_inputs=len(by_rgb),
        expected_input_keys=len(frames_ref),all_bf16_equal=True,learned_states_inspected=False))
    write(dest / 'COMPLETED.json', dict(completed=True,labels=completed,model_load_count=1,
        worker_seconds=time.time()-started,private_truth_read=False,optimizer_steps=0,policy_controls=0))


def summary(code):
    checks = read(OUT / 'CPU-CHECKS.json')
    assert sha(OUT / 'PRIVATE-TRUTH.json') == checks['files']['PRIVATE-TRUTH.json']
    assert read(OUT / 'scoring/COMPLETED.json')['labels'] == 76
    rows = []
    for path in sorted((OUT / 'scoring').glob('*.jsonl')):
        assert sha(path) == read(path.with_suffix('.receipt.json'))['sha256']
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    assert len(rows) == len({r['id'] for r in rows}) == 76
    prior = read(OUT / 'REFERENCE-GRADES.json')
    repeats = [dict(id=r['id'],expected=prior[r['source_id']],actual=r['score'],equal=prior[r['source_id']]==r['score']) for r in rows if r['arm']=='reference']
    truth = {r['id']:r for r in read(OUT / 'PRIVATE-TRUTH.json')}
    paired = [dict(r, labels={a:next(x for x in rows if x['source_id']==uid and x['arm']==a) for a in ARMS}) for uid,r in truth.items()]
    metrics = {}
    for arm in ARMS:
        tp=fp=tn=fn=uncertain=0
        positives, negatives = [], []
        for p in paired:
            row=p['labels'][arm]; actual=p['native_current_success']; predicted=row['accepted']
            tp+=actual and predicted;fp+=(not actual) and predicted
            tn+=(not actual) and (not predicted);fn+=actual and (not predicted)
            uncertain+=row['uncertain']
            if 'expected_score' in row:
                (positives if actual else negatives).append(row['expected_score'])
        auc=None
        if positives and negatives:
            auc=sum((p>n)+.5*(p==n) for p in positives for n in negatives)/(len(positives)*len(negatives))
        metrics[arm]=dict(tp=int(tp),fp=int(fp),tn=int(tn),fn=int(fn),uncertain=int(uncertain),
            precision=tp/(tp+fp) if tp+fp else None,recall=tp/6,AUROC=auc,
            development_pass=fp==0 and tp==6 and all(r['equal'] for r in repeats))
    contrasts={}
    for a,b in (('prefix_grade','anchors_grade'),('prefix_conditions','anchors_conditions'),
                ('prefix_grade','prefix_conditions'),('anchors_grade','anchors_conditions')):
        corrected=[];regressed=[]
        for p in paired:
            actual=p['native_current_success']
            before=p['labels'][a]['accepted']==actual;after=p['labels'][b]['accepted']==actual
            if not before and after:corrected.append(p['id'])
            if before and not after:regressed.append(p['id'])
        contrasts[a+'__'+b]=dict(corrected=corrected,regressed=regressed)
    write(OUT / 'PAIRED-SAMPLE.json', paired)
    write(OUT / 'RESULT.json', dict(completed=True,metrics=metrics,contrasts=contrasts,repeats=repeats,
        current_state_labels=True,candidate_endpoints=18,current_positive=6,current_negative=12,
        production_qualified=False,held_out_validation=False,policy_controls=0,optimizer_steps=0))
    print(json.dumps(metrics,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('prepare','worker','summary'))
    parser.add_argument('--code',type=Path,required=True)
    args=parser.parse_args()
    globals()[args.action](args.code)
