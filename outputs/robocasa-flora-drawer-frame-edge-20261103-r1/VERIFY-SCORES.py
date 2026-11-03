"""Reconcile the downloaded40-grade result against raw labels and input audits."""
import hashlib
import json
import math
from pathlib import Path
import re

root=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(root/'TRANSFER-MANIFEST.json')
for name,value in manifest['files'].items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    assert sha(root/name)==value,name
checks=read(root/'CPU-CHECKS.json')
assert read(root/'scoring/COMPLETED.json')['new_outputs']==40
assert sha(root/'scoring/PROMPTS.json')==checks['prompts_sha256']
prompts={p['id']:p['messages'] for p in read(root/'PREPARED-PROMPTS.json')}
frames,features=read(root/'FRAME-REFERENCE.json'),read(root/'PROCESSOR-REFERENCE.json')
rows,events={},[]
paths=sorted((root/'scoring').glob('*.jsonl'),key=lambda p:(not p.stem.startswith('repeat_both-'),p.name))
assert len(paths)==7
for path in paths:
    batch=[json.loads(line) for line in path.read_text().splitlines()]
    receipt=read(path.with_suffix('.receipt.json'))
    assert receipt['completed'] and receipt['labels']==len(batch) and 1<=len(batch)<=8
    for suffix,key in [('.jsonl','sha256'),('.frames.json','frames_sha256'),
        ('.prompts.json','prompts_sha256'),('.processor.json','processor_sha256')]:
        assert sha(path.with_suffix(suffix))==receipt[key]
    raw,rendered=read(path.with_suffix('.frames.json')),read(path.with_suffix('.prompts.json'))
    assert [r['id'] for r in batch]==[r['id'] for r in raw]==[r['id'] for r in rendered]
    for row,frame,prompt in zip(batch,raw,rendered):
        uid=row['id'];source=uid.rsplit('/',1)[0]
        assert uid not in rows and uid in prompts and source in frames
        assert {k:v for k,v in frame.items() if k!='id'}==frames[source]
        assert prompt['messages_sha256']==hashlib.sha256(json.dumps(prompts[uid],sort_keys=True).encode()).hexdigest()
        assert len(prompt['rendered_sha256'])==64
        assert type(row['score']) is int and 1<=row['score']<=5
        assert re.fullmatch(r'\s*Score:\s*'+str(row['score'])+r'\s*',row['output'])
        assert math.isfinite(row['expected_score']) and 1<=row['expected_score']<=5
        assert set(row['probabilities'])==set('12345')
        assert all(0<=v<=1 for v in row['probabilities'].values())
        assert abs(sum(row['probabilities'].values())-1)<.001
        assert abs(row['progress']-(row['expected_score']-1)/4)<.0002
        assert row['sampled_frames']==64 and row['prefix_frame_count']==row['observed_through']+1
        assert row['score_protocol']=='causal_prefix_terminal_v1'
        rows[uid]=row
    events.extend(read(path.with_suffix('.processor.json')))
assert len(rows)==len(prompts)==40 and set(rows)==set(prompts)
assert {e['id'] for e in events}==set(features) and 19<=len(events)<=40
for event in events:
    ref=features[event['id']]
    assert event['bf16_equal'] and event['bf16_sha256']==ref['bf16_sha256']
    assert event['fp32_equal']==(event['fp32_sha256']==ref['fp32_sha256'])
    assert event['shape']==ref['shape'] and event['grid']==ref['grid']
audit=read(root/'scoring/PROCESSOR-AUDIT.json')
assert audit['events']==events and audit['unique_inputs']==19 and audit['all_bf16_equal']
assert audit['all_fp32_equal']==all(e['fp32_equal'] for e in events)
assert {e['dtype'] for e in events}=={'torch.bfloat16'}
probe,engine=read(root/'allocation/gpu-probe.json'),read(root/'scoring/ENGINE-READY.json')
assert probe['status']=='passed' and len(probe['devices'])==1
assert engine['gpu_uuid']==probe['devices'][0]['uuid']
assert engine['model_load_count']==1 and engine['batch_size']==8 and engine['temperature']==0
old={r['id'].rsplit('/',1)[0]:r['score'] for r in read(root/'RECOVERED.json')['rows']}
anchors=[r for r in rows.values() if r['id'].endswith('/repeat_both')]
assert len(anchors)==len(old)==4
anchor_match=all(r['score']==old[r['id'].rsplit('/',1)[0]] for r in anchors)
refs=read(root/'PREFIX-REFERENCE.json')
paired=read(root/'PAIRED-SAMPLE.json')
assert len(refs)==len(paired)==18
result=read(root/'RESULT.json')
calculated={}
for arm,metric in result['metrics'].items():
    false5=sum(not r['native_ever_success'] and rows[r['id']+'/'+arm]['score']==5 for r in refs)
    post5=sum(r['native_ever_success'] and rows[r['id']+'/'+arm]['score']==5 for r in refs)
    timely=false_episodes=0
    for eid in sorted({r['episode_id'] for r in refs}):
        seq=sorted((r for r in refs if r['episode_id']==eid),key=lambda r:r['control'])
        native=seq[0]['native_first_success']
        accepts=[b['control'] for a,b in zip(seq,seq[1:]) if b['control']-a['control']==50
            and rows[a['id']+'/'+arm]['score']==rows[b['id']+'/'+arm]['score']==5]
        first=accepts[0] if accepts else None
        timely+=native is not None and first is not None and 0<=first-native<=100
        false_episodes+=first is not None and (native is None or first<native)
    closed=[r for r in refs if r['native_ever_success'] and not r['native_current_success'] and r['control']==750]
    assert len(closed)==2
    closed5=sum(rows[r['id']+'/'+arm]['score']==5 for r in closed)
    passes=false5==false_episodes==0 and timely==3 and closed5==2 and anchor_match
    calculated[arm]=dict(false_full_scores=false5,post_success_full_scores=post5,
        correctly_timed_confirmed_episodes=timely,false_accepted_episodes=false_episodes,
        closed_again_full_scores=closed5,passes_development_screen=passes)
    for key,value in calculated[arm].items():assert metric[key]==value,(arm,key)
    for ref,pair in zip(refs,paired):
        assert {k:v for k,v in pair.items() if k!='labels'}==ref
        assert pair['labels'][arm]==rows[ref['id']+'/'+arm]
charge=read(root/'terminal-charge.json')
assert charge['exit_code']==0 and charge['source_commit']==manifest['source_commit']
assert 0<charge['charged_gpu_seconds']<=900
print(json.dumps(dict(verified=True,labels=40,raw_frame_input_audits=40,batch_receipts=7,
    live_processor_events=len(events),unique_live_processor_inputs=19,all_bf16_equal=True,
    all_fp32_equal=audit['all_fp32_equal'],anchor_match=anchor_match,metrics=calculated,
    actual_processor_dtype='torch.bfloat16',bf16_inputs_byte_exact=True,
    fp32_hash_comparison='Actual bfloat16 expanded tofloat32 versus original unrounded CPUfloat32; not like-for-like',
    charged_gpu_seconds=charge['charged_gpu_seconds']),indent=2))
