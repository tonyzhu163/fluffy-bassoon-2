"""Independent checks of downloaded CPU JSON evidence; no model computation."""
import hashlib
import json
from pathlib import Path

out=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=read(out/'CPU-MANIFEST.json')
for name,value in manifest['files'].items():assert sha(out/name)==value,name
checks,receipt=read(out/'CPU-CHECKS.json'),read(out/'CPU-EXIT.json')
assert checks['completed'] and receipt['exit_code']==0 and receipt['gpu_seconds']==0
assert checks['new_outputs']==40 and checks['actual_cpu_processor_calls']==19
assert checks['previous_processor_audit_matches']==4
assert checks['model_weight_loads']==checks['motor_calls']==checks['optimizer_steps']==0
for arm,value in checks['criteria_sha256'].items():assert sha(out/(arm+'.json'))==value
for name,key in [('PUBLIC-JOBS.json','public_sha256'),('RECOVERED.json','recovered_sha256'),
    ('PREPARED-PROMPTS.json','prompts_sha256'),('FRAME-REFERENCE.json','frame_reference_sha256'),
    ('PROCESSOR-REFERENCE.json','processor_reference_sha256')]:
    assert sha(out/name)==checks[key]
items,refs=read(out/'PUBLIC-JOBS.json'),read(out/'PREFIX-REFERENCE.json')
frames,features=read(out/'FRAME-REFERENCE.json'),read(out/'PROCESSOR-REFERENCE.json')
prompts=read(out/'PREPARED-PROMPTS.json')
assert len(items)==len(frames)==len(features)==19 and len(refs)==18 and len(prompts)==40
assert len({i['id'].split('/prefix-')[0] for i in items})==8
assert sum(not r['native_ever_success'] for r in refs)==10
assert sum(r['native_ever_success'] for r in refs)==8
for item in items:
    uid=item['id'];control=item['prefix_controls'];frame=frames[uid];feature=features[uid]
    expected=[round(i*control/63) for i in range(64)]
    assert frame['indices']==expected and frame['shape']==[320,960,3] and frame['frame_count']==64
    assert len(frame['rgb_sha256'])==64
    assert hashlib.sha256(''.join(frame['rgb_sha256']).encode()).hexdigest()==frame['ordered_rgb_hashes_sha256']
    assert feature['grid']==[[32,20,60]] and feature['shape']==[38400,1536]
    assert feature['dtype']=='torch.float32'
    assert len(feature['fp32_sha256'])==len(feature['bf16_sha256'])==64
assert len({p['id'] for p in prompts})==40
previous=read(out.parent/'robocasa-flora-drawer-processor-audit-20261103-r3/RECEIPT.json')['cases']
assert len(previous)==4
for ref in previous:
    uid=ref['id']
    assert frames[uid]['rgb_sha256']==ref['raw_rgb_sha256']
    assert features[uid]['fp32_sha256']==ref['feature_sha256']
    assert features[uid]['bf16_sha256']==ref['bf16_feature_sha256']
anchors=read(out/'RECOVERED.json')['rows']
expected={('000112',300):1,('000064',550):5,('000128',400):1,('000144',250):4}
assert {(r['id'].split('-case-')[1][:6],r['observed_through']):r['score'] for r in anchors}==expected
print(json.dumps(dict(verified=True,source_commit=receipt['source_commit'],
    cpu_seconds=receipt['seconds'],gpu_seconds=0,distinct_inputs=19,raw_frame_hashes=1216,
    processor_references=19,prompts=40,negative_prefixes=10,post_success_prefixes=8)))
