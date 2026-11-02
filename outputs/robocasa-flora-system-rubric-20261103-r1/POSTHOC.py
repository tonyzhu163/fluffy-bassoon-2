"""CPU-only diagnostics on completed grades; neither rule is independently validated."""
import hashlib
import json
from pathlib import Path
import platform
assert platform.system() == 'Linux'
root=Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-flora-system-rubric-20261103-r1')
pairs=json.loads((root/'PAIRED-SAMPLE.json').read_text())
result={'posthoc':True,'independent_validation':False,'production_qualified':False,'source_sha256':hashlib.sha256((root/'PAIRED-SAMPLE.json').read_bytes()).hexdigest(),'simulator_calls':0,'motor_calls':0,'gpu_calls':0,'rules':{}}
for arm in ('revision1_11','revision1_12'):
    negative_p5=[r['labels'][arm]['probabilities']['5'] for r in pairs if not r['native_ever_success'] and r['labels'][arm]['score']==5]
    cutoff=max(negative_p5)
    confidence=[];confirmed=[]
    for eid in sorted({r['episode_id'] for r in pairs}):
        seq=sorted((r for r in pairs if r['episode_id']==eid),key=lambda r:r['control'])
        actual=seq[0]['native_first_success']
        accepted=[r['control'] for r in seq if r['labels'][arm]['score']==5 and r['labels'][arm]['probabilities']['5']>cutoff]
        sustained=[b['control'] for a,b in zip(seq,seq[1:]) if b['control']-a['control']==50 and a['labels'][arm]['score']==b['labels'][arm]['score']==5]
        def record(points):
            first=points[0] if points else None
            return dict(episode_id=eid,native_first_success=actual,first_accepted=first,false_accept=first is not None and (actual is None or first<actual),correctly_timed=actual is not None and first is not None and 0<=first-actual<=100)
        confidence.append(record(accepted));confirmed.append(record(sustained))
    def summarize(cases):return dict(cases=cases,false_accepted_episodes=sum(c['false_accept'] for c in cases),correctly_timed_episodes=sum(c['correctly_timed'] for c in cases))
    result['rules'][arm]={'score5_and_p5_strictly_above_max_sampled_negative':dict(max_sampled_negative_p5=cutoff,**summarize(confidence)),'two_score5_probes_exactly50_controls_apart':summarize(confirmed)}
path=root/'POSTHOC.json'
with path.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
Path('/root/workspace/drawer-system-posthoc.json').write_bytes(path.read_bytes())
for arm,rules in result['rules'].items():
    for rule,m in rules.items():print(arm,rule,m['false_accepted_episodes'],m['correctly_timed_episodes'])
print('SHA256',hashlib.sha256(path.read_bytes()).hexdigest())
