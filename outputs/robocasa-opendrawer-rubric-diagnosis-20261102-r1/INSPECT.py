"""Read retained native metadata/actions and pinned fixture source on Linux CPU."""
import argparse
import ast
import gzip
import base64
import hashlib
import json
import os
from pathlib import Path
import platform

RUN='robocasa-opendrawer-rubric-diagnosis-20261102-r1'
ROOT=Path('/mowenjie/fluffy-bassoon-2')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert platform.system()=='Linux' and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    import torch
    assert not torch.cuda.is_available()
    out=ROOT/'outputs'/RUN;out.mkdir(exist_ok=False)
    selected=read(ROOT/'outputs/robocasa-flora-rubric-failure-sample-20261102-r1/sample/SELECTION.json')['selected']
    wanted=('flora-single-case-000112-rank-06-episode-000001',
        'flora-single-case-000064-rank-00-episode-000001',
        'flora-single-case-000128-rank-00-episode-000006',
        'flora-single-case-000144-rank-02-episode-000004')
    profiles={};episodes=[];checkpoints={}
    for item in selected:
        p=item['pair'];eid=p['id'].split('/')[-1]
        if eid not in wanted:continue
        case_number=int(eid.split('-case-')[1].split('-')[0]);iteration=case_number//64+1
        run='robocasa-opendrawer-flora-first-20261102-r1' if iteration<=4 else 'robocasa-opendrawer-flora-resume-20261102-r1'
        policy=ROOT/'outputs'/run/'policy';profile=read(policy/'profile.json')
        profiles[run]=dict(keys=list(profile),science={k:profile[k] for k in profile if any(t in k for t in ('native','rlinf','horizon','config','runtime','source','monitoring'))})
        request=read(policy/f'labels-{iteration:03d}/request.json');r=request['pending_checkpoint'];assert sha(r['path'])==r['sha256']
        if r['path'] not in checkpoints:checkpoints[r['path']]=torch.load(r['path'],map_location='cpu',weights_only=False)
        saved=checkpoints[r['path']];index=next(i for i,e in enumerate(saved['pending_episodes']) if e['id']==eid)
        ends=[i for i,r in enumerate(saved['pending_records']) if r['done']]
        start=0 if index==0 else ends[index-1]+1;records=saved['pending_records'][start:ends[index]+1]
        monitor=read(Path(profile['monitoring_root'])/'train'/(eid+'.json'))
        episodes.append(dict(id=eid,checkpoint=r,checkpoint_keys=list(saved),episode=saved['pending_episodes'][index],
            monitor=monitor,record_count=len(records),record_keys=list(records[0]),
            record_shapes={k:list(getattr(v,'shape',())) for k,v in records[0].items()},
            has_executed_chunks=all('executed_chunk' in row for row in records),
            executed_shapes=[list(row['executed_chunk'].shape) for row in records] if 'executed_chunk' in records[0] else None))
        torch.save(dict(id=eid,case=monitor['case'],episode=saved['pending_episodes'][index],records=records),out/(eid+'.pt'))
    native=ROOT/'setups/robocasa365-native-20261101-r1/sources/RoboCasa';sources=[]
    for path in sorted((native/'robocasa').rglob('*.py')):
        text=path.read_text()
        if 'TOP_DRAWER' not in text and path.name not in ('kitchen_drawer.py','cabinets.py','fixture.py'):continue
        tree=ast.parse(text);functions=[]
        for node in ast.walk(tree):
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and (
                node.name in ('get_door_state','check_fixture_is_type','get_fixture','get_ep_meta','_check_success','get_env_fns','Drawer') or
                'TOP_DRAWER' in (ast.get_source_segment(text,node) or '') and isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))):
                functions.append(dict(name=node.name,line=node.lineno,source=ast.get_source_segment(text,node)))
        if functions:sources.append(dict(path=str(path),sha256=sha(path),functions=functions))
    audit=dict(completed=True,profiles=profiles,episodes=episodes,sources=sources,simulation_steps=0,model_calls=0)
    path=out/'SOURCE-SURVEY.json';path.write_text(json.dumps(audit,indent=2)+'\n')
    Path('/root/workspace/drawer-diagnosis-survey.txt').write_text(base64.b64encode(gzip.compress(path.read_bytes())).decode()+'\n')
    print(json.dumps(dict(completed=True,episodes=len(episodes),all_have_actions=all(e['has_executed_chunks'] for e in episodes),source_survey_sha256=sha(path),simulation_steps=0,model_calls=0)))
if __name__=='__main__':main()
