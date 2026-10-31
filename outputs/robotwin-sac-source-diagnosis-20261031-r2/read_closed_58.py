"""Closed58 metadata increment only; no active child/process/tensor inspection."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
R=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-continuation-sac-20261030-r1')
closed=58;root=R/'slices'/str(closed).zfill(4)/'experiment';p=root/'summary.json'
s=json.loads(p.read_text());children=json.loads((R/'children.json').read_text());receipt=next(r for r in children if r.get('slice')==closed)
if receipt['exit_code']!=0 or receipt['status']!='verified_wall_budget_stop' or not receipt['round_boundary'] or receipt['source']!=s['source']:raise ValueError('Not the named closed audited boundary')
files=[];episodes=[];metrics=[]
for n in range(56,59):
 for name in ('episodes.jsonl','updates.jsonl'):
  pth=R/'slices'/str(n).zfill(4)/'experiment'/name;raw=pth.read_bytes();files.append(dict(path=str(pth),sha256=hashlib.sha256(raw).hexdigest()))
  for line in raw.splitlines():
   r=json.loads(line)
   if r.get('method')=='sac' and r.get('phase')=='source':(episodes if name=='episodes.jsonl' else metrics).append(r)
source=s['source']['sac']
v=dict(format='robotwin_sac_source98_closed_increment_v1',status='PARTIAL_SOURCE98_TARGET100_NOT_RECORDED',checked_utc=datetime.now(timezone.utc).isoformat(),summary_path=str(root/'summary.json'),summary_sha256=hashlib.sha256((root/'summary.json').read_bytes()).hexdigest(),closed_child=receipt,final_source=source,increment_from95=dict(episodes=len(episodes),successes=sum(r['success'] for r in episodes),actual_actions=sum(r['primitive_steps'] for r in episodes),chunks=sum(r['chunks'] for r in episodes),source_iterations=sorted(set(r['source_iteration'] for r in episodes)),per_goal=[dict(task=t,episodes=sum(r['task']==t for r in episodes),successes=sum(r['success'] for r in episodes if r['task']==t)) for t in s['contract']['source_tasks']],joint_updates=source['updates']-606880,combined_optimizer_calls=source['optimizer_steps']-1820640),last_logged_metrics=metrics[-1],stage100_source_snapshot_recorded='iteration:100' in s.get('stages',{}),stage100_source_validation=[r for r in s.get('source_validation',[]) if r.get('stage_source_iteration')==100],stage100_targets=[r for r in s.get('target',[]) if r.get('stage_source_iteration')==100],parent_exit_present=(R/'exit-code.txt').exists(),metadata_files=files,scope='Single closed-boundary read; no active checkpoint/model/process inspection; prior95analysis immutable')
print(json.dumps(v,indent=2))
