"""One closed SAC95 boundary; stdlib JSON/JSONL only, never active child state."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import hashlib,json,math,statistics
B=Path('/mowenjie/fluffy-bassoon-2/outputs');R=B/'robotwin-native-continuation-sac-20261030-r1'
closed=55;final=R/'slices'/f'{closed:04d}'/'experiment';summary_path=final/'summary.json'
summary=json.loads(summary_path.read_text());source=summary['source']['sac'];last=source['completed_source_iterations']
children=json.loads((R/'children.json').read_text());child=next(row for row in children if row.get('slice')==closed)
if child['status']!='verified_wall_budget_stop' or child['exit_code']!=0 or not child['round_boundary'] or child['source']['sac']!=source:
 raise ValueError('Require the named already-closed audited source boundary')
roots=[B/'robotwin-wholetrial-sac-20261030-r2/experiment']+[R/'slices'/f'{i:04d}'/'experiment' for i in range(1,closed+1)]
episodes={};updates={};conflicts=[];files=[];duplicates=Counter()
for root in roots:
 for kind,name in [('episodes','episodes.jsonl'),('updates','updates.jsonl')]:
  p=root/name
  if not p.exists():continue
  raw=p.read_bytes();files.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
  for line in raw.splitlines():
   row=json.loads(line)
   if row.get('method')!='sac' or row.get('phase')!='source':continue
   if kind=='episodes':
    key=tuple(row.get(k) for k in ('source_iteration','task','episode','seed'))
    if not isinstance(key[0],int) or not 1<=key[0]<=last:continue
    item={k:row.get(k) for k in ('task','episode','seed','success','return_','primitive_steps','chunks','initial_observation_sha256','budget_truncated')};dest=episodes
   else:
    key=row['updates'];item={k:v for k,v in row.items() if k not in ('method','phase')};dest=updates
    if key>source['updates']:raise ValueError('Metric exceeds the declared closed boundary')
   if key in dest:
    duplicates[kind]+=1
    if dest[key]!=item:conflicts.append(dict(kind=kind,key=key,previous=dest[key],current=item,path=str(p)))
   else:dest[key]=item

def stats(values):
 values=list(values)
 return dict(count=len(values),min=min(values),median=statistics.median(values),mean=statistics.mean(values),max=max(values)) if values else None

def metric_summary(rows):
 return {k:stats(r[k] for r in rows if k in r) for k in ('critic_loss','actor_loss','temperature_loss','temperature','target_mean','entropy','actor_gradient_norm','critic_gradient_norm','temperature_gradient_norm','replay_rows')}
chunks_by_iteration={i:sum(r['chunks'] for k,r in episodes.items() if k[0]==i) for i in range(1,last+1)}
cumulative=0;update_thresholds={}
for i,n in chunks_by_iteration.items():cumulative+=n;update_thresholds[i]=max(0,cumulative-500)*20
bins=[]
for lo,hi in ((1,25),(26,50),(51,75),(76,last)):
 rows=[r for k,r in episodes.items() if lo<=k[0]<=hi]
 metrics=[r for clock,r in updates.items() if update_thresholds.get(lo-1,0)<clock<=update_thresholds[hi]]
 bins.append(dict(iterations=[lo,hi],episodes=len(rows),successes=sum(r['success'] for r in rows),actions=sum(r['primitive_steps'] for r in rows),chunks=sum(r['chunks'] for r in rows),per_goal=[dict(task=task,episodes=sum(r['task']==task for r in rows),successes=sum(r['success'] for r in rows if r['task']==task)) for task in summary['contract']['source_tasks']],diagnostics=metric_summary(metrics),logged_update_records=len(metrics)))
per_goal=[]
for task in summary['contract']['source_tasks']:
 rows=[(k,r) for k,r in episodes.items() if r['task']==task]
 per_goal.append(dict(task=task,episodes=len(rows),successes=sum(r['success'] for k,r in rows),actions=sum(r['primitive_steps'] for k,r in rows),chunks=sum(r['chunks'] for k,r in rows),seed_count=len({r['seed'] for k,r in rows}),feature_hash_count=len({r['initial_observation_sha256'] for k,r in rows}),first25=dict(episodes=sum(k[0]<=25 for k,r in rows),successes=sum(r['success'] for k,r in rows if k[0]<=25)),last20=dict(episodes=sum(k[0]>75 for k,r in rows),successes=sum(r['success'] for k,r in rows if k[0]>75))))
validation={str(i):[dict(task=r['task'],successes=sum(e['success'] for e in r['episodes']),episodes=len(r['episodes'])) for r in summary['source_validation'] if r.get('stage_source_iteration')==i] for i in (10,25,50,100)}
code=Path('/mowenjie/fluffy-bassoon-2/worktrees/robotwin-native-continuation-sac-20261030-r1');code_paths=['src/fluffy_bassoon/meta_rl/vla_noise_sac.py','scripts/diagnostics/robotwin_noise_compare.py']
value=dict(format='robotwin_sac_closed_source_metadata_diagnosis_v1',status='PARTIAL' if last<100 else 'SOURCE100_RECORDED_TARGET_COMPLETION_NOT_INFERRED',checked_utc=datetime.now(timezone.utc).isoformat(),boundary_slice=closed,source_iteration=last,operation='JSON/JSONL only; no tensors/replay/frames/active child inspection',final_summary=dict(path=str(summary_path),sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest()),closed_child=child,scientific_source=summary['contract']['source_commit'],final_source=source,recipe=summary['contract']['sac'],runtime=summary['contract']['runtime'],episode_count=len(episodes),successes=sum(r['success'] for r in episodes.values()),episode_action_sum=sum(r['primitive_steps'] for r in episodes.values()),episode_chunk_sum=sum(r['chunks'] for r in episodes.values()),expected_updates_from_observed_chunks=max(0,cumulative-500)*20,episode_return_values=dict(Counter(str(r['return_']) for r in episodes.values())),per_goal=per_goal,round_blocks=bins,source_validation=validation,logged_metric_records=len(updates),logged_update_range=[min(updates),max(updates)],first_logged_metrics=updates[min(updates)],last_logged_metrics=updates[max(updates)],metric_summary=metric_summary(updates.values()),finite_logged_metrics=all(math.isfinite(v) for r in updates.values() for v in r.values() if isinstance(v,(int,float))),duplicates=dict(duplicates),conflicts=conflicts,metadata_files=files,code_files=[dict(path=str(code/p),sha256=hashlib.sha256((code/p).read_bytes()).hexdigest()) for p in code_paths],limitations=['Only closed slice0055/round95 is examined; no source100 or final target completion assertion.','Metrics record the last SGD of20 every25source decisions; summaries are unweighted logged snapshots, not all606880steps.','Per-block metric attribution derives exact cumulative source chunk/update clocks; replay minibatches mix goals, so diagnostics are not per-goal values.','Physical charges use the canonical closed source receipt; repeated metadata deduplication does not remove executed interactions.','Seed/feature hash counts do not establish distinct full physical states.'])
print(json.dumps(value,indent=2,allow_nan=False))
