"""Read-only JSON/JSONL aggregation; no actor/tensor/raw-corpus imports."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
import hashlib,json,math,statistics
B=Path('/mowenjie/fluffy-bassoon-2/outputs')
M='ppo_reset'
roots=[B/'robotwin-wholetrial-ppo_reset-20261030-r2/experiment']
roots+=sorted((B/'robotwin-native-continuation-ppo_reset-20261030-r1').glob('slices/*/experiment'))
roots+=[B/f'robotwin-trained-pair-20261031-r{i}'/M for i in range(1,6)]
roots+=[B/'robotwin-trained-pair-initial-20261031-r1'/M]
roots+=sorted((B/'robotwin-trained-continuation-20261031-r1'/M).glob('slice-*/experiment'))
final=roots[-1];summary=json.loads((final/'summary.json').read_text())
episodes={};updates={};conflicts=[];files=[];duplicates=Counter()
for root in roots:
 for kind,name in [('episodes','episodes.jsonl'),('updates','updates.jsonl')]:
  p=root/name
  if not p.exists():continue
  raw=p.read_bytes();files.append(dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw)))
  for line in raw.splitlines():
   row=json.loads(line)
   if row.get('method')!=M:continue
   if kind=='episodes':
    if row.get('phase')!='source':continue
    key=tuple(row.get(k) for k in ('source_iteration','trial_in_iteration','task','episode','seed'))
    if not isinstance(key[0],int) or not 1<=key[0]<=100:continue
    fields=('task','episode','seed','success','return_','primitive_steps','chunks','initial_observation_sha256','episode_limit','budget_truncated','reset_memory','reset_episode_feedback')
    item={k:row.get(k) for k in fields};dest=episodes
   else:
    key=row['ppo_updates'];item={k:v for k,v in row.items() if k!='method'};dest=updates
   if key in dest:
    duplicates[kind]+=1
    if dest[key]!=item:conflicts.append(dict(kind=kind,key=key,previous=dest[key],current=item,path=str(p)))
   else:dest[key]=item

def stats(values):
 values=list(values)
 return dict(count=len(values),min=min(values),median=statistics.median(values),mean=statistics.mean(values),max=max(values)) if values else None

def panel(rows):
 return [dict(task=r['task'],successes=sum(e['success'] for e in r['episodes']),episodes=len(r['episodes']),actions=sum(e['primitive_steps'] for e in r['episodes']),chunks=sum(e['chunks'] for e in r['episodes']),returns=[e['return_'] for e in r['episodes']],seeds=[e['seed'] for e in r['episodes']],limits=sorted(set(e['episode_limit'] for e in r['episodes']))) for r in rows]
pergoal=[]
for task in summary['contract']['source_tasks']:
 rows=[(key,row) for key,row in episodes.items() if row['task']==task]
 pergoal.append(dict(task=task,episodes=len(rows),successes=sum(r['success'] for _,r in rows),actions=sum(r['primitive_steps'] for _,r in rows),chunks=sum(r['chunks'] for _,r in rows),seed_count=len({r['seed'] for _,r in rows}),initial_hash_count=len({r['initial_observation_sha256'] for _,r in rows}),horizon_failures=sum(not r['success'] and r['primitive_steps']==r['episode_limit'] for _,r in rows),first_quarter=dict(episodes=sum(k[0]<=25 for k,r in rows),successes=sum(r['success'] for k,r in rows if k[0]<=25)),last_quarter=dict(episodes=sum(k[0]>75 for k,r in rows),successes=sum(r['success'] for k,r in rows if k[0]>75))))
bins=[]
for lo,hi in [(1,25),(26,50),(51,75),(76,100)]:
 rows=[r for k,r in episodes.items() if lo<=k[0]<=hi];metrics=[r for i,r in updates.items() if lo<=i<=hi]
 bins.append(dict(iterations=[lo,hi],episodes=len(rows),successes=sum(r['success'] for r in rows),actions=sum(r['primitive_steps'] for r in rows),chunks=sum(r['chunks'] for r in rows),metric_stats={k:stats(r[k] for r in metrics) for k in ('chunks','approximate_kl','clip_fraction','gradient_norm','loss','actor_loss','critic_loss')}))
validation={str(i):panel([r for r in summary['source_validation'] if r.get('stage_source_iteration')==i]) for i in (10,25,50,100)}
targets={str(i):panel([r for r in summary['target'] if r.get('stage_source_iteration')==i]) for i in (10,25,50,100)}
code=Path('/mowenjie/fluffy-bassoon-2/worktrees/robotwin-trained-continuation-code-20261031-r2')
paths=['src/fluffy_bassoon/meta_rl/vla_noise_ppo.py','src/fluffy_bassoon/meta_rl/ppo.py','src/fluffy_bassoon/rl/robotwin_noise.py','scripts/diagnostics/robotwin_noise_compare.py']
out=dict(format='robotwin_reset100_metadata_diagnosis_v1',checked_utc=datetime.now(timezone.utc).isoformat(),operation='JSON/JSONL only; no models/tensors/raw-bank scan',final_summary=dict(path=str(final/'summary.json'),sha256=hashlib.sha256((final/'summary.json').read_bytes()).hexdigest()),final_source=summary['source'][M],scientific_source=summary['contract']['source_commit'],recipe=summary['contract']['ppo'],runtime=summary['contract']['runtime'],per_goal_training=pergoal,training_bins=bins,source_validation=validation,target=targets,update_count=len(updates),update_iterations=sorted(updates),missing_updates=sorted(set(range(1,101))-set(updates)),metric_stats={k:stats(r[k] for r in updates.values()) for k in ('chunks','approximate_kl','clip_fraction','gradient_norm','loss','actor_loss','critic_loss')},finite_metrics=all(math.isfinite(r[k]) for r in updates.values() for k in ('chunks','approximate_kl','clip_fraction','gradient_norm','loss','actor_loss','critic_loss')),total_update_chunks=sum(r['chunks'] for r in updates.values()),total_optimizer_steps=sum(r['optimizer_steps'] for r in updates.values()),source_episode_count=len(episodes),source_episode_action_sum=sum(r['primitive_steps'] for r in episodes.values()),source_episode_chunk_sum=sum(r['chunks'] for r in episodes.values()),episode_return_values=dict(Counter(str(r['return_']) for r in episodes.values())),duplicates=dict(duplicates),conflicts=conflicts,metadata_files=files,code_files=[dict(path=str(code/p),sha256=hashlib.sha256((code/p).read_bytes()).hexdigest()) for p in paths],limitations=['Repeated metadata identities are deduplicated for learning-history description; physical costs are taken from canonical final_source, never deduplicated here.','Seed count/initial feature hash count does not establish distinct full physical states.','Final validation has four fixed scenes per source goal and stochastic action draws; no native source comparator or task-averaged uncertainty from this panel.'])
print(json.dumps(out,indent=2,allow_nan=False))
