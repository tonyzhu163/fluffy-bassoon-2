from pathlib import Path
import hashlib,json,datetime,shutil
base=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-sidney-matched-baselines-20261106-r1')
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
report={'at_shanghai':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'methods':{}}
groups={}
for arm,method in [('ppo02','ppo'),('grpo01','grpo')]:
 root=base/arm
 assert read(root/'EXIT.json')['exit_code']==0
 assert read(root/'RESULT.json')['status']=='complete'
 final=read(root/'final-restore/VERIFIED.json');assert final['status']=='complete'
 assert final['optimizer_steps_per_rank']==[40]*4
 check={'status':'complete','EXIT':read(root/'EXIT.json'),'RESULT':read(root/'RESULT.json'),'final_actor_restore':final,'checkpoint_integrity':{},'gradient_audits':[]}
 for step,expected in [(0,0),(1,20),(2,40)]:
  cp=root/f'sidney-paired-{method}/checkpoints/global_step_{step}'
  manifest=read(cp/'pilot-boundary.json')
  assert manifest['completed'] and manifest['global_step']==step
  signatures=[read(cp/'pilot'/f'paired-actor-signature-{r:02d}.json') for r in range(4)]
  assert [s['optimizer_steps'] for s in signatures]==[expected]*4
  assert len(manifest['workers'])==12 and len(manifest['actor_checkpoint_files'])==5
  check['checkpoint_integrity'][str(step)]={'optimizer_steps_per_rank':[expected]*4,'data_files':[]}
  if step==2:
   for item in manifest['workers']+manifest['actor_checkpoint_files']:
    p=Path(item['path']);actual=sha(p);assert actual==item['sha256'],str(p)
    check['checkpoint_integrity'][str(step)]['data_files'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':actual,'matched':True})
 for r in range(4):
  for micro in range(8):
   f=root/'gradient-audit'/f'rank-{r:02d}-micro-{micro:02d}.json';v=read(f)
   assert v['all_finite'] and v['frozen_nonzero_elements']==v['frozen_optimizer_members']==0
   assert v['synchronized_boundary']==(micro==7)
   if micro==7:assert v['frozen_non_none_views']==0
   check['gradient_audits'].append({k:v[k] for k in ('rank','accumulation_microbatch','synchronized_boundary','frozen_non_none_views','frozen_nonzero_elements','frozen_optimizer_members','all_finite')}|{'path':str(f),'sha256':sha(f),'bytes':f.stat().st_size})
 frozen=[read(f) for f in (root/'actor-invariants').glob('*-after.json')]
 assert len(frozen)==8 and all(x['unchanged'] for x in frozen)
 first=[read(f) for f in sorted((root/'first-replay').glob('*.json'))]
 assert len(first)==4 and all(x['passed'] and x['logprob_max_abs']==0 and x['critic_present']==(method=='ppo') for x in first)
 check['frozen_after_checks']=8;check['first_replay']=first
 check['trainable_update_evidence']=[]
 for r in range(4):
  cp=root/f'sidney-paired-{method}/checkpoints'
  before=read(cp/'global_step_0/pilot'/f'paired-actor-signature-{r:02d}.json')
  after=read(cp/'global_step_2/pilot'/f'paired-actor-signature-{r:02d}.json')
  assert before['trainable_parameters_sha256']!=after['trainable_parameters_sha256']
  gradients=read(root/'actor-invariants'/f'rank-{r:02d}-first-backward.json')['parameters']
  nonzero=sum(g['local_gradient_l2']>0 and 'value_head' not in g['name'] for g in gradients)
  assert nonzero==190
  check['trainable_update_evidence'].append({'rank':r,'trained_parameter_hash_changed':True,'nonzero_noncritic_first_gradient_tensors':nonzero})
 resets=sorted((root/'group-resets').glob('*.json'));assert len(resets)==32
 groups[method]={}
 for f in resets:
  v=read(f);assert v['passed'] and v['raw_public_reference_verified']
  assert v['group_public_reset_identity'] and v['group_exposed_articulation_identity']
  ref=v['raw_replicas'];assert sha(Path(ref['path']))==ref['sha256']
  for g in v['groups']:
   assert g['passed'] and g['replica_count']==8
   key=(g['policy_iteration'],g['round_index'],g['global_group']);assert key not in groups[method]
   groups[method][key]=g
 assert len(groups[method])==256
 check['group_reset_reports']=32;check['initial_groups']=256;check['all_raw_group_replica_sha256_match']=True
 receipts=[]
 for iteration in (1,2):
  for r in range(4):
   f=root/'training-rollouts'/f'iteration-{iteration:04d}-policy-{iteration-1:04d}-rank-{r:02d}.json'
   v=read(f);assert v['completed'] and v['producing_policy_iteration']==iteration-1
   assert v['optimizer_steps_before']==20*(iteration-1)
   assert Path(v['path']).is_file()
   receipts.append(v)
 check['canonical_runtime_SHA_receipts']=receipts
 journals=[]
 for f in sorted((root/'action-accounting').glob('rank-*/*.json')):
  v=read(f)
  assert all(c['resolved'] for c in v['calls'])
  by={}
  for c in v['calls']:by[c['phase']]=by.get(c['phase'],0)+c['executed_or_conservative']
  journals.append({'path':str(f),'calls':len(v['calls']),'all_resolved':True,'phase_counts':by,'charged_conservative_slots':sum(c['executed_or_conservative'] for c in v['calls'])})
 assert len(journals)==4
 check['action_journals']=journals;check['conservative_slots']=sum(x['charged_conservative_slots'] for x in journals)
 assert check['conservative_slots']<=1000000
 check['measured_executed_motor_counts']=False
 spread=read(root/'GROUP-RETURN-SPREAD.json');assert spread['status']=='complete' and len(spread['rows'])==8
 check['group_return_summary']=[{'collection_iteration':i,'groups':sum(x['groups'] for x in spread['rows'] if x['collection_iteration']==i),'constant_groups':sum(x['constant_return_groups'] for x in spread['rows'] if x['collection_iteration']==i),'positive_reward_episodes':sum(x['positive_reward_sum_episodes'] for x in spread['rows'] if x['collection_iteration']==i)} for i in (1,2)]
 check['native_metric_lines']=[line for line in (root/'train.log').read_text(errors='replace').splitlines() if any(k in line for k in ('success_once','actor/run_training','env/run_interact_once','rollout/predict'))]
 check['allocated_bytes']=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file());assert check['allocated_bytes']<=135*2**30
 report['methods'][method]=check
assert groups['ppo'].keys()==groups['grpo'].keys()
fields=['scene_seed','task_description','actual_scene_info','initial_public_observation_sha256','exposed_articulation_sha256']
assert all(all(groups['ppo'][key][f]==groups['grpo'][key][f] for f in fields) for key in groups['ppo'])
report.update(status='complete',all_256_cross_arm_initial_groups_exact=True,cross_arm_key=['policy_iteration','round_index','global_group'],free_gib=shutil.disk_usage(base).free/2**30)
print(json.dumps(report))
