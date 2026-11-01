"""Bounded closed-source JSON/text inventory; never open actor/array payloads."""
from pathlib import Path
import hashlib,json,time
BASE=Path('/mowenjie/fluffy-bassoon-2/outputs');cache={};read_bytes=0;refs={};schedule_cache={}
def read(path):
 global read_bytes
 p=Path(path)
 if str(p) not in cache:
  size=p.stat().st_size
  if size>16*1024**2 or read_bytes+size>128*1024**2:raise ValueError('Bounded metadata exceeded: '+str(p)+' size='+str(size)+' cumulative='+str(read_bytes))
  raw=p.read_bytes();read_bytes+=len(raw);cache[str(p)]=json.loads(raw);refs[str(p)]=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),is_symlink=p.is_symlink(),resolved_path=str(p.resolve()))
 return cache[str(p)]
def stat_payload(path,published_sha):
 p=Path(path);return dict(path=str(p),published_sha256=published_sha,exists=p.is_file(),is_symlink=p.is_symlink(),bytes=p.stat().st_size if p.is_file() else None,byte_SHA_rehashed=False,payload_deserialized=False)
def exposure(index):
 rows=index.get('episode_receipts',[])
 return dict(actual_state_ids=sorted({r['initial_state'] for r in rows}),states40_41={str(s):dict(episodes=sum(r['initial_state']==s for r in rows),case_ids=sorted({r['case_id'] for r in rows if r['initial_state']==s}),scene_seeds=sorted({r['scene_seed'] for r in rows if r['initial_state']==s})) for s in (40,41)})
def leaf(root):
 root=Path(root);ex=root/'experiment';i=read(ex/'index.json');c=read(ex/'contract.json');pubs=i.get('publications',[]);pub=pubs[-1] if pubs else None
 candidates=[]
 for p in pubs:
  if p.get('status')=='raw_only' or Path(p.get('source_checkpoint','')).name=='case-00000-source.pkl':
   episodes=[e for e in i.get('episode_receipts',[]) if int(e['case_id'].rsplit('-',1)[-1])<=p.get('through_case',-1)]
   updates=sorted({e.get('source_updates') for e in episodes})
   candidates.append(dict(publication=p,checkpoint=stat_payload(ex/p['source_checkpoint'],p.get('source_checkpoint_sha256')) if p.get('source_checkpoint') else None,recorded_episode_update_values=updates,all_recorded_updates0=bool(episodes) and updates==[0],pure_initial_actor_hash_metadata=p.get('actor_state_sha256'),original_init_parameter_identity_independently_proven=False))
 schedule=root/'schedule.json';schedule_ref=None
 if schedule.is_file():
  digest=i['schedule_sha256']
  if digest not in schedule_cache:read(schedule);schedule_cache[digest]=refs[str(schedule)]
  schedule_ref=dict(path=str(schedule),bytes=schedule.stat().st_size,published_content_sha256=digest,same_declared_content_sample_ref=schedule_cache[digest],this_copy_byte_SHA_rehashed=str(schedule)==schedule_cache[digest]['path'])
 fixed={k:c.get(k) for k in ('source_commit','learner_seed','horizon','chunk','motor','feature_contract','noise_contract','source_recipe','resets','source_generation_vs_deployment','validation','collection_mode','history_id')}
 result=dict(root=str(root),index=refs[str(ex/'index.json')],contract=refs[str(ex/'contract.json')],schedule=schedule_ref,task={k:v for k,v in c['task'].items() if k!='partitions'},terminal_flags={k:i.get(k) for k in ('completed','all_started_cases_complete','slice_completed','stopped_at_wall_boundary','collection_qualified','smoke_qualified','stopping_reason','continuation_stop_reason')},source_updates=i['source_updates'],episodes=len(i.get('episode_receipts',[])),chunks=i.get('chunks'),actions=i.get('total_action_budget_charge'),final_publication=pub,final_publication_checkpoint=stat_payload(ex/pub['source_checkpoint'],pub.get('source_checkpoint_sha256')) if pub and pub.get('source_checkpoint') else None,terminal_recovery=stat_payload(ex/'source-recovery.pkl',i.get('latest_recovery_sha256')),pre_SGD_candidates=candidates,initial_checkpoint_status='Index-referenced raw_only/update0 candidates only; no regeneration or actor weight load' if candidates else 'No index-referenced case-00000-source.pkl or raw_only checkpoint; original initialized actor unproven/absent under requested search scope',actual_state_exposure=exposure(i),fixed_contract=fixed,continuation=c.get('continuation'),index_continuation=i.get('continuation'))
 return result
plan_path=BASE/'libero40-source-production-plan-cpu-20261031-r1/production.json';plan=read(plan_path)
ledger_path=BASE/'libero40-sac-topup-preparation-20261031-r1/ledger.json';ledger=read(ledger_path)
admission_path=BASE/'libero40-source-pipeline-cpu-20261031-r2/snapshot/admission.json';admission=read(admission_path)
train=[s for s in plan['slots'] if s['slot'].endswith('/train')];assert len(train)==40 and len({s['slot'] for s in train})==40
validation={s['slot'].rsplit('/',1)[0]:s for s in plan['slots'] if s['slot'].endswith('/validation')}
rows=[];ancestry={};unresolved=[]
for ordinal,slot in enumerate(train):
 key=slot['slot'].rsplit('/',1)[0];canonical=ledger['leaves'][slot['slot']]['root'];planned=slot['parent']['root']
 if canonical!=planned:unresolved.append(dict(slot=slot['slot'],ledger_root=canonical,production_plan_root=planned))
 current=leaf(canonical);task=current.pop('task');semantic=task['semantic_task'];bddl=Path('/mowenjie/fluffy-bassoon-2/deps/libero-dsrl-87edbd1-20261029-r1/repo/libero/libero/bddl_files')/task['suite']/(semantic+'.bddl')
 bddl_ref=dict(path=str(bddl),published_sha256=task['bddl_sha256'],exists=bddl.is_file())
 if bddl.is_file():
  raw=bddl.read_bytes();assert len(raw)<1024**2;bddl_ref.update(actual_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),matches=hashlib.sha256(raw).hexdigest()==task['bddl_sha256'])
 chain=[];parent=current.get('continuation',{});parent=parent.get('parent') if isinstance(parent,dict) else None
 for _ in range(5):
  if not parent:break
  pr=parent['root'];chain.append(parent)
  if pr not in ancestry:ancestry[pr]=leaf(pr)
  nxt=ancestry[pr].get('continuation');parent=nxt.get('parent') if isinstance(nxt,dict) else None
 val=leaf(validation[key]['parent']['root']) if key in validation else None
 rows.append(dict(source_ordinal=ordinal,task_key=key,suite=task['suite'],task_id=task['task_id'],semantic_task=semantic,BDDL=bddl_ref,initial_states_published_sha256=task['initial_states_sha256'],canonical_equals_production_plan=canonical==planned,production_parent_metadata=slot['parent'],canonical_terminal_train=current,ancestry=chain,independent_validation=val))
competence_path=BASE/'libero40-source-sac-competence-20261030-r2/experiment/index.json';competence=read(competence_path)
competence_rows=competence.get('receipts',[]);old_state_exposure={}
for row in competence_rows:
 key=row.get('task_key',row.get('source_task_key'));case=row.get('case',{});state=row.get('initial_state',case.get('initial_state'))
 if key is not None:old_state_exposure.setdefault(key,[]).append(dict(initial_state=state,method=row.get('method'),scene_seed=row.get('scene_seed',case.get('scene_seed'))))
result=dict(format='all40_existing_teacher_metadata_inventory_v1',checked_unix=time.time(),metadata_read_files=len(cache),metadata_read_bytes=read_bytes,metadata_refs=list(refs.values()),source_ordinal_basis='Exact train-slot order in bound production plan; interleaved suite IDs',canonical_ledger_ref=refs[str(ledger_path)],production_plan_ref=refs[str(plan_path)],snapshot_admission_ref=refs[str(admission_path)],all40=rows,ancestor_leaves=ancestry,unresolved_competing_leaves=unresolved,historical_competence=dict(index=refs[str(competence_path)],keys=list(competence),state_exposure=old_state_exposure,receipt_shape_examples=competence_rows[:1]),limits='JSON/index/contract/ledger/schedule plus official BDDL text and file stat only; published checkpoint SHAs reused, no model/array/pickle/NPZ payload opened or rehashed, no scheduler/tests/RPC/allocation. Original initialized-actor identity not inferred from source_update0 or filename. Other unindexed historical exposures not exhaustively searched.',new_assessment_outcomes_seen=False,new_source_or_history_admission=False)
print(json.dumps(result,indent=2,allow_nan=False))
