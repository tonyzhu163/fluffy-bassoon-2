import base64,collections,hashlib,json,pathlib,socket,subprocess,sys,time
assert sys.platform=='linux' and socket.gethostname().startswith('job-')
started=time.time();root=pathlib.Path('/mowenjie/fluffy-bassoon-2');r=root/'outputs/dsrl-pi0-task57-20261101-s0-r4';e=r/'experiment';o=root/'outputs/dsrl-pi0-task57-20261101-harvest-r1';o.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
c=read(e/'contract.json');assert c['source_commit']=='00eca3fae5cb8052cee445f87359e26aa18e1a17'
result=read(e/'result.json');motor=read(e/'motor-after.json');before=read(e/'motor-before.json');worker=read(r/'worker-time.json');exit_code=int((r/'exit-code.txt').read_text())
assert result['completed'] and result['updates']>500000 and result['stop_reason']=='optimizer_target'
assert motor['equal'] and motor['before']==motor['after']==before
assert exit_code==worker['exit_code']==0 and worker['prior_gpu_seconds']==660 and worker['cumulative_gpu_seconds']<=36000
recovery=read(e/'recovery.json');recovery['payload_sha256_matches']=sha(e/recovery['path'])==recovery['sha256'];assert recovery['payload_sha256_matches'] and recovery['progress']['updates']==result['updates']
checks=[]
for p in sorted((e/'evaluation-checkpoints').glob('*.json')):
 v=read(p);v['payload_sha256_matches']=sha(p.parent/v['path'])==v['sha256'];assert v['payload_sha256_matches'];checks.append(v)
assert [v['updates'] for v in checks]==list(range(0,500001,10000))
merged={}
for line in (e/'metrics.jsonl').read_text().splitlines():
 v=json.loads(line);merged.setdefault(v['updates'],{}).update(v['values'])
panels=[]
for u,v in sorted(merged.items()):
 if 'evaluation/success_rate' in v:panels.append(dict(optimizer_updates=u,replay_query_rows=v.get('num_online_samples',520),training_primitive_actions=v.get('env_steps',10400),successes=round(v['evaluation/success_rate']*10),episodes=10,panel_actions=round(v['evaluation/avg_episode_len']*10)))
assert [v['optimizer_updates'] for v in panels]==list(range(0,500001,10000))
physical=read(e/'physical/physical-cost.json');roles={};queries={}
for row in physical['episodes']:
 t=roles.setdefault(row['role'],dict(episodes=0,actions=0,successes=0));t['episodes']+=1;t['actions']+=row['actions'];t['successes']+=int(row['success'])
 assert (e/'physical'/row['trace']).is_file()
 if row['role']=='query':
  q=queries.setdefault(row['updates'],dict(episodes=0,actions=0,successes=0));q['episodes']+=1;q['actions']+=row['actions'];q['successes']+=int(row['success'])
for v in panels:
 q=queries[v['optimizer_updates']];assert (q['episodes'],q['actions'],q['successes'])==(v['episodes'],v['panel_actions'],v['successes'])
assert physical['uncertain_actions']==0 and physical['pending'] is None
assert roles['training']['episodes']==result['episodes'] and roles['training']['actions']==result['training_primitives']
assert sum(t['actions'] for t in roles.values())==physical['known_actions']==result['physical_actions']
assert physical['known_actions']+400<=1000000
sources=[];deps=root/'deps/dsrl-pi0-task57-20261101-prep-r4/repo'
for p,pin in [(root/'worktrees/dsrl-pi0-task57-20261101-s0-r4',c['source_commit']),(deps,c['dsrl_commit']),(deps/'openpi',c['openpi_commit']),(deps/'LIBERO',c['libero_commit'])]:
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip();dirty=subprocess.check_output(['git','diff','--name-only'],cwd=p,text=True).strip();assert head==pin and not dirty;sources.append(dict(path=str(p),head=head,tracked_diff=dirty))
bindings={n:read(e/n) for n in ['motor-binding.json','renderer-binding.json','qualification.json']};assert bindings['motor-binding.json']['verified'] and bindings['renderer-binding.json']['verified']
sizes={str(p.name):sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in (root/'outputs').glob('dsrl-pi0-task57-20261101-*') if p.is_dir()};upper=sum(sizes.values())+1024**2;assert upper<64*1024**3
receipts={p.name:read(p) for p in [root/'outputs/dsrl-pi0-task57-20261101-r1/TERMINAL-WATCH-0725-TRAINING.json',root/'outputs/dsrl-pi0-task57-20261101-r1/TERMINAL-WATCH-0725-ACCOUNTING.json'] if p.exists()}
value=dict(hostname=socket.gethostname(),checked_unix=time.time(),contract=c,result=result,motor_terminal=motor,motor_hash_verified=True,worker_time=worker,exit_code=exit_code,recovery=recovery,evaluation_checkpoints=checks,panels=panels,physical_cost={k:v for k,v in physical.items() if k!='episodes'},physical_episode_count=len(physical['episodes']),physical_by_role=roles,query_panels_from_physical_ledger=queries,panel_and_ledger_accounting_matches=True,conservative_phase_actions=physical['known_actions']+400,prior_actions=400,prior_gpu_seconds=660,source_attestations=sources,bindings=bindings,output_bytes=sizes[r.name],all_owned_output_bytes=sizes,conservative_output_bytes_upper_bound=upper,terminal_observer_receipt_names=list(receipts),cpu_harvest=dict(maximum_scheduler_seconds=60,command_timeout_seconds=15,prior_conservative_worker_seconds=1740,conservative_charge_seconds=60,maximum_cumulative_worker_seconds=1800,model_loads=0,extra_physical_actions=0),metadata_elapsed_seconds=time.time()-started,validation_passed=True)
(o/'receipt.json').write_text(json.dumps(value,indent=2)+'\n');print('HARVEST_RECEIPT_BASE64='+base64.b64encode(json.dumps(value,separators=(',',':')).encode()).decode(),flush=True)
