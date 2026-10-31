python3 - <<'PY'
import base64,hashlib,json,pathlib,socket,subprocess,time,traceback
assert socket.gethostname()=='job-psgxs9n5b08w-master-0'
r=pathlib.Path('/mowenjie/fluffy-bassoon-2/outputs/dsrl-pi0-task57-20261101-s0-r4');e=r/'experiment';out=pathlib.Path('/mowenjie/fluffy-bassoon-2/outputs/dsrl-pi0-task57-20261101-r1')
identity='TERMINAL-WATCH-0725'
assert not (out/(identity+'-TRAINING.json')).exists()
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def emit(stage,value):
 p=out/(identity+'-'+stage+'.json');p.write_text(json.dumps(value,indent=2)+'\n')
 print('FINAL_'+stage+'_RECEIPT_BASE64='+base64.b64encode(json.dumps(value,separators=(',',':')).encode()).decode(),flush=True)
try:
 contract=read(e/'contract.json');assert contract['source_commit']=='00eca3fae5cb8052cee445f87359e26aa18e1a17'
 code=pathlib.Path('/mowenjie/fluffy-bassoon-2/worktrees/dsrl-pi0-task57-20261101-s0-r4');deps=pathlib.Path('/mowenjie/fluffy-bassoon-2/deps/dsrl-pi0-task57-20261101-prep-r4/repo')
 sources=[]
 for path,pin in [(code,contract['source_commit']),(deps,contract['dsrl_commit']),(deps/'openpi',contract['openpi_commit']),(deps/'LIBERO',contract['libero_commit'])]:
  head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=path,text=True).strip();dirty=subprocess.check_output(['git','diff','--name-only'],cwd=path,text=True).strip();assert head==pin and not dirty;sources.append(dict(path=str(path),head=head,tracked_diff=dirty))
 cache={}
 def checkpoints():
  result=[]
  for p in sorted((e/'evaluation-checkpoints').glob('*.json')):
   v=read(p);key=(v['path'],v['sha256']);digest=cache.get(key)
   if digest is None:digest=sha(p.parent/v['path']);cache[key]=digest
   v['payload_sha256_matches']=digest==v['sha256'];assert v['payload_sha256_matches'];result.append(v)
  return result
 checkpoints();print('TERMINAL_WATCH_READY: standard-library metadata only; no model loads or physics; existing allocation and shared phase costs unchanged',flush=True)
 deadline=time.time()+900
 while not (e/'result.json').exists() and time.time()<deadline:time.sleep(.5)
 assert (e/'result.json').exists(),'Terminal observation window expired without altering training'
 result=read(e/'result.json');recovery=read(e/'recovery.json');recovery['payload_sha256_matches']=sha(e/recovery['path'])==recovery['sha256'];assert recovery['payload_sha256_matches'];checks=checkpoints()
 while not (e/'motor-after.json').exists() and time.time()<deadline:time.sleep(.1)
 assert (e/'motor-after.json').exists(),'Motor terminal receipt missing'
 motor=read(e/'motor-after.json');before=read(e/'motor-before.json');motor_verified=motor.get('equal') is True and motor['before']==before and motor['after']==before
 physical=read(e/'physical/physical-cost.json');merged={}
 for line in (e/'metrics.jsonl').read_text().splitlines():
  row=json.loads(line);merged.setdefault(row['updates'],{}).update(row['values'])
 panels=[]
 for u,v in sorted(merged.items()):
  if 'evaluation/success_rate' in v:panels.append(dict(optimizer_updates=u,replay_query_rows=v.get('num_online_samples',520),training_primitive_actions=v.get('env_steps',10400),successes=round(v['evaluation/success_rate']*10),episodes=10,panel_actions=round(v['evaluation/avg_episode_len']*10)))
 bindings={n:read(e/n) for n in ['motor-binding.json','renderer-binding.json','qualification.json']}
 size=sum(p.stat().st_size for p in r.rglob('*') if p.is_file());phase_actions=physical['known_actions']+physical['uncertain_actions']+400
 value=dict(hostname=socket.gethostname(),checked_unix=time.time(),contract=contract,source_attestations=sources,result=result,recovery=recovery,evaluation_checkpoints=checks,panels=panels,motor_terminal=motor,motor_hash_verified=motor_verified,bindings=bindings,physical_cost={k:v for k,v in physical.items() if k!='episodes'},physical_episode_count=len(physical['episodes']),conservative_phase_actions=phase_actions,prior_gpu_seconds=660,prior_actions=400,output_bytes=size,worker_time_pending=not (r/'worker-time.json').exists())
 value['validation_passed']=bool(result.get('completed') and result['updates']>500000 and motor_verified and recovery['progress']['updates']==result['updates'] and phase_actions<=1000000 and size<64*1024**3 and bindings['motor-binding.json']['verified'] and bindings['renderer-binding.json']['verified'])
 emit('TRAINING',value)
 until=time.time()+15
 while not (r/'worker-time.json').exists() and time.time()<until:time.sleep(.05)
 if (r/'worker-time.json').exists():
  accounting=read(r/'worker-time.json');accounting['exit_code_file']=(r/'exit-code.txt').read_text().strip() if (r/'exit-code.txt').exists() else None;accounting['checked_unix']=time.time();accounting['phase_actions']=phase_actions;accounting['output_bytes']=sum(p.stat().st_size for p in r.rglob('*') if p.is_file());emit('ACCOUNTING',accounting)
 else:print('FINAL_ACCOUNTING_PENDING: training and frozen-motor receipts preserved; scheduler cleanup accounting still required',flush=True)
except BaseException as error:
 emit('ERROR',dict(checked_unix=time.time(),error=repr(error),traceback=traceback.format_exc(),scientific_source_changed=False,scientific_process_changed=False))
 raise
PY
