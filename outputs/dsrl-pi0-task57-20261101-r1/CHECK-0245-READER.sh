python3 - <<'PY'
import hashlib,json,pathlib,socket,subprocess,time,xml.etree.ElementTree as ET
assert socket.gethostname()=='job-psgxs9n5b08w-master-0'
r=pathlib.Path('/mowenjie/fluffy-bassoon-2/outputs/dsrl-pi0-task57-20261101-s0-r4')
e=r/'experiment'
def read(name):return json.loads((e/name).read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
contract=read('contract.json');progress=read('progress.json');recovery=read('recovery.json');cost=read('physical/physical-cost.json')
assert contract['source_commit']=='00eca3fae5cb8052cee445f87359e26aa18e1a17'
recovery['payload_sha256_matches']=sha(e/recovery['path'])==recovery['sha256']
assert recovery['payload_sha256_matches']
checks=[]
for p in sorted((e/'evaluation-checkpoints').glob('*.json')):
 v=json.loads(p.read_text());v['payload_sha256_matches']=sha(p.parent/v['path'])==v['sha256'];assert v['payload_sha256_matches'];checks.append(v)
bindings={n:read(n) for n in ['motor-binding.json','renderer-binding.json','qualification.json','motor-before.json']}
assert bindings['motor-binding.json']['verified'] and bindings['renderer-binding.json']['verified']
expected=bindings['motor-binding.json']['expected_uuid'];pid=str(bindings['motor-binding.json']['pid'])
smi=subprocess.check_output(['nvidia-smi','-q','-x'],text=True,timeout=25)
start=smi.find('<?xml');start=start if start>=0 else smi.find('<nvidia_smi_log')
contexts=[]
for gpu in ET.fromstring(smi[start:]).findall('gpu'):
 for p in gpu.findall('./processes/process_info'):
  if p.findtext('pid','').strip()==pid:contexts.append(dict(uuid=gpu.findtext('uuid'),type=p.findtext('type','')))
assert contexts and {p['uuid'] for p in contexts}=={expected}
assert any('C' in p['type'] and 'G' in p['type'] for p in contexts)
metrics=[]
for line in (e/'metrics.jsonl').read_text().splitlines():
 row=json.loads(line);v=row['values']
 if any(k.startswith('evaluation/') or k in ['num_online_samples','num_online_trajs','env_steps'] for k in v):metrics.append(row)
size=sum(p.stat().st_size for p in r.rglob('*') if p.is_file())
assert size<64*1024**3 and cost['known_actions']+cost['uncertain_actions']+400<=1000000
v=dict(hostname=socket.gethostname(),checked_unix=time.time(),source_commit=contract['source_commit'],contract=contract,progress=progress,recovery=recovery,evaluation_checkpoints=checks,physical_cost=dict(known_actions=cost['known_actions'],uncertain_actions=cost['uncertain_actions'],episodes=len(cost['episodes']),pending=cost['pending']),bindings=bindings,current_pid_gpu_contexts=contexts,output_bytes=size,metrics=metrics,motor_after_present=(e/'motor-after.json').exists(),worker_time_present=(r/'worker-time.json').exists())
p=pathlib.Path('/mowenjie/fluffy-bassoon-2/outputs/dsrl-pi0-task57-20261101-r1/CHECK-0245-DIRECT.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+chr(10))
print('SNAPSHOT_JSON='+json.dumps(v),flush=True)
PY
