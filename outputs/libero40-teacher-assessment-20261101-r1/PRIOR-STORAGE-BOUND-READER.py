"""Bounded known-path stat proof. No recursive/raw/journal walk or payload load."""
import hashlib,json,os,re,stat,time
from pathlib import Path
root=Path('/mowenjie/fluffy-bassoon-2/outputs/libero40-teacher-assessment-20261101-r1');data=root/'data';started=time.monotonic();deadline=started+40
measured=[];unfamiliar=[];missing=[];stat_calls=0;classified_structural=[]
def guard():
 if time.monotonic()>deadline or stat_calls>=1200:raise TimeoutError('40second/1200stat bound')
def entries(p):
 guard()
 with os.scandir(p) as stream:return {x.name:x for x in stream}
def measure(p,category):
 global stat_calls
 guard();stat_calls+=1;s=os.lstat(p)
 if not stat.S_ISREG(s.st_mode):unfamiliar.append({'path':str(p),'reason':'not regular file'});return
 measured.append(dict(path=str(p.relative_to(root)),category=category,bytes=s.st_size,mtime_ns=s.st_mtime_ns))
def inventory(p,files,dirs,category):
 actual=entries(p)
 for name,x in actual.items():
  if name in files:measure(p/name,category)
  elif name in dirs:
   if not x.is_dir(follow_symlinks=False):unfamiliar.append({'path':str(p/name),'reason':'expected directory'})
  else:unfamiliar.append({'path':str(p/name),'reason':'unclassified shallow entry'})
 for name in set(files)|set(dirs):
  if name not in actual:missing.append(str(p/name))
 return actual
status='held';error=None
try:
 root_files={'assessment-launcher.lock','write-probe.txt','write-probe-gpu.txt','prelaunch-write.txt','exit-code.txt'}
 inventory(root,root_files,{'data','clock','cpu-r1','reset-diagnosis-cpu-r1'},'phase root metadata')
 inventory(root/'cpu-r1',{'junit.xml','cpu-worker-clock.json','zero-gpu-attestation.json','allocation.json','pytest.log','exit-code.txt','summary.json'},set(),'original CPU qualification')
 inventory(root/'reset-diagnosis-cpu-r1',{'analysis.log','junit.xml','cpu-worker-clock.json','zero-gpu-attestation.json','allocation.json','reset-diagnosis.json','analysis-command.json','pytest.log','exit-code.txt','summary.json'},set(),'two-record CPU diagnosis')
 inventory(root/'clock',{'worker-time.json','worker-time.lock'},{'attempts'},'GPU clock metadata')
 inventory(root/'clock/attempts',set(),{'1793502780000000-0000'},'GPU attempt directory')
 inventory(root/'clock/attempts/1793502780000000-0000',{'actual-resources.json','ticket.json','gpu-probe.json','attempt.json','exit-code.txt','console.log'},set(),'closed GPU attempt metadata')
 top=entries(data)
 expected_case_refs={f'case-reference-{i:02d}-{s}.pkl.gz' for i in range(20) for s in [40,41]}|{'case-reference-20-40.pkl.gz'}
 expected_episodes={f'episode-{i:03d}' for i in range(162)}
 top_files={'results-summary.json','loaded-actor-bindings.json','case-results.csv','motor-binding.json','progress.json','assessment-recovery.pkl','motor-final.json','source-provenance.json'}
 servers=[]
 for name,x in top.items():
  p=data/name
  if name in top_files:measure(p,'assessment top metadata and rolling recovery')
  elif name in expected_case_refs:measure(p,'41 original case references')
  elif name in expected_episodes:
   if not x.is_dir(follow_symlinks=False):unfamiliar.append({'path':str(p),'reason':'expected episode directory'})
  elif re.fullmatch(r'server-(?:0|[1-9][0-9]*)-[0-9]+',name) and x.is_dir(follow_symlinks=False):servers.append(name)
  else:unfamiliar.append({'path':str(p),'reason':'unclassified assessment top entry'})
 for name in top_files|expected_case_refs|expected_episodes:
  if name not in top:missing.append(str(data/name))
 if sorted(int(x.split('-')[1]) for x in servers)!=list(range(0,161,8)):unfamiliar.append({'path':str(data),'reason':'unexpected source server ordinal set','servers':servers})
 for name in servers:inventory(data/name,{'simulator-ready.json','renderer-binding.json','simulator.log'},set(),'21 simulator server metadata/logs')
 for i in range(162):
  files=({'trace.pkl.gz','rng-context.pkl.gz','receipt.json'} if i<161 else set())|({'qualification.json'} if i==0 else set())
  inventory(data/f'episode-{i:03d}',files,{'raw-rpc','physical-journal'},'closed episode trace/RNG/receipt or first qualification')
  classified_structural.extend([dict(path=f'data/episode-{i:03d}/raw-rpc',kind='raw RPC subtree; contents NOT enumerated'),dict(path=f'data/episode-{i:03d}/physical-journal',kind='fixed CostLedger JSON subtree; contents NOT enumerated')])
 status='metadata_complete_pending_raw_schema_confirmation' if not unfamiliar and not missing else 'held_unfamiliar_or_missing'
except BaseException as e:error=f'{type(e).__name__}: {e}';status='held_incomplete_metadata'
progress_path=data/'progress.json';progress=json.loads(progress_path.read_text())
failed_path=data/'episode-161/physical-journal/1793504048251783216-reset.json';failed=json.loads(failed_path.read_text())
assert progress['cursor']==161 and progress['costs']=={'all_in_actions':26086,'physical_episodes':161,'uncertain_actions':0}
assert failed['kind']=='reset' and failed['returned'] and failed['actual_steps']==10
closed_primitives=26086-161*10;max_chunks=(closed_primitives+9*161)//10;max_raw=2*max_chunks+162;max_ledger=max_chunks+162
fixed_measurement=sum(x['bytes'] for x in measured)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
print(json.dumps(dict(format='all40_assessment_prior_output_storage_bound_v1',status=status,error=error,output_root=str(root),completed_roles=161,failed_zero_chunk_resets=1,closed_primitive_steps=closed_primitives,chunk_count_upper=max_chunks,raw_RPC_file_count_upper=max_raw,raw_RPC_per_file_bound_bytes=1<<20,raw_RPC_structural_upper_bytes=max_raw*(1<<20),ledger_JSON_count_upper=max_ledger,ledger_JSON_per_file_bound_bytes=4096,ledger_JSON_structural_upper_bytes=max_ledger*4096,measured_non_RPC_bytes=fixed_measurement,conditional_prior_output_upper_bytes=fixed_measurement+max_raw*(1<<20)+max_ledger*4096,measured_files=measured,structural_subtrees=classified_structural,unfamiliar=unfamiliar,missing=missing,actual_stat_calls=stat_calls,maximum_stat_calls=1200,metadata_elapsed_seconds=time.monotonic()-started,metadata_deadline_seconds=40,counter_refs=[dict(path=str(progress_path),sha256=sha(progress_path)),dict(path=str(failed_path),sha256=sha(failed_path))],actor_inputs_copied_into_output=False,actor_inputs_scope='No actor/checkpoint file class is allowed in the measured output tree. Original2.3GB actors remain external references; no actor payload loaded or enumerated here.',actual_recursive_du_bytes=None,raw_schema_confirmation_pending=True,no_runtime_or_storage_admission=True,limits=['Conservative logical-file-byte bound, not exact du or filesystem allocated-block measurement.','Known source-written rawRPC and six-field CostLedger subtrees are structurally bounded without enumeration.','Any unknown shallow file/directory class or exceeded metadata deadline leaves the proof held.','No raw/journal payload forest, array/checkpoint/model deserialization, tests, simulator or new allocation.']),indent=2))
