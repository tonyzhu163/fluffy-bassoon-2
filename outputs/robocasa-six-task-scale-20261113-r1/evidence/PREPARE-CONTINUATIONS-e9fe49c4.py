import hashlib,json,os,socket,sys,time
from pathlib import Path
ROOT=Path('/mnt/bos/adgfyl/strong-plant-archives/robocasa-six-task-scale-20261113-r1')
SOURCE=Path('/mowenjie/fluffy-bassoon-2/sources/matched-final-e9fe49c4')
HERE=SOURCE/'outputs/robocasa-six-task-scale-20261113-r1'
sys.path.insert(0,str(HERE))
from fit_job import checked_resumes
import torch
torch.set_num_threads(4)
OUT=ROOT/'resume-inputs01'
assert not OUT.exists();OUT.mkdir()
PIN='4987585e7427061de30771ee207f9b2b79eedc6a'
CORPORA='85394094fcc159984ffbaf249c93cbced236b723ce4fd6d39fa3e54b5dce4e2e'
JOBS={3:'job-o8p1uyvoxzr3-master-0',6:'job-5zbsgqfx0ax6-master-0'}
DONE={};DEADLINE=1794672000

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(16<<20):h.update(b)
 return h.hexdigest()
def ref(p):return dict(path=str(p),sha256=digest(p))
def read(p):return json.loads(Path(p).read_text())
def write(p,j):p.write_text(json.dumps(j,indent=2)+'\n')
def finite(x):
 if isinstance(x,torch.Tensor):assert torch.isfinite(x).all()
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,(tuple,list)):
  for v in x:finite(v)
state=dict(started_unix=time.time(),hostname=socket.gethostname(),pid=os.getpid(),script_sha256=digest(__file__),source_commit='e9fe49c4a8892f0eec17e5d6c5fada229eac93dc',deadline_unix=DEADLINE,GPU_hours=0,prepared=DONE,submitted=False)
try:
 while len(DONE)<4:
  for size,host in JOBS.items():
   old=ROOT/f'fits{size}r1'
   errors=list(old.glob('*ERROR.json'))
   assert not errors,[str(p) for p in errors]
   if not (old/'EXIT.json').exists():continue
   terminal=read(old/'EXIT.json');launch=read(old/'LAUNCH.json')
   assert launch['hostname']==host and launch['source_commit']==PIN and launch['corpus_size']==size
   assert terminal['ended_unix']<=time.time() and not terminal['completed']
   assert {r['method'] for r in terminal['results']}=={'ad','dpt','dit','ic_iql'}
   assert all(r['exit_code']==0 for r in terminal['results'])
   for method in ['dit','ic_iql']:
    key=f'{size}/{method}'
    if key in DONE:continue
    base=old/method;progress=read(base/'PROGRESS.json');contract=read(base/'CONTRACT.json');method_exit=read(old/f'{method}-EXIT.json')
    assert method_exit['exit_code']==0 and not method_exit['complete'] and not (base/'COMPLETE.json').exists()
    spec=dict(fit_source_commit=PIN,corpus_size=size,methods={method:dict(terminal=ref(old/'EXIT.json'),launch=ref(old/'LAUNCH.json'),progress=ref(base/'PROGRESS.json'),contract=ref(base/'CONTRACT.json'),checkpoint=progress['checkpoint'])})
    path=OUT/f'{size}-{method}.json';assert not path.exists();write(path,spec)
    checked_resumes(path,digest(path),methods=[method],corpus_size=size,fit_pin=PIN,corpora_sha256=CORPORA)
    saved=torch.load(progress['checkpoint']['path'],map_location='cpu',weights_only=False)
    assert saved['format']=='matched_kv_fit_v1' and saved['contract']==contract
    assert saved['progress']['actor_update']==progress['actor_update'] and saved['progress']['value_update']==progress['value_update']==1000
    assert contract['seed']==0 and contract['logical_batch']==8 and contract['actor_updates']==10000
    assert saved['policy'] and saved['values']
    for field in ['policy','values','optimizer','value_optimizer']:finite(saved[field])
    steps={}
    for name,expected in [('optimizer',progress['actor_update']),('value_optimizer',1000)]:
     optimizer=saved[name];assert optimizer['state']
     steps[name]=sorted({int(s['step'].item()) for s in optimizer['state'].values()});assert steps[name]==[expected]
     assert all(g['lr']==contract['learning_rate'] and g['weight_decay']==0 for g in optimizer['param_groups'])
    assert all(k in saved['rng'] for k in ['sampler','python','numpy','torch','cuda'])
    assert len(saved['rng']['cuda'])==1 and saved['flow_generator'].numel()>0
    result=dict(passed=True,spec=ref(path),original_hostname=host,original_terminal=ref(old/'EXIT.json'),method_terminal=ref(old/f'{method}-EXIT.json'),actor_update=progress['actor_update'],value_update=1000,remaining_actor_updates=10000-progress['actor_update'],optimizer_steps=steps,finite_full_state=True,RNG_fields_present=True,checked_unix=time.time(),submitted=False)
    write(OUT/f'{size}-{method}-AUDIT.json',result);DONE[key]=result
    print(json.dumps({'prepared':key,**result}),flush=True)
    del saved
  state.update(checked_unix=time.time(),prepared=DONE,completed=len(DONE)==4);write(OUT/'STATE.json',state)
  if len(DONE)==4:break
  if time.time()>=DEADLINE:raise TimeoutError('No complete original terminal set by observation deadline; no launch performed')
  time.sleep(30)
except BaseException as e:
 state.update(error=repr(e),ended_unix=time.time(),completed=False);write(OUT/'STATE.json',state);raise
state.update(ended_unix=time.time(),completed=True);write(OUT/'STATE.json',state)
