from pathlib import Path
import base64,hashlib,json,subprocess,time,traceback
project=Path('/mowenjie/fluffy-bassoon-2')
root=project/'outputs/libero-gen-prompt-repair-20261111-r1/bootstrap-teacher02'
root.mkdir(parents=True,exist_ok=False)
started=time.time();code=1
sha='0708bddd138c2afa05d06a51698c2955d8f6a54e'
checkout=project/'sources/libero-repair-0708bddd'
store=project/'sources/libero-gen-store-20261111-r1.git'
bundle=root/'deployment.bundle'
try:
 data=base64.b64decode('__BUNDLE_BASE64__',validate=True)
 assert hashlib.sha256(data).hexdigest()=='581cf0a0eaad1bc53cdbfb3c09377b2cb5cda8b79a63934b88398467cca703d4'
 bundle.write_bytes(data)
 def git(where,*args):return subprocess.check_output(['git','-C',str(where),*map(str,args)],text=True).strip()
 print('GIT_VERSION',subprocess.check_output(['git','--version'],text=True).strip(),flush=True)
 print(git(store,'bundle','verify',bundle),flush=True);git(store,'fetch',bundle,'main')
 print('PROMISOR_PACKS',[p.name for p in (store/'objects/pack').glob('*.promisor')],flush=True)
 git(store,'worktree','add','--detach','--no-checkout',checkout,sha)
 git(checkout,'config','--worktree','core.bare','false');git(checkout,'config','--worktree','core.worktree',checkout)
 git(checkout,'sparse-checkout','init','--no-cone')
 patterns='/src/\n/scripts/\n/exploration/libero-scenes/\n/outputs/libero-procedural-qual-20261111-r1/\n/outputs/libero-gen-prompt-repair-20261111-r1/\n/outputs/libero-gen-custom-layouts-20261111-r1/\n/outputs/libero-gen-history-pilot-20261111-r1/\n!/outputs/libero-gen-prompt-repair-20261111-r1/teacher01-failed.png\n'
 subprocess.run(['git','-C',str(checkout),'sparse-checkout','set','--no-cone','--stdin'],input=patterns,text=True,check=True)
 git(checkout,'checkout',sha)
 assert git(checkout,'rev-parse','HEAD')==sha and not git(checkout,'status','--porcelain')
 assert hashlib.sha256((checkout/'outputs/libero-gen-prompt-repair-20261111-r1/teacher.py').read_bytes()).hexdigest()=='3f4b266d1cb9b46b3692c613be8a74dbdb3759b220155c52d0a63e715898c192'
 print('DEPLOY_VERIFIED',sha,str(checkout),'teacher_sha256', '3f4b266d1cb9b46b3692c613be8a74dbdb3759b220155c52d0a63e715898c192',flush=True)
 (root/'deployment.json').write_text(json.dumps({'commit':sha,'checkout':str(checkout),'bundle_sha256':hashlib.sha256(data).hexdigest(),'clean':True},indent=2)+'\n')
 run=project/'outputs/libero-gen-prompt-repair-20261111-r1/teacher02'
 command=['bash',str(checkout/'outputs/libero-gen-prompt-repair-20261111-r1/launch_teacher.sh'),str(project),str(checkout),sha,'teacher02']
 process=subprocess.Popen(command)
 offset=0
 while True:
  log=run/'teacher.log'
  if log.exists():
   with log.open() as stream:
    stream.seek(offset);chunk=stream.read();offset=stream.tell()
   if chunk:print(chunk,end='',flush=True)
  status=process.poll()
  if status is not None:
   code=status;break
  time.sleep(2)
 # Final compact results remain visible in scheduler logs if the development endpoint is unavailable.
 for result in sorted((run/'teacher').glob('*/result.json')):
  r=json.loads(result.read_text());print('TEACHER_RECEIPT',result.parent.name,json.dumps({k:r.get(k) for k in ['count','native_any','native_final','alignment','geometry_observation_hashes_equal','trajectory_sha256']}),flush=True)
 print('TEACHER_CLOCK',(run/'clock.json').read_text() if (run/'clock.json').exists() else 'missing',flush=True)
except Exception:
 traceback.print_exc()
finally:
 receipt={'started':started,'ended':time.time(),'exit_code':code,'gpus':1,'commit':sha}
 (root/'clock.json').write_text(json.dumps(receipt,indent=2)+'\n');print('BOOTSTRAP_CLOCK',json.dumps(receipt),flush=True)
raise SystemExit(code)
