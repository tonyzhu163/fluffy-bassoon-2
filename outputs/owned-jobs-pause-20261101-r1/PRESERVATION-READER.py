import json,hashlib,os,shutil,time
from pathlib import Path
base=Path('/mowenjie/fluffy-bassoon-2/outputs');out=base/'owned-jobs-pause-20261101-r1';out.mkdir(exist_ok=False)
def ref(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest())
def copy(p, dest):
 p=Path(p);d=out/dest;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,d);return ref(d)
r={'format':'owned_jobs_user_pause_preservation_v1','checked_unix':time.time(),'authority_message':'01a19572-3598-7cb1-a201-a619b5f12660','jobs':{}}
root=base/'dsrl-pi05-headroom-20261101-s0-r1';case=root/'experiment/case-learn-44';b=out/'primary';b.mkdir()
for n in ('index.json','contract.json'):copy(case/n,'primary/'+n)
for n in range(3):
 raw=(case/'state.json').read_bytes();s=json.loads(raw);src=case/s['model']['path'];dest=b/f'closed-recovery-attempt-{n}.pkl';os.link(src,dest)
 (b/f'state-attempt-{n}.json').write_bytes(raw);actual=ref(dest)
 if actual['sha256']==s['model']['sha256']:
  r['jobs']['job-2qn8dpw3yvlb']={'snapshot_state':ref(b/f'state-attempt-{n}.json'),'full_checkpoint':actual,'model':s['model'],'training_episodes':s['training']['episodes'],'phase':s['phase'],'next_threshold':s['next_threshold'],'active_panel':s['active_panel'],'closed_checkpoint_verified':True};break
else:raise RuntimeError('No stable primary closed checkpoint')
copy(root/'worker-time.json','primary/worker-time-before-stop.json')
root=base/'libero40-teacher-assessment-20261101-r2';data=root/'data';b=out/'assessment';b.mkdir()
for n in ('progress.json','assessment-recovery.pkl'):
 copy(data/n,'assessment/'+n)
p=json.loads((b/'progress.json').read_text());actual=ref(b/'assessment-recovery.pkl');copy(base/'libero40-teacher-assessment-20261101-r1/clock/worker-time.json','assessment/worker-time-before-stop.json')
r['jobs']['job-bc2ooyztqsfh']={'closed_progress':p,'progress_ref':ref(b/'progress.json'),'current_full_recovery':actual,'matches_last_closed_progress':actual['sha256']==p['recovery']['sha256'],'no_deserialization':True,'resume_limit':'Current recovery may include active next role; ordinary driver refuses active cursor; preserve rather than manufacture closed state.'}
(out/'PRESERVATION-BEFORE-STOP.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
