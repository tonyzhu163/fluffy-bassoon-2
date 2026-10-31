"""One A1 status/cursor, recovery and newest closed corpus sample; no journal replay."""
from pathlib import Path
import datetime,hashlib,json,sys,zipfile
root=Path(sys.argv[1]);child=root/'a1-r1';out=child/'experiment'
def bound(p):
 b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),data=json.loads(b))
def hashed(p,expected=None):
 h=hashlib.sha256();size=0
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):size+=len(b);h.update(b)
 return dict(path=str(p),bytes=size,sha256=h.hexdigest(),expected_sha256=expected,matched=None if expected is None else h.hexdigest()==expected)
x=dict(format='frozen_pi05_a1_due_light_v1',checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id='job-hm6fgvu3p4a6',source_commit='8248b1cefe8757b5e43f2a1b6b503e5b93831659',independent_full_journal_replay=False)
for k,p in (('result',out/'result.json'),('qualification',out/'gpu-qualification.json'),('plan',out/'plan.json'),('clock',root/'gpu-clock/worker-time.json'),('allocation',child/'allocation.json'),('action_snapshot',root/'action-budget/snapshot.json'),('closeout',child/'allocation-closeout.json')):
 if p.exists():x[k]=bound(p)
p=child/'exit-code.txt';x['exit_code']=p.read_text().strip() if p.exists() else None
p=child/'console.log'
if p.exists():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-16000));x['console_tail']=f.read().decode(errors='replace')
rows=[bound(p) for p in sorted(out.glob('dev-*.json'))]+[bound(p) for p in sorted(out.glob('search-*.json'))]+[bound(p) for p in sorted(out.glob('confirm-*.json'))]+[bound(p) for p in sorted(out.glob('native-confirm-*.json'))]
# The result cursor is authoritative for all completed kinds; filesystem rows remain compact proof.
x['episode_summaries']=rows
p=out/'recovery.pt'
if p.exists():x['recovery']=hashed(p)
records=x.get('result',{}).get('data',{}).get('records')
if records is None:records={z['data']['spec']['id']:z['data'] for z in rows}
if records:
 last=max(records.values(),key=lambda z:(Path(z['episode_manifest_ref']['root'])/z['episode_manifest_ref']['path']).stat().st_mtime_ns if z.get('episode_manifest_ref') else 0)
 ref=last.get('episode_manifest_ref')
 if ref:
  p=Path(ref['root'])/ref['path'];m=json.loads(p.read_text());x['newest_episode_manifest']=dict(**hashed(p,ref['sha256']),data=m)
  if m['chunks']:
   chunk=m['chunks'][-1];p=p.parent/chunk['path'];x['newest_chunk_metadata']=dict(**hashed(p,chunk.get('sha256')),data=json.loads(p.read_text()))
x['next_owned_check_shanghai']='2026-10-31T18:34:00+08:00'
print(json.dumps(x))
