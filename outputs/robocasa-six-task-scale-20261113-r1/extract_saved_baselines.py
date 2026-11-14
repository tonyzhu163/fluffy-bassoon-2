import hashlib,json,tarfile,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
COLD=Path('/mnt/bos/adgfyl/strong-plant-archives')
RUN='robocasa-six-task-scale-20261113-r1'
OUT=Path('/tmp/robocasa-baseline-audit-20261114')
OUT.mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def selected(name):
 return name in ('before-evaluation.json','after-evaluation.json','manifest.json','MANIFEST.json','resolved-config.json','flow-contract.json','operator-checkpoint-exit.json','progress.json') or name.startswith('evaluations/') and name.endswith('.json') or name.startswith('canonical-eval/') and name.endswith('complete.json') or name.startswith('environment/') and name.endswith('.json') and any(s in name for s in ('/before-','/after-i030-','/eval_step_030-')) or name.endswith('/global_step_30/pilot-boundary.json')
def one(args):
 task,attempt,prior=args
 receipt=COLD/(('robocasa-six-source-decision-20261112-r1/local01' if prior else RUN)+'/teachers')/task/attempt/'ARCHIVE.json'
 raw=receipt.read_bytes();d=json.loads(raw)
 assert d['verified'] and d['full_archive_SHA_readback'] and d['all_stageback_members_verified']
 assert Path(d['cold']).stat().st_size==d['archive_bytes']
 expected={x['path']:x for x in d['files'] if selected(x['path'])}
 dest=OUT/(task+('-sft' if prior else ''));dest.mkdir(exist_ok=True)
 records={}
 with tarfile.open(d['cold'],'r:') as tar:
  for member in tar:
   if member.name not in expected:continue
   assert member.isfile()
   buf=tar.extractfile(member).read();e=expected[member.name]
   assert len(buf)==e['bytes'] and sha(buf)==e['sha256'],member.name
   p=dest/member.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(buf)
   records[member.name]=e
   if len(records)==len(expected):break
 assert set(records)==set(expected)
 inv={'receipt':str(receipt),'receipt_sha256':sha(raw),'archive_sha256':d['archive_sha256'],'source':d['source'],'files':d['files'],'metadata_verified':list(records),'checked_unix':time.time()}
 (dest/'VERIFIED-INVENTORY.json').write_text(json.dumps(inv,indent=2)+'\n')
 print(json.dumps({'task':task,'prior':prior,'metadata_files':len(records),'after_present':'after-evaluation.json'in records,'step30_eval_present':'evaluations/step-030.json'in records}),flush=True)
with ThreadPoolExecutor(max_workers=3) as pool:
 list(pool.map(one,[('OpenDrawer','attempt01',False),('CoffeeSetupMug','attempt01',False),('PickPlaceCounterToStove','attempt01',False),('CloseToasterOvenDoor','attempt02',False),('TurnOffStove','attempt02',False),('PickPlaceCounterToStove','attempt01',True)]))
