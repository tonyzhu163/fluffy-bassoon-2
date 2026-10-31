"""One bounded closed-RPC sample; JSON/hash only, no model/array decode."""
from pathlib import Path
import base64,bisect,hashlib,json,time
root=Path('/mowenjie/fluffy-bassoon-2/outputs/dsrl-pi05-headroom-20261101-s0-r1/experiment')
directory=root/'case-screen-59';cutoff='1793466701391092359.json'
paths=sorted((directory/'physical-journal').glob('*.json'),key=lambda p:p.name)
names=[p.name for p in paths];i=bisect.bisect_left(names,cutoff)-1
assert i>=0
f=paths[i];raw=f.read_bytes();row=json.loads(raw)
assert row['episode_id']=='screen-126' and row['returned'] and row['outcome_status']=='complete'
def ref(p):
 data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
proof=dict(checked_unix=time.time(),metadata=dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),RPC=row,blobs=[])
for path_key,sha_key in [('input_path','input_sha256'),('outcome_path','outcome_sha256')]:
 p=(directory/'physical-journal'/row[path_key]).resolve();assert p.is_relative_to(root.resolve())
 r=ref(p);assert r['sha256']==row[sha_key];proof['blobs'].append(r)
f=root/'screen-state.json';raw=f.read_bytes();value=json.loads(raw)
proof['recoverable_cursor']=dict(path=str(f),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),phase=value['phase'],screen_cursor=value['screen_cursor'],confirmation_index=value['confirmation_index'],confirmation_cursor=value['confirmation_cursor'],selected_task=value['selected_task'])
proof['cursor_snapshot_base64']=base64.b64encode(raw).decode()
proof['limits']='One closed RPC and its two compressed binary inputs/outcomes byte-checked; no pickle/numerical array decode or historical journal-cost replay. Cursor is a later independent exact-byte snapshot, not the earlier127-episode cost snapshot.'
print(json.dumps(proof,indent=2))
