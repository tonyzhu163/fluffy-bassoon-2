"""Deterministic split; selection sees only source archive metadata."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = Path('/mowenjie/fluffy-bassoon-2/outputs/libero-xemb-ur5e-scale-20261120-r1/gpu01/archive/archive.json')
ARCHIVE_SHA = '37f062958cb63b962ceefb760f0592fdb120da966369ee308b509e2d82449076'
COMMANDS = ('nom','yaw90','yaw180','yawm90','scale_lo','scale_hi','grip_inv')
EXCLUDED = {'Panda-0000','Panda-0001','Panda-0002','Panda-0003',
            'Sawyer-0084','Sawyer-0085','Sawyer-0086','Sawyer-0087'}
FIELDS = ('index','shard','arm','suite','task_id','init_state_id','seed','embodiment','physics','passive_path',
          'passive_controls','passive_first_proposal_sha256','passive_noise_sha256')


def select(raw):
    assert hashlib.sha256(raw).hexdigest() == ARCHIVE_SHA
    rows = [r for r in json.loads(raw)['trials'] if r['arm'] in ('Panda','Sawyer') and r['split']=='train']
    tasks = sorted({(r['suite'],r['task_id']) for r in rows})
    assert len(rows)==1344 and len(tasks)==24 and all(sum(t[0]==s for t in tasks)==6 for s,_ in tasks)
    selected = []
    for rank,(suite,tid) in enumerate(tasks):
        split = 'val' if rank%6 in (2,5) else 'train'
        for body,arm in enumerate(('Panda','Sawyer')):
            for j in range(4):
                command = COMMANDS[(rank+body+j)%7]
                pool = [r for r in rows if (r['suite'],r['task_id'],r['arm'],r['embodiment'],r['passive_controls']) ==
                        (suite,tid,arm,command,15) and f"{arm}-{r['index']:04d}" not in EXCLUDED]
                r = min(pool,key=lambda x:hashlib.sha256(x['passive_path'].encode()).digest())
                selected.append(dict(split=split,id=f"{arm}-{r['index']:04d}",**{k:r[k] for k in FIELDS}))
    assert len({r['id'] for r in selected})==192 and sum(r['split']=='train' for r in selected)==128
    return dict(archive_sha256=ARCHIVE_SHA,rows=selected)


if __name__=='__main__':
    print(json.dumps(select(sys.stdin.buffer.read()),indent=2))
