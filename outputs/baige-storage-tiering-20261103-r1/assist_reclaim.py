"""Assist only leaf unlinks in the final already verified private quarantine."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import stat
import time

BASE=Path('/mowenjie/fluffy-bassoon-2/outputs')
PRODUCER=BASE/'baige-storage-tiering-20261103-r3'
AUDIT=BASE/'baige-storage-tiering-20261103-r6'
REL=Path('code/wam-ttt')
SOURCE=Path('/mowenjie')/REL
QUARANTINE=SOURCE.with_name(SOURCE.name+'.tiering-'+PRODUCER.name)

if __name__=='__main__':
    AUDIT.mkdir(exist_ok=False)
    manifest=json.loads((PRODUCER/'code__wam-ttt.json').read_text())
    assert manifest['source']==str(SOURCE)
    assert manifest['bos_readback_verified'] and manifest['archive_source_compare']
    assert not SOURCE.exists() and not SOURCE.is_symlink()
    obj=Path(manifest['bos_archive'])
    assert obj.parent==Path('/mnt/bos/adgfyl/strong-plant-archives/20261103-rl-checkpoint-space-r1/parallel-r3')
    assert obj.stat().st_size==manifest['archive_bytes']
    assert obj.with_name(obj.name+'.sha256').read_text().split()[0]==manifest['sha256']
    rows=json.loads((PRODUCER/'code__wam-ttt.inventory.json').read_text())
    root=next(r for r in rows if r[0]==str(REL))
    s=QUARANTINE.lstat()
    assert stat.S_ISDIR(s.st_mode) and not QUARANTINE.is_symlink()
    assert (s.st_dev,s.st_ino)==(root[7],root[8])
    paths=[]
    for row in rows:
        if not stat.S_ISDIR(row[1]):
            suffix=Path(row[0]).relative_to(REL)
            assert '..' not in suffix.parts
            paths.append(QUARANTINE/suffix)
    # r4 owns directory removals and final receipts. Work backwards through
    # the remaining suffix so we do not chase its forward unlink queue.
    selected=list(reversed(paths[133000:]))
    plan={'source':str(SOURCE),'quarantine':str(QUARANTINE),'sha256':manifest['sha256'],
          'workers':64,'selected_known_leaves':len(selected),'skip_forward_prefix':133000,
          'guard':'Only immutable inventoried leaves under matching device/inode private quarantine; complete GNU tar comparison and BOS SHA256 readback already passed. Never remove directories or original namespaces. r4 remains sole finalization owner.',
          'unlink_semantics':'Same as qualified r4 leaf unlink: Path.unlink, FileNotFoundError is idempotent success.',
          'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    (AUDIT/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    def unlink(path):
        try: path.unlink();return 1
        except FileNotFoundError:return 0
    begin=time.monotonic();removed=0;attempted=0
    with ThreadPoolExecutor(max_workers=64) as pool:
        for result in pool.map(unlink,selected):
            removed+=result;attempted+=1
            if attempted%10000==0:
                print(time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                      f'assist {attempted}/{len(selected)} known leaves; {removed} removed',flush=True)
    summary={'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
             'attempted_known_leaves':attempted,'removed_by_assistant':removed,
             'seconds':time.monotonic()-begin,'complete':True,'directory_owner':'r4'}
    (AUDIT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
