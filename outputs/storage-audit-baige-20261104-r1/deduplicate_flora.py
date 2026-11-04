"""Coalesce identical immutable Flora shards; preserve both model paths."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import time

AUDIT = Path('/mowenjie/fluffy-bassoon-2/outputs/storage-audit-baige-20261104-r1')
OLD = Path('/mowenjie/fluffy-bassoon/inputs/flora-rewarder-031299e4-20261028-r1')
CURRENT = Path('/mowenjie/fluffy-bassoon-2/inputs/flora-rewarder-031299e4-20261101-r1')
spec = importlib.util.spec_from_file_location('archive', AUDIT / 'archive.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert not m.users(OLD), m.users(OLD)
def stable(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_nlink)

rows = []
for old in sorted(OLD.glob('model-*.safetensors')):
    current = CURRENT / old.name
    a, b = old.stat(), current.stat()
    assert a.st_size == b.st_size and a.st_dev == b.st_dev
    assert stat.S_ISREG(a.st_mode) and stat.S_ISREG(b.st_mode)
    ha, hb = m.digest(old), m.digest(current)
    assert ha == hb, old
    assert stable(old.stat()) == stable(a) and stable(current.stat()) == stable(b), 'Changed while hashing'
    # These are frozen downloaded shards, outside all active run worktrees.
    # The canonical current inode is never modified; atomically redirect only
    # the verified identical legacy directory entry to it.
    if a.st_ino == b.st_ino:
        reclaimed = 0
    else:
        assert a.st_nlink == 1, 'Unexpected existing external hardlinks'
        assert not m.users(OLD), m.users(OLD)
        temp = old.with_name(old.name + '.deduplicate-storage-audit-20261104-r1')
        assert not temp.exists()
        os.link(current, temp)
        assert temp.stat().st_ino == b.st_ino
        assert stable(old.stat()) == stable(a) and current.stat().st_mtime_ns == b.st_mtime_ns
        os.replace(temp, old)
        reclaimed = a.st_blocks * 512
    assert old.stat().st_ino == current.stat().st_ino
    rows.append({'legacy': str(old), 'canonical': str(current), 'sha256': ha,
                 'bytes': a.st_size, 'reclaimed_allocated_bytes': reclaimed,
                 'shared_device': b.st_dev, 'shared_inode': b.st_ino,
                 'both_paths_preserved': True})
    m.save(AUDIT / 'FLORA-DEDUPLICATION.json', {'complete': False, 'shards': rows})
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), old.name, 'HASH MATCH / BOTH PATHS PRESERVED', flush=True)
assert len(rows) == 11
result = {'complete': True, 'shards': rows,
          'reclaimed_allocated_bytes': sum(r['reclaimed_allocated_bytes'] for r in rows),
          'method': 'Two separately hashed frozen model copies now share PFS inodes; no content change or path removal. Keep weights immutable.',
          'independent_copy_command': 'cp --reflink=auto --preserve=all <legacy-shard> <temporary-shard>; mv <temporary-shard> <legacy-shard>',
          'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
m.save(AUDIT / 'FLORA-DEDUPLICATION.json', result)
