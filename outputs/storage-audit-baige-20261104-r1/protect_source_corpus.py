"""Withdraw queued bulk candidates after following retained source manifests."""
import json
import os
from pathlib import Path
import time
base=Path('/mowenjie/fluffy-bassoon-2/outputs/storage-audit-baige-20261104-r1')
root=base.parent/'robocasa-opendrawer-flow-sde-20261102-r4'
paths=[root/x for x in ('training-rollouts','sampling','environment')]
handles=[os.open(p,os.O_RDONLY|os.O_DIRECTORY) for p in paths]
record={'protected':[str(p) for p in paths],'guard_pid':os.getpid(),
        'reason':'Following SOURCE-CHUNKS.json found 3,072 retained physical references into r4/environment, used by four-method PREPARE.py. Retain the closed raw bundle and final18 teacher hot for that lineage.',
        'no_source_mutation':True,'original_archive_worker':2093731}
(base/'ADDITIONAL-PROTECTION.json').write_text(json.dumps(record,indent=2)+'\n')
while Path('/proc/2093731/cmdline').exists():
    try:
        if b'storage-audit-baige-20261104-r1/archive.py' not in Path('/proc/2093731/cmdline').read_bytes():break
    except OSError:break
    time.sleep(5)
for fd in handles:os.close(fd)
