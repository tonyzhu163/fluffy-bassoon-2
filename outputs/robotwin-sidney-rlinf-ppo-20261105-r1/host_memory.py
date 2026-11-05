"""Observe the allocated container's host memory without changing Ray guards."""
import argparse
import json
import os
from pathlib import Path
import time


parser=argparse.ArgumentParser()
parser.add_argument('output',type=Path)
parser.add_argument('--once',action='store_true')
parser.add_argument('--minimum-gib',type=int,default=512)
args=parser.parse_args();root=args.output;once=args.once
assert args.minimum_gib>0
base=Path('/sys/fs/cgroup')
if (base/'memory.max').exists():
    limit_path=base/'memory.max';current_path=base/'memory.current';stat_path=base/'memory.stat'
else:
    base=base/'memory';limit_path=base/'memory.limit_in_bytes';current_path=base/'memory.usage_in_bytes';stat_path=base/'memory.stat'
limit_text=limit_path.read_text().strip()
limit=int(limit_text) if limit_text!='max' else os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
assert limit>=args.minimum_gib*2**30,f'Host-memory allocation too small: {limit/2**30:.1f}GiB'
peak=0
while True:
    used=int(current_path.read_text());peak=max(peak,used)
    stats=dict(line.split() for line in stat_path.read_text().splitlines())
    value=dict(at_unix=time.time(),limit_bytes=limit,current_bytes=used,peak_observed_bytes=peak,
        minimum_allocation_bytes=args.minimum_gib*2**30,
        headroom_bytes=limit-used,anonymous_bytes=int(stats.get('anon',stats.get('total_rss',0))),
        file_cache_bytes=int(stats.get('file',stats.get('total_cache',0))),ray_memory_guard_unchanged=True)
    target=root/'host-memory.json';temporary=target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(target)
    if once:print(json.dumps(value));break
    time.sleep(10)
