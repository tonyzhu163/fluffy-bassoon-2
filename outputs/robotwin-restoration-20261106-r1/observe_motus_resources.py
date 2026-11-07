"""Observe only this exclusive job's attested GPUs and cgroup; no model calls."""
import csv
import copy
import io
import json
import os
import queue
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from fluffy_bassoon.baige_gpu import mapping
from fluffy_bassoon.meta_rl.rlinf_native_ppo import write


root = Path(sys.argv[1])
assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
expected = {row['uuid'] for row in mapping()}
expected_count = int(sys.argv[2]) if len(sys.argv) > 2 else 4
assert expected_count in (1, 4, 8) and len(expected) == expected_count
stopping = False


def stop(*unused):
    global stopping
    stopping = True


signal.signal(signal.SIGTERM,stop)
signal.signal(signal.SIGINT,stop)
base = Path('/sys/fs/cgroup')
if (base/'memory.max').exists():
    limit_path,current_path = base/'memory.max',base/'memory.current'
else:
    base = base/'memory'
    limit_path,current_path = base/'memory.limit_in_bytes',base/'memory.usage_in_bytes'
limit = limit_path.read_text().strip()
limit = int(limit) if limit != 'max' else os.sysconf('SC_PAGE_SIZE')*os.sysconf('SC_PHYS_PAGES')
report = dict(samples=0,interval_seconds=1,host_limit_bytes=limit,host_peak_bytes=0,gpus={},
              resolution_limit='NVML1s samples can miss transients; actor Torch allocator peaks are recorded separately')
live = Path(os.environ.get('TASK_RESOURCE_LIVE',str(root/'resources.json')))
archive = root/'resources.json'
pending = queue.Queue(maxsize=1)
archive_status = dict(last_error=None)


def publish_archive():
    while True:
        value = pending.get()
        if value is None:return
        try:
            write(archive,value)
            archive_status['last_error']=None
        except Exception as error:
            archive_status['last_error']=f'{type(error).__name__}: {error}'


publisher = threading.Thread(target=publish_archive,name='resource-archive',daemon=True)
if live != archive:publisher.start()


def offer(value):
    try:pending.put_nowait(value)
    except queue.Full:pass  # At most one bounded archive write is queued.


last_saved = 0
while True:
    output = subprocess.check_output(['nvidia-smi',f'--id={",".join(sorted(expected))}',
        '--query-gpu=uuid,memory.total,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=10)
    rows = list(csv.reader(io.StringIO(output)))
    assert {row[0].strip() for row in rows} == expected
    for uuid,total,free,used,utilization in rows:
        uuid = uuid.strip()
        total,free,used = [int(float(value)*2**20) for value in (total,free,used)]
        gpu = report['gpus'].setdefault(uuid,dict(total_bytes=total,minimum_free_bytes=free,
            maximum_used_bytes=used,maximum_utilization_percent=0))
        gpu['minimum_free_bytes'] = min(gpu['minimum_free_bytes'],free)
        gpu['maximum_used_bytes'] = max(gpu['maximum_used_bytes'],used)
        gpu['maximum_utilization_percent'] = max(gpu['maximum_utilization_percent'],int(utilization))
    report.update(samples=report['samples']+1,at_unix=time.time(),
                  host_peak_bytes=max(report['host_peak_bytes'],int(current_path.read_text())))
    report['sampled_reserve_passed'] = all(row['minimum_free_bytes'] >= max(8*2**30,.08*row['total_bytes'])
                                         for row in report['gpus'].values())
    report['archive_last_error'] = archive_status['last_error']
    write(live,report)
    if live != archive and (stopping or time.monotonic()-last_saved >= 10):
        offer(copy.deepcopy(report))
        last_saved = time.monotonic()
    if stopping:
        if live != archive:
            until=time.monotonic()+35
            try:pending.put(None,timeout=35)
            except queue.Full:pass
            publisher.join(max(0,until-time.monotonic()))
            report['archive_writer_still_active_at_shutdown']=publisher.is_alive()
            write(live,report)
        break
    time.sleep(1)
