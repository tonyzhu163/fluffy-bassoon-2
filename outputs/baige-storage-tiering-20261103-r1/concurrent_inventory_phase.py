"""Fresh r3 continuation with qualified concurrent metadata reads."""
import importlib.util
import json
import os
from pathlib import Path
import queue
import stat
import threading
import time

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/baige-storage-tiering-20261103-r1')
spec = importlib.util.spec_from_file_location('parallel_phase', BASE / 'parallel_phase.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)
m = p.m
m.RUN = 'baige-storage-tiering-20261103-r3'
m.AUDIT = BASE.parent / m.RUN
m.BOS = m.BOS.parent / 'parallel-r3'
m.STAGE = Path('/tmp') / m.RUN
reference_inventory = m.inventory


def inventory(source):
    pending = queue.Queue()
    pending.put(source)
    rows, errors = [], []
    def worker():
        while True:
            path = pending.get()
            try:
                if path is None:
                    return
                s = path.lstat()
                rows.append([str(path.relative_to(m.ROOT)), s.st_mode, s.st_size,
                             s.st_mtime_ns, s.st_uid, s.st_gid, s.st_blocks,
                             s.st_dev, s.st_ino, None,
                             os.readlink(path) if stat.S_ISLNK(s.st_mode) else None])
                if stat.S_ISDIR(s.st_mode):
                    for child in path.iterdir():
                        pending.put(child)
            except Exception as error:
                errors.append((str(path), str(error)))
            finally:
                pending.task_done()
    workers = [threading.Thread(target=worker) for _ in range(8)]
    for thread in workers:
        thread.start()
    pending.join()
    for _ in workers:
        pending.put(None)
    for thread in workers:
        thread.join()
    if errors:
        raise RuntimeError(str(errors[:5]))
    rows.sort(key=lambda row: row[0])
    return rows


if __name__ == '__main__':
    m.AUDIT.mkdir(exist_ok=False)
    qualification_source = Path('/mowenjie/archive_manifests/20261020-inactive-reclaim')
    begin = time.monotonic()
    expected = reference_inventory(qualification_source)
    reference_seconds = time.monotonic() - begin
    begin = time.monotonic()
    actual = inventory(qualification_source)
    concurrent_seconds = time.monotonic() - begin
    assert actual == expected
    m.save(m.AUDIT / 'INVENTORY-QUALIFICATION.json', {
        'source': str(qualification_source), 'entries': len(actual),
        'all_fields_exact_match': True, 'metadata_workers_per_archive': 8,
        'reference_seconds': reference_seconds, 'concurrent_seconds': concurrent_seconds,
        'payload_or_model_execution': False})
    m.save(m.AUDIT / 'MIGRATION.json', {
        'replaces': 'baige-storage-tiering-20261103-r2',
        'reason': 'Serial metadata latency; r2 stopped before packing, upload or reclaim.',
        'first_phase': 'First six r1 sources already verified and cold-moved; r1 stopped at the recorded boundary.',
        'ownership': 'Own only r1 original plan sources after its first six, disjoint from completed first phase.',
        'verification': 'Same GNU tar content comparison, object SHA256 readback and unchanged-source checks. Eight metadata reads per archive; three archives maximum. Ignore only sibling-induced hardlink count changes.'})
    m.inventory = inventory
    m.log('Concurrent metadata inventory qualified; starting remaining disjoint sources')
    p.main()
