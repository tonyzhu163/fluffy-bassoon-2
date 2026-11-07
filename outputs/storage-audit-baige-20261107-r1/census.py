"""Metadata-only storage census. Remote output goes to /tmp, never PFS."""
import collections
import concurrent.futures
import heapq
import json
import os
from pathlib import Path
import queue
import shutil
import stat
import threading
import time

ROOT = '/mowenjie'
OUT = '/tmp/storage-audit-baige-20261107-r1'
os.makedirs(OUT, exist_ok=True)
started = time.time()
device = os.lstat(ROOT).st_dev
before = shutil.disk_usage(ROOT)._asdict()
work = queue.Queue()
lock = threading.Lock()
shared = {}
results = []
progress = collections.Counter()
errors = []
workers = 24


def prefixes(rel):
    parts = rel.split('/') if rel else []
    if parts and parts[0] in ('.secrets', '.codex', '.claude', '.ssh'):
        return [parts[0]]
    depth = 4
    if len(parts) > 2 and parts[1] in ('outputs', 'inputs', 'envs', 'setups', 'cache'):
        depth = 6
    return ['.'] + ['/'.join(parts[:i]) for i in range(1, min(depth, len(parts)) + 1)]


def rank(rel):
    return (0 if rel.startswith('fluffy-bassoon-2/inputs/') else
            1 if rel.startswith('fluffy-bassoon-2/setups/') else
            3 if '/cache/' in rel or rel.startswith('.cache/') else 2, rel)


def bucket(rel):
    parts = rel.split('/')
    if len(parts) >= 3 and parts[1] in ('outputs', 'inputs', 'envs', 'setups'):
        if parts[2] == 'archive' and len(parts) >= 4:
            return '/'.join(parts[:4])
        return '/'.join(parts[:3])
    return '/'.join(parts[:2])


def file_kind(rel):
    name = rel.rsplit('/', 1)[-1].lower()
    suffix = Path(name).suffix
    if suffix in ('.pt', '.pth', '.safetensors', '.ckpt', '.bin', '.distcp', '.msgpack'):
        return 'weights-checkpoints'
    if suffix in ('.npz', '.npy', '.h5', '.hdf5', '.parquet', '.arrow', '.tfrecord'):
        return 'arrays-trajectories'
    if suffix in ('.mp4', '.mkv', '.avi', '.webm', '.mov'):
        return 'videos'
    if suffix in ('.gz', '.zst', '.zip', '.tar', '.whl', '.bz2', '.xz'):
        return 'archives-packages'
    if suffix in ('.log', '.jsonl', '.json', '.csv', '.txt', '.yaml', '.yml'):
        return 'logs-records-configs'
    return 'other'


def save(name, value):
    path = Path(OUT, name)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, separators=(',', ':')) + '\n')
    tmp.replace(path)


def worker():
    allocated = collections.Counter()
    logical = collections.Counter()
    entries = collections.Counter()
    kinds = collections.Counter()
    ages = collections.Counter()
    newest = collections.defaultdict(float)
    large = []
    symlinks = []
    big = []
    local_count = 0

    def count(rel, s, unique=True):
        blocks = s.st_blocks * 512
        regular = stat.S_ISREG(s.st_mode)
        for key in prefixes(rel):
            entries[key] += 1
            logical[key] += s.st_size if regular else 0
            if unique:
                allocated[key] += blocks
        if unique and regular:
            b = bucket(rel)
            kinds[b + '|' + file_kind(rel)] += blocks
            age = max(0, started - s.st_mtime) / 86400
            ages[b + '|' + ('<1d' if age < 1 else '1-3d' if age < 3 else
                            '3-7d' if age < 7 else '7-14d' if age < 14 else '>=14d')] += blocks
        if regular:
            newest[bucket(rel)] = max(newest[bucket(rel)], s.st_mtime)

    while True:
        item = work.get()
        if item is None:
            work.task_done()
            break
        try:
            with os.scandir(item) as scan:
                for ent in scan:
                    path = ent.path
                    rel = os.path.relpath(path, ROOT)
                    try:
                        s = ent.stat(follow_symlinks=False)
                        if s.st_dev != device:
                            continue
                        local_count += 1
                        regular = stat.S_ISREG(s.st_mode)
                        linked = regular and s.st_nlink > 1
                        count(rel, s, unique=not linked)
                        if linked:
                            with lock:
                                key = (s.st_dev, s.st_ino)
                                old = shared.get(key)
                                if old is None or rank(rel) < rank(old[0]):
                                    shared[key] = (rel, s)
                        if stat.S_ISDIR(s.st_mode):
                            if ent.name != '.snapshots':
                                work.put(path)
                        elif regular and not rel.startswith(('.secrets/', '.codex/', '.claude/', '.ssh/')):
                            if s.st_size >= 256 << 20:
                                record = (s.st_blocks * 512, path, s.st_size, s.st_nlink, s.st_mtime, s.st_ino)
                                heapq.heappush(large, record)
                                if len(large) > 200:
                                    heapq.heappop(large)
                                big.append(record)
                        elif stat.S_ISLNK(s.st_mode) and len(symlinks) < 300:
                            if '/outputs/' in rel or '/inputs/' in rel or rel.count('/') < 2:
                                symlinks.append((rel, os.readlink(path)))
                    except OSError as exc:
                        with lock:
                            progress['errors'] += 1
                            if len(errors) < 100:
                                errors.append((rel, type(exc).__name__))
        except OSError as exc:
            with lock:
                progress['errors'] += 1
                if len(errors) < 100:
                    errors.append((item, type(exc).__name__))
        finally:
            with lock:
                progress['entries'] += local_count
                progress['directories'] += 1
            local_count = 0
            work.task_done()
    results.append((allocated, logical, entries, kinds, ages, newest, large, symlinks, big))


work.put(ROOT)
threads = [threading.Thread(target=worker, daemon=True) for _ in range(workers)]
for thread in threads:
    thread.start()
while work.unfinished_tasks:
    time.sleep(10)
    with lock:
        p = dict(progress, elapsed_seconds=round(time.time() - started),
                 queued=work.qsize(), shared_inodes=len(shared),
                 free_bytes=shutil.disk_usage(ROOT).free)
    save('progress.json', p)
    print(json.dumps(p), flush=True)
for _ in threads:
    work.put(None)
for thread in threads:
    thread.join()
a, logical, entries, kinds, ages = [collections.Counter() for _ in range(5)]
newest = collections.defaultdict(float)
large, symlinks, big = [], [], []
for result in results:
    for dst, src in zip((a, logical, entries, kinds, ages), result[:5]):
        dst.update(src)
    for key, value in result[5].items():
        newest[key] = max(newest[key], value)
    large.extend(result[6])
    symlinks.extend(result[7])
    big.extend(result[8])
for rel, s in shared.values():
    for key in prefixes(rel):
        a[key] += s.st_blocks * 512
    b = bucket(rel)
    kinds[b + '|' + file_kind(rel)] += s.st_blocks * 512
    age = max(0, started - s.st_mtime) / 86400
    ages[b + '|' + ('<1d' if age < 1 else '1-3d' if age < 3 else '3-7d' if age < 7 else '7-14d' if age < 14 else '>=14d')] += s.st_blocks * 512
rootstat = os.lstat(ROOT)
a['.'] += rootstat.st_blocks * 512
data = dict(complete=True, started_unix=started, finished_unix=time.time(),
            elapsed_seconds=time.time()-started, filesystem_before=before,
            filesystem_after=shutil.disk_usage(ROOT)._asdict(),
            entries_scanned=progress['entries'], errors_count=progress['errors'],
            errors=errors, shared_inodes_counted_once=len(shared),
            allocated_bytes_by_prefix=dict(a), logical_bytes_by_prefix=dict(logical),
            entries_by_prefix=dict(entries), allocated_bytes_by_kind=dict(kinds),
            allocated_bytes_by_file_age=dict(ages), newest_mtime_by_bucket=dict(newest),
            largest_files=sorted(large, reverse=True)[:300], symlinks=symlinks,
            files_at_least_256MiB=sorted(big, reverse=True),
            note='Live metadata scan; symlinks not followed; cross-device and .snapshots skipped. '
                 'Shared inodes counted once, preferring current inputs then setup then stable paths. '
                 'Secret/chat roots disclose aggregate totals only; no file contents read.')
save('census.json', data)
print('COMPLETE', json.dumps({k:v for k,v in a.items() if k.count('/') < 1}), flush=True)
