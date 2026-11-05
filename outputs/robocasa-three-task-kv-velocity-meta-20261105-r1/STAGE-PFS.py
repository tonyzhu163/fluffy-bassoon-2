"""Copy only indexed immutable source payloads from BOS to PFS, checking every SHA."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-free-gib', type=int, default=96)
    args = parser.parse_args()
    assert args.minimum_free_gib >= 96
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / '.stage.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    probe = args.output / '.write-probe'
    with probe.open('wb') as f:
        f.write(b'pfs source staging\n'); f.flush(); os.fsync(f.fileno())
    probe.unlink()
    assembly = json.loads(args.assembly.read_text())
    sources, indexes = {}, {}
    for task, source in assembly['source_assemblies'].items():
        path = Path(source['path'])
        assert sha(path.read_bytes()) == source['sha256']
        record = json.loads(path.read_text()); index = path.parent / 'TAR-INDEX.json'
        rows = sorted(json.loads(index.read_text()), key=lambda row: row['offset_data'])
        sources[task] = dict(archive=record['provenance']['bos_archive'],
            index_path=str(index), index_sha256=sha(index.read_bytes()),
            members=len(rows), bytes=sum(row['bytes'] for row in rows))
        indexes[task] = rows
    total = sum(value['bytes'] for value in sources.values())
    progress = {task: dict(members=0, bytes=0, bos_read_bytes=0, retries=0, finished=False)
                for task in sources}
    mutex = threading.Lock(); cancelled = threading.Event(); started = time.monotonic()

    def snapshot(**extra):
        with mutex:
            copied = sum(value['bytes'] for value in progress.values())
            result = dict(checked_shanghai=datetime.datetime.now().astimezone().isoformat(),
                elapsed_seconds=time.monotonic()-started, total_bytes=total, verified_bytes=copied,
                total_members=sum(len(rows) for rows in indexes.values()),
                verified_members=sum(value['members'] for value in progress.values()),
                free_PFS_bytes=os.statvfs(args.output).f_bavail*os.statvfs(args.output).f_frsize,
                tasks=json.loads(json.dumps(progress)), **extra)
        write_json(args.output / 'PROGRESS.json', result)
        return result

    def capacity():
        with mutex:
            remaining = total - sum(value['bytes'] for value in progress.values())
        fs = os.statvfs(args.output)
        if fs.f_bavail*fs.f_frsize - remaining < args.minimum_free_gib*2**30:
            raise OSError('Insufficient PFS capacity for remaining members plus reserved free space')

    def scan_task(task):
        folder = args.output / task; folder.mkdir(exist_ok=True)
        pending = []
        for row in indexes[task]:
            path = folder / (row['sha256'] + '.pt.gz')
            if path.exists():
                raw = path.read_bytes()
                if len(raw) != row['bytes'] or sha(raw) != row['sha256']:
                    raise ValueError(f'Existing cache member changed: {path}')
                with mutex:
                    progress[task]['members'] += 1; progress[task]['bytes'] += len(raw)
            else:
                pending.append(row)
        return task, pending

    def copy_task(task):
        folder = args.output / task
        pending = pending_by_task[task]
        position = 0
        while position < len(pending):
            if cancelled.is_set():
                raise RuntimeError('Another source copy failed; retaining verified members')
            capacity()
            group = [pending[position]]; position += 1
            start = group[0]['offset_data']; end = start + group[0]['bytes']
            while position < len(pending):
                row = pending[position]; next_end = row['offset_data'] + row['bytes']
                if row['offset_data'] - end > 65536 or next_end-start > 64*2**20:
                    break
                group.append(row); position += 1; end = next_end
            payloads = None
            for attempt in range(5):
                fd = None
                try:
                    fd = os.open(sources[task]['archive'], os.O_RDONLY)
                    parts, received = [], 0
                    while received < end-start:
                        chunk = os.pread(fd, end-start-received, start+received)
                        if not chunk:
                            raise OSError('Short source archive read')
                        parts.append(chunk); received += len(chunk)
                    raw = b''.join(parts)
                    payloads = [(row, raw[row['offset_data']-start:row['offset_data']-start+row['bytes']])
                                for row in group]
                    if any(len(data) != row['bytes'] or sha(data) != row['sha256'] for row, data in payloads):
                        raise ValueError(f'Source byte/hash verification failed for {task} offset {start}')
                    with mutex:
                        progress[task]['bos_read_bytes'] += len(raw)
                    break
                except (OSError, ValueError):
                    with mutex:
                        progress[task]['retries'] += 1
                    if attempt == 4:
                        raise
                    time.sleep(min(.25*2**attempt, 2.))
                finally:
                    if fd is not None:
                        try:
                            os.close(fd)
                        except OSError:
                            pass  # read-only descriptor retired, never close the integer twice
            for row, data in payloads:
                path = folder / (row['sha256'] + '.pt.gz')
                tmp = path.with_suffix('.partial')
                with tmp.open('wb') as f:
                    f.write(data); f.flush(); os.fsync(f.fileno())
                if sha(tmp.read_bytes()) != row['sha256']:
                    raise ValueError(f'PFS readback verification failed: {tmp}')
                tmp.replace(path)
                with mutex:
                    progress[task]['members'] += 1; progress[task]['bytes'] += len(data)
            del raw, parts, payloads
        with mutex:
            progress[task]['finished'] = True

    snapshot(completed=False, sources=sources)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            pending_by_task = dict(pool.map(scan_task, sources))
            futures = [pool.submit(copy_task, task) for task in sources]
            while True:
                done, pending = wait(futures, timeout=5)
                failures = [future.exception() for future in done if future.exception() is not None]
                snapshot(completed=False, errors=[str(error) for error in failures])
                if failures:
                    cancelled.set()
                if not pending:
                    for future in futures:
                        future.result()
                    break
        assert all(value['finished'] for value in progress.values())
        capacity()
        manifest = dict(format='flow_source_member_cache_v1', completed=True,
            assembly_sha256=sha(args.assembly.read_bytes()), sources=sources,
            total_bytes=total, minimum_free_GiB=args.minimum_free_gib,
            verification='Every original compressed member SHA256 and complete PFS readback match',
            completed_shanghai=datetime.datetime.now().astimezone().isoformat())
        write_json(args.output / 'CACHE.json', manifest)
        print(json.dumps(snapshot(completed=True)), flush=True)
    except BaseException as error:
        snapshot(completed=False, failed=True, error=repr(error))
        raise


if __name__ == '__main__':
    main()
