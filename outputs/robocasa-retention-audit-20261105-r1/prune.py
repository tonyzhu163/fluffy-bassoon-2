"""Apply reviewed bulk-file decisions; preserve all other experiment records."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import errno
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import threading
import time

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-retention-audit-20261105-r1')
ROOT = Path('/mowenjie')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def users(source):
    found = []
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            if str(source).encode() in (process / 'cmdline').read_bytes():
                found.append([process.name, 'command reference'])
        except OSError:
            pass
        paths = [process / 'cwd', process / 'exe']
        try:
            paths.extend((process / 'fd').iterdir())
        except OSError:
            pass
        for p in paths:
            try:
                target = os.readlink(p)
                if target == str(source) or target.startswith(str(source) + '/'):
                    found.append([process.name, target])
            except OSError:
                pass
    return found


def open_parent(rootfd, relative, directories):
    """Resolve each component beneath the pinned root, rejecting symlinks."""
    fd = os.dup(rootfd)
    prefix = Path('.')
    try:
        for part in Path(relative).parts:
            assert part not in ('', '..', '/')
            nextfd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nextfd
            prefix /= part
            expected = directories[str(prefix)]
            current = os.fstat(fd)
            assert (current.st_dev, current.st_ino) == (expected[5], expected[6])
        return fd
    except BaseException:
        os.close(fd)
        raise


def identity_matches(s, row):
    return (s.st_mode, s.st_size, s.st_mtime_ns, s.st_dev, s.st_ino) == (row[1], row[2], row[3], row[5], row[6])


def delete_group(source, rootfd, parent, rows, directories, stop, known_code, classify):
    result = {'source': source, 'parent': parent, 'files': 0, 'logical_bytes': 0,
              'allocated_payload_bytes': 0, 'code_files': 0, 'preserved': []}
    fd = open_parent(rootfd, parent, directories)
    try:
        for row in rows:
            if stop.is_set():
                result['preserved'].append([row[0], 'source changed or another group failed'])
                continue
            name = Path(row[0]).name
            try:
                current = os.stat(name, dir_fd=fd, follow_symlinks=False)
                assert stat.S_ISREG(current.st_mode) and identity_matches(current, row), 'file identity or metadata changed'
                if row[7] == 'bulk':
                    assert classify(row[0], row[2]) == row[8] and row[8], 'unreviewed payload class'
                else:
                    assert row[7] == 'one_off_code'
                    filefd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
                    with os.fdopen(filefd, 'rb') as stream:
                        data = stream.read()
                    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                    assert blob in known_code, 'code is not recoverable from pinned Git'
                    assert identity_matches(os.stat(name, dir_fd=fd, follow_symlinks=False), row)
                    result['code_files'] += 1
                os.unlink(name, dir_fd=fd)
                result['files'] += 1
                result['logical_bytes'] += row[2]
                result['allocated_payload_bytes'] += row[4] * 512
            except Exception as error:
                stop.set()
                result['preserved'].append([row[0], repr(error)])
    finally:
        os.close(fd)
    return result


def main():
    review = read(BASE / 'REVIEW.json')
    assert review['bulk_prune_review_complete']
    assert sha(BASE / 'AUDIT.json') == review['audit_sha256']
    assert sha(BASE / 'PLAN.json') == review['plan_sha256']
    assert sha(BASE / 'audit.py') == review['classifier_sha256']
    plan, audit = read(BASE / 'PLAN.json'), read(BASE / 'AUDIT.json')
    allowed = {r['source'] for r in plan['sources']}
    known = set()
    spec = importlib.util.spec_from_file_location('reviewed_audit', BASE / 'audit.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = []
    roots = {}
    directories = {}
    stopping = {}
    retained = {}
    skipped = []
    before = os.statvfs(ROOT)
    start_free = before.f_bavail * before.f_frsize
    for entry in audit['sources']:
        if entry['status'] != 'audited_hot_source':
            continue
        source = entry['source']
        assert source in allowed
        p = Path(source)
        assert p.parent in (ROOT / 'fluffy-bassoon/outputs', ROOT / 'fluffy-bassoon-2/outputs')
        if p.is_symlink():
            skipped.append({'source': source, 'reason': 'source became a symlink'})
            continue
        if entry['errors'] or entry['protected']:
            skipped.append({'source': source, 'reason': 'audit errors or protected content',
                            'errors': entry['errors'], 'protected': entry['protected']})
            continue
        busy = users(p)
        if busy:
            skipped.append({'source': source, 'reason': 'process dependency', 'users': busy})
            continue
        inventory_path = Path(entry['inventory'])
        assert inventory_path.parent == BASE / 'private-inventories' and not inventory_path.is_symlink()
        assert sha(inventory_path) == entry['inventory_sha256']
        inventory = read(inventory_path)
        assert inventory['source'] == source and inventory['root_identity'] == entry['root_identity']
        try:
            s = p.lstat()
        except OSError as error:
            skipped.append({'source': source, 'reason': 'source unavailable', 'error': repr(error)})
            continue
        if [s.st_dev, s.st_ino, s.st_mtime_ns] != inventory['root_identity']:
            skipped.append({'source': source, 'reason': 'root identity or modification time changed since audit'})
            continue
        fd = os.open(p, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        assert (os.fstat(fd).st_dev, os.fstat(fd).st_ino) == (s.st_dev, s.st_ino)
        roots[source] = fd
        directories[source] = {r[0]: r for r in inventory['rows'] if stat.S_ISDIR(r[1])}
        stopping[source] = threading.Event()
        retained[source] = [r for r in inventory['rows'] if r[7] == 'keep' and not stat.S_ISDIR(r[1])]
        parents = defaultdict(list)
        for row in inventory['rows']:
            if row[7] in ('bulk', 'one_off_code'):
                assert stat.S_ISREG(row[1])
                assert not Path(row[0]).is_absolute() and '..' not in Path(row[0]).parts
                parents[str(Path(row[0]).parent)].append(row)
        groups.extend((source, parent, rows) for parent, rows in parents.items())
    # Independent directories run concurrently; files in one directory are
    # deleted sequentially to avoid contention on GPFS directory write locks.
    groups.sort(key=lambda g: sum(r[2] for r in g[2]), reverse=True)
    output = BASE / 'prune-receipts'
    output.mkdir(exist_ok=False)
    results = []
    summary = {'finished': False, 'directory_groups_completed': 0, 'directory_groups': len(groups),
               'files_deleted': 0, 'deleted_logical_bytes': 0, 'deleted_allocated_payload_bytes': 0,
               'recoverable_one_off_code_files_deleted': 0, 'preserved': [], 'skipped_sources': skipped,
               'checked_shanghai': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    save(BASE / 'PRUNE-SUMMARY.json', summary)
    receipt_log = (output / 'receipts.jsonl').open('x', buffering=1 << 20)
    last_write = time.monotonic()
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(delete_group, source, roots[source], parent, rows,
                               directories[source], stopping[source], known, module.bulk_reason): (source, parent)
                   for source, parent, rows in groups}
        for future in as_completed(futures):
            source, parent = futures[future]
            try:
                result = future.result()
            except Exception as error:
                stopping[source].set()
                result = {'source': source, 'parent': parent, 'files': 0, 'logical_bytes': 0,
                          'allocated_payload_bytes': 0, 'code_files': 0, 'preserved': [[parent, repr(error)]]}
            receipt_log.write(json.dumps(result) + '\n')
            summary['directory_groups_completed'] += 1
            summary['files_deleted'] += result['files']
            summary['deleted_logical_bytes'] += result['logical_bytes']
            summary['deleted_allocated_payload_bytes'] += result['allocated_payload_bytes']
            summary['recoverable_one_off_code_files_deleted'] += result['code_files']
            if result['preserved']:
                summary['preserved'].append(result)
            if time.monotonic() - last_write > 2:
                receipt_log.flush()
                summary['checked_shanghai'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
                save(BASE / 'PRUNE-SUMMARY.json', summary)
                last_write = time.monotonic()
    receipt_log.close()
    summary['receipt_file'] = str(output / 'receipts.jsonl')
    summary['receipt_sha256'] = sha(output / 'receipts.jsonl')
    # Verify the metadata identities of retained records; no content is moved
    # or rewritten and no directory outside the selected source roots is read.
    def check_retained(item):
        source, row = item
        p = Path(source) / row[0]
        assert identity_matches(p.lstat(), row), ('retained record changed', str(p))
        return 1
    with ThreadPoolExecutor(max_workers=24) as pool:
        retained_checked = sum(pool.map(check_retained, ((source, row) for source, records in retained.items() for row in records)))
    for fd in roots.values():
        os.close(fd)
    now = os.statvfs(ROOT)
    summary.update(finished=True, retained_records_checked=retained_checked,
                   filesystem_free_before=start_free, filesystem_free_after=now.f_bavail * now.f_frsize,
                   no_active_job_folder_scan=True, GPU_jobs_launched=0)
    save(BASE / 'PRUNE-SUMMARY.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
