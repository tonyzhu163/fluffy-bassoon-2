"""Run-local complete cold archive verification and selective bulk reclamation."""
import argparse
import concurrent.futures
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import sys

HERE = Path(__file__).resolve().parent
LOCK = threading.Lock()
STATE = {}
BULK_SUFFIXES = ('.pt', '.pth', '.pt.gz', '.pth.gz', '.npz', '.npy', '.distcp',
                 '.safetensors', '.bin', '.pkl', '.pickle', '.mp4', '.mkv',
                 '.webm', '.avi', '.h5', '.hdf5', '.tar', '.tar.gz', '.tar.zst')
COMPACT_SUFFIXES = ('.json', '.jsonl', '.json.gz', '.jsonl.gz', '.yaml', '.yml',
                    '.toml', '.ini', '.cfg', '.csv', '.tsv', '.txt', '.log',
                    '.log.gz', '.md', '.py', '.sh', '.xml', '.png', '.jpg', '.jpeg')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing-' + str(os.getpid()) + '-' + str(threading.get_ident()))
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def stamp(path):
    s = path.lstat()
    return [s.st_mode, s.st_size, s.st_mtime_ns, s.st_uid, s.st_gid,
            s.st_dev, s.st_ino, s.st_nlink, s.st_blocks * 512]


def inventory(root, anchor, hashing=True):
    rows, stack = [], [root]
    while stack:
        path = stack.pop()
        rel = path.relative_to(anchor).as_posix()
        assert not Path(rel).is_absolute() and '..' not in Path(rel).parts
        assert not any(p in ('.secrets', '.ssh', '.aws', '.bce', '.env', '.netrc') for p in Path(rel).parts)
        before = stamp(path); mode = before[0]
        assert stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode), rel
        row = {'path': rel, 'stamp': before}
        if stat.S_ISREG(mode) and hashing:
            row['sha256'] = sha(path)
            assert stamp(path) == before, ('changed while hashing', rel)
        elif stat.S_ISLNK(mode):
            row['target'] = os.readlink(path)
        if stat.S_ISDIR(mode):
            stack.extend(path.iterdir())
        rows.append(row)
    return sorted(rows, key=lambda x: x['path'])


def metadata(rows):
    return [{k: v for k, v in row.items() if k != 'sha256'} for row in rows]


def bulk(row, preserve):
    if not stat.S_ISREG(row['stamp'][0]):
        return False
    parts = Path(row['path']).parts
    if any(part in preserve for part in parts):
        return False
    name = parts[-1].lower()
    if name.endswith(COMPACT_SUFFIXES):
        return False
    return name.endswith(BULK_SUFFIXES)


def readers(source):
    hits = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().split(b'\0')
            if any(str(source).encode() in arg for arg in cmd):
                hits.append({'pid': proc.name, 'kind': 'command_reference'})
            refs = [proc / 'cwd', proc / 'exe'] + list((proc / 'fd').iterdir())
        except OSError:
            continue
        for ref in refs:
            try:
                target = os.readlink(ref)
                if target == str(source) or target.startswith(str(source) + '/'):
                    hits.append({'pid': proc.name, 'kind': 'filesystem_reference', 'path': target})
            except OSError:
                pass
        # mmap readers may have closed their original descriptor.
        try:
            if str(source) + '/' in (proc / 'maps').read_text(errors='replace'):
                hits.append({'pid': proc.name, 'kind': 'mmap_reference'})
        except OSError:
            pass
    return hits


def verify_restore(before, restored_root, anchor):
    actual = inventory(restored_root, anchor)
    assert [r['path'] for r in actual] == [r['path'] for r in before], 'Restored member coverage differs'
    original_links, restored_links = {}, {}
    for left, right in zip(before, actual):
        a, b = left['stamp'], right['stamp']
        assert a[0] == b[0] and a[3:5] == b[3:5], ('mode/owner', left['path'])
        assert a[2] == b[2], ('mtime', left['path'])
        if stat.S_ISREG(a[0]):
            assert a[1] == b[1] and left['sha256'] == right['sha256'], ('bytes', left['path'])
            original_links.setdefault(tuple(a[5:7]), []).append(left['path'])
            restored_links.setdefault(tuple(b[5:7]), []).append(right['path'])
        elif stat.S_ISLNK(a[0]):
            assert left['target'] == right['target'], ('symlink', left['path'])
    assert sorted(sorted(v) for v in original_links.values()) == sorted(sorted(v) for v in restored_links.values()), 'Hardlink topology differs'
    return len(actual)


def pack_compare(source, anchor, archive, log):
    rel = source.relative_to(anchor).as_posix()
    with log.open('ab') as stream:
        subprocess.run(['tar', '--format=posix', '--sort=name', '--acls', '--xattrs',
                        '--numeric-owner', '--sparse', '-I', 'zstd -1 -T2', '-cf',
                        str(archive), '-C', str(anchor), rel], stdout=stream, stderr=stream, check=True)
        subprocess.run(['tar', '--acls', '--xattrs', '-I', 'zstd', '-df', str(archive),
                        '-C', str(anchor)], stdout=stream, stderr=stream, check=True)


def restore(archive, target, log):
    target.mkdir(exist_ok=False)
    with log.open('ab') as stream:
        subprocess.run(['tar', '--acls', '--xattrs', '--same-owner', '-I', 'zstd',
                        '-xf', str(archive), '-C', str(target)],
                       stdout=stream, stderr=stream, check=True)


def copy_archive(source, dest):
    digest = hashlib.sha256(); count = 0
    with source.open('rb') as inp, dest.open('xb') as out:
        for block in iter(lambda: inp.read(8 << 20), b''):
            out.write(block); digest.update(block); count += len(block)
        out.flush()
        try:
            os.fsync(out.fileno())
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOTSUP):
                raise
    assert dest.stat().st_size == count
    return digest.hexdigest(), count


def release_payloads(root, rows, preserve, before_each=None):
    selected = [r for r in rows if bulk(r, preserve)]
    retained = [r for r in rows if stat.S_ISREG(r['stamp'][0]) and not bulk(r, preserve)]
    released, logical = [], 0
    # An inode may have several selected hardlinks; account allocation once and
    # only when the final link is actually unlinked.
    for row in selected:
        path = root / row['path']
        current = stamp(path); expected = row['stamp']
        assert current[:7] == expected[:7], ('Changed before unlink', row['path'])
        # Only earlier links from this same verified selection may change nlink.
        if before_each is not None:
            before_each(path)
        assert sha(path) == row['sha256'], ('Source bytes changed before unlink', row['path'])
        assert stamp(path)[:7] == expected[:7], ('Changed during final hash', row['path'])
        physical = current[8] if current[7] == 1 else 0
        path.unlink()
        released.append({'path': row['path'], 'logical_bytes': expected[1],
                         'released_allocated_bytes': physical, 'sha256': row['sha256']})
        logical += expected[1]
    for row in retained:
        path = root / row['path']
        # Retained files may share an inode with an archived link. Contents must
        # stay exact even if link counts legitimately decrease.
        assert stamp(path)[:7] == row['stamp'][:7], ('Retained file changed', row['path'])
    assert all(not (root / row['path']).exists() for row in selected)
    return released, retained


def qualify():
    assert os.uname().sysname == 'Linux'
    with tempfile.TemporaryDirectory(prefix='outputs-retention-qualification-', dir='/tmp') as tmp:
        base = Path(tmp); source = base / 'source'; folder = source / 'outputs/fixture'
        folder.mkdir(parents=True)
        (folder / 'model.pt').write_bytes(bytes(range(256)) * 8192)
        os.link(folder / 'model.pt', folder / 'model-linked.pt')
        (folder / 'sample.pt.gz').write_bytes(b'opaque-compressed-tensor\0' * 200)
        (folder / 'keep.json').write_text('{"result":"retained"}\n')
        (folder / 'gradient-audit').mkdir()
        (folder / 'gradient-audit/reference.pt').write_bytes(b'numerical reference')
        (folder / 'model-link').symlink_to('model.pt')
        (folder / 'space and unicode-测量.txt').write_text('preserved\n')
        (folder / 'model.pt').chmod(0o640)
        rows = inventory(folder, source)
        obj = base / 'archive.tar.zst'; log = base / 'commands.log'
        pack_compare(folder, source, obj, log)
        cold = base / 'cold.tar.zst'; digest, size = copy_archive(obj, cold)
        assert sha(cold) == digest
        stage = base / 'restored'; restore(cold, stage, log)
        count = verify_restore(rows, stage / 'outputs/fixture', stage)
        assert metadata(inventory(folder, source, False)) == metadata(rows)
        # Refuse a same-size mutation even when its mtime is restored.
        altered = stage / 'outputs/fixture/sample.pt.gz'; old = altered.stat()
        altered.write_bytes(b'X' * old.st_size); os.utime(altered, ns=(old.st_atime_ns, old.st_mtime_ns))
        refused = False
        try:
            verify_restore(rows, stage / 'outputs/fixture', stage)
        except AssertionError:
            refused = True
        assert refused
        held = subprocess.Popen([sys.executable, '-c',
            'import sys; f=open(sys.argv[1],"rb"); print("ready",flush=True); sys.stdin.read()',
            str(folder / 'model.pt')], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            assert held.stdout.readline().strip() == 'ready'
            assert any(int(r['pid']) == held.pid for r in readers(folder))
        finally:
            held.terminate(); held.wait(timeout=10)
        # Refuse a changed source before releasing it; then restore the fixture.
        victim = folder / 'sample.pt.gz'; raw = victim.read_bytes(); original = victim.stat()
        victim.write_bytes(b'Z' * len(raw))
        assert metadata(inventory(folder, source, False)) != metadata(rows)
        victim.write_bytes(raw); os.utime(victim, ns=(original.st_atime_ns, original.st_mtime_ns))
        released, kept = release_payloads(source, rows, ['gradient-audit'])
        assert sum(r['released_allocated_bytes'] for r in released) > 0
        assert len(released) == 3 and (folder / 'keep.json').exists()
        assert (folder / 'gradient-audit/reference.pt').read_bytes() == b'numerical reference'
        result = {'status': 'passed', 'restored_entries': count,
                  'archive_bytes': size, 'archive_sha256': digest,
                  'full_restore_byte_metadata_and_hardlink_identity': True,
                  'same_size_same_mtime_corruption_rejected': True,
                  'changed_source_rejected': True, 'compact_and_numerical_files_retained': True,
                  'live_open_reader_detected': True,
                  'hardlinked_bulk_release_accounted_once': True, 'GPU_allocations': 0,
                  'script_sha256': sha(Path(__file__).resolve()), 'checked_unix': time.time()}
        print(json.dumps(result), flush=True)


def update(key, phase, **extra):
    with LOCK:
        STATE['items'].setdefault(key, {}).update(phase=phase, checked_unix=time.time(), **extra)
        STATE['filesystem_now'] = shutil.disk_usage(STATE['root'])._asdict()
        STATE['checked_unix'] = time.time()
        save(HERE / 'STATE.json', STATE)
    print(json.dumps({'key': key, 'phase': phase, **extra}), flush=True)


def scheduler_gate():
    data = json.loads((HERE / 'SCHEDULER-CURRENT.json').read_text())
    assert data['complete'] and data['all_clusters'] and data['all_queues']
    assert data['no_selected_candidate_consumers']
    assert 0 <= time.time() - data['observed_unix'] <= 25 * 60, 'Refresh scheduler attestation'
    return data


def archive_one(item, plan, stage_root):
    root = Path(plan['root']); source = root / item['relative_path']
    key = item['relative_path'].replace('/', '__'); log = HERE / (key + '.log')
    receipt = HERE / (key + '.json'); stage = stage_root / key
    stage.mkdir(exist_ok=False)
    assert not receipt.exists()
    scheduler_gate(); assert not readers(source), readers(source)
    update(key, 'inventory')
    before = inventory(source, root)
    inv = HERE / (key + '.inventory.json'); save(inv, before)
    compact = [r for r in before if stat.S_ISREG(r['stamp'][0]) and not bulk(r, plan['preserve_numerical_directories'])]
    payload = [r for r in before if bulk(r, plan['preserve_numerical_directories'])]
    assert payload, 'No eligible bulk payloads'
    obj = stage / 'archive.tar.zst'; dest = Path(item['archive'])
    assert not obj.exists() and not dest.exists()
    update(key, 'packing', entries=len(before), payload_files=len(payload))
    pack_compare(source, root, obj, log)
    assert metadata(inventory(source, root, False)) == metadata(before)
    assert obj.stat().st_size + sum(r['stamp'][1] for r in before if stat.S_ISREG(r['stamp'][0])) < 80 * 2**30
    update(key, 'uploading', archive_bytes=obj.stat().st_size)
    digest, count = copy_archive(obj, dest)
    assert sha(obj) == digest
    update(key, 'cold_readback', archive_bytes=count)
    assert sha(dest) == digest
    update(key, 'complete_restore', archive_bytes=count)
    restored = stage / 'restored'; restore(dest, restored, log)
    coverage = verify_restore(before, restored / item['relative_path'], restored)
    assert metadata(inventory(source, root, False)) == metadata(before), 'Source changed'
    assert not readers(source), readers(source)
    admission = scheduler_gate()
    item_receipt = {'status': 'verified', 'source': str(source), 'archive': str(dest),
                    'archive_sha256': digest, 'archive_bytes': count, 'source_entries': len(before),
                    'source_inventory_sha256': sha(inv), 'restored_entries': coverage,
                    'source_compare': True, 'full_BOS_readback': True, 'complete_restore_verified': True,
                    'compact_files_kept': len(compact), 'payload_files': len(payload),
                    'source_commit': os.environ['TASK_SOURCE'], 'script_sha256': sha(Path(__file__).resolve()),
                    'scheduler_observed_unix': admission['observed_unix'],
                    'restore_command': item['restore_command'], 'payload_released': False,
                    'verified_unix': time.time()}
    save(receipt, item_receipt)
    save(dest.with_name(dest.name + '.receipt.json'), item_receipt)
    save(dest.with_name(dest.name + '.inventory.json'), before)
    with dest.with_name(dest.name + '.sha256').open('x') as stream:
        stream.write(digest + '  ' + dest.name + '\n')
    pointer = source.with_name(source.name + '.outputs-retention-20261107-r1.json')
    assert not pointer.exists(); save(pointer, item_receipt)
    # Publish a root-local pointer only after the original tree is fully verified.
    save(source / 'COLD-ARCHIVE-outputs-retention-20261107-r1.json', item_receipt)
    update(key, 'releasing_verified_bulk', archive_bytes=count)
    def before_each(path):
        scheduler_gate()
        # Directory-wide reader check was just completed; check the exact file
        # again immediately before its final content hash and unlink.
        assert not readers(path), readers(path)
    released, kept = release_payloads(root, before, plan['preserve_numerical_directories'], before_each)
    assert all(sha(root / row['path']) == row['sha256'] for row in kept)
    item_receipt.update(status='complete', payload_released=True,
                        released_allocated_bytes=sum(r['released_allocated_bytes'] for r in released),
                        released_logical_bytes=sum(r['logical_bytes'] for r in released),
                        released_files=len(released), completed_unix=time.time(),
                        filesystem_after=shutil.disk_usage(root)._asdict())
    save(HERE / (key + '.released.json'), released)
    for target in (receipt, pointer, source / 'COLD-ARCHIVE-outputs-retention-20261107-r1.json', dest.with_name(dest.name + '.receipt.json')):
        save(target, item_receipt)
    shutil.rmtree(stage)
    update(key, 'complete', released_allocated_bytes=item_receipt['released_allocated_bytes'],
           archive_bytes=count, compact_files_kept=len(kept))
    return item_receipt


def run():
    assert os.uname().sysname == 'Linux'
    plan = json.loads((HERE / 'PLAN.json').read_text())
    qualification = json.loads((HERE / 'QUALIFICATION.json').read_text())
    assert qualification['status'] == 'passed' and qualification['script_sha256'] == sha(Path(__file__).resolve())
    assert os.environ['TASK_SOURCE'] and os.environ['TASK_SCRIPT_SHA256'] == sha(Path(__file__).resolve())
    assert os.environ['TASK_PLAN_SHA256'] == sha(HERE / 'PLAN.json')
    assert os.path.ismount('/mnt/bos/adgfyl')
    root = Path(plan['root']); cold = Path(plan['cold_root'])
    assert root == Path('/mowenjie/fluffy-bassoon-2')
    assert cold == Path('/mnt/bos/adgfyl/strong-plant-archives') / plan['run_id']
    assert not cold.exists(); cold.mkdir()
    assert shutil.disk_usage('/tmp').free > plan['minimum_tmp_free_bytes']
    assert shutil.disk_usage(root).free > 512 * 2**20
    probe = cold / 'WRITE-PROBE.txt'; probe.write_bytes(b'verified outputs archive\n')
    assert probe.read_bytes() == b'verified outputs archive\n'
    stage_root = Path('/tmp') / plan['run_id']; stage_root.mkdir(exist_ok=False)
    rels = [i['relative_path'] for i in plan['selected']]
    assert len(rels) == 11 and len(rels) == len(set(rels))
    assert all(not b.startswith(a + '/') and not a.startswith(b + '/') for n, a in enumerate(rels) for b in rels[n+1:])
    for rel in rels:
        assert rel.startswith('outputs/') and '..' not in Path(rel).parts
        assert rel not in plan['deferred']
        assert not any(rel == p or rel.startswith(p + '/') or p.startswith(rel + '/') for p in plan['excluded_hot_paths'] if p.startswith('outputs/'))
        assert (root / rel).is_dir() and not (root / rel).is_symlink()
        assert (root / rel).resolve() == root / rel
    STATE.update(status='running', root=str(root), run_id=plan['run_id'], pid=os.getpid(),
                 source_commit=os.environ['TASK_SOURCE'], script_sha256=sha(Path(__file__).resolve()),
                 started_unix=time.time(), filesystem_before=shutil.disk_usage(root)._asdict(),
                 GPU_allocations=0, items={})
    save(HERE / 'STATE.json', STATE)
    completed, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:
        futures = {pool.submit(archive_one, item, plan, stage_root): item for item in sorted(plan['selected'], key=lambda i: i['allocated_bytes'])}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                failure = {'source': item['source'], 'error': repr(error)}; failures.append(failure)
                update(item['relative_path'].replace('/', '__'), 'failed_preserved', error=repr(error))
    result = {'status': 'complete' if not failures else 'incomplete',
              'source_commit': os.environ['TASK_SOURCE'], 'script_sha256': sha(Path(__file__).resolve()),
              'completed': completed, 'preserved_failures': failures,
              'released_allocated_bytes': sum(r['released_allocated_bytes'] for r in completed),
              'archive_bytes': sum(r['archive_bytes'] for r in completed),
              'filesystem_before': STATE['filesystem_before'], 'filesystem_after': shutil.disk_usage(root)._asdict(),
              'finished_unix': time.time(), 'GPU_allocations': 0}
    save(HERE / 'RESULT.json', result); save(cold / 'RESULT.json', result)
    STATE.update(status=result['status'], completed_unix=time.time()); save(HERE / 'STATE.json', STATE)
    if not failures:
        stage_root.rmdir()
    print(json.dumps({'FINAL': result['status'], 'released_allocated_bytes': result['released_allocated_bytes'], 'failures': failures}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args()
    qualify() if args.qualify else run()
