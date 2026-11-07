"""Exact approved redundant-file cleanup, preserving verified cold recovery."""
from pathlib import Path
import argparse, collections, hashlib, importlib.util, json, os, shutil, stat
import struct, subprocess, tempfile, time

HERE = Path(__file__).resolve().parent
HELPER = HERE.parent / 'outputs-retention-20261107-r1/archive_outputs.py'
assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == '3d58eaa4a18a82e719bf1ce28c8527baa6b43b1e9c2811402b449d6d61acffe4'
spec = importlib.util.spec_from_file_location('retention', HELPER)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def emit(phase, **data):
    record = dict(phase=phase, checked_unix=time.time(), pid=os.getpid(), **data)
    m.save(HERE / 'STATE.json', record)
    print(json.dumps(record), flush=True)


def gate():
    s = json.loads((HERE / 'SCHEDULER.json').read_text())
    assert all(s[k] for k in ['complete', 'all_clusters', 'all_queues', 'no_selected_candidate_consumers'])
    assert 0 <= time.time() - s['observed_unix'] <= 25 * 60


def record(path, root):
    before = m.stamp(path); assert stat.S_ISREG(before[0]) and not path.is_symlink()
    row = dict(path=path.relative_to(root).as_posix(), stamp=before, sha256=m.sha(path))
    assert m.stamp(path) == before
    return row


def readers(path):
    hits = m.readers(path)
    # File mappings can outlive their open descriptor; the shared helper's
    # directory-prefix check does not cover an exact regular-file path.
    for p in Path('/proc').glob('[0-9]*'):
        if int(p.name) == os.getpid(): continue
        try:
            for line in (p / 'maps').read_text(errors='replace').splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].removesuffix(' (deleted)') == str(path):
                    hits.append(dict(pid=p.name, kind='exact_mmap_reference'))
        except OSError: pass
    return hits


def pack(rows, root, stage, cold):
    listing = stage / 'members.txt'
    listing.write_bytes(b''.join(r['path'].encode() + b'\0' for r in rows))
    local = stage / 'selected-files.tar.zst'; log = HERE / 'tar.log'
    with log.open('ab') as stream:
        subprocess.run(['tar', '--format=posix', '--acls', '--xattrs', '--numeric-owner', '--sparse',
                        '-I', 'zstd -1 -T2', '-cf', str(local), '-C', str(root),
                        '--null', '--no-recursion', '-T', str(listing)], stdout=stream, stderr=stream, check=True)
        subprocess.run(['tar', '--acls', '--xattrs', '-I', 'zstd', '-df', str(local), '-C', str(root)],
                       stdout=stream, stderr=stream, check=True)
    emit('uploading_selected_recovery', archive_bytes=local.stat().st_size)
    digest, count = m.copy_archive(local, cold)
    assert m.sha(local) == digest
    emit('independent_cold_readback', archive_bytes=count)
    assert m.sha(cold) == digest
    emit('complete_staged_restore', archive_bytes=count)
    restored = stage / 'restored'; m.restore(cold, restored, log)
    actual = sorted(p.relative_to(restored).as_posix() for p in restored.rglob('*') if p.is_file())
    assert actual == sorted(r['path'] for r in rows)
    original_groups, restored_groups = collections.defaultdict(list), collections.defaultdict(list)
    for row in rows:
        path = restored / row['path']; a, b = row['stamp'], m.stamp(path)
        assert a[:5] == b[:5] and m.sha(path) == row['sha256'], row['path']
        original_groups[tuple(a[5:7])].append(row['path'])
        restored_groups[tuple(b[5:7])].append(row['path'])
    assert sorted(sorted(x) for x in original_groups.values()) == sorted(sorted(x) for x in restored_groups.values())
    return dict(archive=str(cold), archive_sha256=digest, archive_bytes=count,
                source_compare=True, full_cold_readback=True, full_restore_verified=True, restored_files=len(rows))


def release(rows, root, callback):
    released, prior_links = [], collections.Counter()
    for row in rows:
        path = root / row['path']; expected = row['stamp'].copy(); key = tuple(expected[5:7])
        expected[7] -= prior_links[key]
        assert m.stamp(path) == expected, ('source changed', row['path'])
        callback(path)
        assert m.sha(path) == row['sha256'] and m.stamp(path) == expected
        physical = expected[8] if expected[7] == 1 else 0
        path.unlink(); prior_links[key] += 1
        released.append(dict(path=row['path'], sha256=row['sha256'], released_allocated_bytes=physical))
        m.save(HERE / 'RELEASED.json', released)
    return released


def qualify():
    assert os.uname().sysname == 'Linux'
    with tempfile.TemporaryDirectory(prefix='redundant-cleanup-fixture-') as tmp:
        root = Path(tmp) / 'root'; root.mkdir(); (root / 'a.bin').write_bytes(b'fixture' * 1000)
        os.link(root / 'a.bin', root / 'b.bin'); (root / 'keep.json').write_text('{}')
        rows = [record(root / name, root) for name in ['a.bin', 'b.bin']]
        stage = Path(tmp) / 'stage'; stage.mkdir()
        receipt = pack(rows, root, stage, Path(tmp) / 'cold.tar.zst')
        released = release(rows, root, lambda p: None)
        assert len(released) == 2 and sum(r['released_allocated_bytes'] for r in released) == rows[0]['stamp'][8]
        assert (root / 'keep.json').read_text() == '{}'
        changed = root / 'changed.bin'; changed.write_bytes(b'original')
        original = record(changed, root); changed.write_bytes(b'modified')
        os.utime(changed, ns=(original['stamp'][2], original['stamp'][2]))
        rejected = False
        try: release([original], root, lambda p: None)
        except AssertionError: rejected = True
        assert rejected and changed.exists()
        report = dict(status='passed', script_sha256=m.sha(Path(__file__)), restored_files=receipt['restored_files'],
                      hardlink_release_counted_once=True, changed_file_rejected=True,
                      unselected_file_preserved=True, checked_unix=time.time())
    # Qualification uses a separate output directory; its RELEASED list is never an execution receipt.
    (HERE / 'RELEASED.json').unlink()
    m.save(HERE / 'QUALIFICATION.json', report); print(json.dumps(report), flush=True)


def run():
    assert os.uname().sysname == 'Linux'
    plan = json.loads((HERE / 'PLAN.json').read_text()); root = Path(plan['root']); cold = Path(plan['cold_root'])
    assert root == Path('/mowenjie/fluffy-bassoon-2') and cold.name == HERE.name
    assert m.sha(Path(__file__)) == os.environ['TASK_SCRIPT_SHA256']
    assert m.sha(HERE / 'PLAN.json') == os.environ['TASK_PLAN_SHA256']
    q = json.loads((HERE / 'QUALIFICATION.json').read_text()); assert q['status'] == 'passed' and q['script_sha256'] == m.sha(Path(__file__))
    assert not cold.exists() and os.path.ismount('/mnt/bos/adgfyl')
    gate(); before = shutil.disk_usage(root)._asdict()
    cold.mkdir(); probe = cold / 'WRITE-PROBE.txt'; probe.write_bytes(b'verified redundant-file recovery\n'); assert probe.read_bytes() == b'verified redundant-file recovery\n'
    stage = Path('/tmp') / HERE.name; assert not stage.exists() and shutil.disk_usage('/tmp').free > 32 * 2**30; stage.mkdir()
    selected = plan['files']; assert len(selected) == 323 and len({r['path'] for r in selected}) == 323
    assert all(Path(p).is_file() and p not in {r['path'] for r in selected} for p in plan['excluded_bundle_paths'])
    emit('inventory_and_protected_hashes')
    rows = []
    for item in selected:
        p = Path(item['path']); assert p.is_relative_to(root) and p.resolve() == p
        assert item['kind'] in ['duplicate_archive', 'objects_zip', 'truncated_model', 'deployment_bundle']
        assert not readers(p), p
        row = record(p, root); row['kind'] = item['kind']; rows.append(row)
    m.save(HERE / 'SOURCE-INVENTORY.json', rows)
    motor = root / 'cache/robotwin-ppo-pilot-20261105-r1-restore02/pi05-motus-effc6e1a-r1'
    assets = root / 'cache/robotwin-sidney-objects-20261105-r1/restored-families-assets02'
    protected = m.inventory(motor, root) + m.inventory(assets, root)
    excluded_bundles = [p for p in [*root.glob('*.bundle'), *(root / 'bundles').glob('*.bundle')]
                       if str(p) not in {r['path'] for r in selected}]
    compact_root = root / 'cache/robotwin-ppo-pilot-20261105-r1'
    extra = [p for p in compact_root.rglob('*') if p.is_file() and str(p) not in {r['path'] for r in selected}]
    extra += excluded_bundles + [root / 'sources/robotwin-baseline-smoke-20261028-r1/RoboTwin/assets/objects/.fb2-archive-manifest.json']
    protected += [record(p, root) for p in extra]
    m.save(HERE / 'PROTECTED-INVENTORY.json', protected)
    duplicate = next(r for r in rows if r['kind'] == 'duplicate_archive')
    assert duplicate['sha256'] == plan['existing_cold_sha256']
    emit('verifying_existing_cold_duplicate')
    assert m.sha(Path(plan['existing_cold_archive'])) == duplicate['sha256']
    assert next(r['sha256'] for r in rows if r['kind'] == 'objects_zip') == plan['object_zip_sha256']
    broken = next(r for r in rows if r['kind'] == 'truncated_model')
    with (root / broken['path']).open('rb') as stream:
        n = struct.unpack('<Q', stream.read(8))[0]; assert n < 20 * 1024**2; header = json.loads(stream.read(n))
    assert broken['stamp'][1] < 8 + n + max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
    emit('checking_retained_git_object_closure')
    repo = root / 'repo'; reachable = set(subprocess.check_output(['git', '-C', str(repo), 'rev-list', '--all'], text=True).splitlines()); tips = set()
    for row in rows:
        if row['kind'] != 'deployment_bundle': continue
        lines = subprocess.check_output(['git', '-C', str(repo), 'bundle', 'list-heads', str(root / row['path'])], text=True).splitlines()
        heads = {line.split()[0] for line in lines}; assert heads and heads <= reachable; tips.update(heads)
    objects = subprocess.check_output(['git', '-C', str(repo), 'rev-list', '--objects', '--stdin'], input='\n'.join(sorted(tips))+'\n', text=True)
    object_ids = '\n'.join(line.split()[0] for line in objects.splitlines())+'\n'
    checked = subprocess.check_output(['git', '-C', str(repo), 'cat-file', '--batch-check'], input=object_ids, text=True)
    assert all(' missing' not in line for line in checked.splitlines()) and len(checked.splitlines()) == len(object_ids.splitlines())
    m.save(HERE / 'GIT-CLOSURE.json', dict(bundle_files=320, retained_tips=len(tips), available_objects=len(checked.splitlines()), all_advertised_tips_reachable=True))
    packed = [r for r in rows if r['kind'] != 'duplicate_archive']
    emit('packing_selected_recovery')
    receipt = pack(packed, root, stage, cold / 'selected-files.tar.zst')
    assert sum(p.stat().st_size for p in stage.rglob('*') if p.is_file()) < plan['maximum_staging_bytes']
    assert all(m.stamp(root / row['path']) == row['stamp'] for row in rows)
    m.save(cold / 'SOURCE-INVENTORY.json', rows); m.save(cold / 'RECOVERY.json', receipt)
    m.save(HERE / 'RECOVERY.json', receipt)
    # Publish recovery mapping before any unlink; all exact original paths are inventoried.
    pointer = dict(run=plan['run_id'], source_commit=os.environ['TASK_SOURCE'], archive=receipt,
                   original_duplicate_archive=plan['existing_cold_archive'], duplicate_sha256=duplicate['sha256'], inventory=str(cold / 'SOURCE-INVENTORY.json'))
    for p in [root / ('COLD-CLEANUP-'+HERE.name+'.json'), motor.parent / ('COLD-CLEANUP-'+HERE.name+'.json'), assets.parent / ('COLD-CLEANUP-'+HERE.name+'.json')]:
        assert not p.exists(); m.save(p, pointer)
    emit('releasing_verified_selected_files')
    def admission(path):
        gate(); assert not readers(path), path
    released = release(rows, root, admission)
    emit('final_protected_hash_reconciliation', released_files=len(released))
    for row in protected:
        p = root / row['path']; assert m.stamp(p) == row['stamp'], row['path']
        if stat.S_ISREG(row['stamp'][0]): assert m.sha(p) == row['sha256'], row['path']
        elif stat.S_ISLNK(row['stamp'][0]): assert os.readlink(p) == row['target']
    assert all(not os.path.lexists(root / r['path']) for r in released)
    result = dict(status='complete', source_commit=os.environ['TASK_SOURCE'], script_sha256=m.sha(Path(__file__)),
                  released_files=len(released), released_allocated_bytes=sum(r['released_allocated_bytes'] for r in released),
                  protected_entries=len(protected), protected_hashes_and_stamps_unchanged=True,
                  recovery=receipt, existing_cold_duplicate_verified=True, filesystem_before=before,
                  filesystem_after=shutil.disk_usage(root)._asdict(), completed_unix=time.time(), GPU_allocations=0)
    for name in ['PLAN.json', 'SOURCE-INVENTORY.json', 'PROTECTED-INVENTORY.json', 'GIT-CLOSURE.json', 'RELEASED.json', 'QUALIFICATION.json', 'cleanup.py', 'SCHEDULER.json']:
        data = (HERE / name).read_bytes(); target = cold / name
        assert not target.exists() or target.read_bytes() == data
        target.write_bytes(data); assert target.read_bytes() == data
    shutil.rmtree(stage); result['staging_removed'] = not stage.exists()
    m.save(HERE / 'RESULT.json', result); m.save(cold / 'RESULT.json', result)
    emit('complete', **result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--qualify', action='store_true'); args = parser.parse_args()
    try:
        qualify() if args.qualify else run()
    except Exception as error:
        emit('failed_preserved', error=repr(error)); raise
