"""Run-local cold archival, with exact source comparison and BOS readback."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time

RUN = 'storage-audit-baige-20261104-r1'
ROOT = Path('/mowenjie/fluffy-bassoon-2')
AUDIT = ROOT / 'outputs' / RUN
BOS = Path('/mnt/bos/adgfyl/strong-plant-archives') / RUN
STAGE = Path('/tmp') / RUN


def save(path, value):
    tmp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def inventory(source):
    rows = []
    stack = [source]
    while stack:
        p = stack.pop()
        s = p.lstat()
        assert stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode) or stat.S_ISLNK(s.st_mode), p
        rows.append([str(p.relative_to(ROOT)), s.st_mode, s.st_size, s.st_mtime_ns,
                     s.st_uid, s.st_gid, s.st_blocks, s.st_dev, s.st_ino, s.st_nlink,
                     os.readlink(p) if stat.S_ISLNK(s.st_mode) else None])
        if stat.S_ISDIR(s.st_mode):
            stack.extend(p.iterdir())
    return sorted(rows)


def users(source):
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            if str(source) in cmd:
                found.append([proc.name, 'command_reference'])
        except OSError:
            pass
        paths = [proc / 'cwd', proc / 'exe']
        try:
            paths.extend((proc / 'fd').iterdir())
        except OSError:
            pass
        for p in paths:
            try:
                target = os.readlink(p)
                if target == str(source) or target.startswith(str(source) + '/'):
                    found.append([proc.name, str(p), target])
            except OSError:
                pass
    return found


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while b := f.read(16 << 20):
            h.update(b)
    return h.hexdigest()


def log(key, stage):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), key, stage, flush=True)


def archive(rel):
    source = ROOT / rel
    key = rel.replace('/', '__')
    receipt = AUDIT / (key + '.json')
    if receipt.exists():
        old = json.loads(receipt.read_text())
        assert old.get('source_reclaimed'), 'Unfinished receipt; manual reconcile required'
        return old
    assert source.is_dir() and not source.is_symlink(), source
    assert not users(source), users(source)
    before = inventory(source)
    assert all(not any(x in Path(r[0]).parts for x in ('.secrets', '.ssh', '.aws', '.bce', '.env', '.netrc')) for r in before)
    save(AUDIT / (key + '.inventory.json'), before)
    allocated = sum(r[6] for r in before) * 512
    log(key, f'inventory {len(before)} entries / {allocated/(1<<30):.2f} GiB')
    local = STAGE / (key + '.tar.zst')
    dest = BOS / local.name
    assert not local.exists() and not dest.exists()
    with (AUDIT / (key + '.log')).open('a') as out:
        subprocess.run(['tar', '--sort=name', '--acls', '--xattrs', '--numeric-owner',
                        '-I', 'zstd -1 -T2', '-cf', str(local), '-C', str(ROOT), rel],
                       stdout=out, stderr=out, check=True)
        log(key, 'packed; comparing source')
        subprocess.run(['tar', '-I', 'zstd', '-df', str(local), '-C', str(ROOT)],
                       stdout=out, stderr=out, check=True)
        listing = subprocess.check_output(['tar', '-I', 'zstd', '-tf', str(local)], text=True)
    assert sorted(x.rstrip('/') for x in listing.splitlines()) == sorted(r[0] for r in before), 'Archive member coverage differs'
    assert inventory(source) == before, 'Source changed during packing'
    log(key, 'source exact / member coverage exact; uploading')
    h = hashlib.sha256()
    with local.open('rb') as f, dest.open('xb') as out:
        while b := f.read(16 << 20):
            out.write(b)
            h.update(b)
        out.flush()
        os.fsync(out.fileno())
    expected = h.hexdigest()
    assert local.stat().st_size == dest.stat().st_size
    log(key, 'uploaded; complete BOS SHA256 readback')
    assert digest(dest) == expected, 'BOS checksum mismatch'
    assert inventory(source) == before, 'Source changed during upload'
    assert not users(source), users(source)
    item = {'source': str(source), 'archive': str(dest), 'sha256': expected,
            'archive_bytes': dest.stat().st_size, 'source_allocated_bytes': allocated,
            'source_logical_bytes': sum(r[2] for r in before if stat.S_ISREG(r[1])),
            'entries': len(before), 'source_compare': True, 'exact_member_coverage': True,
            'bos_readback_sha256_verified': True, 'source_reclaimed': False,
            'verified_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'restore_command': f"tar --acls --xattrs -I zstd -xf '{dest}' -C '{ROOT}'"}
    save(receipt, item)
    save(BOS / receipt.name, item)
    shutil.copyfile(AUDIT / (key + '.inventory.json'), BOS / (key + '.inventory.json'))
    (BOS / (dest.name + '.sha256')).write_text(expected + '  ' + dest.name + '\n')
    # Source reclaimed only after the complete restorable object was verified.
    pointer = source.with_name(source.name + '.bos-archive.json')
    assert not pointer.exists()
    save(pointer, item)
    quarantine = source.with_name(source.name + '.archiving-' + RUN)
    assert not quarantine.exists()
    source.rename(quarantine)
    shutil.rmtree(quarantine)
    item['source_reclaimed'] = True
    item['reclaimed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    for p in (receipt, BOS / receipt.name, pointer):
        save(p, item)
    local.unlink()
    log(key, 'VERIFIED / RECLAIMED')
    return item


def main():
    assert os.path.ismount('/mnt/bos/adgfyl')
    BOS.mkdir(exist_ok=False)
    STAGE.mkdir(exist_ok=False)
    probe = BOS / 'WRITE-PROBE.txt'
    probe.write_text(RUN + '\n')
    assert probe.read_text() == RUN + '\n'
    probe2 = AUDIT / 'WRITE-PROBE.txt'
    probe2.write_text(RUN + '\n')
    assert probe2.read_text() == RUN + '\n'
    plan = json.loads((AUDIT / 'CANDIDATES.json').read_text())
    start = shutil.disk_usage(ROOT)._asdict()
    save(AUDIT / 'SPACE-BEFORE.json', start)
    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fs = {pool.submit(archive, rel): rel for rel in plan['selected']}
        for f in concurrent.futures.as_completed(fs):
            try:
                results.append(f.result())
            except Exception as e:
                failures.append({'source': fs[f], 'error': repr(e)})
                log(fs[f], repr(e))
            save(AUDIT / 'SUMMARY.json', {'completed': results, 'preserved': failures,
                 'reclaimed_allocated_bytes': sum(x['source_allocated_bytes'] for x in results),
                 'filesystem_before': start, 'filesystem_now': shutil.disk_usage(ROOT)._asdict(),
                 'finished': False})
    summary = {'completed': results, 'preserved': failures,
               'reclaimed_allocated_bytes': sum(x['source_allocated_bytes'] for x in results),
               'filesystem_before': start, 'filesystem_after': shutil.disk_usage(ROOT)._asdict(),
               'finished': True, 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    save(AUDIT / 'SUMMARY.json', summary)
    save(BOS / 'SUMMARY.json', summary)
    log(RUN, 'FINISHED')


if __name__ == '__main__':
    main()
