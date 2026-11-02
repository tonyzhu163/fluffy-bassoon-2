"""One-off, fail-closed move of inactive Baige assets to the existing BOS mount."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time

ROOT = Path('/mowenjie')
RUN = 'baige-storage-tiering-20261103-r1'
AUDIT = ROOT / 'fluffy-bassoon-2/outputs' / RUN
BOS = Path('/mnt/bos/adgfyl/strong-plant-archives/20261103-rl-checkpoint-space-r1')
STAGE = Path('/tmp') / RUN


def log(message):
    print(time.strftime('%Y-%m-%dT%H:%M:%S%z'), message, flush=True)


def save(path, data):
    with path.open('w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def inventory(source):
    rows = []
    stack = [source]
    while stack:
        p = stack.pop()
        s = p.lstat()
        rows.append([str(p.relative_to(ROOT)), s.st_mode, s.st_size,
                     s.st_mtime_ns, s.st_uid, s.st_gid, s.st_blocks,
                     s.st_dev, s.st_ino, s.st_nlink,
                     os.readlink(p) if stat.S_ISLNK(s.st_mode) else None])
        if stat.S_ISDIR(s.st_mode):
            stack.extend(p.iterdir())
    rows.sort(key=lambda r: r[0])
    return rows


def users(source):
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        paths = [proc / 'cwd', proc / 'exe']
        try:
            paths.extend((proc / 'fd').iterdir())
        except (OSError, PermissionError):
            pass
        for p in paths:
            try:
                target = os.readlink(p)
                if target == str(source) or target.startswith(str(source) + '/'):
                    found.append({'pid': int(proc.name), 'kind': p.parent.name,
                                  'path': target})
            except OSError:
                pass
    return found


def command(args, label):
    log(label)
    with (AUDIT / 'commands.log').open('a') as out:
        p = subprocess.Popen(args, stdout=out, stderr=out)
        while True:
            try:
                rc = p.wait(timeout=25)
                if rc:
                    raise RuntimeError(f'{label}: exit {rc}; see commands.log')
                return
            except subprocess.TimeoutExpired:
                log(label + ' in progress')


def digest(path):
    h = hashlib.sha256()
    count = 0
    last = time.monotonic()
    with path.open('rb') as f:
        while chunk := f.read(16 * 1024**2):
            h.update(chunk)
            count += len(chunk)
            if time.monotonic() - last > 25:
                log(f'readback {path.name}: {count / 1024**3:.2f} GiB')
                last = time.monotonic()
    return h.hexdigest()


def archive(rel):
    source = ROOT / rel
    key = rel.replace('/', '__')
    manifest = AUDIT / (key + '.json')
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old.get('source_moved'):
            log('Already moved: ' + rel)
            return old
        raise RuntimeError('Existing unfinished record: ' + str(manifest))
    assert source.is_dir() and not source.is_symlink(), source
    assert not (str(source).startswith('/mowenjie/fluffy-bassoon'))
    assert rel != 'envs/wam-libero-plus-py310'
    assert not users(source), f'Open process dependency: {source}'
    for name in ('.secrets', '.ssh', '.aws', '.bce', '.env'):
        assert not (source / name).exists(), f'Keep local authentication material: {source / name}'
    before = inventory(source)
    log(f'{rel}: {len(before)} entries, {sum(r[6] for r in before)*512/1024**3:.2f} GiB allocated')
    inv = AUDIT / (key + '.inventory.json')
    save(inv, before)
    local = STAGE / (key + '.tar.zst')
    dest = BOS / local.name
    assert not local.exists() and not dest.exists()
    command(['tar', '--sort=name', '--acls', '--xattrs', '--numeric-owner',
             '-I', 'zstd -1 -T2', '-cf', str(local), '-C', str(ROOT), rel], 'pack ' + rel)
    command(['tar', '-I', 'zstd', '-df', str(local), '-C', str(ROOT)], 'compare ' + rel)
    assert inventory(source) == before, f'Source changed while packing: {source}'
    h = hashlib.sha256()
    copied = 0
    last = time.monotonic()
    with local.open('rb') as inp, dest.open('xb') as out:
        while chunk := inp.read(16 * 1024**2):
            out.write(chunk)
            h.update(chunk)
            copied += len(chunk)
            if time.monotonic() - last > 25:
                log(f'upload {rel}: {copied / 1024**3:.2f}/{local.stat().st_size / 1024**3:.2f} GiB')
                last = time.monotonic()
        out.flush()
        os.fsync(out.fileno())
    expected = h.hexdigest()
    assert dest.stat().st_size == copied == local.stat().st_size
    assert digest(dest) == expected, f'Object readback mismatch: {dest}'
    assert inventory(source) == before, f'Source changed while uploading: {source}'
    assert not users(source), f'New open process dependency: {source}'
    item = dict(source=str(source), bos_archive=str(dest), archive_bytes=copied,
                source_allocated_bytes=sum(r[6] for r in before)*512,
                source_logical_bytes=sum(r[2] for r in before if stat.S_ISREG(r[1])),
                entries=len(before), sha256=expected, archive_source_compare=True,
                bos_readback_verified=True, source_moved=False,
                verified_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                restore_command=f"tar --acls --xattrs -I zstd -xf '{dest}' -C /mowenjie")
    save(manifest, item)
    save(BOS / manifest.name, item)
    shutil.copyfile(inv, BOS / inv.name)
    (BOS / (dest.name + '.sha256')).write_text(expected + '  ' + dest.name + '\n')
    # Cold move: the complete verified BOS copy remains recoverable. Only this
    # selected, unchanged source tree is reclaimed; no other payload is pruned.
    quarantine = source.with_name(source.name + '.tiering-' + RUN)
    assert not quarantine.exists()
    pointer = source.with_name(source.name + '.bos-archive.json')
    save(pointer, item)
    os.rename(source, quarantine)
    shutil.rmtree(quarantine)
    item['source_moved'] = True
    item['moved_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    for p in (manifest, BOS / manifest.name, pointer):
        save(p, item)
    local.unlink()
    log('VERIFIED COLD MOVE COMPLETE ' + rel)
    return item


def main():
    assert os.path.ismount('/mnt/bos/adgfyl')
    AUDIT.mkdir(parents=True, exist_ok=True)
    STAGE.mkdir(exist_ok=True)
    plan = json.loads((AUDIT / 'PLAN.json').read_text())
    before = shutil.disk_usage(ROOT)
    save(AUDIT / 'SPACE-BEFORE.json', dict(before._asdict(), time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    results = []
    failed = []
    for rel in plan['sources']:
        try:
            results.append(archive(rel))
        except Exception as error:
            failed.append({'source': rel, 'error': str(error)})
            log('SOURCE PRESERVED / NEEDS REVIEW ' + rel + ': ' + str(error))
        summary = dict(completed=results, preserved=failed,
                       free_now=shutil.disk_usage(ROOT).free,
                       before_free=before.free)
        save(AUDIT / 'SUMMARY.json', summary)
    after = shutil.disk_usage(ROOT)
    summary['after_free'] = after.free
    summary['net_free_increase_bytes'] = after.free - before.free
    summary['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    save(AUDIT / 'SUMMARY.json', summary)
    save(BOS / 'SUMMARY.json', summary)
    log('FINISHED ' + json.dumps({k: v for k, v in summary.items() if k not in ('completed', 'preserved')}))


if __name__ == '__main__':
    main()
