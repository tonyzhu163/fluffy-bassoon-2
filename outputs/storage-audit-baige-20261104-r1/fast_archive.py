"""Archive stale environments with exact parallel GNU tar partitions."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import shutil
import stat
import subprocess
import tempfile
import threading
import time

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/storage-audit-baige-20261104-r1')
def load(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + '.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
m = load('archive')
c = load('parallel_compare')
ROOT = Path('/mowenjie')
AUDIT = BASE / 'workspace-r2'
BOS = m.BOS / 'workspace-r2'
STAGE = m.STAGE / 'workspace-r2'


def inventory(source):
    pending = queue.Queue()
    pending.put(source)
    rows, errors = [], []
    def worker():
        while True:
            p = pending.get()
            try:
                if p is None: return
                s = p.lstat()
                assert stat.S_ISREG(s.st_mode) or stat.S_ISLNK(s.st_mode) or stat.S_ISDIR(s.st_mode), p
                # Link counts can change when a sibling archived environment is
                # reclaimed. Content, mode, size, time, owner and inode cannot.
                rows.append([str(p.relative_to(ROOT)), s.st_mode, s.st_size, s.st_mtime_ns,
                             s.st_uid, s.st_gid, s.st_blocks, s.st_dev, s.st_ino, None,
                             os.readlink(p) if stat.S_ISLNK(s.st_mode) else None])
                if stat.S_ISDIR(s.st_mode):
                    for child in p.iterdir(): pending.put(child)
            except Exception as e: errors.append([str(p), repr(e)])
            finally: pending.task_done()
    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads: t.start()
    pending.join()
    for _ in threads: pending.put(None)
    for t in threads: t.join()
    assert not errors, errors[:5]
    return sorted(rows)


def reclaim(source, rows):
    quarantine = source.with_name(source.name + '.archiving-' + m.RUN + '-workspace-r2')
    assert not quarantine.exists()
    original = source.stat()
    rootrow = next(r for r in rows if r[0] == str(source.relative_to(ROOT)))
    assert (original.st_dev, original.st_ino) == (rootrow[7], rootrow[8])
    source.rename(quarantine)
    def target(row): return quarantine / Path(row[0]).relative_to(source.relative_to(ROOT))
    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(lambda r: target(r).unlink(), [r for r in rows if not stat.S_ISDIR(r[1])]))
        dirs = [r for r in rows if stat.S_ISDIR(r[1])]
        for depth in sorted({len(Path(r[0]).parts) for r in dirs}, reverse=True):
            list(pool.map(lambda r: target(r).rmdir(), [r for r in dirs if len(Path(r[0]).parts) == depth]))
    assert not quarantine.exists()


def archive(rel, workers=8):
    source = ROOT / rel
    key = rel.replace('/', '__')
    assert source.is_dir() and not source.is_symlink()
    assert not m.users(source), m.users(source)
    before = inventory(source)
    assert all(not any(x in Path(r[0]).parts for x in ('.secrets','.ssh','.aws','.bce','.env','.netrc')) for r in before)
    m.save(AUDIT / (key + '.inventory.json'), before)
    m.log(key, f'parallel inventory {len(before)} entries / {sum(r[6] for r in before)*512/(1<<30):.2f} GiB')
    parts = [[] for _ in range(workers)]
    for row in before:
        # Put every member of a hardlink group in the same tar partition.
        identity = (row[7], row[8]) if stat.S_ISREG(row[1]) else tuple(os.fsencode(row[0]))
        parts[int.from_bytes(hashlib.sha256(repr(identity).encode()).digest()[:8], 'big') % workers].append(row[0])
    assert sorted(n for part in parts for n in part) == [r[0] for r in before]
    def pack(i):
        names = parts[i]
        local = STAGE / (key + f'.part-{i:02d}.tar.zst')
        members = AUDIT / (key + f'.part-{i:02d}.names')
        members.write_bytes(b''.join(os.fsencode(n) + b'\0' for n in names))
        with (AUDIT / (key + f'.part-{i:02d}.log')).open('wb') as out:
            subprocess.run(['tar','--sort=name','--acls','--xattrs','--numeric-owner','--no-recursion',
                            '--null','--verbatim-files-from','-I','zstd -1 -T1','-cf',str(local),
                            '-C',str(ROOT),'-T',str(members)],stdout=out,stderr=out,check=True)
            assert sorted(c.archive_names(local)) == sorted(names)
            subprocess.run(['tar','-I','zstd','-df',str(local),'-C',str(ROOT)],stdout=out,stderr=out,check=True)
        m.log(key, f'partition {i} packed / exact GNU tar source comparison passed')
        return local
    with ThreadPoolExecutor(max_workers=workers) as pool:
        archives = list(pool.map(pack, range(workers)))
    assert inventory(source) == before
    dest = BOS / (key + '.tar.zst')
    h = hashlib.sha256()
    assert not dest.exists()
    # Concatenated Zstandard frames contain concatenated GNU tar streams.
    # --ignore-zeros makes GNU tar read and restore the complete stream.
    with dest.open('xb') as out:
        for local in archives:
            with local.open('rb') as f:
                while b := f.read(16 << 20):
                    out.write(b); h.update(b)
        out.flush(); os.fsync(out.fileno())
    assert dest.stat().st_size == sum(x.stat().st_size for x in archives)
    assert m.digest(dest) == h.hexdigest()
    assert inventory(source) == before
    assert not m.users(source), m.users(source)
    receipt = {'source': str(source), 'archive': str(dest), 'sha256': h.hexdigest(),
               'archive_bytes': dest.stat().st_size, 'entries': len(before),
               'source_allocated_bytes': sum(r[6] for r in before) * 512,
               'source_logical_bytes': sum(r[2] for r in before if stat.S_ISREG(r[1])),
               'exact_member_coverage': True, 'source_compare': True,
               'comparison_method': f'{workers} disjoint GNU tar archives; hardlink groups kept together; every member compared',
               'bos_readback_sha256_verified': True, 'source_reclaimed': False,
               'restore_command': f"tar --ignore-zeros --acls --xattrs -I zstd -xf '{dest}' -C '{ROOT}'",
               'format': 'Concatenated Zstandard frames / GNU tar streams: --ignore-zeros REQUIRED',
               'verified_at': time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    paths = [AUDIT / (key + '.json'), BOS / (key + '.json'), source.with_name(source.name + '.bos-archive.json')]
    assert not paths[2].exists()
    for p in paths: m.save(p, receipt)
    shutil.copyfile(AUDIT/(key+'.inventory.json'), BOS/(key+'.inventory.json'))
    (BOS/(dest.name+'.sha256')).write_text(h.hexdigest()+'  '+dest.name+'\n')
    reclaim(source, before)
    receipt['source_reclaimed'] = True
    receipt['reclaimed_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    for p in paths: m.save(p, receipt)
    for x in archives: x.unlink()
    m.log(key, 'VERIFIED / RECLAIMED')
    return receipt


def qualify():
    global ROOT, AUDIT, BOS, STAGE
    original = ROOT, AUDIT, BOS, STAGE
    work = Path(tempfile.mkdtemp(prefix='storage-partition-qualification-'))
    ROOT, AUDIT, BOS, STAGE = [work / n for n in ('root','audit','bos','stage')]
    for p in (ROOT,AUDIT,BOS,STAGE): p.mkdir()
    source = ROOT/'fixture'; (source/'nested').mkdir(parents=True)
    for i in range(30): (source/'nested'/f'file {i}').write_bytes(bytes([i])*333)
    (source/'link').symlink_to('nested/file 1'); os.link(source/'nested'/'file 2', source/'hardlink')
    (source/'odd\nname[*]').write_bytes(b'literal filename')
    serial = m.ROOT; m.ROOT = ROOT
    expected = m.inventory(source); actual = inventory(source); m.ROOT = serial
    assert [r[:9]+r[10:] for r in expected] == [r[:9]+r[10:] for r in actual]
    r = archive('fixture', workers=4)
    subprocess.run(r['restore_command'], shell=True, check=True)
    for i in range(30): assert (source/'nested'/f'file {i}').read_bytes() == bytes([i])*333
    assert (source/'hardlink').stat().st_ino == (source/'nested'/'file 2').stat().st_ino
    assert (source/'link').is_symlink() and (source/'odd\nname[*]').read_bytes() == b'literal filename'
    # Mutation with preserved size and mtime must be detected by GNU tar.
    target=source/'nested'/'file 3'; s=target.stat(); target.write_bytes(b'x'*s.st_size); os.utime(target,ns=(s.st_atime_ns,s.st_mtime_ns))
    neg=subprocess.run(['tar','--ignore-zeros','-I','zstd','-df',r['archive'],'-C',str(ROOT)],capture_output=True)
    assert neg.returncode != 0
    ROOT, AUDIT, BOS, STAGE = original
    m.save(BASE/'FAST-QUALIFICATION.json', {'passed':True,'parallel_inventory_matches_serial':True,
         'partitioned_full_restore':True,'hardlinks_symlinks_literal_names_preserved':True,
         'same_size_same_mtime_content_corruption_detected':True,'GPU_used':0})
    shutil.rmtree(work)


def main():
    qualify()
    for p in (AUDIT,BOS,STAGE): p.mkdir(exist_ok=False)
    plan=json.loads((BASE/'WORKSPACE-CANDIDATES.json').read_text())
    m.save(AUDIT/'CANDIDATES.json',plan)
    start=shutil.disk_usage(ROOT)._asdict();m.save(AUDIT/'SPACE-BEFORE.json',start)
    completed, failures=[],[]
    for absolute in plan['selected']:
        try: completed.append(archive(str(Path(absolute).relative_to(ROOT))))
        except Exception as e:
            failures.append({'source':absolute,'error':repr(e)});m.log(absolute,repr(e))
        summary={'completed':completed,'preserved':failures,'finished':False,
                 'reclaimed_allocated_bytes':sum(r['source_allocated_bytes'] for r in completed),
                 'filesystem_before':start,'filesystem_now':shutil.disk_usage(ROOT)._asdict()}
        m.save(AUDIT/'SUMMARY.json',summary)
    summary.update(finished=True,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),filesystem_after=shutil.disk_usage(ROOT)._asdict())
    m.save(AUDIT/'SUMMARY.json',summary);m.save(BOS/'SUMMARY.json',summary)

if __name__ == '__main__':
    import sys
    qualify() if '--qualify-only' in sys.argv else main()
