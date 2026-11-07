"""Read-only exact tree inventory and content/metadata qualification helpers."""
from pathlib import Path
import hashlib, json, os, stat, time


def stamp(path):
    s = path.lstat()
    return [s.st_mode, s.st_size, s.st_mtime_ns, s.st_uid, s.st_gid,
            s.st_dev, s.st_ino, s.st_nlink, s.st_blocks * 512]


def sha(path, progress=None):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
            if progress: progress(len(block))
    return h.hexdigest()


def save(path, obj):
    tmp = path.with_name(path.name+'.writing-'+str(os.getpid()))
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def scan(source, anchor, progress=None):
    rows, stack = [], [source]; device = source.lstat().st_dev
    while stack:
        p = stack.pop(); s = stamp(p)
        assert s[5] == device, ('Nested filesystem is not included', str(p))
        assert stat.S_ISREG(s[0]) or stat.S_ISDIR(s[0]) or stat.S_ISLNK(s[0]), ('Unsupported file type', str(p))
        row = dict(path=p.relative_to(anchor).as_posix(), stamp=s, ctime_ns=p.lstat().st_ctime_ns)
        if stat.S_ISLNK(s[0]): row['target'] = os.readlink(p)
        if stat.S_ISDIR(s[0]): stack.extend(p.iterdir())
        rows.append(row)
        if progress: progress(len(rows), 0)
    return sorted(rows, key=lambda r: r['path'])


def hash_rows(rows, anchor, progress=None):
    cache = {}; total = 0
    for n, row in enumerate(rows):
        if not stat.S_ISREG(row['stamp'][0]): continue
        p = anchor / row['path']; assert stamp(p) == row['stamp'] and p.lstat().st_ctime_ns == row['ctime_ns']
        key = tuple(row['stamp'][5:7])
        if key not in cache:
            def bump(count):
                nonlocal total
                total += count
                if progress: progress(n+1, total)
            cache[key] = sha(p, bump)
            assert stamp(p) == row['stamp'] and p.lstat().st_ctime_ns == row['ctime_ns']
        row['sha256'] = cache[key]
    if progress: progress(len(rows), total)
    return rows


def same_metadata(a, b):
    return [{k:v for k,v in r.items() if k!='sha256'} for r in a] == [{k:v for k,v in r.items() if k!='sha256'} for r in b]


def verify(expected, actual):
    assert [r['path'] for r in expected] == [r['path'] for r in actual], 'Member coverage differs'
    left, right = {}, {}
    for a, b in zip(expected, actual):
        x, y = a['stamp'], b['stamp']; assert x[:5] == y[:5], ('mode/size/mtime/owner', a['path'])
        if stat.S_ISREG(x[0]):
            assert a['sha256'] == b['sha256'], ('content', a['path'])
            left.setdefault(tuple(x[5:7]), []).append(a['path']); right.setdefault(tuple(y[5:7]), []).append(b['path'])
        elif stat.S_ISLNK(x[0]): assert a['target'] == b['target']
    assert sorted(sorted(v) for v in left.values()) == sorted(sorted(v) for v in right.values()), 'Internal hardlink topology differs'


if __name__ == '__main__':
    out = Path(__file__).resolve().parent; source = Path('/mowenjie/anyviewphys'); last = [0.0]
    def progress(count, size):
        if time.monotonic()-last[0] > 5:
            save(out/'METADATA-STATE.json', dict(pid=os.getpid(), entries=count, checked_unix=time.time())); last[0]=time.monotonic()
    assert os.uname().sysname == 'Linux'
    rows = scan(source, source.parent, progress); save(out/'METADATA.json', rows)
    seen=set(); allocation=logical=0
    for row in rows:
        s=row['stamp']; key=tuple(s[5:7])
        if key in seen:continue
        seen.add(key); allocation+=s[8]
        if stat.S_ISREG(s[0]):logical+=s[1]
    save(out/'METADATA-SUMMARY.json', dict(entries=len(rows), unique_inodes=len(seen), allocated_bytes=allocation,
         unique_regular_logical_bytes=logical, completed_unix=time.time(), inventory_sha256=sha(out/'METADATA.json')))
