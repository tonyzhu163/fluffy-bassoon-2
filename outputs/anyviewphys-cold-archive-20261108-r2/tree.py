"""Read-only exact tree inventory and content/metadata qualification helpers."""
from pathlib import Path
import collections, concurrent.futures, hashlib, json, os, stat, threading, time


def stamp(path):
    return stat_stamp(path.lstat())


def stat_stamp(s):
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
    initial=source.lstat();device=initial.st_dev
    def row(p,s):
        value=stat_stamp(s);assert s.st_dev==device,('Nested filesystem is not included',str(p))
        assert stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode) or stat.S_ISLNK(s.st_mode),('Unsupported file type',str(p))
        result=dict(path=p.relative_to(anchor).as_posix(),stamp=value,ctime_ns=s.st_ctime_ns)
        if stat.S_ISLNK(s.st_mode):result['target']=os.readlink(p)
        return result
    def directory(p):
        found=[];children=[]
        with os.scandir(p) as entries:
            for entry in entries:
                s=entry.stat(follow_symlinks=False);path=Path(entry.path);found.append(row(path,s))
                if stat.S_ISDIR(s.st_mode):children.append(path)
        return found,children
    rows=[row(source,initial)];todo=collections.deque([source]);pending=set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        while todo or pending:
            while todo and len(pending)<32:pending.add(pool.submit(directory,todo.popleft()))
            done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                found,children=future.result();rows.extend(found);todo.extend(children)
            if progress:progress(len(rows),0)
    return sorted(rows, key=lambda r: r['path'])


def hash_rows(rows, anchor, progress=None):
    groups={};cache={};total=finished=0;lock=threading.Lock()
    for row in rows:
        if stat.S_ISREG(row['stamp'][0]):groups.setdefault(tuple(row['stamp'][5:7]),row)
    def worker(item):
        nonlocal total,finished
        key,row=item;p=anchor/row['path'];s=p.lstat();assert stat_stamp(s)==row['stamp'] and s.st_ctime_ns==row['ctime_ns']
        def bump(count):
            nonlocal total
            with lock:
                total+=count
                if progress:progress(finished,total)
        digest=sha(p,bump);s=p.lstat();assert stat_stamp(s)==row['stamp'] and s.st_ctime_ns==row['ctime_ns']
        with lock:finished+=1
        return key,digest
    todo=collections.deque(groups.items());pending=set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        while todo or pending:
            while todo and len(pending)<8:pending.add(pool.submit(worker,todo.popleft()))
            done,pending=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                key,digest=future.result();cache[key]=digest
    for row in rows:
        if stat.S_ISREG(row['stamp'][0]):row['sha256']=cache[tuple(row['stamp'][5:7])]
    if progress: progress(len(rows), total)
    return rows


def same_metadata(a, b):
    return [{k:v for k,v in r.items() if k!='sha256'} for r in a] == [{k:v for k,v in r.items() if k!='sha256'} for r in b]


def verify(expected, actual):
    assert [r['path'] for r in expected] == [r['path'] for r in actual], 'Member coverage differs'
    left, right = {}, {}
    for a, b in zip(expected, actual):
        x, y = a['stamp'], b['stamp']
        assert x[0] == y[0] and x[2:5] == y[2:5], ('mode/mtime/owner', a['path'])
        if stat.S_ISREG(x[0]):
            assert x[1] == y[1] and a['sha256'] == b['sha256'], ('content', a['path'])
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
