"""Audit only the human-selected closed experiment folders before bulk pruning."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import queue
import stat
import threading
import time

ROOT = Path('/mowenjie')
BASE = ROOT / 'fluffy-bassoon-2/outputs/robocasa-retention-audit-20261105-r1'
OLD = ROOT / 'fluffy-bassoon-2/outputs/experiment-bundles-cold-archive-20261104-r1'
SECRET = {'.secrets', '.ssh', '.aws', '.bce', '.netrc', '.env'}
ARRAYS = {'.npz', '.npy', '.pkl', '.pickle', '.h5', '.hdf5', '.arrow', '.parquet'}
WEIGHTS = {'.pt', '.ckpt', '.safetensors'}
VIDEO = {'.mp4', '.avi', '.mkv', '.webm', '.mov'}
IMAGE = {'.png', '.jpg', '.jpeg', '.webp', '.gif'}
RAW_CONTEXT = {'corpus', 'replay', 'buffer', 'replay_buffer', 'rollouts', 'trajectories',
               'chunks', 'observations', 'images', 'frames', 'rgb', 'videos'}


def save(path, data):
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def bulk_reason(relative, size):
    """Keep text, unknown formats, symlinks and small numerical result records."""
    p = Path(relative)
    cache_parts = set(p.parts[:-1])
    if ('torch_compile_cache' in cache_parts or 'triton' in cache_parts or 'flashinfer' in cache_parts) and any('cache' in q for q in p.parts[:-1]):
        return 'regenerable run-local compiler cache'
    suffix = p.suffix.lower()
    parts = {v.lower() for v in p.parts[:-1]}
    if p.name.endswith('.pt.gz') and 'environment' in p.parts and 'chunk' in p.name:
        return 'retired compressed evaluation environment chunk'
    if suffix in WEIGHTS:
        return 'retired checkpoint or policy tensor'
    if suffix == '.pth' and size > 1 << 20:
        return 'retired model payload; small Python path configuration stays'
    if suffix in ARRAYS:
        result_name = any(x in p.stem.lower() for x in ('result', 'metric', 'reference', 'qualification', 'initial_state', 'seed'))
        if not parts.intersection(RAW_CONTEXT) and size <= 2 << 20 and result_name:
            return None
        return 'raw trajectory, replay, training array or serialized data'
    if suffix in VIDEO:
        return 'retired generated video'
    if (p.name.lower().endswith(('.pkl.gz', '.pickle.gz'))
            and parts.intersection({'raw', 'rpc', 'traces', 'corpus', 'replay', 'buffer', 'replay_buffer'})):
        return 'retired compressed raw RPC or trajectory snapshot'
    if suffix in IMAGE and parts.intersection(RAW_CONTEXT):
        return 'raw observation or video frame'
    if suffix in {'.pyc', '.pyo'}:
        return 'regenerable Python bytecode'
    if suffix in {'.dat', '.mmap', '.memmap'} and parts.intersection(RAW_CONTEXT):
        return 'raw replay backing store'
    if suffix == '.bin' and (parts.intersection({'models', 'weights', 'checkpoints', 'policies'}) or 'model' in p.name.lower()):
        return 'retired model binary'
    return None


def representative(relative, size):
    p = Path(relative)
    if any(v.lower() in RAW_CONTEXT for v in p.parts[:-1]):
        return False
    if size > 2 << 20:
        return False
    name = p.name.lower()
    return (p.suffix.lower() == '.md' and any(s in name for s in ('result', 'evaluat', 'interpret', 'failure', 'method', 'summary'))
            or name in {'summary.json', 'result.json', 'results.json', 'evaluation.json', 'config.json', 'contract.json'}
            or name in {'episodes.jsonl', 'progress.jsonl', 'updates.jsonl', 'metrics.jsonl', 'rates.csv'})


def excerpt(path, size):
    with path.open('rb') as stream:
        start = stream.read(min(size, 4096))
        if size > 8192:
            stream.seek(-4096, 2)
            tail = stream.read()
        else:
            tail = stream.read()
    return (start + (b'\n[... middle retained in original file ...]\n' if size > 8192 else b'') + tail).decode(errors='replace')


def main():
    plan = json.loads((BASE / 'PLAN.json').read_text())
    cold = {}
    known = {}
    rows = {r['source']: [] for r in plan['sources']}
    notes = {r['source']: {'code': [], 'reports': [], 'protected': [], 'errors': []} for r in plan['sources']}
    roots = {}
    work = queue.Queue()
    device = ROOT.stat().st_dev
    for source in rows:
        p = Path(source)
        assert p.parent in (ROOT / 'fluffy-bassoon/outputs', ROOT / 'fluffy-bassoon-2/outputs')
        assert p.name.startswith('robocasa') and p.name not in plan['protected_roots']
        if not p.exists():
            continue
        if source in cold:
            assert not p.exists()
            continue
        assert p.is_dir() and not p.is_symlink()
        s = p.lstat()
        assert s.st_dev == device
        roots[source] = [s.st_dev, s.st_ino, s.st_mtime_ns]
        work.put((source, p))

    def worker():
        while True:
            item = work.get()
            try:
                if item is None:
                    return
                source, p = item
                relative = p.relative_to(source)
                s = p.lstat()
                assert s.st_dev == device
                kind = 'keep'
                reason = None
                if any(part in SECRET for part in relative.parts):
                    notes[source]['protected'].append(str(relative))
                elif stat.S_ISDIR(s.st_mode):
                    for child in p.iterdir():
                        work.put((source, child))
                elif stat.S_ISREG(s.st_mode):
                    reason = bulk_reason(relative, s.st_size)
                    if reason:
                        kind = 'bulk'
                    elif p.suffix.lower() in ('.py', '.sh') and s.st_size <= 1 << 20:
                        data = p.read_bytes()
                        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
                        code = {'relative': str(relative), 'bytes': s.st_size, 'git_blob': blob,
                                'sha256': hashlib.sha256(data).hexdigest()}
                        if blob in known:
                            code.update(recoverable=known[blob])
                            kind, reason = 'one_off_code', 'byte-identical old orchestration code recoverable from pinned Git'
                        notes[source]['code'].append(code)
                    if kind == 'keep' and representative(relative, s.st_size):
                        notes[source]['reports'].append({'relative': str(relative), 'bytes': s.st_size,
                                                        'excerpt': excerpt(p, s.st_size)})
                elif not stat.S_ISLNK(s.st_mode):
                    notes[source]['protected'].append(str(relative))
                # relative, mode, size, time, allocated blocks, dev, inode, category
                rows[source].append([str(relative), s.st_mode, s.st_size, s.st_mtime_ns,
                                     s.st_blocks, s.st_dev, s.st_ino, kind, reason])
            except Exception as error:
                if item is not None:
                    notes[item[0]]['errors'].append([str(item[1]), repr(error)])
            finally:
                work.task_done()

    threads = [threading.Thread(target=worker) for _ in range(24)]
    for thread in threads:
        thread.start()
    work.join()
    for _ in threads:
        work.put(None)
    for thread in threads:
        thread.join()
    out = BASE / 'private-inventories'
    out.mkdir(exist_ok=False)
    reviews = BASE / 'private-reviews'
    reviews.mkdir(exist_ok=False)
    summaries = []
    for entry in plan['sources']:
        source = entry['source']
        key = source.removeprefix('/mowenjie/').replace('/', '__')
        if source in cold:
            receipt = cold[source]
            summaries.append(dict(entry, status='full_bundle_already_in_BOS', archive=receipt['archive'],
                                  sha256=receipt['sha256'], source_payload_bytes=receipt['source_allocated_bytes']))
            continue
        if source not in roots:
            summaries.append(dict(entry,status='absent_no_scan'))
            continue
        records = sorted(rows[source])
        save(out / (key + '.json'), {'source': source, 'root_identity': roots[source], 'rows': records})
        save(reviews / (key + '.json'), notes[source])
        bulk = [r for r in records if r[7] == 'bulk']
        code = [r for r in records if r[7] == 'one_off_code']
        kept = [r for r in records if r[7] == 'keep' and stat.S_ISREG(r[1])]
        types = Counter()
        for r in bulk:
            types[Path(r[0]).suffix.lower()] += r[4] * 512
        summaries.append(dict(entry, status='audited_hot_source', root_identity=roots[source],
                              inventory=str(out / (key + '.json')), review=str(reviews / (key + '.json')),
                              entries=len(records), bulk_files=len(bulk), bulk_allocated_bytes=sum(r[4] for r in bulk) * 512,
                              bulk_logical_bytes=sum(r[2] for r in bulk), bulk_by_extension=dict(types),
                              recoverable_one_off_files=len(code), retained_files=len(kept),
                              retained_allocated_bytes=sum(r[4] for r in kept) * 512,
                              code_files=len(notes[source]['code']), result_records_read=len(notes[source]['reports']),
                              protected=notes[source]['protected'], errors=notes[source]['errors'],
                              retained_large_files=[{'relative': r[0], 'bytes': r[2]} for r in kept if r[2] > 16 << 20]))
    result = {'complete': True, 'checked_shanghai': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'sources': summaries, 'bulk_allocated_bytes': sum(r.get('bulk_allocated_bytes', 0) for r in summaries),
              'bulk_files': sum(r.get('bulk_files', 0) for r in summaries),
              'retained_allocated_bytes': sum(r.get('retained_allocated_bytes', 0) for r in summaries),
              'policy': plan['retention_policy'], 'no_active_job_folder_scan': True, 'GPU_jobs_launched': 0}
    save(BASE / 'AUDIT.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'policy')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
