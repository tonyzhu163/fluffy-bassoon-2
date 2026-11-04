"""Plan and retire the remaining reviewed serialized observation/RPC payloads."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import threading
import time

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/legacy-experiment-retention-audit-20261104-r2')
REASON = 'obsolete serialized observations or raw RPC journal; compact outcome records retained'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def eligible(relative):
    p = Path(relative)
    name = p.name.lower()
    if not name.endswith(('.pkl.gz', '.pickle.gz')):
        return False
    if any(word in name for word in ('recovery', 'checkpoint', 'selected', 'best')):
        return False
    return (name.startswith('episode-') or name in {'trace.pkl.gz', 'trajectory.pkl.gz'}
            or any(any(word in part.lower() for word in ('rpc', 'journal', 'raw', 'traces', 'replay')) for part in p.parts[:-1]))


def make_plan():
    assert not (BASE / 'ADDITIONAL-TRACE-PLAN.json').exists()
    sources = []
    unclassified = []
    for entry in read(BASE / 'AUDIT.json')['sources']:
        if entry['status'] != 'audited_hot_source':
            continue
        inventory = read(Path(entry['inventory']))
        selected = []
        for row in inventory['rows']:
            if row[7] != 'keep' or not stat.S_ISREG(row[1]) or not row[0].endswith('.gz'):
                continue
            if eligible(row[0]):
                selected.append(row[:7] + ['bulk', REASON])
            else:
                unclassified.append({'source': entry['source'], 'relative': row[0], 'bytes': row[2]})
        if selected:
            sources.append({'source': entry['source'], 'inventory': entry['inventory'],
                            'root_identity': entry['root_identity'], 'rows': selected})
    result = {'sources': sources, 'files': sum(len(r['rows']) for r in sources),
              'allocated_payload_bytes': sum(row[4] * 512 for r in sources for row in r['rows']),
              'logical_bytes': sum(row[2] for r in sources for row in r['rows']),
              'unclassified_compressed_files_retained': unclassified,
              'basis': 'Observed gzip Pickle protocol5 trajectory/RPC data in physical-journal, raw-rpc and episode trace layouts. Scalar outcomes, summaries, configs, logs, cited recovery checkpoints and compact evaluation context/RNG records stay.'}
    save(BASE / 'private-additional-trace-plan.json', result)
    save(BASE / 'ADDITIONAL-TRACE-PLAN.json', {k: v for k, v in result.items() if k != 'sources'})
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'unclassified_compressed_files_retained')}, indent=2))


def apply():
    main = read(BASE / 'PRUNE-SUMMARY.json')
    assert main['finished'] and not main['preserved'] and not main['skipped_sources']
    assert read(BASE / 'COLD-RECOVERY.json')['finished']
    review = read(BASE / 'ADDITIONAL-TRACE-REVIEW.json')
    data = (BASE / 'private-additional-trace-plan.json').read_bytes()
    assert review['review_complete'] and hashlib.sha256(data).hexdigest() == review['plan_sha256']
    plan = json.loads(data)
    spec = importlib.util.spec_from_file_location('qualified_prune', BASE / 'prune.py')
    prune = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prune)
    assert hashlib.sha256((BASE / 'prune.py').read_bytes()).hexdigest() == review['primitive_sha256']
    # Confirm the narrow extension/layout classifier does not admit code,
    # logs, evidence archives or model recovery snapshots.
    assert eligible('experiment/physical-journal/123-input.pkl.gz')
    assert eligible('experiment/raw-rpc/123-inference.pkl.gz')
    assert eligible('experiment/case-00/native/episode-001.pkl.gz')
    assert not eligible('thin-evidence.tar.gz') and not eligible('console.log.gz')
    assert not eligible('experiment/raw/recovery.pkl.gz') and not eligible('unknown.py')
    roots = {}
    stopping = {}
    groups = []
    skipped = []
    for entry in plan['sources']:
        source = entry['source']
        p = Path(source)
        assert p.parent in (Path('/mowenjie/fluffy-bassoon/outputs'), Path('/mowenjie/fluffy-bassoon-2/outputs'))
        assert not p.is_symlink()
        s = p.lstat()
        assert (s.st_dev, s.st_ino) == tuple(entry['root_identity'][:2])
        busy = prune.users(p)
        if busy:
            skipped.append({'source': source, 'users': busy})
            continue
        rootfd = os.open(p, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        assert (os.fstat(rootfd).st_dev, os.fstat(rootfd).st_ino) == (s.st_dev, s.st_ino)
        roots[source] = rootfd
        stopping[source] = threading.Event()
        inventory = read(Path(entry['inventory']))
        directories = {r[0]: r for r in inventory['rows'] if stat.S_ISDIR(r[1])}
        parents = defaultdict(list)
        for row in entry['rows']:
            assert eligible(row[0]) and row[7:] == ['bulk', REASON]
            parents[str(Path(row[0]).parent)].append(row)
        groups.extend((source, parent, rows, directories) for parent, rows in parents.items())
    groups.sort(key=lambda g: sum(r[2] for r in g[2]), reverse=True)
    summary = {'finished': False, 'groups': len(groups), 'groups_completed': 0, 'files_deleted': 0,
               'deleted_logical_bytes': 0, 'deleted_allocated_payload_bytes': 0,
               'preserved': [], 'skipped_sources': skipped, 'GPU_jobs_launched': 0,
               'no_active_job_folder_scan': True}
    save(BASE / 'ADDITIONAL-TRACE-SUMMARY.json', summary)
    last = time.monotonic()
    with (BASE / 'private-additional-trace-receipts.jsonl').open('x') as receipts:
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {}
            for source, parent, rows, directories in groups:
                classifier = lambda relative, size: REASON if eligible(relative) else None
                future = pool.submit(prune.delete_group, source, roots[source], parent, rows, directories,
                                     stopping[source], set(), classifier)
                futures[future] = (source, parent)
            for future in as_completed(futures):
                source, parent = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    stopping[source].set()
                    result = {'source': source, 'parent': parent, 'files': 0, 'logical_bytes': 0,
                              'allocated_payload_bytes': 0, 'preserved': [[parent, repr(error)]]}
                receipts.write(json.dumps(result) + '\n')
                summary['groups_completed'] += 1
                summary['files_deleted'] += result['files']
                summary['deleted_logical_bytes'] += result['logical_bytes']
                summary['deleted_allocated_payload_bytes'] += result['allocated_payload_bytes']
                if result['preserved']:
                    summary['preserved'].append(result)
                if time.monotonic() - last > 2:
                    receipts.flush(); os.fsync(receipts.fileno())
                    summary['checked_shanghai'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
                    save(BASE / 'ADDITIONAL-TRACE-SUMMARY.json', summary)
                    last = time.monotonic()
        receipts.flush(); os.fsync(receipts.fileno())
    for fd in roots.values():
        os.close(fd)
    summary.update(finished=True, checked_shanghai=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    save(BASE / 'ADDITIONAL-TRACE-SUMMARY.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    make_plan() if '--plan-only' in sys.argv else apply()
