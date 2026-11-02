"""Bounded parallel continuation; imports the immutable qualified first-phase mover."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import shutil
import time

BASE = Path('/mowenjie/fluffy-bassoon-2/outputs/baige-storage-tiering-20261103-r1')
spec = importlib.util.spec_from_file_location('first_phase_archive', BASE / 'archive.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.RUN = 'baige-storage-tiering-20261103-r2'
m.AUDIT = BASE.parent / m.RUN
m.BOS = m.BOS / 'parallel-r2'
m.STAGE = Path('/tmp') / m.RUN

# Removing a verified sibling environment may legitimately change a shared
# inode's hardlink count. Content, inode identity, mode, size, mtime and symlink
# target remain strict; GNU tar compares every archived file's actual bytes.
original_inventory = m.inventory
def inventory(source):
    rows = original_inventory(source)
    for row in rows:
        row[9] = None
    return rows
m.inventory = inventory


def main():
    assert Path('/mnt/bos/adgfyl').is_mount()
    for p in (m.AUDIT, m.BOS, m.STAGE):
        p.mkdir(exist_ok=True)
    old_plan = json.loads((BASE / 'PLAN.json').read_text())
    sources = old_plan['sources'][6:]
    # Prioritize large, entirely historical project groups.
    sources = ['experiments', 'strong-plant', 'fictional-chainsaw'] + [
        x for x in sources if x not in ('experiments', 'strong-plant', 'fictional-chainsaw')]
    plan = dict(old_plan, run_id=m.RUN, sources=sources,
                parent_phase=str(BASE), max_parallel_archives=3,
                phase_change='First phase retains only its first six data/checkpoint sources; this phase owns all later sources. Stop the first-phase driver after those six verified moves, before it can reclaim another source. No source overlap is allowed.',
                verification=old_plan['verification'] + ' Ignore sibling-induced hardlink-count changes only; preserve all other checks.')
    m.save(m.AUDIT / 'PLAN.json', plan)
    m.save(m.BOS / 'PLAN.json', plan)
    before = shutil.disk_usage('/mowenjie')
    m.save(m.AUDIT / 'SPACE-BEFORE.json', dict(before._asdict(), time=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    completed, preserved = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(m.archive, rel): rel for rel in sources}
        for future in concurrent.futures.as_completed(futures):
            rel = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                preserved.append({'source': rel, 'error': str(error)})
                m.log('SOURCE PRESERVED / NEEDS REVIEW ' + rel + ': ' + str(error))
            summary = dict(completed=completed, preserved=preserved,
                           free_now=shutil.disk_usage('/mowenjie').free,
                           before_free=before.free)
            m.save(m.AUDIT / 'SUMMARY.json', summary)
    summary['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    summary['after_free'] = shutil.disk_usage('/mowenjie').free
    m.save(m.AUDIT / 'SUMMARY.json', summary)
    m.save(m.BOS / 'SUMMARY.json', summary)
    m.log('PARALLEL PHASE FINISHED')


if __name__ == '__main__':
    main()
