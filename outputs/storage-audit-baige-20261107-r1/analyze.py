"""Summarize the completed metadata census; no remote or payload operations."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
raw = HERE / 'census.json'
data = json.loads(raw.read_text())
assert data['complete']
a = data['allocated_bytes_by_prefix']
projects = {'fluffy-bassoon-2', 'fluffy-bassoon', 'anyviewphys', 'worldrank'}
categories = Counter()
pieces = []


def classify(path):
    parts = path.split('/')
    leaf = parts[-1]
    if len(parts) == 2 and parts[0] in projects:
        if leaf in ('outputs', 'runs'):
            return 'Experiment outputs'
        if leaf in ('inputs', 'models', 'data'):
            return 'Models and datasets'
        if leaf in ('envs', 'setups'):
            return 'Environments and simulator setup'
        if leaf in ('cache', 'caches'):
            return 'Caches'
        if leaf in ('repo', 'worktrees', 'sources', 'deps', 'third_party', 'tools', 'snapshots'):
            return 'Code, worktrees, source dependencies and tools'
    if parts[0] in ('.cache', '.uv-cache', '.pytest_cache'):
        return 'Caches'
    if parts[0] in ('models', 'data'):
        return 'Models and datasets'
    if parts[0] in ('envs', 'libs', '.uv-python', '.uv-tools'):
        return 'Environments and simulator setup'
    return 'Other files and metadata'


for path, size in a.items():
    if path == '.' or '/' in path:
        continue
    if path in projects:
        children = [(p, b) for p, b in a.items() if p.startswith(path + '/') and p.count('/') == 1]
        for child, amount in children:
            category = classify(child)
            categories[category] += amount
            pieces.append({'path': child, 'allocated_bytes': amount, 'category': category})
        remainder = size - sum(b for _, b in children)
        categories['Other files and metadata'] += remainder
    else:
        category = classify(path)
        categories[category] += size
        pieces.append({'path': path, 'allocated_bytes': size, 'category': category})
total = sum(b for p, b in a.items() if p != '.' and '/' not in p)
assert sum(categories.values()) == total
runs = []
for path, size in a.items():
    parts = path.split('/')
    if len(parts) == 3 and parts[0] in projects and parts[1] in ('outputs', 'runs'):
        if parts[2] == 'archive':
            runs.extend({'path': p, 'allocated_bytes': b} for p, b in a.items()
                        if p.startswith(path + '/') and p.count('/') == 3)
        else:
            runs.append({'path': path, 'allocated_bytes': size})
runs.sort(key=lambda r: r['allocated_bytes'], reverse=True)
old = json.loads((HERE.parent / 'storage-audit-baige-20261104-r1/REMAINING-SUMMARY.json').read_text())
prior = old['allocated_bytes_by_prefix']
deltas = sorted(({'path': row['path'], 'now_bytes': row['allocated_bytes'],
                  'nov4_bytes': prior.get(row['path'], 0),
                  'delta_bytes': row['allocated_bytes'] - prior.get(row['path'], 0)}
                 for row in pieces), key=lambda r: r['delta_bytes'], reverse=True)
duplicates = defaultdict(dict)
for blocks, path, size, links, mtime, inode in data['files_at_least_256MiB']:
    duplicates[(Path(path).name, size)].setdefault(inode, (blocks, path, links))
potential = []
for (name, size), inodes in duplicates.items():
    if len(inodes) > 1:
        values = sorted(inodes.values(), reverse=True)
        potential.append({'basename': name, 'logical_bytes_each': size,
                          'distinct_inodes': len(inodes),
                          'allocation_beyond_one_copy_bytes': sum(v[0] for v in values[1:]),
                          'paths': [v[1] for v in values],
                          'qualification': 'Same basename and size only; contents may differ. Not a reclaim estimate.'})
potential.sort(key=lambda r: r['allocation_beyond_one_copy_bytes'], reverse=True)
summary = {k: data[k] for k in ('started_unix', 'finished_unix', 'elapsed_seconds',
                                'entries_scanned', 'errors_count', 'errors',
                                'shared_inodes_counted_once', 'filesystem_before', 'filesystem_after')}
summary.update(unique_allocated_bytes=total,
               categories_bytes=dict(categories),
               top_level_bytes={p: b for p, b in a.items() if p != '.' and '/' not in p},
               category_components=sorted(pieces, key=lambda r: r['allocated_bytes'], reverse=True),
               largest_runs=runs, nov4_component_deltas=deltas,
               potential_same_name_size_groups=potential,
               raw_census_bytes=raw.stat().st_size,
               raw_census_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
               reconciliation_gap_bytes=data['filesystem_after']['used'] - total,
               note=data['note'])
refresh_path = HERE / 'volatile-large-file-refresh.json'
if refresh_path.exists():
    refresh = json.loads(refresh_path.read_text())
    change = sum(r['old_allocated'] - r['now_allocated'] for r in refresh['changes'])
    adjusted_categories = dict(categories)
    adjusted_categories['Experiment outputs'] -= change
    adjusted_roots = dict(summary['top_level_bytes'])
    adjusted_roots['fluffy-bassoon'] -= change
    summary['post_scan_refresh'] = {
        'receipt': refresh_path.name,
        'captured_unix': refresh['captured_unix'],
        'paths_checked': refresh['paths_checked'],
        'changes': refresh['changes'],
        'allocation_reduction_bytes': change,
        'adjusted_unique_allocated_bytes': total - change,
        'adjusted_categories_bytes': adjusted_categories,
        'adjusted_top_level_bytes': adjusted_roots,
        'note': 'Refresh of original-repository output files >=256MiB, after the full census. '
                'Original raw census and directory CSV remain unchanged. This is still a moving window.'}
(HERE / 'SUMMARY.json').write_text(json.dumps(summary, indent=2) + '\n')
with (HERE / 'DIRECTORIES.csv').open('w') as f:
    writer = csv.writer(f)
    writer.writerow(['path_under_mowenjie', 'unique_allocated_GiB', 'logical_path_GiB', 'entries'])
    for path, amount in sorted(a.items(), key=lambda row: row[1], reverse=True):
        if path != '.':
            writer.writerow([path, round(amount/2**30, 6), round(data['logical_bytes_by_prefix'].get(path, 0)/2**30, 6), data['entries_by_prefix'].get(path, 0)])
for label, amount in categories.most_common():
    print(f'{amount/2**30:10.2f} GiB {label}')
print('MEASURED_GiB', total/2**30, 'GAP_GiB', summary['reconciliation_gap_bytes']/2**30)
for row in runs[:30]:
    print(f"{row['allocated_bytes']/2**30:10.2f} GiB {row['path']}")
