"""Within-scene repeatability of success in the audited Stage 1 AD readout.

    tar -xzf ../robocasa-three-task-kv-velocity-meta-20261105-r1/monitor/STAGE1-METADATA22.tar.gz -C DIR
    python stage1_repeatability.py DIR
"""
import collections
import glob
import json
import sys

rows = collections.defaultdict(lambda: collections.defaultdict(dict))
for path in glob.glob(sys.argv[1] + '/readouts/*/ad/*/*/*/bank-*/RESULT.json'):
    data = json.load(open(path))
    for e in data['episodes']:
        rows[data['interface'], data['mode'], data['task']][e['case']['native_seed']][e['attempt']] = bool(e['success_once'])
print('interface mode task cases rate zero-of-six six-of-six ICC')
for key in sorted(rows):
    counts = [sum(v.values()) for v in rows[key].values() if len(v) == 6]
    n, p = len(counts), sum(counts) / (6 * len(counts))
    between = 6 * sum((k / 6 - p) ** 2 for k in counts) / (n - 1)
    within = sum(k * (1 - k / 6) ** 2 + (6 - k) * (k / 6) ** 2 for k in counts) / (n * 5)
    print(*key, n, round(p, 2), counts.count(0), counts.count(6), round((between - within) / (between + 5 * within), 2))
