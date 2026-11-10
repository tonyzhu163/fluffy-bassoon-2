"""Gate 1 table from finished jobs, one argument per job directory. Standard library only.

Per task: single-attempt success, best-of-R success, scenes that always fail or always
succeed, mixed scenes, and the scene intraclass correlation.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path


def summarize(scenes, repeats):
    counts = []
    for scene, group in sorted(scenes.items()):
        group = sorted(group, key=lambda r: r['repeat'])
        assert [r['repeat'] for r in group] == list(range(repeats)), (scene, 'incomplete repeats')
        assert all(r['noise_seed'] == scene + 700000 + 1000 * r['repeat'] for r in group)
        assert len({(r['initial_sha256'], r['initial_observation_sha256']) for r in group}) == 1
        counts.append(sum(r['success'] for r in group))
    n, k = len(counts), repeats
    p = sum(counts) / (n * k)
    between = k * sum((c / k - p) ** 2 for c in counts) / (n - 1)
    within = sum(c * (1 - c / k) for c in counts) / (n * (k - 1))
    denom = between + (k - 1) * within
    return dict(scenes=n, repeats=k, successes_by_scene=counts, single_attempt_rate=round(p, 4),
                best_of_repeats_rate=round(sum(c > 0 for c in counts) / n, 4),
                always_fail=counts.count(0), always_succeed=counts.count(k),
                mixed=sum(0 < c < k for c in counts),
                expected_always_fail_if_independent=round(n * (1 - p) ** k, 2),
                icc=round((between - within) / denom, 3) if denom else None)


def main(directories):
    episodes = defaultdict(lambda: defaultdict(list))
    repeats = {}
    for directory in map(Path, directories):
        assert json.loads((directory / 'exit.json').read_text())['exit_code'] == 0, directory
        probe = json.loads((directory / 'gpu.json').read_text())
        assert probe['status'] == 'passed' and len(probe['devices']) == 1
        rows = [json.loads(line) for line in (directory / 'events.jsonl').read_text().splitlines()]
        assert rows[-1]['kind'] == 'complete' and rows[-1]['check'] == 'gate1', directory
        repeats.setdefault(rows[-1]['task'], rows[-1]['repeats'])
        assert repeats[rows[-1]['task']] == rows[-1]['repeats']
        for row in rows:
            if row['kind'] == 'episode':
                assert (directory / row['archive']).stat().st_size == row['archive_bytes']
                episodes[row['task']][row['scene']].append(row)
    print(json.dumps({task: dict(**summarize(scenes, repeats[task]),
                                 controls=sum(r['controls'] for g in scenes.values() for r in g))
                      for task, scenes in sorted(episodes.items())}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1:])
