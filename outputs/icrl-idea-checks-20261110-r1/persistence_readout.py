"""Rebuild the persistence panel from qualified jobs, one argument per job directory."""
import json
import sys
from collections import defaultdict
from pathlib import Path


def summarize(scenes):
    counts = []
    for scene in range(983000, 983008):
        group = sorted(scenes[scene], key=lambda r: r["repeat"])
        assert [r["repeat"] for r in group] == list(range(4))
        assert all(r["noise_seed"] == scene + 700000 + 1000 * r["repeat"] for r in group)
        assert len({(r["initial_sha256"], r["initial_observation_sha256"]) for r in group}) == 1
        counts.append(sum(r["success"] for r in group))
    assert len(scenes) == 8
    n, k = len(counts), 4
    p = sum(counts) / (n * k)
    between = k * sum((c / k - p) ** 2 for c in counts) / (n - 1)
    within = sum(c * (1 - c / k) for c in counts) / (n * (k - 1))
    denom = between + (k - 1) * within
    return dict(successes_by_scene=counts, successes=sum(counts), episodes=n*k,
                zero_of_four=counts.count(0), four_of_four=counts.count(4),
                expected_zero_if_independent=round(n * (1 - p) ** k, 2),
                icc=round((between - within) / denom, 3) if denom else None)


def main(directories):
    episodes = defaultdict(lambda: defaultdict(list))
    candidates = defaultdict(list)
    for directory in map(Path, directories):
        assert json.loads((directory / "exit.json").read_text())["exit_code"] == 0
        probe = json.loads((directory / "gpu.json").read_text())
        assert probe["status"] == "passed" and len(probe["devices"]) == 1
        rows = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        assert rows[-1]["kind"] == "complete" and rows[-1]["check"] == "persistence"
        for row in rows:
            if row["kind"] == "episode":
                episodes[row["task"]][row["scene"]].append(row)
            elif row["kind"] == "reset_check":
                candidates[row["task"]].append(row)
    results = {}
    for task, scenes in sorted(episodes.items()):
        selected = [r for r in candidates[task] if r["selected"]]
        for scene, group in scenes.items():
            starts = [r for r in selected if r["scene"] == scene]
            assert sorted(r["repeat"] for r in starts) == sorted(r["repeat"] for r in group)
            assert len({r["signature"] for r in starts}) == 1
        results[task] = dict(**summarize(scenes), reset_candidates=len(candidates[task]),
                             selected_starts=len(selected),
                             controls=sum(r["controls"] for group in scenes.values() for r in group))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
