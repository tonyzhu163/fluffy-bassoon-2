"""Score fixed actions, per-scene upper bounds and repeat-held-out choices."""
from pathlib import Path
import json
import sys

ARMS = ("native", "wait", "base_y_minus", "base_y_plus", "base_x_plus")
SCENES = tuple(range(983000, 983008))
REPEATS = (0, 1)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main(paths):
    grid = {}
    root = Path(__file__).resolve().parent
    reference = rows(root.parent / "icrl-idea-checks-20261110-r1/closedrawer02/events.jsonl")
    prior = {(r["scene"], r["repeat"]): r for r in reference if r["kind"] == "episode"}
    for path in map(Path, paths):
        assert json.loads((path / "exit.json").read_text())["exit_code"] == 0
        probe = json.loads((path / "gpu.json").read_text())
        assert probe["status"] == "passed" and len(probe["devices"]) == 1
        events = rows(path / "events.jsonl")
        assert events[-1]["kind"] == "complete"
        for row in events:
            if row["kind"] != "episode":
                continue
            scene, repeat, arm = row["scene"], row["repeat"], row["arm"]
            key = (scene, repeat, arm)
            assert key not in grid
            old = prior[scene, repeat]
            assert all(old[field] == row[field] for field in
                       ("initial_sha256", "initial_observation_sha256", "noise_seed", "task"))
            if arm == "native":
                assert old["success"] == row["success"]
            grid[key] = row
    assert set(grid) == {(s, r, a) for s in SCENES for r in REPEATS for a in ARMS}

    def wins(scene, repeat, arm):
        return int(grid[scene, repeat, arm]["success"])

    def best(scenes, repeats):
        return max(ARMS, key=lambda arm: sum(wins(s, r, arm) for s in scenes for r in repeats))

    fixed = {}
    for arm in ARMS:
        fixed[arm] = dict(successes=sum(wins(s, r, arm) for s in SCENES for r in REPEATS),
                          gains=sum(wins(s,r,arm) and not wins(s,r,"native") for s in SCENES for r in REPEATS),
                          losses=sum(wins(s,r,"native") and not wins(s,r,arm) for s in SCENES for r in REPEATS))
    per_scene = []
    transferred = [0, 0]
    oracle = 0
    for s in SCENES:
        both = best([s], REPEATS)
        oracle += sum(wins(s,r,both) for r in REPEATS)
        choices = [best([s], [1-r]) for r in REPEATS]
        for r in REPEATS:
            transferred[r] += wins(s, r, choices[r])
        per_scene.append(dict(scene=s, side=grid[s,0,"native"]["side"], fixture=grid[s,0,"native"]["fixture"],
                              outcomes={a:[wins(s,r,a) for r in REPEATS] for a in ARMS},
                              best_on_both=both, selected_on_other_repeat=choices))
    fixed_on_other = [best(SCENES,[1-r]) for r in REPEATS]
    result = dict(episodes=len(grid),controls=sum(r["controls"] for r in grid.values()),
                  native_exact_terminals=sum(grid[s,r,"native"]["terminal_sha256"]==prior[s,r]["terminal_sha256"]
                                             for s in SCENES for r in REPEATS), fixed=fixed,
                  per_scene_oracle_successes=oracle, per_scene_other_repeat_successes=transferred,
                  fixed_on_other_repeat=fixed_on_other,
                  fixed_other_repeat_successes=[sum(wins(s,r,fixed_on_other[r]) for s in SCENES) for r in REPEATS],
                  scenes=per_scene)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
