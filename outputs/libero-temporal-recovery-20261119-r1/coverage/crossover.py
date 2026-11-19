"""Read the saved physical menus. Run from any directory and redirect stdout to counts.json."""
import json
from collections import defaultdict
from pathlib import Path

outputs = Path(__file__).resolve().parents[2]
menus = {}
modes = ["default", "paced", "short_wait", "gripper_wait"]
timings = ["BASE", "S20"]


def load(study, name):
    path = outputs / f"libero-{study}-20261119-r1" / name
    if name.endswith(".jsonl"):
        return [json.loads(line) for line in path.read_text().splitlines()]
    return json.loads(path.read_text())


def summarize(name, groups):
    strict, pair_only = {}, {}
    reachable = 0
    for key, conditions in sorted(groups.items()):
        assert len(conditions) == 2
        a, b = map(set, conditions.values())
        reachable += bool(a and b)
        if a - b and b - a:
            target = pair_only if a & b else strict
            target[key] = conditions
    menus[name] = dict(groups=len(groups), reachable_both=reachable, strict=strict,
                       strict_count=len(strict), pair_only_with_common_winner=pair_only)


def cadence_pairs(rows, fresh=False):
    groups = defaultdict(dict)
    for row in rows:
        if fresh:
            key = f"{row['suite']}-{row['task']}-p7"
            cond, hits = row["timing"], row["hits"]
        else:
            key = f"{row['suite']}-{row['task_id']}-r{row['repeat']}"
            cond, hits = row["variant"], row
        groups[key][cond] = [m for m in modes[:2] if hits[m]]
    return groups


summarize("original two-cadence development",
          cadence_pairs(load("retry-headroom", "paired01-summary.json")["trials"]))
archive = load("first-controls", "paired-evidence.json")
for split, data in archive.items():
    groups = {}
    for row in data["rows"]:
        groups[row["key"]] = {c: [m for m, bits in zip(modes, row["modes"]) if bits[i]]
                              for i, c in enumerate(timings)}
    summarize(f"four cadences {split}", groups)
summarize("fresh first-controls fixed cadences",
          cadence_pairs(load("first-controls", "live-result.json")["rows"], fresh=True))

for split in ("source", "development", "prospective"):
    rows = load("moving-rack", f"{split}-branches.jsonl")
    blocks = defaultdict(list)
    for r in rows:
        blocks[r["pose"]].append(r)
    for block in blocks.values():
        assert len(block) == 6 and len({(r["regime"], r["option"]) for r in block}) == 6
        assert len({r["physics"] for r in block}) == len({r["prefix"] for r in block}) == 1
    def rack_groups(endpoint):
        return {p: {c: [r["option"] for r in block if r["regime"] == c and r[endpoint]]
                    for c in ("BASE", "D02")} for p, block in blocks.items()}
    for endpoint in ("hit", "final"):
        summarize(f"moving rack {split} {endpoint}", rack_groups(endpoint))
    if split == "prospective":
        for file, name in (("current-camera", "camera"), ("live-motor", "live")):
            for row in load("moving-rack", file + ".json")["rows"]:
                blocks[row["pose"]].append(dict(row, option=name))
        for endpoint in ("hit", "final"):
            summarize(f"moving rack prospective with live and camera {endpoint}", rack_groups(endpoint))

for family, data in load("failure-diagnosis", "replay.json").items():
    groups = defaultdict(dict)
    for r in data["tapes"]:
        for goal in (0, 1):
            # Action order from panel.ACTIONS: (goal, paced), paced=False first.
            key = f"pose{r['query_pose']}-goal{goal}"
            winners = [mode for j, mode in enumerate(["switch_to_five", "continue_paced"])
                       if r["query"][2 * goal + j][goal]]
            assert groups[key].setdefault(r["timing"], winners) == winners
    summarize(f"failure diagnosis {family}", groups)

phase = load("release-phase", "phase.json")
release = dict(practice_tapes=len(phase["practices"]), retry_tapes=len(phase["queries"]),
               retry_pose_goal=[[r["pose"], r["goal"]] for r in phase["queries"]],
               retry_modes=sorted({r["mode"] for r in phase["queries"]}), crossed_groups=0)
print(json.dumps(dict(menus=menus, release_phase=release), indent=2))
