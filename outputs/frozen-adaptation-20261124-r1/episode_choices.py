"""Post hoc bound for choosing one complete, actually executed deployment mode per start."""
import argparse
import hashlib
import json
from pathlib import Path


def read_panel(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def evaluate(rows, modes):
    alternative_modes = [mode for mode in modes if mode not in ("raw", "tuned")]
    cases = []
    for arm, case, reward in rows:
        if set(reward) != set(modes) or any(value not in (0, 1) for value in reward.values()):
            raise ValueError((arm, case, reward))
        cases.append(dict(arm=arm, case=case, rewards=reward))
    if len({(row["arm"], row["case"]) for row in cases}) != len(cases):
        raise ValueError("Repeated native start")
    outcomes = [row["rewards"] for row in cases]
    result = {"n": len(outcomes), "cases": cases}
    for baseline in ("raw", "tuned"):
        failed = [reward for reward in outcomes if not reward[baseline]]
        result[baseline] = dict(
            successes=len(outcomes) - len(failed),
            failures=len(failed),
            rescued_by_any_new_mode=sum(any(reward[mode] for mode in alternative_modes) for reward in failed),
        )
    result["best_if_tuned_or_any_new_mode"] = sum(
        any(reward[mode] for mode in ["tuned", *alternative_modes]) for reward in outcomes
    )
    result["best_if_any_mode"] = sum(any(reward.values()) for reward in outcomes)
    return result


def main():
    args = argparse.ArgumentParser()
    args.add_argument("progress", type=Path)
    args.add_argument("cue", type=Path)
    args.add_argument("output", type=Path)
    ns = args.parse_args()
    motion, motion_sha = read_panel(ns.progress)
    cue, cue_sha = read_panel(ns.cue)
    arms = ("Kinova3", "IIWA")
    motion_rows = []
    for arm in arms:
        for row in motion[arm]["tasks"]:
            query = row["query"]
            motion_rows.append((arm, f"{query['suite']}/{query['task_id']}@{query['init_state_id']}", row["success"]))
    cue_rows = [(row["arm"], f"libero_goal/3@{row['layout']}", row["success"]) for row in cue["rows"]]
    panels = dict(
        motion=evaluate(motion_rows, ("raw", "tuned", "fixed", "progress")),
        cue=evaluate(cue_rows, ("raw", "tuned", "clock", "visual")),
    )
    if len(motion_rows) != 16 or len(cue_rows) != 8 or not cue["status"] == "passed" or cue["fixture"]:
        raise ValueError("Incomplete native endpoint set")
    panels["inputs_sha256"] = dict(progress=motion_sha, cue=cue_sha)
    ns.output.write_text(json.dumps(panels, indent=2) + "\n")


if __name__ == "__main__":
    main()
