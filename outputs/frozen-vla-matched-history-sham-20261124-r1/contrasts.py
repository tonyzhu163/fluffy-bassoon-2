"""Summarize verified endpoints by robot and shared task/layout with fixed-panel sensitivity."""
import json
from pathlib import Path
import sys


def contrast(rows, left, right):
    differences = [int(v["endpoints"][left]["success"])-int(v["endpoints"][right]["success"]) for v in rows]
    return dict(pairs=len(rows), rescues=differences.count(1), losses=differences.count(-1),
                ties=differences.count(0), difference=sum(differences)/len(rows))


def summarize(paired):
    if paired["fixture"] or paired["cases"] != 8 or paired["episodes"] != 24:
        raise ValueError("Expected the eight complete native triplets")
    rows = paired["rows"]
    starts = sorted({(v["case"]["suite"], v["case"]["init_state_id"]) for v in rows})
    result = {}
    for arm in ("all", "Kinova3", "Panda"):
        subset = [v for v in rows if arm == "all" or v["case"]["arm"] == arm]
        cell = {}
        for label, left, right in (("history", "real", "sham"), ("graph", "sham", "plain")):
            cell[label] = contrast(subset, left, right)
            sensitivity = {}
            for suite, layout in starts:
                keep = [v for v in subset if (v["case"]["suite"], v["case"]["init_state_id"]) != (suite, layout)]
                sensitivity[f"{suite}/{layout}"] = contrast(keep, left, right)["difference"]
            cell[label]["leave_one_shared_start_out"] = sensitivity
        cell["successes"] = {m: sum(v["endpoints"][m]["success"] for v in subset) for m in ("real", "sham", "plain")}
        result[arm] = cell
    return result


if __name__ == "__main__":
    print(json.dumps(summarize(json.loads(Path(sys.argv[1]).read_text())), indent=2))
