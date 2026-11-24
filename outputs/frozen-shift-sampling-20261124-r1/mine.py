"""List archived Kinova3 drawer-and-bowl episodes by layout. Archived native outcomes, no new rollout."""
import json
from pathlib import Path
import sys

OUTPUTS = Path(__file__).resolve().parents[1]
LABELS = ("adapter", "method", "mode", "branch", "group", "arm_name", "treatment", "name", "policy", "donor")
UNTREATED = {"raw", "repeat", "untreated", "continue", "control", "nominal", "own", "default", "none", "frozen"}


def identity(node):
    case = node.get("case") if isinstance(node.get("case"), dict) else node
    arm = case.get("arm", node.get("target"))
    layout = case.get("init_state_id", node.get("layout"))
    suite, task = case.get("suite"), case.get("task_id")
    if arm != "Kinova3" or not isinstance(layout, int) or isinstance(layout, bool):
        return None
    if suite is not None and (suite, task) != ("libero_goal", 3):
        return None
    if case.get("variant", "BASE") != "BASE":
        return None
    return layout, suite is not None


def walk(node, trail, rows, source):
    if isinstance(node, dict):
        found = identity(node) if isinstance(node.get("success"), bool) else None
        if found:
            reset = node.get("reset") if isinstance(node.get("reset"), dict) else {}
            label = "/".join(str(node[key]) for key in LABELS if isinstance(node.get(key), (str, int)))
            rows.append(dict(
                file=source, layout=found[0], task_stated=found[1], label=label or "/".join(trail[-2:]),
                success=node["success"], controls=node.get("controls", node.get("steps")),
                noise=(reset.get("noise_sha256") or node.get("noise_sha256") or "")[:10],
                opening=(node.get("opening_proposal_sha256") or node.get("untreated_opening_proposal_sha256") or "")[:10],
                seed=(node.get("case") or {}).get("seed") if isinstance(node.get("case"), dict) else node.get("seed"),
            ))
        for key, value in node.items():
            walk(value, (*trail, str(key)), rows, source)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            walk(value, (*trail, str(index)), rows, source)


def collect(root=OUTPUTS):
    rows = []
    for path in sorted(Path(root).rglob("*.json")):
        relative = path.relative_to(root)
        if relative.parts[0] in ("archive", Path(__file__).parent.name) or path.stat().st_size > 2e7:
            continue
        try:
            walk(json.loads(path.read_text()), (), rows, str(relative))
        except (ValueError, UnicodeDecodeError, RecursionError):
            continue
    return rows


def untreated(row):
    return any(part.lower() in UNTREATED for part in row["label"].replace("-", "/").split("/"))


if __name__ == "__main__":
    rows = collect()
    out = Path(__file__).with_name("archived-episodes.json")
    out.write_text(json.dumps(rows, indent=1) + "\n")
    for row in rows:
        print(row["layout"], int(row["success"]), row["controls"], row["noise"], row["opening"], row["seed"],
              "U" if untreated(row) else "-", "T" if row["task_stated"] else "?", row["label"][:40], row["file"], sep="\t")
    print(len(rows), file=sys.stderr)
