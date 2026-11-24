"""Tabulate native episodes in evidence/ and write summary.json. Reads reward only here, as audit data."""
import json
from math import comb
from pathlib import Path

here = Path(__file__).parent
rows = {}
for path in sorted(here.glob("evidence/*/*.json")):
    e = json.loads(path.read_text())
    assert e["weights_unchanged"] and e["parameters"] == 450_046_176 and e["evidence"] == "native-reward"
    rows[(path.parent.name, e["task"], e["arm"], e["layout"])] = e
tags = sorted({k[:2] for k in rows})
arms = [a for a in ("Panda", "Kinova3", "IIWA", "UR5e") if any(k[2] == a for k in rows)]
summary = dict(cells={}, totals={}, paired={}, repeats={})
for tag, task in tags:
    layouts = sorted({k[3] for k in rows if k[:2] == (tag, task)})
    print(f"\n{tag}, libero_goal/{task}\n| Layout | " + " | ".join(arms) + " |\n| ---: |" + " --- |" * len(arms))
    for layout in layouts:
        cells = []
        for arm in arms:
            e = rows.get((tag, task, arm, layout))
            cells.append("" if e is None else f"{'success' if e['success'] else 'fail'} {e['controls']}")
            if e is not None:
                summary["cells"][f"{tag}/goal{task}/{arm}/{layout}"] = dict(
                    success=e["success"], controls=e["controls"], prompt=e["replans"][0]["prompt"],
                    noise_seed=e["noise_seed"], command_digest=e["command_digest"], scene=e["scene_fingerprint"])
        print(f"| {layout} | " + " | ".join(cells) + " |")
    for arm in arms:
        got = [e["success"] for k, e in rows.items() if k[:3] == (tag, task, arm)]
        if got:
            summary["totals"][f"{tag}/goal{task}/{arm}"] = [sum(got), len(got)]
    print("| total | " + " | ".join("{}/{}".format(*summary["totals"].get(f"{tag}/goal{task}/{arm}", ["", ""])) for arm in arms) + " |")
    for arm in arms[1:]:
        both = [(rows[(tag, task, "Panda", l)], rows[(tag, task, arm, l)]) for l in layouts
                if (tag, task, "Panda", l) in rows and (tag, task, arm, l) in rows]
        if not both:
            continue
        # Matched resets and noise: same scene fingerprint and first noise draw on both arms.
        assert all(a["scene_fingerprint"] == b["scene_fingerprint"]
                   and a["replans"][0]["noise_digests"] == b["replans"][0]["noise_digests"] for a, b in both)
        only_panda = sum(a["success"] and not b["success"] for a, b in both)
        only_arm = sum(b["success"] and not a["success"] for a, b in both)
        n = only_panda + only_arm
        p = min(1.0, 2 * sum(comb(n, i) for i in range(max(only_panda, only_arm), n + 1)) / 2 ** n) if n else 1.0
        summary["paired"][f"{tag}/goal{task}/Panda-vs-{arm}"] = dict(
            starts=len(both), panda_only=only_panda, arm_only=only_arm, exact_sign_p=round(p, 4))
        print(f"Panda vs {arm}: {len(both)} matched starts, Panda only {only_panda}, {arm} only {only_arm}, exact sign p {p:.3f}")
for (tag, task, arm, layout), e in rows.items():
    for other in ("libero-prompt-noise2", "libero-prompt-same"):
        r = rows.get((other, task, arm, layout))
        if tag == "libero-prompt" and r is not None:
            summary["repeats"][f"{other}/goal{task}/{arm}/{layout}"] = dict(
                first=e["success"], repeat=r["success"], identical_commands=e["command_digest"] == r["command_digest"])
(here / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
print("\nrepeats", json.dumps(summary["repeats"]))
