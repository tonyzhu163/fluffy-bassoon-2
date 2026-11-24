"""Summarize the rotation-branch episodes into summary.json. Run from this directory."""
import json
from math import comb
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYOUTS, OFFSETS = (14, 15, 17, 36, 37, 38), (0, 1000003)
ARMS = ("kinova3-raw", "kinova3-mapped", "panda-raw", "panda-flipped")


def interval(k, n, level=0.95):
    """Exact Clopper-Pearson interval by bisection on the binomial tail."""
    def tail(p, lo):
        return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in (range(k, n + 1) if lo else range(k + 1)))
    def solve(lo):
        a, b = 0.0, 1.0
        for _ in range(60):
            m = (a + b) / 2
            if (tail(m, lo) < (1 - level) / 2) == lo:
                a = m
            else:
                b = m
        return (a + b) / 2
    return [0.0 if k == 0 else solve(True), 1.0 if k == n else solve(False)]


def sign_test(wins, losses):
    n = wins + losses
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(max(wins, losses), n + 1)) / 2**n)


cells = {}
for arm in ARMS:
    for off in OFFSETS:
        for layout in LAYOUTS:
            path = next((HERE / "evidence" / f"{arm}-noise{off}").glob(f"*-layout{layout:02d}.json"))
            d = json.loads(path.read_text())
            x = [r["state"][3] for r in d["replans"]]
            assert d["weights_unchanged"] and d["noise_seed"] - d["seed"] == off
            cells[arm, off, layout] = dict(success=d["success"], controls=d["controls"], rot_x=[min(x), max(x)],
                                           scene=d["scene_fingerprint"], noise=d["replans"][0]["noise_digests"][0],
                                           commands=d["command_digest"])
for off in OFFSETS:
    for layout in LAYOUTS:
        group = [cells[a, off, layout] for a in ARMS]
        assert len({g["noise"] for g in group}) == 1
        assert group[0]["scene"] == group[1]["scene"] and group[2]["scene"] == group[3]["scene"]
out = {"arms": {}, "pairs": {}}
for arm in ARMS:
    mine = [v for (a, _, _), v in cells.items() if a == arm]
    k = sum(v["success"] for v in mine)
    out["arms"][arm] = dict(successes=k, episodes=len(mine), interval95=[round(v, 3) for v in interval(k, len(mine))],
                            rot_x=[round(min(v["rot_x"][0] for v in mine), 2), round(max(v["rot_x"][1] for v in mine), 2)],
                            solved=[[off, layout] for (a, off, layout), v in cells.items() if a == arm and v["success"]])
for name, base, treated in (("kinova3 mapped vs raw", "kinova3-raw", "kinova3-mapped"),
                            ("panda flipped vs raw", "panda-raw", "panda-flipped")):
    gained = sum(cells[treated, o, l]["success"] and not cells[base, o, l]["success"] for o in OFFSETS for l in LAYOUTS)
    lost = sum(cells[base, o, l]["success"] and not cells[treated, o, l]["success"] for o in OFFSETS for l in LAYOUTS)
    out["pairs"][name] = dict(gained=gained, lost=lost, sign_test_p=round(sign_test(gained, lost), 4))
on = out["arms"]["kinova3-mapped"]["successes"] + out["arms"]["panda-raw"]["successes"]
out["source_branch_vs_target_branch"] = dict(source=[on, 24], target=[out["arms"]["kinova3-raw"]["successes"]
                                             + out["arms"]["panda-flipped"]["successes"], 24])
total_gained = out["pairs"]["kinova3 mapped vs raw"]["gained"] + out["pairs"]["panda flipped vs raw"]["lost"]
total_against = out["pairs"]["kinova3 mapped vs raw"]["lost"] + out["pairs"]["panda flipped vs raw"]["gained"]
out["both_directions"] = dict(as_predicted=total_gained, against=total_against,
                              sign_test_p=round(sign_test(total_gained, total_against), 4))
(HERE / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
