"""Per-start success frequency over independent noise streams, from panel output directories."""
import json
from math import comb
from pathlib import Path
import sys


def exact_interval(successes, total, level=.95):
    """Clopper-Pearson by bisection."""
    def tail(p, low):  # P(X >= s) if low else P(X <= s)
        terms = range(successes, total + 1) if low else range(successes + 1)
        return sum(comb(total, k) * p ** k * (1 - p) ** (total - k) for k in terms)

    def solve(low):
        a, b = 0., 1.
        for _ in range(60):
            mid = (a + b) / 2
            if (tail(mid, low) < (1 - level) / 2) == low:
                a = mid
            else:
                b = mid
        return (a + b) / 2

    return (0. if successes == 0 else solve(True)), (1. if successes == total else solve(False))


def summarize(folders):
    report = {}
    for folder in folders:
        for path in sorted(Path(folder).glob("Kinova3-[0-9]*.json")):
            cell = json.loads(path.read_text())
            raw = cell["branches"]["raw"]
            draws = {name: b for name, b in cell["branches"].items() if name.startswith("noise-")}
            wins = sum(b["success"] for b in draws.values())
            report[str(cell["case"]["init_state_id"])] = dict(
                group=cell["group"], raw=raw["success"], repeat=cell["branches"]["repeat"]["success"],
                successes=wins, draws=len(draws), interval95=exact_interval(wins, len(draws)) if draws else None,
                controls={name: b["controls"] for name, b in draws.items()},
                distinct_openings=len({b["replans"][0]["selected_proposal_sha256"] for b in draws.values()}),
                errors=sorted(cell["errors"]))
    failed = [row for row in report.values() if row["group"] == "development_failures"]
    wins, draws = sum(r["successes"] for r in failed), sum(r["draws"] for r in failed)
    return dict(starts=report, failed_starts=dict(successes=wins, draws=draws,
                                                  interval95_pooled=exact_interval(wins, draws) if draws else None,
                                                  starts_with_any_success=sum(r["successes"] > 0 for r in failed)))


if __name__ == "__main__":
    print(json.dumps(summarize(sys.argv[1:]), indent=1))
