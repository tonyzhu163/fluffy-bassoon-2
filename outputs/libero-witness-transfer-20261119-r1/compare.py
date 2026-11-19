"""Compare paired physical decisions and the value of observing each information set."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def ceiling(cases, scope):
    groups = defaultdict(lambda: [0,0])
    for r in cases:
        key = (*r["triple"], *r["order"])
        if scope != "commands_and_rewards":
            key += tuple(r["variants"])
        if scope != "history_only":
            key += tuple(r["rewards"])
        outcomes = groups[key]
        for action in (0,1):
            outcomes[action] += r["query_hits"][action]
    return sum(max(outcomes) for outcomes in groups.values())


def compare(directory):
    records = {arm:json.loads((directory / f"{arm}01/{arm}.json").read_text()) for arm in ("robot","vlm","rgb","motion")}
    base = records["robot"]["cases"]
    methods = {}
    for arm, result in records.items():
        for r, other in zip(base,result["cases"],strict=True):
            for key in ("triple","order","variants","hidden","rewards","query_hits","strict"):
                assert r[key]==other[key]
        methods[arm]=dict(completion=result["physical_completion"]["numerator"],
                          strict=result["strict"]["native"]["numerator"],
                          swapped=result["strict"]["swapped"]["numerator"],
                          native=result["cyclic_all_assignments"]["numerator"],
                          two_zero=sum(r["native"] for r in result["cases"] if not any(r["rewards"])),
                          by_query={q:sum(r["native"] for r in result["cases"] if r["triple"][2]==q)
                                    for q in range(24,30)})
    pairs = {}
    for name, hit_key, rows in (
        ("placements","correct","physical_practices"), ("native","native","cases")):
        scores = [(r[hit_key], v[hit_key]) for r,v in
                  zip(records["robot"][rows],records["vlm"][rows],strict=True)]
        pairs[name]=dict(robot_better=sum(a>b for a,b in scores),
                         vlm_better=sum(a<b for a,b in scores),equal=sum(a==b for a,b in scores))
    return dict(arms=methods, ceilings={scope:ceiling(base,scope) for scope in
                ("commands_and_rewards","history_only","exact_history_and_reward")},
                reachable=sum(any(r["query_hits"]) for r in base),assignments=len(base),
                two_zero_assignments=sum(not any(r["rewards"]) for r in base),
                strict_assignments=sum(r["strict"] for r in base),robot_vs_general=pairs)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("json",type=Path)
    args=p.parse_args()
    args.json.write_text(json.dumps(compare(args.directory),indent=2)+"\n")
