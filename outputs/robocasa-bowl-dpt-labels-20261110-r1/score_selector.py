"""Score a held-out selector by looking up the actual physical branch it chose."""
import argparse
import json
from pathlib import Path


def score(coverage, diagnostic):
    scenes={r["seed"]:r for r in coverage["scenes"]}
    histories={(r["seed"],r["goal"]):r for r in diagnostic["trials"]}
    records=[]
    for (seed,goal),r in sorted(histories.items()):
        query=r["query"]
        assert seed!=query
        outcomes=scenes[query]["outcomes"]
        flipped=histories[seed,1-goal]
        assert flipped["query"]==query and flipped["arm"]==r["arm"]
        choices={"public_support_and_reward":r["full"],"reward_masked":r["blind"],
                 "current_observation":r["current"],"counterfactual_reward":flipped["full"]}
        records.append({"support":seed,"query":query,"goal":goal,"choice":choices,
                        "success":{k:int(outcomes[c][goal]) for k,c in choices.items()},
                        "oracle":int(any(o[goal] for o in outcomes))})
    return {"cases":len(records),"scores":{k:sum(r["success"][k] for r in records) for k in records[0]["success"]},
            "oracle":sum(r["oracle"] for r in records),"per_case":records}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("coverage",type=Path)
    p.add_argument("diagnostic",type=Path)
    args=p.parse_args()
    print(json.dumps(score(json.loads(args.coverage.read_text()),json.loads(args.diagnostic.read_text())),indent=2))
