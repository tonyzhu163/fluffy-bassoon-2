"""Independent reward, intervention timing and pairing audit for the native feedback run."""
import hashlib
import json
from math import sqrt
from pathlib import Path
import sys


def require(x, message):
    if not x:
        raise ValueError(message)


def vecnorm(x):
    return sqrt(sum(v*v for v in x))


def read(p):
    return json.loads(p.read_bytes())


def native(body):
    r = body["record"]
    rewards = r["native_rewards"]
    require(len(rewards)==r["controls"] and all(v in (0,1) for v in rewards), "bad reward tape")
    require(bool(max(rewards)) == r["success"] and r["native_return"] == sum(rewards), "bad success flag")
    require(r["terminated"] or r["truncated"], "unterminated")
    require(sum(len(c["cmd"]) for c in body["chunks"]) == r["controls"], "action count")
    require(r["reset"]["renderer_vendor"] == "NVIDIA Corporation", "renderer")
    return dict(success=r["success"],controls=r["controls"],native_return=r["native_return"])


def delta_stats(body):
    actions = [cmd for chunk in body["chunks"] for cmd in chunk["cmd"]]
    changes = [cmd for chunk in body["corrections"] for cmd in chunk]
    require(len(actions)==len(changes), "correction length")
    changed=[]
    residual_norms=[]
    for i,(issued,delta) in enumerate(zip(actions,changes),1):
        base=[a-b for a,b in zip(issued,delta)]
        require(all(x==0 for x in delta[3:]), "wrist/gripper was altered")
        require(max(map(abs,issued))<=1 and vecnorm(delta[:3])<=vecnorm(base[:3])+1e-6, "correction bound")
        if any(delta):
            changed.append(i)
            residual_norms.append(vecnorm(delta[:3]))
    require(not changed or min(changed)>=16, "feedback changed a calibration-prefix command")
    return dict(first_changed_control=min(changed, default=None), altered_controls=len(changed),
                mean_change_among_altered=sum(residual_norms)/len(changed) if changed else 0)


def main(p):
    rows=[]
    sha={}
    raw_summary=read(p/"summary.json")
    require(raw_summary["model_sha256"]=="19607168a6ff7425f461db94211b49b09b0bb6259ca896559a5d1ead6e2432e9",
            "wrong source weights")
    for (suite,tid) in (("libero_goal",3),("libero_10",8),("libero_goal",6)):
        for start in (16,17):
            key=f"{suite}-{tid}-{start}"
            trials={}
            for mode in ("raw","static","feedback","stale"):
                f=p/f"{key}-{mode}.json"
                sha[f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
                trials[mode]=read(f)
            ref=trials["raw"]["record"]
            results={}
            for mode,d in trials.items():
                r=d["record"]
                c=d["case"]
                require((c["suite"],c["task_id"],c["arm"],c["init_state_id"],c["variant"]) ==
                        (suite,tid,"Kinova3",start,"BASE"),"task")
                require(r["reset"]==ref["reset"] and r["opening_proposal_sha256"]==ref["opening_proposal_sha256"],
                        "different openings")
                results[mode]=native(d)
                if mode in ("feedback","stale"):
                    require(d["chunks"][:3]==trials["raw"]["chunks"][:3], "prefix mismatch")
                    results[mode].update(delta_stats(d))
            rows.append(dict(suite=suite,task_id=tid,start=start,results=results,
                             opening_sha256=ref["opening_proposal_sha256"],noise_sha256=ref["reset"]["noise_sha256"]))
    contrasts={}
    for a,b in (("feedback","raw"),("feedback","static"),("feedback","stale"),("stale","raw"),("static","raw")):
        diff=[int(r["results"][a]["success"])-int(r["results"][b]["success"]) for r in rows]
        contrasts[f"{a}_minus_{b}"]=dict(n=len(diff), gains=diff.count(1), losses=diff.count(-1),difference=sum(diff)/len(diff))
    require(len(raw_summary["rows"])==len(rows),"missing runner aggregate")
    for s,r in zip(raw_summary["rows"],rows):
        for m in r["results"]:
            require(all(r["results"][m][k]==s["results"][m][k] for k in ("controls","success","native_return")),
                    "runner summary mismatch")
    gpu=read(p/"gpu.json")
    return dict(launch=(p/"launch.txt").read_text().strip(),exit=(p/"exit.txt").read_text().strip(),
                gpu=gpu["devices"][0]["cuda_name"],rows=rows,contrasts=contrasts,episode_sha256=sha)


if __name__=="__main__":
    print(json.dumps(main(Path(sys.argv[1])),indent=2,allow_nan=False))
