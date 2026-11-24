"""Portable readback of the archived controls, recorded native endpoints and matched starts."""
from collections import Counter
from math import comb, sqrt
from statistics import NormalDist
import json
import tarfile

from material import DEV, HERE, digest, read, sha

HIST = (14,15,16,17,31)
FRESH = tuple(range(40,48))
EVIDENCE = HERE/"evidence"
CASES = EVIDENCE/"cases"


def contrast(cases, rows, a, b, common):
    changes = []
    counts = [0,0]
    limits = []
    for row in rows:
        pair = [cases[row,k] for k in (a,b)]
        limit = min(r["tape_length"] for r in pair) if common else None
        outcome = [int(r["first_success"] is not None and (limit is None or r["first_success"]<=limit)) for r in pair]
        for i, win in enumerate(outcome):
            counts[i] += win
        changes.append(outcome[0]-outcome[1])
        limits.append(limit)
    win,loss = changes.count(1),changes.count(-1)
    n = win+loss
    p = min(1.,2*sum(comb(n,i) for i in range(min(win,loss)+1))/2**n)
    return dict(branches=[a,b], rows=rows, controls=limits, successes=counts, win=win, loss=loss, tie=len(rows)-n,
                differences=changes, delta_pp=100*(win-loss)/len(rows), exact_two_sided_discordant_sign_p=p)



def completion_counts(cases, rows, branch):
    outcomes = [cases[row,branch] for row in rows]
    n = len(rows)
    wins = sum(r["first_success"] is not None for r in outcomes)
    z = NormalDist().inv_cdf(.975)
    center = (wins + z*z/2)/(n + z*z)
    radius = z*sqrt(wins*(n-wins)/n + z*z/4)/(n + z*z)
    return dict(n=n, completions=wins, censored=sum(r["stop"]=="tape_end" for r in outcomes),
                horizon=sum(r["stop"]=="horizon" for r in outcomes),
                descriptive_wilson_95=[max(0.,center-radius),min(1.,center+radius)])


def main():
    rc,start,end = map(int,(EVIDENCE/"exit.txt").read_text().split())
    assert rc == 0
    target, manifest = read()
    assert manifest == json.loads((HERE/"inputs/manifest.json").read_bytes())
    donor = {}
    with tarfile.open(DEV/"inputs/raw-tapes.tar.gz") as archive:
        for row in (34,35):
            name=f"gpu01/episodes/IIWA-{row}-raw.json"
            blob = archive.extractfile(name).read()
            assert sha(blob) == json.loads((DEV/"inputs/manifest.json").read_bytes())["members"][name]
            x = json.loads(blob)
            assert x["native"]["success"] and x["query"]["init_state_id"] == row
            donor[f"IIWA{row}"] = ([a for c in x["calls"] for a in c["sent"]],x["native"]["reset"]["noise_sha256"])
    data = {}
    rows = []
    for row in (*HIST,*FRESH):
        path = CASES/f"start-{row}.json"
        s = json.loads(path.read_bytes())
        assert s["environment"]["init_state_id"] == row and s["body"]["arm"] == "Kinova3"
        assert s["environment"]["seed"] == (41810300 if row in HIST else 42710300)+row
        branches = (["raw"] if row in HIST else []) + (["tuned"] if row == 31 else []) + list(donor)
        item = {"row":row}
        for branch in branches:
            r = json.loads((CASES/f"case-{row}-{branch}.json").read_bytes())
            assert r["row"] == row and r["branch"] == branch and r["start_sha256"] == sha(path.read_bytes())
            if branch in donor:
                commands, noise = donor[branch]
            else:
                _,rec,env,commands = target[row,branch]
                noise = rec["reset"]["noise_sha256"]
                assert s["environment"]["released_row_fingerprint"] == env["released_row_fingerprint"]
                item["historical_"+branch] = dict(success=rec["success"],controls=rec["controls"])
            n = r["controls"]
            assert 0 < n <= len(commands) == r["tape_length"] and r["tape_noise_sha256"] == noise
            assert r["submitted_sha256"] == digest(commands[:n]) and r["action_sha256"] == digest(commands)
            success = r["first_success"] is not None
            assert r["first_success"] == (n if success else None)
            assert r["reward_sum"] == int(success) and r["preceding_zero_nonterminal_controls"] == n-1
            assert r["final_reward_success_terminated_truncated"] == [float(success),success,success,n==300 and not success]
            assert r["stop"] == ("success" if success else "horizon" if n==300 else "tape_end")
            assert success or n == len(commands)
            data[row,branch] = r
            item[branch] = dict(stop=r["stop"],controls=n)
        rows.append(item)
    assert len(data) == len(list(CASES.glob("case-*.json"))) == 32
    starts = [json.loads(f.read_bytes())["environment"] for f in CASES.glob("start-*.json")]
    assert len({x["released_row_fingerprint"] for x in starts}) == len({x["scene_fingerprint"] for x in starts}) == 13
    runtime = [json.loads((CASES/f"runtime-{cpu}.json").read_bytes()) for cpu in (30,31)]
    assert {x["cpu"] for x in runtime} == {30,31} and len({x["study_pin"] for x in runtime}) == 1
    assert len({x["host"] for x in runtime}) == 1
    contrasts = {}
    for name,panel,conditions in (("primary",HIST,[("IIWA35","raw"),("IIWA34","raw"),("IIWA35","IIWA34")]),
                                  ("fresh",FRESH,[("IIWA35","IIWA34")]),
                                  ("tuned",(31,),[("IIWA35","tuned"),("IIWA34","tuned")])):
        contrasts[name] = {f"{a}-{b}":{scope:contrast(data,panel,a,b,common)
            for scope,common in (("full_recorded",False),("common_budget",True))} for a,b in conditions}
    score = dict(wall_seconds=end-start, intervals_epoch=[start,end], study_pin=runtime[0]["study_pin"], rows=rows,
                 controls=sum(r["controls"] for r in data.values()), stops=dict(Counter(r["stop"] for r in data.values())),
                 historical_physics_equal=[row for row in HIST if data[row,"raw"]["historical_reset_comparison"]["physics_equal"]],
                 historical_post_settle_equal=[row for row in HIST if data[row,"raw"]["historical_reset_comparison"]["post_settle_equal"]],
                 full_tape_counts={
                     "primary":{b:completion_counts(data,HIST,b) for b in ("IIWA35","IIWA34","raw")},
                     "fresh":{b:completion_counts(data,FRESH,b) for b in ("IIWA35","IIWA34")}},
                 contrasts=contrasts)
    (EVIDENCE/"summary.json").write_text(json.dumps(score,indent=2)+"\n")
    print(json.dumps(score, indent=2))


if __name__ == "__main__":
    main()
