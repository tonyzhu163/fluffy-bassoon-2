"""Check pairing and read the three preregistered contrasts from the saved native episodes."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from cue import ARMS, CUE, IDENTITY, LAYOUTS, MODEL_SHA, MODES, TASK, require
from contract import WEIGHT_SHA
from readback import interval, native_commands
from ur5e_motor import NOISE_SEED_OFFSET, array_digest


def verify(obj):
    rec, calls, mode, q = obj["native"], obj["calls"], obj["mode"], obj["query"]
    reset = rec["reset"]
    n = rec["controls"]
    h, width = obj["shape"]
    correction = IDENTITY if mode == "raw" else obj["support"]["tuned"]
    require(rec["inferred"] == correction and reset["prompt"] == rec["released_prompt"] == TASK and
            reset["horizon"] == 300 and obj["model_sha"] == MODEL_SHA and obj["weight_manifest_sha"] == WEIGHT_SHA,
            "correction, horizon, files or native task changed")
    require(q["seed"] == 42710300+q["init_state_id"] and mode in MODES and q["arm"] in ARMS and
            (q["suite"], q["task_id"], q["variant"]) == ("libero_goal", 3, "BASE"), "unregistered case")
    require(0 < n <= 300 and len(calls) == (n+4)//5 and calls[0]["public"] == reset["public"] and
            calls[0]["inputs"] == reset["motor_frame0"] and calls[0]["proposal"] == rec["opening_proposal_sha256"],
            "native query count or opening")
    seed = q["seed"]+NOISE_SEED_OFFSET
    bank = np.random.default_rng(seed).standard_normal((60, h, width)).astype("f4")
    require(reset["noise_seed"] == seed and reset["noise_sha256"] == array_digest(bank), "full native noise")
    first = next((r["t"] for r in calls if mode in ("visual","clock") and r[mode] >= .5), None)
    rewards = []
    for i, r in enumerate(calls):
        size = min(5,n-5*i)
        require(r["t"] == 5*i and r["fire"] == (r["t"] == first) and
                r["prompt"] == (CUE if r["fire"] else TASK) and r["noise"] == array_digest(bank[i]),
                "event, prompt or actual query noise")
        proposal = np.asarray(r["first_five"], dtype=np.float32)
        sent = np.asarray(r["sent"], dtype=np.float32)
        require(proposal.shape == (5,7) and np.array_equal(sent,native_commands(proposal[:size],correction))
                and len(r["rewards"]) == size, "unmatched executed controls")
        require(i == len(calls)-1 or not (r["terminated"] or r["truncated"]), "query after termination")
        rewards.extend(r["rewards"])
    require(rewards == rec["native_rewards"] and all(x in (0.,1.) for x in rewards) and
            rec["native_return"] == sum(rewards) == int(rec["success"]) and bool(rewards[-1]) == rec["success"]
            and (calls[-1]["terminated"], calls[-1]["truncated"]) == (rec["terminated"],rec["truncated"]),
            "native terminal reward")
    if not obj["fixture"]:
        require(rec["terminated"] == rec["success"] and rec["terminated"] != rec["truncated"] and
                (not rec["truncated"] or n == reset["horizon"]), "incomplete native endpoint")


def first_fire(calls, mode=None):
    return next((r["t"] for r in calls if (r[mode] >= .5 if mode else r["fire"])), None)


def order(v, c):
    if v is None or c is None:
        return "neither" if v == c else ("clock_only" if v is None else "visual_only")
    return "tied" if v == c else ("visual_before" if v < c else "visual_after")


def action_change(a, b):
    first_proposal = first_action = None
    equal_history = True
    for x,y in zip(a["calls"],b["calls"]):
        require(x["noise"] == y["noise"] and x["t"] == y["t"], "different noise at shared replan")
        matched = x["public"] == y["public"] and x["inputs"] == y["inputs"]
        if equal_history:
            require(matched and (x["visual"],x["clock"]) == (y["visual"],y["clock"]),
                    "physical or policy inputs diverged before commands")
        if matched:
            require(x["prompt"] != y["prompt"] or x["proposal"] == y["proposal"],
                    "unequal frozen draws on identical input, prompt and noise")
            if x["prompt"] != y["prompt"] and x["proposal"] != y["proposal"] and first_proposal is None:
                first_proposal = x["t"]
            if x["prompt"] != y["prompt"] and first_action is None:
                for j,(left,right) in enumerate(zip(x["sent"],y["sent"])):
                    if left != right:
                        first_action = dict(query_t=x["t"], control=x["t"]+j+1, left=left, right=right,
                                            noise=x["noise"], inputs=x["inputs"],
                                            left_prompt=x["prompt"], right_prompt=y["prompt"])
                        break
        if x["sent"] != y["sent"]:
            equal_history = False
        if equal_history:
            require((x["rewards"],x["terminated"],x["truncated"]) == (y["rewards"],y["terminated"],y["truncated"]),
                    "reward or termination diverged on identical commands")
    if equal_history:
        require(len(a["calls"]) == len(b["calls"]), "unequal length on same history")
    return dict(first_different_proposal_query_t=first_proposal, first_different_issued_action=first_action)


def paired(group):
    raw = group["raw"]
    opening = raw["calls"][0]
    for mode,obj in group.items():
        require(obj["env"] == raw["env"] and obj["body"] == raw["body"] and obj["query"] == raw["query"] and
                obj["support"] == raw["support"] and obj["native"]["reset"] == raw["native"]["reset"] and
                obj["shape"] == raw["shape"], "unpaired physical opening or support")
        first = obj["calls"][0]
        require(all(first[k] == opening[k] for k in ("public","inputs","noise","prompt","proposal","first_five")),
                "unmatched raw opening proposal")
        if mode != "raw":
            require(first["sent"] == group["tuned"]["calls"][0]["sent"], "unmatched tuned opening commands")
    changes = {a+"-"+b: action_change(group[a],group[b]) for a,b in
               (("clock","tuned"),("visual","tuned"),("visual","clock"))}
    v,c = (first_fire(group[m]["calls"]) for m in ("visual","clock"))
    if v == c:
        require(group["visual"]["calls"] == group["clock"]["calls"], "equal cue times must tie")
    tuned = group["tuned"]["calls"]
    tv,tc = (first_fire(tuned,m) for m in ("visual","clock"))
    return dict(actual=dict(visual=v, clock=c, order=order(v,c)),
                same_tuned_observations=dict(visual=tv, clock=tc, order=order(tv,tc)), changes=changes)


def summarize(folder, cpu, pin):
    rows, across = [], {}
    for arm in ARMS:
        for layout in LAYOUTS:
            group = {}
            for m in MODES:
                obj = json.loads((folder/"episodes"/f"{arm}-{layout}-{m}.json").read_bytes())
                require(obj["mode"] == m and obj["fixture"] is cpu and obj["pin"] == pin and
                        (obj["query"]["arm"],obj["query"]["init_state_id"]) == (arm, layout), "miskeyed tape")
                verify(obj)
                group[m] = obj
            base = group["raw"]
            evidence = paired(group)
            key = [base["env"][k] for k in ("released_row_fingerprint","scene_fingerprint","bddl_sha256")]
            key += [base["native"]["reset"][k] for k in ("noise_sha256","prompt","horizon")]
            require(key == across.setdefault(layout,key), "unpaired cross-body released scene or noise")
            rows.append(dict(arm=arm, layout=layout, released_row_sha=key[0], **evidence,
                             success={m:int(obj["native"]["success"]) for m,obj in group.items()},
                             controls={m:obj["native"]["controls"] for m,obj in group.items()}))
    result = dict(status="passed", pin=pin, fixture=cpu, episodes=len(rows)*len(MODES), paired_starts=len(rows),
                  rows=rows, controls=sum(sum(r["controls"].values()) for r in rows))
    if cpu:
        result["routing"] = [json.loads((folder/f"routing-{arm}.json").read_bytes()) for arm in ARMS]
        return result
    for which in ("actual","same_tuned_observations"):
        result[which+"_timing_counts"] = dict(Counter(r[which]["order"] for r in rows))
    result["endpoints"] = {}
    for arm in (*ARMS, "both"):
        rs = [r for r in rows if arm == "both" or r["arm"] == arm]
        totals = {m:sum(r["success"][m] for r in rs) for m in MODES}
        contrasts = {}
        for b in ("tuned", "clock", "raw"):
            d = [r["success"]["visual"]-r["success"][b] for r in rs]
            layout_sums = [sum(v for v,r in zip(d,rs) if r["layout"] == l) for l in LAYOUTS]
            contrasts["visual-minus-"+b] = dict(rescues=d.count(1), losses=d.count(-1), ties=d.count(0),
                rescue_denominator=len(rs)-totals[b], loss_denominator=totals[b],
                layout_differences=[v/(len(rs)/4) for v in layout_sums], effect_pp=100*sum(d)/len(rs),
                layout_bootstrap_95_pp=interval(layout_sums,len(rs)))
        result["endpoints"][arm] = dict(n=len(rs),successes=totals, contrasts=contrasts)
    return result
