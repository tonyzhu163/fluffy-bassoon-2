"""Fixed source material and paired outcome reducer. Usable without a simulator."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "frozen-transfer-progress-20261124-r1"))
from contract import IDENTITY, MANIFEST_SHA, RELEASE_SHA, WEIGHT_SHA, cases, require, save
from readback import fp, interval, native_commands

ROWS = tuple(range(40, 48))
MODES = ("raw", "tuned", "IIWA34", "IIWA35")
CUT = {"IIWA34": 105, "IIWA35": 115}
TASK = "open the top layer of the drawer and put the bowl inside"
BANK_SHA = "1cd5354ade9ab9f323a518515f961df96123b7daeca017df1f62d3e9ff18794f"
MEMBERS = {34: "a610024dc57b1f8783d3dfe000454414db48b1915f858f3759abb9d700b754c9",
           35: "c5d8b4ba1a858b1f1c2b245cb1ec447d09439580ff3a1a404724042d4fe37d34"}


def sources():
    archive = HERE.parent / "frozen-transfer-donor-20261124-r1/inputs/raw-tapes.tar.gz"
    tapes, info = {}, {}
    with tarfile.open(archive) as ar:
        for row, checksum in MEMBERS.items():
            name = f"gpu01/episodes/IIWA-{row}-raw.json"
            data = ar.extractfile(name).read()
            require(hashlib.sha256(data).hexdigest() == checksum, name)
            obj = json.loads(data)
            rec = obj["native"]
            require(obj["query"] == dict(arm="IIWA", suite="libero_goal", task_id=3, variant="BASE",
                                         init_state_id=row, seed=42710300+row) and obj["mode"] == "raw"
                    and not obj["fixture"] and rec["inferred"] == IDENTITY and rec["success"]
                    and rec["released_prompt"] == TASK and obj["body"]["gripper"] == "PandaGripper", name)
            vals = [a for call in obj["calls"] for a in call["sent"]]
            a = np.asarray(vals, np.float32)
            require(a.shape == (rec["controls"], 7) and np.array_equal(a, vals) and
                    np.isfinite(a).all() and np.max(np.abs(a)) <= 1, "source actions")
            first = int(np.flatnonzero(a[:, 6] > 0)[0])
            mode = f"IIWA{row}"
            cut = first // 5 * 5
            require(cut == CUT[mode] and (a[:first, 6] < 0).all(), "source-only handoff")
            tapes[mode] = a[:cut].copy()
            info[mode] = dict(member=name, member_sha256=checksum, source_controls=len(a),
                              first_positive_index=first, cut=cut, prefix_sha256=fp(a[:cut]),
                              original_pin=obj["pin"], initial_grip=float(a[0, 6]))
    return tapes, info


def verify(obj, tapes):
    q, rec, calls = obj["query"], obj["native"], obj["calls"]
    m, cpu, support = obj["mode"], obj["fixture"], obj["support"]
    reset, env, body = rec["reset"], obj["env"], obj["body"]
    row = q["init_state_id"]
    require(row in ROWS and q == dict(arm="Kinova3", suite="libero_goal", task_id=3, variant="BASE",
                                      init_state_id=row, seed=42710300+row) and m in MODES, "case")
    require(body["arm"] == reset["arm"] == "Kinova3" and body["model_class"] == "MountedKinova3" and
            body["controller"] == "OSC_POSE" and body["gripper"] == "PandaGripper" and
            body["joint_count"] == 7 and body["control_freq"] == 20, "target body")
    require(env["source_pin"] == RELEASE_SHA and env["manifest_sha256"] == MANIFEST_SHA and
            env["released_row_fingerprint"] == obj["release_row"] and env["perturbation_plan"]["case_id"] == "BASE"
            and all(env[k] == q[k] for k in ("suite", "task_id", "init_state_id", "seed"))
            and reset["physics"] == env["initial_state_fingerprint"] != support["physics"]
            and reset["horizon"] == 300 and reset["worker_step_index"] == 0 and
            reset["prompt"] == rec["released_prompt"] == TASK and obj["weight_manifest"] == WEIGHT_SHA,
            "release, reset or checkpoint")
    require("llvmpipe" in reset["renderer"].lower() if cpu else reset["renderer_vendor"] == "NVIDIA Corporation",
            "renderer")
    correction = IDENTITY if m == "raw" else support["tuned"]
    require(rec["inferred"] == correction, "controller")
    n, cut = rec["controls"], CUT.get(m, 0)
    shape = obj["shape"]
    seed = 817117 + q["seed"]
    noise = np.random.default_rng(seed).standard_normal((60, *shape)).astype("f4")
    require(reset["noise_seed"] == seed and reset["noise_sha256"] == fp(noise) and
            0 < n <= 300 and len(calls) == (n+4)//5, "horizon or noise schedule")
    public = reset["public"]
    rewards = []
    for i, r in enumerate(calls):
        t, k = 5*i, min(5, n-5*i)
        prefix = t < cut
        require(r["t"] == t and r["public"] == public and r["inputs"] == r["expected_inputs"] and
                r["noise"] == fp(noise[i]) and r["prompt"] == TASK and r["prefix"] == prefix, "policy routing")
        require(r["motor"] == ("fixture" if cpu and (m not in CUT or t != cut) else "pi05_libero"),
                "motor identity")
        if i == 0:
            require(r["inputs"] == reset["motor_frame0"] and r["proposal"] == rec["opening_proposal_sha256"],
                    "opening")
        plan = np.asarray(r["first_five"], np.float32)
        require(plan.shape == (5, 7) and np.isfinite(plan).all(), "proposal")
        if prefix:
            expected = tapes[m][t:t+k]
        elif m == "raw":
            expected = np.clip(plan[:k], -1, 1)
        else:
            expected = native_commands(plan, correction)[:k]
        steps = r["native_steps"]
        issued = np.asarray([s["action"] for s in steps], dtype="f4")
        require(issued.shape == (k, 7) and issued.tobytes() == expected.tobytes(), "physical action")
        require([s["index"] for s in steps] == list(range(t+1, t+k+1)), "native step indices")
        for s in steps:
            require(s["reward"] == float(s["success"]) and s["success"] == s["terminated"] and
                    s["truncated"] == (s["index"] == 300 and not s["success"]) and
                    (s["index"] == n or not (s["terminated"] or s["truncated"])), "step outcome")
            rewards.append(s["reward"])
        public = r["after_public"]
        if m in CUT and t == cut:
            require(r["inputs"] != calls[0]["inputs"] and r["inputs"] != calls[i-1]["inputs"] and
                    r["public"] != calls[i-1]["public"], "stale handoff")
            if cpu:
                require(not np.array_equal(issued, np.clip(plan[:k], -1, 1)), "fixture did not distinguish correction")
    last = calls[-1]["native_steps"][-1]
    require(all(last[k] == rec[k] for k in ("terminated", "truncated", "success")) and
            rewards == rec["native_rewards"] and sum(rewards) == rec["native_return"] == int(rec["success"]),
            "native motor endpoint")
    if cpu:
        require(n == cut+5 and not (last["terminated"] or last["truncated"]), "fixture failed to reach boundary")
    else:
        require(last["terminated"] != last["truncated"], "missing native endpoint")


def summarize(out, pin, cpu):
    tapes, source_info = sources()
    registered_support, = (x["support"] for x in cases("Kinova3") if x["query"]["suite"] == "libero_goal"
                            and x["query"]["task_id"] == 3)
    rows = ROWS[:1] if cpu else ROWS
    pairs = []
    require({p.name for p in (out/"episodes").glob("*.json")} ==
            {f"{row}-{m}.json" for row in rows for m in MODES}, "episode inventory")
    for row in rows:
        group = {}
        for m in MODES:
            obj = json.loads((out/"episodes"/f"{row}-{m}.json").read_bytes())
            require(obj["mode"] == m and obj["query"]["init_state_id"] == row and obj["pin"] == pin
                    and obj["fixture"] is cpu and obj["support"] == registered_support, "record key")
            verify(obj, tapes)
            group[m] = obj
        ref = group["raw"]
        for obj in group.values():
            require(all(obj[k] == ref[k] for k in ("env", "body", "query", "shape", "support", "release_row"))
                    and obj["native"]["reset"] == ref["native"]["reset"] and all(
                        obj["calls"][0][k] == ref["calls"][0][k] for k in
                        ("public", "inputs", "noise", "prompt", "proposal", "first_five")), "unpaired opening")
        pairs.append(dict(row=row, release_row=ref["release_row"], physics=ref["native"]["reset"]["physics"],
                          success={m:int(v["native"]["success"]) for m,v in group.items()},
                          controls={m:v["native"]["controls"] for m,v in group.items()},
                          handoffs={m:group[m]["native"]["controls"] > CUT[m] for m in CUT}))
    require(len(set(p["release_row"] for p in pairs)) == len(rows), "duplicate released row")
    result = dict(pin=pin, status="passed", fixture=cpu, episodes=len(rows)*4, paired_starts=len(rows),
                  sources=source_info, rows=pairs, support=registered_support)
    if not cpu:
        result["successes"] = {m:sum(p["success"][m] for p in pairs) for m in MODES}
        contrasts = {}
        for m, b in [(m,b) for m in CUT for b in ("tuned","raw")] + [("tuned","raw")]:
            d = [p["success"][m]-p["success"][b] for p in pairs]
            contrasts[f"{m}-{b}"] = dict(rescues=[r for r,v in zip(rows,d) if v==1],
                                         losses=[r for r,v in zip(rows,d) if v==-1],
                                         ties=d.count(0), effect_pp=100*sum(d)/len(d),
                                         paired_layout_bootstrap_95_pp=interval(d,len(d)))
        result["contrasts"] = contrasts
    return result
