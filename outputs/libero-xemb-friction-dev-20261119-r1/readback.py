"""Read the 32 native traces and their immutable source tapes, without simulation."""
import hashlib
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
rows = [json.loads(line) for line in (out / "branches.jsonl").read_text().splitlines()]
sources = json.loads((out / "sources.json").read_text())["sources"]
recorded = json.loads((out / "score.json").read_text())
pairs = json.loads((out / "pairs.json").read_text())
modes = ("default", "paced")
mats = ("ordinary", "reduced")
keyed = {(r["task"], r["mode"], r["material"]): r for r in rows}
tasks = sorted({r["task"] for r in rows})
assert len(rows) == 32 and len(tasks) == 8 and set(keyed) == set(product(tasks, modes, mats))
returns, stops, details, physical = {}, Counter(), {}, {}
archives = Counter()
equal = Counter()
n_pre, max_wrist, exact_wrist = 0, 0., 0
for task in tasks:
    details[task] = {}
    taskrows = [r for r in rows if r["task"] == task]
    for field in ("physics", "public", "cameras", "frame_state"):
        equal["task_quartet_" + field] += all(r[field] == taskrows[0][field] for r in taskrows)
    for mode in modes:
        source = sources[f"{task}-{mode}"]
        folder = Path(source["folder"])
        assert hashlib.sha256((folder / "public.npz").read_bytes()).hexdigest() == source["npz_sha256"]
        s = source["record"]
        with np.load(folder / "public.npz") as f:
            ref = {k: f[k] for k in ("issued", "rewards", "positions")}
        branch = []
        trace = []
        for mat in mats:
            row = keyed[task, mode, mat]
            with np.load(out / f"{task}-{mode}-{mat}.npz") as f:
                tr = {k: f[k] for k in f.files}
            trace.append(tr)
            rewards, flags = tr["rewards"], tr["flags"]
            n = len(rewards)
            assert n == row["controls"] <= s["controls"]
            assert not flags[:-1].any() and np.array_equal(rewards, flags[:, 0])
            assert float(rewards.sum()) == row["native_return"] and rewards[-1] == row["final_reward"]
            assert n == len(ref["issued"]) or flags[-1, 1:].any()
            assert np.array_equal(tr["issued"], ref["issued"][:n])
            ret = int(rewards.sum())
            assert ret in (0, 1) and bool(ret) == row["success"]
            returns[task, mode, mat] = ret
            stops[mat + "_" + row["stop"]] += 1
            branch.append(dict(material=mat, controls=n, ret=ret, stop=row["stop"],
                               source_length=s["controls"], horizon=s["identity"]["horizon"]))
            archives[mat + "_physics_match"] += row["archive_physics_match"]
            for field, same in row["archive_public_match"].items():
                archives[mat + "_" + field] += same
            if mat == "ordinary":
                err = np.linalg.norm(tr["robot0_eef_pos"][1:].astype(float) - ref["positions"][:n], axis=1)
                archives["success_matches"] += bool(ret) == s["success"]
                archives["length_matches"] += n == s["controls"]
                archives["exact_reward_tapes"] += np.array_equal(rewards, ref["rewards"])
                archives["reward_positions_compared"] += n
                archives["exact_wrist_tapes"] += np.array_equal(tr["robot0_eef_pos"][1:], ref["positions"])
                exact_wrist += int(np.count_nonzero(err == 0))
                max_wrist = max(max_wrist, float(err.max()))
        details[task][mode] = branch
        p = pairs[task][mode]
        last = p["last_complete_precontact_control"]
        assert last >= 0
        n_pre += last + 1
        pre_eq = all(np.array_equal(trace[0][k][:last+1], trace[1][k][:last+1]) for k in trace[0]
                     if k not in ("issued", "rewards", "flags"))
        steps_eq = all(np.array_equal(trace[0][k][:last], trace[1][k][:last]) for k in ("issued", "rewards", "flags"))
        assert pre_eq == all(p["precontact_equal"].values()) and steps_eq == p["precontact_rewards_flags_equal"]
        equal["paired_precontact_state"] += pre_eq
        equal["paired_precontact_actions_rewards_flags"] += steps_eq
        equal["paired_frame_state"] += p["frame_state_equal"]
        equal["paired_reset_physics"] += p["physics_equal"]
        for camera, same in p["camera_equal"].items():
            equal["paired_" + camera] += same
        grouped = sorted({(v[0],v[1],*v[2:]) for v in p["changed_mu"].values() if v[0] in ("object", "fixture")})
        physical[f"{task}-{mode}"] = dict(first_contact=[r["first_finger"] for r in
                  (keyed[task,mode,mats[0]], keyed[task,mode,mats[1]])], changed_objects_or_fixtures=grouped,
                  common_changed_geom_pairs=len(p["changed_mu"]), last_complete_precontact=last)

maps = {"/".join((a,b)): sum(returns[t,a,mats[0]] + returns[t,b,mats[1]] for t in tasks)
        for a,b in product(modes,repeat=2)}
fixed = max(maps[f"{m}/{m}"] for m in modes)
task_fixed = sum(max(sum(returns[t,m,s] for s in mats) for m in modes) for t in tasks)
oracle = sum(max(returns[t,m,s] for m in modes) for t,s in product(tasks,mats))
assert (maps, fixed, max(maps.values()), task_fixed, oracle) == (recorded["material_maps"],
        recorded["best_fixed"], recorded["material_only"], recorded["task_fixed"], recorded["task_and_material"])
counts = {m: {a: sum(returns[t,m,a] for t in tasks) for a in mats} for m in modes}
outcomes = {m: {t: [returns[t,m,a] for a in mats] for t in tasks if returns[t,m,mats[0]] != returns[t,m,mats[1]]}
            for m in modes}
result = dict(branch_count=len(rows), task_count=len(tasks), total_native_controls=sum(r["controls"] for r in rows),
              native_reward_equals_binary_success_all=True, fixed_material_scores=counts, material_maps=maps,
              best_fixed=fixed, best_material_map=max(maps.values()), task_fixed=task_fixed, task_and_material=oracle,
              changed_outcomes=outcomes, stops=stops, archive=archives, paired_and_quartet=equal,
              precontact_pair_samples=n_pre, exact_wrist_controls=exact_wrist, max_ordinary_wrist_error_m=max_wrist,
              physical=physical, details=details)
(out / "readback.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({k:v for k,v in result.items() if k not in ("physical","details")}, indent=2))
