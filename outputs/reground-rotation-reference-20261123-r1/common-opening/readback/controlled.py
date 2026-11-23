"""Independently verify a sealed set of three controlled arms against its one raw capture."""
import argparse
from functools import lru_cache
import re
import subprocess

from native_read import *

CODE = "c3efd5042a29c68248c07b2d6375d367df6378ef"
READER = "c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4"
SIM = {"sim", "body_pos", "body_quat", "timestep", "cur_time", "grip", "done", "shift/None"}
OBS = {"pixels/image","pixels/image2","robot_state/eef/pos","robot_state/eef/quat","robot_state/eef/mat",
       "robot_state/gripper/qpos","robot_state/gripper/qvel","robot_state/joints/pos","robot_state/joints/vel"}


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def source_tasks():
    site = ROOT/"env/lib/python3.12/site-packages/libero/libero/benchmark"
    tree = ast.parse((site/"libero_suite_task_map.py").read_bytes())
    suites = ast.literal_eval(next(node.value for node in tree.body if isinstance(node,ast.Assign)
                                   and node.targets[0].id == "libero_task_map"))
    language, = load_functions(site/"__init__.py", ("grab_language_from_filename",))
    return {f"{suite.removeprefix('libero_')}_{i}":language(stem+".bddl")
            for suite,names in suites.items() for i,stem in enumerate(names)}


def typed_digest(fields):
    # Include each leaf's path, dtype, shape and SHA256 inside the encompassing digest.
    payload = [[k,v.dtype.str,list(v.shape),digest(v)] for k,v in sorted(fields.items())]
    return hashlib.sha256(json.dumps(payload,separators=(",",":")).encode()).hexdigest()


@lru_cache(None)
def regenerated_noise(chunk_seed):
    raw = subprocess.check_output([str(Path(__file__).resolve().parent/"noise"), str(chunk_seed)])
    return np.frombuffer(raw, dtype=np.float32).reshape(50,32)


def read_lane(z, lane, key, text):
    task, i, rep = key
    nonce = seed_for(i,rep)
    chunk = seed_for(nonce,0)
    check(z["episodes"][lane] == f"{task}/{i}/{rep}" and z["tasks"][lane] == text
          and z["scene_seed"][lane] == 0 and z["noise_seed"][lane] == nonce and z["chunk_seed"][lane] == chunk, key)
    def subset(group):
        start = f"{lane}/{group}/"
        return {k[len(start):]:z[k] for k in z.files if k.startswith(start)}
    obs,sim = subset("obs"),subset("sim")
    check(obs.keys() == OBS and sim.keys() == SIM, (key, obs.keys(), sim.keys()))
    check(all(np.isfinite(v).all() for v in (*obs.values(),*sim.values())), key)
    for camera in ("pixels/image","pixels/image2"):
        check(obs[camera].shape == (256,256,3) and obs[camera].dtype == np.uint8, (key,camera))
    check(sim["body_pos"].ndim == 2 and sim["body_pos"].shape[1] == 3 and
          sim["body_quat"].shape == (len(sim["body_pos"]),4) and sim["sim"].ndim == 1
          and sim["sim"].size > 0 and sim["timestep"].item() == 10 and not sim["done"].item()
          and sim["shift/None"].size == 0 and sim["grip"].shape == (2,), key)
    state = {k.removeprefix("robot_state/"):v for k,v in obs.items() if k.startswith("robot_state/")}
    rotation = state["eef/mat"]
    x = np.concatenate((state["eef/pos"],rotation[:,0],rotation[:,1],state["gripper/qpos"],
                        state["joints/pos"],state["joints/vel"])).astype(np.float32)
    actual_x = z[f"{lane}/X0"]
    check(actual_x.shape == (25,) and actual_x.dtype == np.float32 and digest(x) == digest(actual_x), key)
    noise = z["noise"][lane]
    check(noise.shape == (50,32) and noise.dtype == np.float32 and digest(noise) == digest(regenerated_noise(chunk)), key)
    p = z["proposal"][lane]
    check(p.shape == (50,7) and p.dtype == np.float32 and np.isfinite(p).all(), key)
    evidence = dict(noise_seed=nonce, chunk_seed=chunk, noise_sha256=digest(noise), proposal50_sha256=digest(p),
                    proposal10_sha256=digest(p[:10]), x0_sha256=digest(actual_x), public_pose_sha256=typed_digest(state),
                    obs_sha256=typed_digest(obs), sim_sha256=typed_digest(sim), agentview_sha256=digest(obs["pixels/image"]),
                    wrist_sha256=digest(obs["pixels/image2"]))
    return evidence, state, p


def verify_weights(path,s):
    weights = read(path/"weights.json")
    models = read(ROOT/f"reference-native/fitted/seed{s}.json")["models"]
    check(len(weights) == len(models) == 4 and set(weights) == {m["dir"] for m in models}, weights)
    check({(m["role"],m["kind"]) for m in models} == {("nom","fwd"),("nom","idm"),("dep","fwd"),("dep","idm")}, models)
    for entry,sha in weights.items():
        check(re.fullmatch(r"runs/models/[0-9a-f]+",entry),entry)
        check(sha == file_hash(ROOT/"reference-native"/entry/"model.pt"), entry)
    return weights


def verify_shard(s,h,keys,predicates):
    stage = ROOT/f"common-opening/shards/seed{s}-shard{h:03d}"
    manifests = {a:read(stage/a/"opening.json") for a in ("capture",*ARMS)}
    cap = manifests["capture"]
    ident = {"episodes":list(map(list,keys)),"seed":s,"shard":h,"cell":"rot_z/90","policy":POLICY,"upstream":PIN,"code":CODE}
    names = {f"batch{i}.npz" for i in range(len(keys)//4)}
    check(cap["identity"] == ident and set(cap["batches"]) == names and
          {f.name for f in (stage/"capture").glob("*.npz")} == names and
          cap["reset_history"] == "fresh process per native batch", stage)
    check(all(m == cap for m in manifests.values()), f"arm receipts differ at {stage}")
    hist = cap["historical_agentview"]
    check([tuple(r["key"]) for r in hist] == keys, hist)
    limit = read(ROOT/"upstream/configs/speed_envelope.json")["pi05"]
    arms = {a:read_arm(stage/a,s,a,keys,limit) for a in ARMS}
    man = {a:read(stage/a/"manifest.json") for a in ARMS}
    native_cfg = next(j["cfg"] for j in read(ROOT/"rotation-jobs.json") if j["seed"] == s)
    cfg = {"base":{}, "reground":native_cfg, "symmeta":{"reader_seed":41,"reader_chunk":5,"max_updates":24}}
    attempts = {}
    for a,m in man.items():
        check(m["cfg"] == cfg[a], a)
        attempt = Path(m["argv"].split()[-1])
        check(attempt.parent == ROOT/"common-opening/attempts", attempt)
        selection = read(attempt/"selection.json")
        check(selection["seed"] == s and selection["shard"] == h and selection["episodes"] == ident["episodes"], attempt)
        attempts[a] = dict(host=m["host"], attempt=attempt.name,
                           selection=selection["original_proposal_mismatches"])
        cfg_id = ",".join(f"{k}={v}" for k,v in sorted(m["cfg"].items())) or "default"
        check(all(r["csv"]["cfg_id"] == cfg_id for r in arms[a][0]), (a,cfg_id))
    m = man["symmeta"]
    check(m["reader"]["weights_sha256"] == m["checkpoint_sha256"] == READER and
          m["code_revision"] == CODE and m["native_settle"] == m["policy_n_exec"] == 10 and not m["cpu_fixture"], m)
    weights = verify_weights(stage/"reground",s)
    orig_cost = read(original(s,h,"symmeta")/"calibration.json")
    cost = read(stage/"symmeta/calibration.json")
    check([row_key(c) for c in orig_cost] == [row_key(c) for c in cost] == keys, stage)
    descriptions = source_tasks()
    proposal, stream = predicates
    rows = []
    for batch in range(len(keys)//4):
        name = f"batch{batch}.npz"
        fp = stage/"capture"/name
        check(file_hash(fp) == cap["batches"][name], fp)
        with np.load(fp,allow_pickle=False) as z:
            globals_ = {"episodes","scene_seed","tasks","noise_seed","chunk_seed","noise","proposal"}
            required = {f"{j}/{g}/{k}" for j in range(4) for g,leaves in (("sim",SIM),("obs",OBS)) for k in leaves}
            check(set(z.files) == globals_ | required | {f"{j}/X0" for j in range(4)}, fp)
            check(z["proposal"].shape == (4,50,7) and z["noise"].shape == (4,50,32), fp)
            for lane in range(4):
                idx = 4*batch+lane
                key = keys[idx]
                facts,state,p = read_lane(z,lane,key,descriptions[key[0]])
                for c in (orig_cost[idx],cost[idx]):
                    q,u = c["queries"][0],c["updates"][0]
                    check(c["adapt_seed"] == s and c["noise_seed"] == facts["noise_seed"] and c["scene_seed"] == 0
                          and q["start_control"] == 0 and q["chunk_seed"] == facts["chunk_seed"]
                          and q["estimate"] == [0,1,1] and c["native_resets"] == 1 and c["settle_controls"] == 10
                          and c["probe_controls"] == c["extra_resets"] == 0, key)
                    check(np.array_equal(state["eef/pos"],u["before_eef"]) and np.array_equal(state["gripper/qpos"],u["before_jaw_qpos"])
                          and q["policy_eef"] == q["public_eef"] == u["before_eef"], key)
                q = cost[idx]["queries"][0]
                check(facts["proposal50_sha256"] == q["proposal_sha256"], key)
                check(facts["agentview_sha256"] == hist[idx]["current"] == cost[idx]["updates"][0]["before_sha256"]
                      and hist[idx]["previous"] == orig_cost[idx]["updates"][0]["before_sha256"], key)
                trace = {a:arms[a][1][idx] for a in ARMS}
                for a in ARMS:
                    check(trace[a]["A_vla"].shape == (10,7) and digest(trace[a]["A_vla"]) == facts["proposal10_sha256"]
                          and digest(trace[a]["X"][0]) == facts["x0_sha256"], (a,key))
                check(stream(trace["symmeta"],trace["base"],10) and
                      all(proposal(trace[a],trace[b],10) for a,b in PAIRS), key)
                c = cost[idx]
                check([q["start_control"] for q in c["queries"]] == list(range(0,c["scored_controls"],10))
                      and c["scored_controls"] == arms["symmeta"][0][idx]["metrics"]["steps"]
                      and [q["chunk_seed"] for q in c["queries"]] ==
                      [seed_for(facts["noise_seed"],j) for j in range(len(c["queries"]))],key)
                rows.append(dict(seed=s,shard=h,key=key,batch=batch,lane=lane,opening=facts,
                                 historical_agentview=hist[idx]["previous"],
                                 historical_stream_proposal50=orig_cost[idx]["queries"][0]["proposal_sha256"],
                                 arms={a:arms[a][0][idx] for a in ARMS}))
    return dict(seed=s,shard=h,rows=rows,tapes=cap["batches"],attempts=attempts,weights=weights,
                opening_receipt_sha256={a:file_hash(stage/a/"opening.json") for a in manifests})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("work",type=Path)
    p.add_argument("native_grid",type=Path)
    args=p.parse_args()
    plans=grid()
    predicates=load_functions(args.native_grid,("equal_proposal","equal_stream_prefix"))
    out=args.work/"controlled"
    out.mkdir(exist_ok=True)
    for (s,h),keys in plans.items():
        dest=out/(label(s,h)+".json.gz")
        if dest.exists() or not (ROOT/f"common-opening/shards/seed{s}-shard{h:03d}/reground/opening.json").exists():
            continue
        if not all((ROOT/f"common-opening/shards/seed{s}-shard{h:03d}"/a/"opening.json").exists() for a in ARMS):
            continue
        result=verify_shard(s,h,keys,predicates)
        tmp=dest.with_suffix(".part")
        with gzip.open(tmp,"wt") as f:
            json.dump(result,f,separators=(",",":"),allow_nan=False)
        tmp.replace(dest)
        print("verified",s,h,len(keys),flush=True)


if __name__ == "__main__":
    main()
