"""Append to references.py on stdin for read-only execution on the Baige CPU host."""
from copy import deepcopy
from itertools import product

MODES = ("tuned", "switch", "source", "corrected")
PAIRS = (("source","tuned"),("source","switch"),("source","corrected"),
         ("switch","tuned"),("corrected","tuned"),("corrected","switch"))
PUBLIC = {"agentview_image","robot0_eye_in_hand_image","robot0_joint_pos","robot0_joint_vel",
          "robot0_eef_pos","robot0_eef_quat_xyzw","robot0_gripper_qpos"}
IMAGE_KEYS = (("agentview_image","observation/image"),("robot0_eye_in_hand_image","observation/wrist_image"))
METHODS = {"separate_float32_products", "reader_matmul"}


def xy_rotate(v, deg):
    theta = math.radians(deg)
    c, s = np.float32(math.cos(theta)), np.float32(math.sin(theta))
    matrix = np.array([[c,s],[-s,c]], np.float32)
    return {"separate_float32_products": v[...,0,None]*matrix[0] + v[...,1,None]*matrix[1],
            "reader_matmul": v @ matrix}


def inverse(plan, correction):
    v = np.clip(plan, -1, 1)
    factor = np.float32(1 / np.clip(correction["xy_scale"],.25,4))
    answer = {}
    for method, xy in xy_rotate(v[:,:2], -correction["yaw_deg"]).items():
        preclip = v.copy()
        preclip[:,:2] = xy*factor
        preclip[:,2] *= factor
        preclip[:,6] *= np.float32(correction["gripper_sign"])
        answer[method] = (preclip, np.clip(preclip,-1,1))
    return answer


def episode(obj, task, row, mode, pin, fixture, reference, tape, lookups, kernels):
    tag = f"{task}/{row}/{mode}"
    rec, env, body = obj["native"], obj["env"], obj["body"]
    reset = rec["reset"]
    n, horizon, cut = rec["controls"], reference["horizon"], len(tape)
    seed = reference["seed_base"] + row
    query = dict(arm="Kinova3",suite=reference["suite"],task_id=reference["task_id"],variant="BASE",init_state_id=row,seed=seed)
    support = reference["support"]
    correction = support["tuned"]
    check(obj["fixture"] is fixture and obj["query"] == query and obj["mode"] == mode and obj["task"] == task and obj["pin"] == pin
          and obj["support"] == support and rec["inferred"] == correction and obj["cut"] == cut
          and obj["shape"] == [10,32] and obj["weight_manifest"] == WEIGHTS, tag+" identity")
    check(obj["accepted_reset"] in (1,2,3) and len(obj["rejected_before_inference"]) == obj["accepted_reset"]-1, tag+" reset admission")
    check(obj["release_row"] == reference["release_rows"][row] == env["released_row_fingerprint"]
          and env["manifest_sha256"] == MANIFEST and env["source_pin"] == RELEASE and env["bddl_sha256"] == reference["bddl_sha256"]
          and env["perturbation_plan"]["operator"] == "none" and env["perturbation_plan"]["case_id"] == "BASE"
          and env["alignment"]["method"] == "joint_limited_grip_site_ik"
          and all(env[k] == query[k] for k in ("suite","task_id","seed","init_state_id"))
          and reset["physics"] == env["initial_state_fingerprint"] != support["physics"]
          and reset["scene_fingerprint"] == env["scene_fingerprint"] and reset["perturbation_plan"] == env["perturbation_plan_fingerprint"]
          and reset["horizon"] == horizon and reset["worker_step_index"] == 0 and reset["prompt"] == reference["prompt"] == rec["released_prompt"],
          tag+" release and physical start")
    schema = reset["action_schema"]
    check(body["model_class"] == reset["model_class"] == "MountedKinova3" and body["arm"] == reset["arm"] == "Kinova3"
          and body["joint_count"] == 7 and body["cameras"] == ["agentview","robot0_eye_in_hand"] and body["gripper"] == "PandaGripper"
          and body["controller"] == schema["controller"] == "OSC_POSE" and body["control_freq"] == schema["control_freq"] == 20
          and schema["low"] == [-1]*7 and schema["high"] == [1]*7 and schema["dtype"] == "float32"
          and schema["shape"] == [7] and schema["control_mode"] == "relative",tag+" embodiment")
    check("llvmpipe" in reset["renderer"].lower() if fixture else reset["renderer_vendor"] == "NVIDIA Corporation",tag+" renderer")
    bank = np.random.default_rng(seed+817117).standard_normal(((horizon+4)//5,10,32)).astype("<f4")
    check(reset["noise_seed"] == seed+817117 and reset["noise_sha256"] == fp(bank),tag+" absolute noise bank")
    anchor = np.asarray(obj["anchor"],"<f4")
    public = reset["public"]
    calls = obj["calls"]
    check(fp(anchor) == public["robot0_eef_pos"] and 0 < n <= horizon and len(calls) == (n+4)//5,tag+" anchor/length")
    actions, rewards = [], []
    stages = {}
    handback = None
    for i,cell in enumerate(calls):
        t, k = 5*i, min(5,n-5*i)
        ctx = f"{tag} call {i}"
        raw = mode == "switch" and t < cut
        donor = mode in ("source","corrected") and t < cut
        pre = t < cut
        check(cell["t"] == t == len(rewards) and cell["raw_stage"] is raw and cell["prefix"] is donor and cell["public"] == public
              and set(public) == PUBLIC == set(cell["after_public"]) and cell["noise"] == fp(bank[i])
              and cell["prompt"] == reference["prompt"] and cell["motor"] == ("fixture" if fixture else "pi05_libero"),
              ctx+" query ordering, input or noise")
        inputs = cell["inputs"]
        check(set(inputs) == {"observation/state","observation/image","observation/wrist_image"},ctx+" input allowlist")
        for native_key,motor_key in IMAGE_KEYS:
            image_pairs = lookups.setdefault(motor_key,{})
            check(image_pairs.setdefault(public[native_key],inputs[motor_key]) == inputs[motor_key],ctx+" camera input changes for identical frame")
        original = np.asarray(cell["raw_state"],cell["state_dtype"])
        tuned = np.asarray(cell["tuned_state"],cell["state_dtype"])
        pos = np.asarray(original[:3],"<f4")
        check(original.dtype.str == "<f8" and original.shape == tuned.shape == (8,)
              and np.isfinite(original).all() and np.array_equal(original[3:],tuned[3:])
              and fp(pos) == public["robot0_eef_pos"] and fp(np.asarray(original[-2:],"<f4")) == public["robot0_gripper_qpos"],
              ctx+" current native proprioception")
        angles = lookups.setdefault("axis_angle", {})
        check(angles.setdefault(public["robot0_eef_quat_xyzw"], original[3:6].tobytes()) == original[3:6].tobytes(),
              ctx+" rotation conversion consistency")
        delta = pos-anchor
        for method,xy in xy_rotate(delta[:2],correction["yaw_deg"]).items():
            expected = anchor + np.array([*xy,delta[2]],"<f4") if t else pos
            if not np.array_equal(expected,tuned[:3]):
                kernels["state"].discard(method)
        check(kernels["state"] and inputs["observation/state"] == fp(original if raw else tuned),ctx+" actual routed model state")
        if i == 0:
            check(inputs == reset["motor_frame0"] and cell["proposal"] == rec["opening_proposal_sha256"],ctx+" opening motor")
        plan = np.asarray(cell["first_five"],"<f4")
        check(plan.shape == (5,7) and np.isfinite(plan).all(),ctx+" decoded proposal")
        predicted = lookups.setdefault("model",{})
        model_key = (cell["prompt"],tuple(sorted(inputs.items())),cell["noise"],cell["motor"])
        value = (cell["proposal"],plan.tobytes())
        check(predicted.setdefault(model_key,value) == value,ctx+" identical frozen query")
        full = tape[t:t+5] if donor else plan
        raw_action = np.clip(full,-1,1)
        variants = inverse(full,correction)
        steps = cell["native_steps"]
        sent = np.array([s["action"] for s in steps],"<f4")
        check(len(steps) == k and sent.shape == (k,7) and np.isfinite(sent).all() and np.abs(sent).max() <= 1,
              ctx+" terminal chunk width")
        calibrated = not raw and not (donor and mode == "source")
        if calibrated:
            kernels["action"].intersection_update(m for m,(_,cmds) in variants.items() if cmds[:k].tobytes() == sent.tobytes())
            check(kernels["action"],ctx+" inverse native commands")
        else:
            check(raw_action[:k].tobytes() == sent.tobytes(),ctx+" raw native commands")
        stage = stages.setdefault("before" if pre else "after",Counter())
        stage["calls"] += 1
        stage["raw_tuned_states_distinct"] += int(fp(original) != fp(tuned))
        stage["controls"] += k
        stage["issued_different_from_unconverted_input"] += int(np.any(sent != raw_action[:k],axis=1).sum())
        stage["issued_different_from_raw_motor"] += int(np.any(sent != np.clip(plan[:k],-1,1),axis=1).sum())
        for j,s in enumerate(steps,1):
            idx = t+j
            check(s["index"] == idx and s["reward"] == int(s["success"]) and s["terminated"] is s["success"]
                  and s["truncated"] is (idx == horizon and not s["success"]) and
                  (idx == n or not (s["terminated"] or s["truncated"])),ctx+" native reward / earliest terminal")
            rewards.append(s["reward"])
        actions.append(sent)
        if t == cut and mode != "tuned":
            handback = dict(query=i,first_control=t+1,motor=cell["motor"],native_commands=k,used_raw_stage=raw,
                            state_changed=inputs["observation/state"] != fp(original),
                            commands_changed=int(np.any(sent != np.clip(plan[:k],-1,1),axis=1).sum()),
                            distinct_from_previous_input=inputs != calls[i-1]["inputs"],
                            distinct_from_previous_public=public != calls[i-1]["public"])
        public = cell["after_public"]
    end = calls[-1]["native_steps"][-1]
    check(all(rec[k] == end[k] for k in ("success","terminated","truncated"))
          and rec["native_rewards"] == rewards and sum(rewards) == rec["native_return"] == int(end["success"]),tag+" recorded endpoint")
    if fixture:
        check(n == cut+5 and not end["success"] and not end["truncated"],tag+" qualification cap")
    else:
        check(end["terminated"] != end["truncated"] and (end["success"] or n == horizon),tag+" complete native episode")
    commands = np.concatenate(actions)
    return dict(success=int(end["success"]),controls=n,calls=len(calls),last_chunk=len(calls[-1]["native_steps"]),
                terminal="fixture cap" if fixture else ("success" if end["success"] else "horizon"),
                accepted_reset=obj["accepted_reset"],handback=handback,stages=stages,commands_sha256=sha(commands.tobytes())),commands


def paired_readback(groups):
    result = {}
    for a,b in PAIRS:
        hist = Counter({0:1})
        rescues,losses = [],[]
        for task,rows in groups.items():
            differences = []
            for row,r in rows.items():
                d = r["modes"][a]["success"]-r["modes"][b]["success"]
                differences.append(d)
                if d:
                    (rescues if d > 0 else losses).append(f"{task}/{row}")
            m = len(differences)
            within = Counter(map(sum,product(differences,repeat=m)))
            check(sum(within.values()) == m**m,"enumeration per task")
            joined = Counter()
            for u,v in hist.items():
                for x,y in within.items():
                    joined[u+x] += v*y
            hist = joined
        n = sum(map(len,groups.values()))
        total = sum(hist.values())
        ci = []
        for numerator in (1,39):
            seen = 0
            for v,count in sorted(hist.items()):
                seen += count
                if 40*seen >= total*numerator:
                    ci.append(100*v/n)
                    break
        result[f"{a}-{b}"] = dict(rescues=rescues,losses=losses,ties=n-len(rescues)-len(losses),
            effect_pp=100*(len(rescues)-len(losses))/n,exact_paired_bootstrap_95_pp=ci)
    return result


def corruption_tests(raw,task,pin,ref,tape):
    cases = (("raw_input_erased","switch"),("raw_action_erased","switch"),("handoff_stays_raw","switch"),
             ("handoff_noise_restarted","switch"),("direct_prefix_one_bit","source"),("inverse_bypassed","corrected"),
             ("postcut_source_replay","source"),("nonterminal_reward","tuned"),("extra_step_after_stop","corrected"),
             ("short_nonterminal_chunk","switch"))
    rejected = {}
    for error,mode in cases:
        obj = deepcopy(raw[mode])
        calls = obj["calls"]
        first, early, hand = calls[0], calls[1], calls[len(tape)//5]
        if error == "raw_input_erased":
            early["inputs"]["observation/state"] = fp(np.asarray(early["tuned_state"],early["state_dtype"]))
        elif error == "raw_action_erased":
            a = np.array(early["first_five"],"<f4")
            early["native_steps"][0]["action"] = inverse(a,ref["support"]["tuned"])["separate_float32_products"][1][0].tolist()
        elif error == "handoff_stays_raw":
            hand["inputs"]["observation/state"] = fp(np.array(hand["raw_state"],hand["state_dtype"]))
        elif error == "handoff_noise_restarted":
            hand["noise"] = first["noise"]
        elif error == "direct_prefix_one_bit":
            action = np.array(first["native_steps"][0]["action"],"<f4")
            action.view("<u4")[0] ^= np.uint32(1)
            first["native_steps"][0]["action"] = action.tolist()
        elif error == "inverse_bypassed":
            first["native_steps"][0]["action"] = tape[0].tolist()
        elif error == "postcut_source_replay":
            hand["native_steps"][0]["action"] = tape[0].tolist()
        elif error == "extra_step_after_stop":
            first["native_steps"][2].update(success=True,terminated=True,reward=1.)
        elif error == "short_nonterminal_chunk":
            early["native_steps"].pop()
        elif error == "nonterminal_reward":
            early["native_steps"][0]["reward"] = 1.
        try:
            episode(obj,task,20,mode,pin,True,ref,tape,{},dict(action=set(METHODS),state=set(METHODS)))
        except ValueError as exc:
            rejected[error] = str(exc)
        else:
            raise ValueError("reader failed corruption test "+error)
    return rejected


def verify_code(repo):
    run = repo/"outputs"/RUN
    old = repo/"outputs/frozen-transfer-demonstration-20261124-r1"
    shim = ROOT/"outputs/frozen-vla-native-transfer-20261123-r1"/("src-"+NATIVE)
    openpi = Path("/mowenjie/fluffy-bassoon/third_party/openpi-215abfb")
    for path,pin in ((shim,NATIVE),(RELEASE_DIR,RELEASE),(openpi,"215abfb217dbac7d5f1273282331b9b1866c0479")):
        check(subprocess.check_output(["git","-C",str(path),"rev-parse","HEAD"],text=True).strip() == pin,"runtime pin")
    paths = list(run/name for name in ("PLAN.md","registration.json","confirmation.py","protocol.py","audit.py","launch.sh"))
    paths += [old/"run.py",old/"audited_worker.py",shim/"src/fluffy_bassoon/xemb/embodiment.py",
              shim/"outputs/libero-xemb-ur5e-sysid-20261120-r1/ur5e_motor.py",
              shim/"outputs/libero-xemb-ur5e-sysid-20261120-r1/ur5e_sim_worker.py",shim/"outputs/libero-crossarm-20261120-r1/arm_env.py",
              openpi/"src/openpi/policies/policy.py", openpi/"src/openpi/models/pi0.py"]
    check(sha((run/"PLAN.md").read_bytes()) == read(run/"registration.json")["plan_sha256"],"locked plan")
    installed = ROOT/"setups/libero-meta-learning-20261117-r1"
    for suffix in ["env_wrapper.py","bddl_base_domain.py","problems/libero_tabletop_manipulation.py",
                   "problems/libero_kitchen_tabletop_manipulation.py"]:
        rel = Path("simulator-source/libero/libero/envs")/suffix
        checked_bytes(installed/rel,sha((RELEASE_DIR/rel).read_bytes()))
        paths.append(installed/rel)
    gripper = installed/"env/lib/python3.11/site-packages/robosuite/models/grippers/panda_gripper.py"
    checked_bytes(gripper,"a3760a8fc4599fa6f79914304b82466f3c1bda1a945943ca693bb57f00dc59eb")
    paths.append(gripper)
    return {str(p):sha(p.read_bytes()) for p in paths}


def check_model_files(directory, full):
    directory = Path(directory)
    manifest = json.loads(checked_bytes(directory/"READY.json",WEIGHTS))
    check(manifest["checkpoint_path"] == str(directory) and manifest["status"] == "verified", "frozen checkpoint")
    count = 0
    for spec in manifest["files"]:
        file = directory/spec["name"]
        check(file.stat().st_size == spec["bytes"],"checkpoint byte length")
        count += spec["bytes"]
        if full:
            h = hashlib.sha256()
            with file.open("rb") as f:
                for block in iter(lambda:f.read(8<<20),b""):
                    h.update(block)
            check(h.hexdigest() == spec["sha256"],"checkpoint file "+spec["name"])
    return dict(path=str(directory),manifest_sha256=WEIGHTS,files=len(manifest["files"]),bytes=count,
                all_file_bytes_hashed=full)


def read_attempt(pin,attempt,job=None):
    cpu()
    start_cpu,start_wall = time.process_time(),time.monotonic()
    repo = checkout(pin)
    folder = ROOT/"outputs"/RUN/attempt
    receipt = (folder/"exit.txt").read_text().split()
    launch = dict(x.split("=",1) for x in (folder/"launch.txt").read_text().split())
    check(receipt[0] == "0" and len(receipt) == 3 and launch["start"] == receipt[1] and launch["pin"] == pin,"successful launch")
    fixture = attempt.startswith("cpu")
    origin = read(folder/"origin.json")
    check(origin["pin"] == pin and origin["host"] == launch["host"] and origin["fixture"] is fixture
          and origin["lock"] == read(repo/"outputs"/RUN/"registration.json") and origin["shape"] == [10,32]
          and origin["policy"] == "/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1", "origin/registered lock")
    receipts = ["launch.txt","exit.txt","origin.json","readback.json"]
    runtime = dict(host=origin["host"],attempt=attempt,pin=pin,start_epoch=int(receipt[1]),end_epoch=int(receipt[2]),
                   elapsed_s=int(receipt[2])-int(receipt[1]))
    if fixture:
        check(launch["attempt"] == "cpu" and origin["cpu_affinity"] == [30,31] and not job,"fixture provenance")
    else:
        check(job and launch["attempt"] == "gpu" and origin["host"] == job+"-master-0" and 0 < runtime["elapsed_s"] <= 90*60,"job receipt")
        q = read(folder.parent/"cpu01/qualified.json")
        check(q["pin"] == pin and q["fixture"] is True and (folder.parent/"cpu01/exit.txt").read_text().split()[0]=="0","same-pin prequalification")
        probe = read(folder/"gpu.json")
        check(probe["status"] == "passed" and probe["hostname"] == origin["host"] and probe["allocation_type"] == "baige_container"
              and len(probe["devices"]) == 1 and probe["devices"][0]["cuda_name"] == "NVIDIA GeForce RTX 4090","probe")
        runtime.update(job=job,device=probe["devices"][0]["cuda_name"],egl_device=origin["egl_device"])
        receipts.append("gpu.json")
    result_name = "qualified.json" if fixture else "result.json"
    receipts.append(result_name)
    refs,tapes = references(repo)
    sums = {name:sha((folder/name).read_bytes()) for name in receipts}
    check(read(folder/"readback.json") == dict(status="reproduced",episodes=8 if fixture else 48,pin=pin),"writer verification")
    check(json.loads(json.dumps({k:v["release_rows"] for k,v in refs.items()})) == origin["release_rows"]
          and origin["support"] == {k:v["support"] for k,v in refs.items()},"independent row banks and PR7 support")
    rows = [20] if fixture else ROWS
    names = {f"{task}-{row}-{m}.json" for task in T for row in rows for m in MODES}
    check({p.name for p in (folder/"episodes").glob("*.json")} == names,"episode inventory")
    kernels = dict(action=set(METHODS),state=set(METHODS))
    groups,tests,lookup,noise_banks = {},{}, {},set()
    author = read(folder/result_name)
    check(author["episodes"] == len(names) and author["pin"] == pin and author["status"] == "passed" and author["fixture"] is fixture,
          "saved summary provenance")
    for task in T:
        table = groups[task] = {}
        for row in rows:
            reports,raw,commands = {},{},{}
            for m in MODES:
                name = f"episodes/{task}-{row}-{m}.json"
                data = (folder/name).read_bytes()
                sums[name] = sha(data)
                obj = raw[m] = json.loads(data)
                reports[m],commands[m] = episode(obj,task,row,m,pin,fixture,refs[task],tapes[task],lookup,kernels)
            base = raw["tuned"]
            reset = base["native"]["reset"]
            first = base["calls"][0]
            check(reset["noise_sha256"] not in noise_banks,"reused noise bank across case")
            noise_banks.add(reset["noise_sha256"])
            for m,obj in raw.items():
                check(all(obj[k] == base[k] for k in ("env","support","body","query","release_row","anchor","shape"))
                      and obj["native"]["reset"] == reset and all(obj["calls"][0][k] == first[k] for k in
                      ("inputs","public","noise","prompt","proposal","first_five","raw_state","tuned_state")),f"unpaired {task}/{row}/{m}")
            h = min(len(tapes[task]),len(commands["source"]),len(commands["corrected"]))
            same = commands["source"][:h] == commands["corrected"][:h]
            unchanged = np.where(np.all(same,axis=1))[0].tolist()
            first_diffs = {}
            for a,b in PAIRS:
                common = min(len(commands[a]),len(commands[b]))
                diff = np.where(np.any(commands[a][:common] != commands[b][:common],axis=1))[0]
                before = int(diff[0]) if len(diff) else common
                for c1,c2 in zip(raw[a]["calls"][:before//5],raw[b]["calls"][:before//5]):
                    check(c1["public"] == c2["public"] and c1["after_public"] == c2["after_public"],
                          f"divergent observed physical trajectories under identical issued prefixes {task}/{row} {a}-{b}")
                first_diffs[a+"-"+b] = int(diff[0])+1 if len(diff) else None
            table[row] = dict(released_row=base["release_row"],scene=reset["scene_fingerprint"],physics=reset["physics"],
                post_settle_scene=base["env"]["post_settle_scene_fingerprint"],anchor=base["anchor"],noise_bank=reset["noise_sha256"],
                noise_seed=reset["noise_seed"],opening_proposal=first["proposal"],modes=reports,first_different_control=first_diffs,
                actual_common_donor_controls=h,donor_controls_identical=len(unchanged),identical_prefix_indices_zero_based=unchanged,
                changed_donor_components_by_channel=(~same).sum(axis=0).tolist(),
                max_abs_donor_change=float(np.abs(commands["source"][:h]-commands["corrected"][:h]).max()))
            other = author["tasks"][task]["rows"][str(row)]
            check(other["physics"] == reset["physics"] and other["noise_sha256"] == reset["noise_sha256"] and
                  all(all(other["modes"][m][k] == reports[m][k] for k in ("controls","success","calls","accepted_reset")) for m in MODES)
                  and other["observed_prefix_difference"]["identical_controls"] == len(unchanged),"writer row agreement")
            if fixture:
                tests[task] = corruption_tests(raw,task,pin,refs[task],tapes[task])
    outcomes = {}
    if not fixture:
        for t,r in groups.items():
            by_mode = {m:sum(i["modes"][m]["success"] for i in r.values()) for m in MODES}
            paired = paired_readback({t:r})
            check(author["tasks"][t]["successes"] == by_mode and author["tasks"][t]["contrasts"] == paired,t+" writer effects")
            outcomes[t] = dict(successes=by_mode,contrasts=paired)
        pooled = paired_readback(groups)
        check(author["secondary_pooled"] == pooled,"writer stratified secondary effects")
        outcomes["secondary_pooled_contrasts"] = pooled
    clip = {}
    for task,tape in tapes.items():
        method = sorted(kernels["action"])[0]
        raw_clip = [inverse(tape[t:t+5],refs[task]["support"]["tuned"])[method] for t in range(0,len(tape),5)]
        pre,post = [np.concatenate(x) for x in zip(*raw_clip)]
        clip[task] = dict(source_prefix_controls=len(tape),identical_control_count=int(np.all(post==tape,axis=1).sum()),
                          unchanged_components_by_channel=(post==tape).sum(axis=0).tolist(),
                          changes_erased_by_clipping_by_channel=((post==tape)&(pre!=tape)).sum(axis=0).tolist())
    return dict(verified_utc=datetime.now(timezone.utc).isoformat(), runtime=runtime,read_only_host=os.uname().nodename,
                native_controls=sum(m["controls"] for g in groups.values() for r in g.values() for m in r["modes"].values()),
                policy_calls=sum(m["calls"] for g in groups.values() for r in g.values() for m in r["modes"].values()),
                episodes=len(names),paired_starts=len(noise_banks),rows=groups,contrasts=outcomes,clipping=clip,
                rejected_record_corruptions=tests,consistent_kernels={k:sorted(v) for k,v in kernels.items()},
                source=refs,code_sha256=verify_code(repo),checkpoint=check_model_files(origin["policy"],not fixture),
                record_and_receipt_sha256=sums,
                cpu_seconds=time.process_time()-start_cpu,wall_seconds=time.monotonic()-start_wall)


if __name__ == "__main__" and sys.argv[1] == "attempt":
    print(json.dumps(read_attempt(*sys.argv[2:]),indent=2,allow_nan=False))
