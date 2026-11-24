"""Read back native steps, actions, openings and camera inputs using the physical wire receipts."""
import numpy as np
from cohort import MANIFEST_SHA, RELEASE_PIN, array_sha, require


def opening(obj):
    rec = obj["native"]
    return dict(reset=rec["reset"], env=obj["env"], body=obj["body"], proposal_sha256=rec["opening_proposal_sha256"],
                motor_inputs=obj["chunks"][0]["motor_inputs"], first_noise=obj["chunks"][0]["noise"])


def inspect_episode(row, method, obj, cpu=False):
    rec, env, body = obj["native"], obj["env"], obj["body"]
    r, ref = rec["reset"], row["reference"]
    require(env["source_pin"]==RELEASE_PIN and env["manifest_sha256"]==MANIFEST_SHA and
            (env["suite"],env["task_id"],env["init_state_id"],env["seed"]) ==
            (row["suite"],row["task"],row["layout"],row["seed"]) and env["perturbation_plan"]["case_id"]=="BASE" and
            all(env[k]==ref[k] for k in ("bddl_sha256","released_row_fingerprint")) and
            env["released_row_fingerprint"]!=ref["support_row_fingerprint"] and
            env["initial_state_fingerprint"]==r["physics"]!=row.get("support_physics") and
            r["scene_fingerprint"]==env["scene_fingerprint"] and r["perturbation_plan"]==env["perturbation_plan_fingerprint"],
            "official physical reset")
    require(r["arm"]==body["arm"]==row["arm"] and r["joint_count"]==body["joint_count"]==(6 if row["arm"]=="UR5e" else 7)
            and body["controller"]=="OSC_POSE" and body["gripper"]=="PandaGripper" and body["control_freq"]==20
            and body["model_class"]==r["model_class"] and r["worker_step_index"]==0 and
            (r["renderer_vendor"]=="NVIDIA Corporation" if not cpu else "llvmpipe" in r["renderer"].lower()), "model/render")
    schema = r["action_schema"]
    require(schema==dict(shape=[7],dtype="float32",controller="OSC_POSE",control_mode="relative",control_freq=20,
                         low=[-1.]*7,high=[1.]*7), "relative control schema")
    require(r["horizon"]==ref["horizon"] and r["prompt"]==rec["released_prompt"] and r["prompt"] and
            rec["inferred"]==dict(yaw_deg=0.,xy_scale=1.,gripper_sign=1), "nominal motor interface")
    if not cpu:
        require(r["prompt"]==row["prompt"], "released task prompt")
        require(all(body[k]==row["support_model"][k] for k in ("xml_sha256","arm","model_class","joint_count","gripper","controller")),
                "query model vs support")
    horizon = r["horizon"]
    n = rec["controls"]
    require(0<n<=horizon and len(obj["chunks"])==(n+4)//5, "native duration")
    noise_seed = row["seed"]+817117
    noise = np.random.default_rng(noise_seed).standard_normal(((horizon+4)//5,10,32)).astype(np.float32)
    require(r["noise_seed"]==noise_seed and array_sha(noise)==r["noise_sha256"], "actual policy noise bank")
    a = row["estimates"][method]
    inv = None if a is None else np.linalg.pinv(np.array(a["A"],dtype=a["dtype"]))
    truth = np.array(row["true_map"],np.float64)
    rewards = []
    bounds = np.zeros(2, np.int64)
    saturated = 0
    discrepancy = 0.
    for j,block in enumerate(obj["chunks"]):
        raw = np.array(block["raw"],np.float32)
        require(raw.shape==(10,7) and np.isfinite(raw).all() and block["noise"]==array_sha(noise[j]), "policy call")
        if j==0:
            require(array_sha(raw)==rec["opening_proposal_sha256"] and block["motor_inputs"]==r["motor_frame0"],
                    "opening actual inference")
        steps = block["steps"]
        count = min(5,n-5*j)
        require(len(steps)==count, "reward count")
        q = np.clip(raw[:5],-1.,1.)
        issued = q.copy()
        if inv is not None:
            issued[:,:2] = np.clip(q[:,:2].astype(np.float64) @ inv.T,-1.,1.)
        for i,s in enumerate(steps):
            u = np.array(s["issued"],np.float32)
            actual = np.array(s["physical"],np.float32)
            np.testing.assert_array_equal(issued[i],u)
            expected = u.copy()
            expected[:2] = truth @ u[:2]
            np.testing.assert_array_equal(actual,expected)
            require(np.max(np.abs(actual))<=1, "infeasible received command")
            hits = np.abs(u[:2])==1.
            bounds += hits
            saturated += int(np.any(hits))
            discrepancy += float(np.linalg.norm(actual[:2].astype(float)-q[i,:2]))
            rewards.append(s["reward"])
            require(s["index"]==len(rewards) and type(s["success"]) is bool and s["reward"]==int(s["success"])
                    and s["terminated"] is s["success"] and s["truncated"] is (len(rewards)==horizon and not s["success"])
                    and (len(rewards)==n or not(s["terminated"] or s["truncated"])), "native reward and flags")
    require(all(steps[-1][k]==rec[k] for k in ("success","terminated","truncated"))
            and rewards==rec["native_rewards"] and sum(rewards)==rec["native_return"]==int(rec["success"]) and
            (cpu or rec["terminated"]!=rec["truncated"]), "terminal native endpoint")
    return dict(success=rec["success"],controls=n,xy_saturated_controls=saturated,xy_bound_counts=bounds.tolist(),
                total_desired_to_received_xy=discrepancy,native_return=rec["native_return"])
