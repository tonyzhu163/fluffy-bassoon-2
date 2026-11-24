"""Portable final CPU readback of the complete cohort and eligibility of archived controls."""
import argparse
import gzip
from itertools import combinations
import json
import math
from pathlib import Path

import numpy as np

from release import HERE, KIN, UR, LAUNCH, ORIGINAL, ROOT, checked, digest, original, read
from rollout import TRACES, TRANSPORT, MOTOR_READY

def fp(a):
    a = np.ascontiguousarray(a)
    return digest(repr((a.shape,a.dtype.str)).encode()+a.tobytes())


def planar(values, degrees, factor, grip):
    # The portable UR verifier uses independent float64 scalar products.
    angle = math.radians(degrees)
    cs,sn = math.cos(angle),math.sin(angle)
    v = np.asarray(values,dtype=np.float64)
    w = v.copy()
    w[:,0] = factor*(cs*v[:,0]-sn*v[:,1])
    w[:,1] = factor*(sn*v[:,0]+cs*v[:,1])
    w[:,2] *= factor
    w[:,6] *= grip
    return np.clip(w,-1,1)


PACKETS = {"UR5e":"be817675e72acca4ebb34a8b6f48d774aa51077a54d12a86ca28cc80456332b2",
           "Kinova3":"ef18976059a9a8944ab88bbc144c43b41aee8ba8eb8bb7bb9c9b8e05e502d8f7"}
KIN_ORIGINAL_RELEASE = "50e320040e06c70d2b0924f14afc24f0608e5951bfc7bcec37f5781384c7820f"
SIM_PIN = "e5500c690c1498eca1b7050350654964190cdf70"
SIM_MANIFEST = "71a6e4e25a2a4a162844d77136e7bff8e6f69c839d2d8194620c4afeabde2bba"
PUBLIC = {"observation/image","observation/wrist_image","observation/state"}


def episode(row, name, ep):
    """Use the original independent float64 reference with its float32 rounding bound."""
    rec,reset,env,body = ep["native"],ep["native"]["reset"],ep["env"],ep["body"]
    q,ref,c,true = row["query"],row["reference"],row["corrections"][name],row["true_map"]
    assert rec["inferred"]==c and rec["released_prompt"]==reset["prompt"]==row["prompt"]
    assert reset["horizon"]==ref["horizon"] and reset["worker_step_index"]==0 and reset["renderer_vendor"]=="NVIDIA Corporation"
    assert body["arm"]==q["arm"]==reset["arm"] and body["cameras"]==["agentview","robot0_eye_in_hand"]
    assert body["joint_count"]==reset["joint_count"]=={"UR5e":6,"Kinova3":7}[q["arm"]]
    assert body["controller"]=="OSC_POSE" and body["gripper"]=="PandaGripper"
    assert body["model_class"]==reset["model_class"] and body.get("joint_names")
    assert env["source_pin"]==SIM_PIN and env["manifest_sha256"]==SIM_MANIFEST and env["perturbation_plan"]["case_id"]=="BASE"
    assert all(env[k]==q[k] for k in ("suite","task_id","init_state_id","seed"))
    assert all(env[k]==ref[k] for k in ("bddl_sha256","released_row_fingerprint"))
    assert env["initial_state_fingerprint"]==reset["physics"] and reset["physics"] not in (row["calibration_physics"],row.get("support_physics",row.get("physics")))
    assert env["scene_fingerprint"]==reset["scene_fingerprint"] and env["perturbation_plan_fingerprint"]==reset["perturbation_plan"]
    assert set(reset["motor_frame0"])==PUBLIC and reset["noise_seed"]==q["seed"]+817117
    if q["arm"]=="Kinova3":
        assert body==row["model"]
    else:
        assert reset["action_schema"]==row["action_schema"] and body["model_class"]==row["model_class"]
    horizon,n = ref["horizon"],rec["controls"]
    noise = np.random.default_rng(reset["noise_seed"]).standard_normal(((horizon+4)//5,10,32)).astype("f4")
    assert fp(noise)==reset["noise_sha256"] and 0<n<=horizon and len(ep["chunks"])==(n+4)//5
    u = 2**-24
    gamma6 = 6*u/(1-6*u)
    inv = 1/float(np.clip(c["xy_scale"],0.25,4))
    forward = true["xy_scale"]
    worst = [0.,0.]
    rewards,physical,issued,proposals = [],[],[],[]
    for i,b in enumerate(ep["chunks"]):
        raw = np.asarray(b["raw"],"f4")
        assert raw.shape==(10,7) and np.isfinite(raw).all() and fp(noise[i])==b["noise"] and set(b["motor_inputs"])==PUBLIC
        if i==0:
            assert fp(raw)==rec["opening_proposal_sha256"] and b["motor_inputs"]==reset["motor_frame0"]
        count = min(5,n-5*i)
        wire = np.asarray(b["issued"],"f4")
        actual = np.asarray([s["action"] for s in b["steps"]],"f4")
        assert actual.shape==wire.shape==(count,7)
        expected_wire = planar(np.clip(raw[:count],-1,1),-c["yaw_deg"],inv,c["gripper_sign"])
        expected_physical = planar(wire,true["yaw_deg"],forward,true["gripper_sign"])
        for j,(got,want,scale) in enumerate(((wire,expected_wire,inv),(actual,expected_physical,forward))):
            residual = float(np.abs(got[:,:3]-want[:,:3]).max())
            assert np.array_equal(got[:,3:],want[:,3:]) and residual<=2*gamma6*scale,(name,row["index"],i,j,residual)
            worst[j]=max(worst[j],residual)
        for s in b["steps"]:
            rewards.append(s["reward"])
            assert s["index"]==len(rewards) and type(s["success"]) is bool and s["reward"]==int(s["success"])
            assert s["terminated"] is s["success"] and s["truncated"] is (len(rewards)==horizon and not s["success"])
            assert (s["terminated"] or s["truncated"]) is (len(rewards)==n)
        physical.append(actual)
        issued.append(wire)
        proposals.append(raw)
    last=ep["chunks"][-1]["steps"][-1]
    assert all(rec[k] is last[k] for k in ("terminated","truncated","success")) and rec["terminated"]!=rec["truncated"]
    assert rewards==rec["native_rewards"] and sum(rewards)==rec["native_return"]==int(rec["success"])
    summary = dict(success=rec["success"],controls=n,return_sum=sum(rewards),terminated=rec["terminated"],
                   truncated=rec["truncated"],max_action_residual=worst)
    return summary, (np.concatenate(physical),np.concatenate(issued),np.stack(proposals))


def differences(left,right):
    """Control index and policy-call index are 1-based. Later physical controls exclude control one."""
    a,w,p = left
    b,v,q = right
    n,k = min(len(a),len(b)),min(len(p),len(q))
    pa = np.any(a[:n]!=b[:n],axis=1)
    wa = np.any(w[:n]!=v[:n],axis=1)
    rp = np.any(p[:k]!=q[:k],axis=(1,2))
    first = lambda mask: int(np.flatnonzero(mask)[0]+1) if mask.any() else None
    return dict(common_steps=n,physical_changed=int(pa.sum()),later_physical_changed=int(pa[1:].sum()),
                after_first_chunk_changed=int(pa[5:].sum()),first_physical_difference=first(pa),
                different_lengths=len(a)!=len(b),first_issued_difference=first(wa),first_raw_call_difference=first(rp),
                common_calls=k,later_raw_calls_changed=int(rp[1:].sum()),
                same_physical_tape=bool(len(a)==len(b) and not pa.any()))


def opening(ep):
    return dict(reset=ep["native"]["reset"],body=ep["body"],environment=ep["env"],
                opening_proposal=ep["native"]["opening_proposal_sha256"],motor_inputs=ep["chunks"][0]["motor_inputs"],
                opening_noise=ep["chunks"][0]["noise"],first_raw=ep["chunks"][0]["raw"])


def group(folder,row,names,binding,key):
    path = folder/"pairs"/f"{row['index']:02d}.json.gz"
    raw = path.read_bytes()
    g = json.loads(gzip.decompress(raw))
    assert g["index"]==row["index"] and g[key]==binding and set(g["arms"])==set(names)==set(g["execution_order"])
    assert len(g["execution_order"])==len(names)
    ref = opening(g["arms"][names[0]])
    data,actions = {},{}
    for m in names:
        assert opening(g["arms"][m])==ref
        data[m],actions[m] = episode(row,m,g["arms"][m])
    return ref,data,actions,digest(raw)


def summarize(rows,eligible):
    draws = np.random.default_rng(1124).integers(8,size=(20000,8))
    clusters = [[r for r in rows if r["shard"]==s] for s in range(8)]
    assert all(len(g)==8 and len({(r["suite"],r["task_id"]) for r in g})==1 for g in clusters)
    names = list(rows[0]["new"]) + (list(rows[0]["old"]) if eligible else [])
    def tab(items):
        return {m:sum(r["success"][m] for r in items) for m in names}
    def discord(items,a,b):
        return [sum(r["success"][a] and not r["success"][b] for r in items),
                sum(r["success"][b] and not r["success"][a] for r in items)]
    def delta(items,a,b):
        x,y=discord(items,a,b)
        return 100*(x-y)/len(items)
    contrasts=[("A_real","B_real")]
    if "frozen_warp" in names:
        contrasts.extend(("frozen_warp",m) for m in ("B_real","A_real"))
    for m in ("A_real","B_real"):
        contrasts.extend((m,n) for n in ("shipped","true_map") if n in names)
    if eligible:
        contrasts.append(("A_24","B_24"))
    stats={}
    for a,b in contrasts:
        task = np.asarray([delta(g,a,b) for g in clusters])
        dist=task[draws].mean(axis=1)
        result=dict(effect_pp=float(task.mean()),ci95_pp=np.quantile(dist,[.025,.975]).tolist(),
                    rescues_losses=discord(rows,a,b),per_task_pp=task.tolist())
        if (a,b)==("A_real","B_real"):
            result["ci97_5_pp"]=np.quantile(dist,[.0125,.9875]).tolist()
        key=f"{a} minus {b}"
        result["strata"]={}
        for col in ("angle_deg","regime"):
            result["strata"][col]={}
            for val in sorted({r[col] for r in rows}):
                subset=[r for r in rows if r[col]==val]
                result["strata"][col][str(val)]=dict(n=len(subset),effect_pp=delta(subset,a,b),rescues_losses=discord(subset,a,b))
        diffs = [r["action_differences"].get(f"{a}/{b}",r["action_differences"].get(f"{b}/{a}")) for r in rows]
        result["action_checks"]=dict(first_physical_changed=sum(d["first_physical_difference"]==1 for d in diffs),
            different_after_first=sum(d["later_physical_changed"]>0 for d in diffs),
            different_after_five=sum(d["after_first_chunk_changed"]>0 for d in diffs),
            later_proposals_changed=sum(d["later_raw_calls_changed"]>0 for d in diffs),
            same_complete_physical_tapes=sum(d["same_physical_tape"] for d in diffs))
        stats[key]=result
    return dict(starts=len(rows),successes=tab(rows),per_task=[
                dict(task=f"{g[0]['suite']}/{g[0]['task_id']}",shard=g[0]["shard"],successes=tab(g),
                     rescues_losses=discord(g,"A_real","B_real")) for g in clusters],contrasts=stats)


def main(body, attempt,job):
    shared=ROOT/HERE.name
    folder=shared/attempt
    packet=read(shared/"release01"/f"{body}.json",PACKETS[body])
    runtime=json.loads((folder/"runtime.json").read_bytes())
    launch=dict(field.split("=",1) for field in (folder/"launch.txt").read_text().split())
    assert runtime["host"]==launch["host"]==job+"-master-0" and runtime["pin"]==launch["pin"] and runtime["body"]==body
    assert (runtime["original"],runtime["deps"])==TRACES[body] and runtime["release_sha256"]==PACKETS[body]
    assert runtime["motor_manifest_sha256"]==TRANSPORT and runtime["motor_ready_sha256"]==MOTOR_READY
    assert runtime["worker_sha256"]==digest((HERE.parent/UR/"audited_worker.py").read_bytes())
    assert (folder/"exit.txt").read_text().split()[0]=="0"
    old_dir,_,old_hash=ORIGINAL[body]
    checked(old_dir/"launch.txt",LAUNCH[body])
    oldsites={r["index"]:r for r in original(body)}
    old_binding=old_hash if body=="UR5e" else KIN_ORIGINAL_RELEASE
    binding_field="panel_sha256" if body=="UR5e" else "release_sha256"
    old_available=(old_dir/"exit.txt").is_file() and (old_dir/"exit.txt").read_text().split()[0]=="0"
    expected={f"{r['index']:02d}.json.gz" for r in packet["rows"]}
    assert len(expected)==64 and expected=={p.name for p in (folder/"pairs").iterdir()}
    if old_available:
        assert expected=={p.name for p in (old_dir/"pairs").iterdir()}
    result=[]
    for r in packet["rows"]:
        names=["A_real","B_real"] + (["frozen_warp"] if body=="UR5e" else [])
        op,new,acts,sha=group(folder,r,names,PACKETS[body],"release_sha256")
        archive,old_sha,unmatched={},None,["original incomplete"]
        opening_check = None
        ar = oldsites[r["index"]]
        assert r["true_map"]==ar["true_map"] and r["query"]==ar["query"] and r["reference"]==ar["reference"]
        if old_available:
            old_names=list(ar["corrections"])
            oldop,baseline,arch_acts,old_sha=group(old_dir,ar,old_names,old_binding,binding_field)
            unmatched=[k for k in op if op[k]!=oldop[k]]
            diff = np.abs(np.subtract(oldop["first_raw"],op["first_raw"]))
            opening_check = dict(reset_differences=[k for k in op["reset"] if op["reset"][k]!=oldop["reset"][k]],
                environment_differences=[k for k in op["environment"] if op["environment"][k]!=oldop["environment"][k]],
                motor_input_differences=[k for k in op["motor_inputs"] if op["motor_inputs"][k]!=oldop["motor_inputs"][k]],
                max_abs_raw=float(diff.max()),changed_raw_components=int(np.count_nonzero(diff)),
                old_opening=oldop["opening_proposal"])
            for m in old_names:
                label=m+"_24" if m in ("A","B") else m
                archive[label]=baseline[m]
                acts[label]=arch_acts[m]
            if not unmatched:
                for m in names:
                    for o in old_names:
                        other=o+"_24" if o in ("A","B") else o
                        if r["corrections"][m]==ar["corrections"][o]:
                            if not all(np.array_equal(x,y) for x,y in zip(acts[m],acts[other])):
                                unmatched.append(f"identical correction produced different actions: {m}/{other}")
        comparisons={f"{a}/{b}":differences(acts[a],acts[b]) for a,b in combinations(acts,2)}
        result.append(dict(**{k:r[k] for k in ("index","suite","task_id","shard","slot","angle_deg","regime","query_layout","noise_seed")},
            new=new,old=archive,new_tape_sha256=sha,old_tape_sha256=old_sha,query_physics=op["reset"]["physics"],
            opening=op["opening_proposal"],opening_inputs=op["motor_inputs"],noise_sha256=op["reset"]["noise_sha256"],
            cross_job_mismatch=unmatched,cross_job_opening=opening_check,action_differences=comparisons,
            success={m:v["success"] for m,v in (new|archive).items()}))
    eligible=all(not r["cross_job_mismatch"] for r in result)
    stats=summarize(result,eligible)
    if old_available:
        stats["historical_success_counts"]={m:sum(r["old"][m]["success"] for r in result) for m in result[0]["old"]}
        stats["historical_cross_job_qualification"]=dict(
            reset_equal=sum(not r["cross_job_opening"]["reset_differences"] for r in result),
            motor_inputs_equal=sum(not r["cross_job_opening"]["motor_input_differences"] for r in result),
            raw_opening_equal=sum(r["cross_job_opening"]["changed_raw_components"]==0 for r in result),
            max_abs_raw=max(r["cross_job_opening"]["max_abs_raw"] for r in result))
    return dict(body=body,job=job,attempt=attempt,runtime=runtime,gpu_exit=(folder/"exit.txt").read_text().strip(),
                release_sha256=PACKETS[body],original= str(old_dir),original_launch_sha256=LAUNCH[body],
                archive_eligible=eligible,matched_old_openings=sum(not r["cross_job_mismatch"] for r in result),
                original_complete=old_available,summary=stats,rows=result)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("body",choices=PACKETS)
    p.add_argument("attempt")
    p.add_argument("job")
    p.add_argument("output",type=Path)
    args=p.parse_args()
    result=main(args.body,args.attempt,args.job)
    with args.output.open("x") as f:
        json.dump(result,f,separators=(",",":"),allow_nan=False)
        f.write("\n")
    print(args.body,result["summary"]["starts"],args.output)
