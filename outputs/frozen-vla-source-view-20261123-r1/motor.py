"""Within-process frozen proposal contrasts and fresh native target scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from live import HERE, Simulator
from ur5e_motor import (NOISE_SEED_OFFSET, array_digest, load_frozen_pi05, run_episode,
                        to_openpi_observation, write_json)
from ur5e_sim_worker import UR5eSimulator

MODES = ("raw","control","panda")
STEPS = (0,30,45,60,90)
CAMS = ("agentview","robot0_eye_in_hand")


def case(i):
    return dict(arm="Kinova3",suite="libero_goal",task_id=3,variant="BASE",init_state_id=i,seed=41810300+i)


def propose(policy, item, noise):
    a = np.asarray(policy.infer(item,noise=noise)["actions"],dtype=np.float32)
    if a.shape != (10,7) or not np.isfinite(a).all():
        raise ValueError("Invalid action chunk")
    return a


def paired_frames(policy,out):
    frames = HERE/"evidence/cpu01"
    for i in (14,15):
        meta = json.loads((frames/f"layout-{i}.json").read_text())
        if meta["case"] != case(i) or [r["step"] for r in meta["frames"]] != list(STEPS):
            raise ValueError("Unexpected archived frame list")
        rng = np.random.default_rng(case(i)["seed"]+NOISE_SEED_OFFSET)
        # Same initial noise sequence as the stock 300-control episode.
        bank = rng.standard_normal((60,10,32)).astype(np.float32)
        for r in meta["frames"]:
            n = r["step"]
            noise = bank[n//5]
            f = frames/f"s{i}-t{n:03d}.npz"
            with np.load(f,allow_pickle=False) as z:
                proprio = {k:z[k] for k in ("robot0_joint_pos","robot0_joint_vel","robot0_eef_pos",
                                           "robot0_eef_quat_xyzw","robot0_gripper_qpos")}
                items = {}
                for mode in MODES:
                    public = dict(proprio, **{c+"_image":z[mode+"_"+c] for c in CAMS})
                    items[mode] = to_openpi_observation(public,meta["instruction"])
            states = [array_digest(v["observation/state"]) for v in items.values()]
            if len(set(states)) != 1:
                raise ValueError("Proprioception changed")
            hashes = {k:{field:array_digest(value) for field,value in item.items() if field != "prompt"}
                      for k,item in items.items()}
            if hashes["raw"] != hashes["control"] or hashes["control"] == hashes["panda"]:
                raise ValueError("Preprocessed motor control or intervention failed")
            if policy is None:
                stats = {}
                for key in ("observation/image","observation/wrist_image"):
                    a, b = items["control"][key], items["panda"][key]
                    stats[key] = dict(shape=list(a.shape), changed=int(np.any(a!=b,axis=-1).sum()))
                write_json(out/f"motor-input-{i}-{n:03d}.json",dict(hashes=hashes,images=stats,noise=array_digest(noise)))
                continue
            proposals = {m:propose(policy,items[m],noise) for m in MODES}
            proposals["raw_repeat"] = propose(policy,items["raw"],noise)
            row = dict(layout=i,step=n,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),input_hashes=hashes,
                       noise=array_digest(noise),instruction=meta["instruction"],
                       actions={k:v.tolist() for k,v in proposals.items()},differences={})
            for a,b in (("control","raw"),("raw_repeat","raw"),("panda","control")):
                delta = proposals[a]-proposals[b]
                row["differences"][a+"-minus-"+b] = dict(first_action=delta[0].tolist(),
                    rms_by_dimension=np.sqrt(np.mean(delta*delta,axis=0)).tolist(),max_abs=float(abs(delta).max()),
                    first_five_bounded_l2=float(np.linalg.norm(np.clip(proposals[a][:5],-1,1)-np.clip(proposals[b][:5],-1,1))))
            write_json(out/f"proposals-{i}-{n:03d}.json",row)


class AuditCalls:
    """Measure actual motor inputs and proposals without modifying either."""
    def __init__(self, policy):
        self.policy = policy
        self._model = getattr(policy,"_model",None)
        self.calls = []

    def infer(self,item,*,noise):
        result = self.policy.infer(item,noise=noise)
        self.calls.append(dict(input={k:array_digest(v) for k,v in item.items() if k!="prompt"},
                               prompt=item["prompt"], noise=array_digest(noise),
                               proposal=array_digest(np.asarray(result["actions"],np.float32))))
        return result


def native(policy,out,*,mesa=False):
    scores = []
    for i in (16,17):
        rows = {}
        for mode in MODES:
            trace = out/f"views-{i}-{mode}.jsonl"
            # Only the isolated EGL simulator child receives this environment adjustment.
            with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}):
                endpoint = UR5eSimulator(allow_mesa=mesa) if mode=="raw" else Simulator(mode,trace,mesa=mesa)
            audit = AuditCalls(policy)
            try:
                with endpoint as sim:
                    row = run_episode(sim,audit,case(i),get_embodiment_spec("nom"),
                                      max_controls=6 if mesa else None,canonicalize_quat=False)
            except RuntimeError as exc:
                # A future unrepresentable pose censors the branch; preserve prior frames and other starts.
                if mode == "panda" and trace.exists() and "invalid" in json.loads(trace.read_text().splitlines()[-1]):
                    write_json(out/f"inadmissible-{i}.json",dict(case=case(i),mode=mode,reason=str(exc)))
                    break
                raise
            finally:
                write_json(out/f"calls-{i}-{mode}.json",audit.calls)
            write_json(out/f"native-{i}-{mode}.json",dict(case=case(i),mode=mode,**row))
            if mode == "raw":
                raw_calls = audit.calls
            elif mode == "control" and raw_calls != audit.calls:
                raise ValueError("Target render changed motor input, noise or proposal in matched native loop")
            if not mesa and (not (row["terminated"] or row["truncated"])
                             or row["reset"]["renderer_vendor"] != "NVIDIA Corporation"
                             or bool(max(row["native_rewards"])) != row["success"]):
                raise ValueError("Incomplete native evaluation")
            if mode != "raw":
                ref = rows["raw"]
                for k in ("horizon","prompt","physics","perturbation_plan","scene_fingerprint","noise_sha256","action_schema"):
                    if row["reset"][k]!=ref["reset"][k]:
                        raise ValueError(f"Different score reset in {k}")
                if any(row["reset"]["public"][k]!=v for k,v in ref["reset"]["public"].items() if not k.endswith("_image")):
                    raise ValueError("Mismatched target sensors")
                if mode == "control" and (ref["reset"]!=row["reset"] or ref["opening_proposal_sha256"]!=row["opening_proposal_sha256"]
                                           or ref["controls"]!=row["controls"] or ref["native_rewards"]!=row["native_rewards"]):
                    raise ValueError("Same-input native control differed; inspect raw records")
                if mesa and (ref["native_rewards"]!=row["native_rewards"] or row["controls"]!=6):
                    raise ValueError("Fixed-control qualification mismatch")
            rows[mode]=row
        scores.append(dict(layout=i,paired=len(rows)==3,outcomes={m:r["success"] for m,r in rows.items()}))
    write_json(out/"native-pairs.json",scores)


class CpuProbe:
    def infer(self,item,*,noise):
        # IPC smoke only; ignore pixels and issue a fixed open-jaw motion.
        return dict(actions=np.tile(np.array([.1,-.1,.15,0,.05,0,-.9],np.float32),(10,1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output",type=Path)
    parser.add_argument("--cpu-smoke",action="store_true")
    args = parser.parse_args()
    if args.cpu_smoke:
        if os.environ.get("CUDA_VISIBLE_DEVICES") or len(os.sched_getaffinity(0))!=2:
            raise ValueError("Expected two CPU cores")
        paired_frames(None,args.output)
        native(CpuProbe(),args.output,mesa=True)
    else:
        policy=load_frozen_pi05()
        paired_frames(policy,args.output)
        native(policy,args.output)
