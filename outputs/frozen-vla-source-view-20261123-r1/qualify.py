"""Two-core software-rendered, target-action-only drawer exposures."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from arm_env import ArmEnv, CAMERAS
from source_view import Exposure, Sibling, activate, compose
from ur5e_sim_worker import _probe_egl

STEPS = (0, 30, 45, 60, 90)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def case(layout):
    return dict(suite="libero_goal", task_id=3, arm="Kinova3", variant="BASE",
                init_state_id=layout, seed=41810300+layout)


def qualify(tapes, out, layout):
    spec = case(layout)
    tape_file = tapes / f"libero_goal-3-{layout}-Kinova3.json"
    tape = json.loads(tape_file.read_bytes())
    if tape["case"] != spec:
        raise ValueError("Unexpected archived case")
    with ExitStack() as stack:
        control = Sibling("Kinova3", layout, spec["seed"])
        stack.callback(control.close)
        panda = Sibling("Panda", layout, spec["seed"])
        stack.callback(panda.close)
        target = stack.enter_context(ArmEnv("Kinova3","libero_goal",3))
        expo = Exposure()
        with expo.capture():
            first = target.reset(layout,spec["seed"])
        native = target._env
        lineage = first["environment_reproducibility"]
        for k in ("released_row_fingerprint","scene_fingerprint","perturbation_plan_fingerprint","bddl_sha256"):
            if not all(x[k] == lineage[k] for x in (tape["full_reset"], control.initial["environment_reproducibility"],
                                                     panda.initial["environment_reproducibility"])):
                raise ValueError(f"Scene origin mismatch in {k}")
        rec = dict(case=spec, tape_sha256=hashlib.sha256(tape_file.read_bytes()).hexdigest(),
                   target_reset=lineage, historical_reset=tape["record"]["reset"]["physics"],
                   target_model=first["model_identity"], source_model=panda.initial["model_identity"],
                   instruction=first["language_instruction"], frames=[])
        actions = [(cmd, xyz, jaw, chunk["q1"] if i == len(chunk["cmd"])-1 else None)
                   for chunk in tape["chunks"] for i,(cmd,xyz,jaw) in
                   enumerate(zip(chunk["cmd"],chunk["xyz"],chunk["jaw"],strict=True))]
        obs = first["observation"]
        errors = []
        for n in range(max(STEPS)+1):
            if n:
                cmd, xyz, jaw, quat = actions[n-1]
                activate(native.sim)
                if n in STEPS:
                    expo = Exposure()
                    with expo.capture():
                        step = target.step(np.array(cmd,np.float32))
                else:
                    step = target.step(np.array(cmd,np.float32))
                obs = step["observation"]
                if step["terminated"] or step["truncated"] or step["reward"] != tape["record"]["native_rewards"][n-1]:
                    raise ValueError("Replay ended or reward sequence changed in selected prefix")
                er = [n, float(np.linalg.norm(obs["robot0_eef_pos"]-xyz)),
                      float(np.max(np.abs(obs["robot0_gripper_qpos"]-jaw)))]
                if quat is not None:
                    er.append(float((Rotation.from_quat(quat)*Rotation.from_quat(obs["robot0_eef_quat_xyzw"]).inv()).magnitude()))
                errors.append(er)
            if n not in STEPS:
                continue
            state_before = native.get_sim_state().copy()
            ed, exposure = expo.read(native, obs)
            ed.time = exposure["exposure_time"]
            ca, pa = control.sync(native,ed), panda.sync(native,ed)
            m = native.sim.model._model
            drawer = float(ed.qpos[m.joint("wooden_cabinet_1_top_level").qposadr[0]])
            row = dict(step=n, drawer_q_m=drawer, exposure=exposure, control=ca, panda=pa, images={})
            arrays = {k:v for k,v in obs.items() if not k.endswith("_image")}
            imgs = []
            for cam in CAMERAS:
                raw = obs[cam+"_image"]
                arrays["raw_"+cam] = raw
                if pa["admissible"]:
                    sham, swap, stats = compose(raw,control,panda,cam)
                    row["images"][cam] = stats
                    arrays["control_"+cam],arrays["panda_"+cam] = sham,swap
                    imgs.append(np.concatenate([raw,sham,swap],axis=1)[::-1])
            activate(native.sim)
            if not np.array_equal(state_before,native.get_sim_state()):
                raise ValueError("Sibling changed physical target state")
            row["target_state_sha256"] = hashlib.sha256(state_before.tobytes()).hexdigest()
            frame = f"s{layout}-t{n:03d}"
            np.savez_compressed(out/(frame+".npz"),**arrays)
            if imgs:
                Image.fromarray(np.concatenate(imgs,axis=0)).save(out/(frame+".jpg"),quality=90)
            rec["frames"].append(row)
            rec["replay_errors_step_m_m_rad"] = errors
            save(out/f"layout-{layout}.json", rec)
            print(frame,round(drawer,5),pa["admissible"],row["images"],flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tapes",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("layouts",nargs="*",type=int,default=[14,15])
    args = p.parse_args()
    start=time.monotonic()
    renderer = _probe_egl(True)
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus)!=2 or os.environ["CUDA_VISIBLE_DEVICES"] or "llvmpipe" not in renderer[1].lower():
        raise ValueError((cpus,renderer))
    info=dict(cpu_affinity=cpus, renderer=renderer, pin=os.environ["VIEW_PIN"], layouts=args.layouts, steps=STEPS)
    save(args.out/"runtime.json",info)
    for layout in args.layouts:
        qualify(args.tapes,args.out,layout)
    save(args.out/"runtime.json", dict(info,elapsed_s=time.monotonic()-start))
