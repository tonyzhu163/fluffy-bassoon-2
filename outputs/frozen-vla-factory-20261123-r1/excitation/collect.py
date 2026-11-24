"""Collect matched 15-control supports in the pinned, headless native simulator."""
import argparse
from contextlib import ExitStack
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from inputs import BASE, HERE, OUTPUTS, RELEASE, SEED, designs, rot, spread, supports

sys.path.insert(0, str(OUTPUTS / "libero-crossarm-20261120-r1"))


def digest(value):
    return hashlib.sha256(np.asarray(value).tobytes()).hexdigest()


def rollout(env, transform, commands):
    """Private map exists on the actuator side. Return public commands and FK only."""
    import mujoco
    from sim_reset import tool_state

    scratch = mujoco.MjData(env.sim.model._model)
    points = [tool_state(env, scratch)["tcp_xyz"][:2]]
    reset = digest(env.get_sim_state())
    for vector in commands:
        physical = transform @ vector
        assert np.max(np.abs(physical)) <= 1
        action = np.r_[physical, 0., 0., 0., 0., -1.].astype(np.float32)
        for _ in range(5):
            _, _, done, _ = env.step(action.copy())
            if done or env.check_success():
                raise RuntimeError("Task terminated during the fixed support")
        points.append(tool_state(env, scratch)["tcp_xyz"][:2])
    return dict(x=commands.tolist(), y=np.diff(points, axis=0).tolist(),
                reset=reset, controls=15)


def collect_child(pipe, env, transform, commands):
    try:
        pipe.send((True, rollout(env, transform, commands)))
    except BaseException:
        pipe.send((False, traceback.format_exc()))
    finally:
        pipe.close()


def fork_rollout(env, transform, commands):
    # One child at a time. The parent stays at the settled state, including
    # controller history, observable caches and RNG; no partial reset on reuse.
    ctx = mp.get_context("fork")
    read, write = ctx.Pipe(duplex=False)
    child = ctx.Process(target=collect_child, args=(write, env, transform, commands))
    child.start()
    write.close()
    try:
        ok, data = read.recv()
    finally:
        read.close()
        child.join()
    if not ok or child.exitcode:
        raise RuntimeError(data)
    return data


def collect_task(task, ref, candidates, angles):
    from arm_env import (ArmEnv, ControlEnv, _identity, _reset_wrapper, _set_controller_mode,
                         _settle_environment, load_panda_row, scene_state, transfer_scene)

    suite, task_id = task
    seed = ref["support_seed"]
    assert ref["support_layout"] == 11
    cfg = ArmEnv("UR5e", suite, task_id)
    args = dict(bddl_file_name=str(cfg.bddl), gripper_types="PandaGripper",
                initialization_noise=None, control_freq=20, hard_reset=False,
                use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    with ExitStack() as stack:
        np.random.seed(seed)
        donor = ControlEnv(robots=["Panda"], **args)
        stack.callback(donor.close)
        env = ControlEnv(robots=["UR5e"], **args)
        stack.callback(env.close)
        for wrapper in (donor, env):
            _reset_wrapper(wrapper, seed)
        load_panda_row(donor, cfg._states[11])
        transfer_scene(donor, env, scene_state(donor))
        identity = _identity(env, "UR5e")
        _settle_environment(env)
        env.set_perturbation_seed(seed)
        env.begin_perturbation_rollout()
        _set_controller_mode(env, "relative")
        assert not env.check_success()
        q0 = env.sim.data.qpos[env.robots[0]._ref_joint_pos_indexes]
        q0_err = float(np.max(np.abs(q0 - ref["q0"])))
        assert q0_err < 2e-6  # existing active replay joint tolerance
        task_name = f"{suite}/{task_id}"
        reset = digest(env.get_sim_state())
        receipt = dict(task=task_name, reset=reset, q0_error_rad=q0_err,
                       seed=seed, layout=11, robot=identity["xml_sha256"],
                       gripper=identity["gripper_xml_sha256"])
        result = []
        for motor in ("pi05", "pi0"):
            panel = [r for r in candidates if r["task"] == task_name and r["motor"] == motor]
            assert len(panel) == 12
            panel.sort(key=lambda r: (spread(designs(r["x"])["passive"]), r["index"]))
            for group, template in (("low", panel[2]), ("high", panel[9])):
                for sign in (1, -1):
                    transform = rot(angles[0]) @ np.diag([1.45, sign * .70]) @ rot(angles[1])
                    for amplitude in (.155, .62):
                        for design, unit in designs(template["x"]).items():
                            cmd = amplitude * unit
                            assert np.allclose(np.linalg.norm(cmd, axis=1), amplitude)
                            rec = fork_rollout(env, transform, cmd)
                            assert rec.pop("reset") == reset and rec.pop("controls") == 15
                            result.append(dict(task=task_name, motor=motor, template=group,
                                               tape_sha=template["tape_sha"], index=template["index"],
                                               sign=sign, amplitude=amplitude, design=design,
                                               truth=transform.tolist(), **rec))
        return result, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "supports.json.gz")
    args = parser.parse_args()
    assert not os.environ.get("CUDA_VISIBLE_DEVICES")
    release = os.environ["CROSSARM_RELEASE"]
    pin = subprocess.check_output(["git", "-C", release, "rev-parse", "HEAD"], text=True).strip()
    assert pin == RELEASE
    candidates, refs = supports()
    rng = np.random.default_rng(SEED)
    rows, receipts = [], []
    for task, ref in sorted(refs.items()):
        data, receipt = collect_task(task, ref, candidates, rng.uniform(-np.pi, np.pi, 2))
        rows.extend(data)
        receipts.append(receipt)
        print(f"{receipt['task']} {len(data)} supports, q0 error {receipt['q0_error_rad']:.3g} rad",
              flush=True)
    assert len(rows) == 384 and len(receipts) == 8
    payload = json.dumps(dict(base=BASE, release=RELEASE, seed=SEED, tasks=receipts, rows=rows),
                         separators=(",", ":"), allow_nan=False).encode()
    args.out.write_bytes(gzip.compress(payload, mtime=0))


if __name__ == "__main__":
    main()
