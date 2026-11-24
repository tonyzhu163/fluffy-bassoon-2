"""Simulator worker for Track 1: cross-arm native OSC_POSE vs JOINT_POSITION decoders."""
from multiprocessing.connection import Connection
import sys
import traceback

import mujoco
import numpy as np

from arm_env import CAMERAS, _public, _register_arm, _reset_wrapper
from diag_cpu import CrossArmDecoder, GeneralChain
from libero.libero.envs.env_wrapper import ControlEnv
from native import JOINT, OSC, Pair
from sim_reset import scene_state, transfer_scene
from ur5e_sim_worker import _probe_egl


def _step_checked(env, action, dof):
    lo, hi = env.env.action_spec
    a = np.asarray(action, dtype=np.float64)
    if a.shape != lo.shape or not np.isfinite(a).all() or np.any(a < lo) or np.any(a > hi):
        raise ValueError(f"Illegal native action shape={a.shape} range=[{a.min()},{a.max()}]")
    raw, reward, done, _ = env.step(a)
    success = bool(env.check_success())
    return _public(raw, dof), float(reward), success


def run(fd, fixture):
    pair = None
    arm_envs = {}
    with Connection(fd) as conn:
        try:
            vendor, renderer = _probe_egl(fixture)
            while True:
                op, v = conn.recv()
                if op == "case":
                    for e in arm_envs.values():
                        e.close()
                    arm_envs.clear()
                    if pair is not None:
                        pair.close()
                    suite, layout, seed, arm = v["suite"], v["layout"], v["seed"], v["arm"]
                    pair = Pair(suite, layout, seed, cameras=False)
                    pair.start(OSC)
                    src_env = pair.envs[OSC]
                    src_scene = scene_state(src_env)
                    if arm != "Panda":
                        _register_arm(arm)
                    for ctrl in (OSC, JOINT):
                        np.random.seed(seed)
                        e = ControlEnv(
                            bddl_file_name=str(pair.bddl),
                            robots=[arm],
                            controller=ctrl,
                            gripper_types="PandaGripper",
                            control_freq=20,
                            use_camera_obs=True,
                            has_offscreen_renderer=True,
                            camera_heights=256,
                            camera_widths=256,
                            camera_names=list(CAMERAS),
                        )
                        arm_envs[ctrl] = e
                    chain = GeneralChain(arm_envs[JOINT])
                    cur_arm = arm
                    cur_seed = seed
                    payload = dict(
                        state=pair.full_hash,
                        physics=pair.reset_hash,
                        row=pair.row_hash,
                        bddl=pair.bddl.name,
                        horizon=pair.horizon,
                        prompt=pair.prompt,
                        dof=chain.dof,
                        renderer=renderer,
                        vendor=vendor,
                    )
                elif op == "start":
                    method = v
                    ctrl = OSC if method == OSC else JOINT
                    env = arm_envs[ctrl]
                    _reset_wrapper(env, cur_seed)
                    transfer_scene(src_env, env, src_scene)
                    env.robots[0].gripper.current_action = src_env.robots[0].gripper.current_action.copy()
                    env.set_perturbation_seed(cur_seed)
                    raw = env.begin_perturbation_rollout()
                    obs = _public(raw, chain.dof)
                    sensors = env.env._observables
                    q0 = sensors["robot0_joint_pos"]._sensor({})
                    decoder = None if method == OSC else CrossArmDecoder(chain, q0, method)
                    t = 0
                    payload = dict(
                        obs=obs,
                        controller=env.robots[0].controller.name,
                        dof=chain.dof,
                    )
                elif op == "chunk":
                    if not 1 <= len(v) <= 5 or t >= pair.horizon:
                        raise ValueError("Invalid motor chunk")
                    actions, rewards = [], []
                    sensors = env.env._observables
                    for canonical in v:
                        if decoder is None:
                            a = np.asarray(canonical, dtype=np.float64)
                        else:
                            q = sensors["robot0_joint_pos"]._sensor({})
                            v_jnt = sensors["robot0_joint_vel"]._sensor({})
                            a = decoder.action(q, v_jnt, canonical)
                        obs, reward, success = _step_checked(env, a, chain.dof)
                        t += 1
                        actions.append(a.tolist())
                        rewards.append(reward)
                        if success or t == pair.horizon:
                            break
                    payload = dict(
                        obs=obs,
                        native=actions,
                        rewards=rewards,
                        success=bool(rewards[-1]),
                        truncated=(t == pair.horizon and not rewards[-1]),
                    )
                elif op == "close":
                    conn.send((True, None))
                    break
                else:
                    raise ValueError(op)
                conn.send((True, payload))
        except EOFError:
            pass
        except Exception:
            conn.send((False, traceback.format_exc()))
        finally:
            for e in arm_envs.values():
                e.close()
            if pair is not None:
                pair.close()


if __name__ == "__main__":
    run(int(sys.argv[1]), "--fixture" in sys.argv[2:])
