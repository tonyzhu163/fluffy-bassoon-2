"""Render and step the two native environments in the LIBERO interpreter."""
import json
from multiprocessing.connection import Connection
from pathlib import Path
import sys
import traceback

import numpy as np

from native import Decoder, JOINT, OSC, Pair, digest
from ur5e_sim_worker import _probe_egl


def run(fd, fixture):
    cal = json.loads((Path(__file__).parent/"evidence/calibration.json").read_text())
    pair = None
    with Connection(fd) as conn:
        try:
            vendor, renderer = _probe_egl(fixture)
            while True:
                op, v = conn.recv()
                if op == "case":
                    if pair is not None:
                        pair.close()
                    pair = Pair(**v, cameras=True)
                    payload = dict(state=pair.full_hash, physics=pair.reset_hash, row=pair.row_hash,
                                   bddl=pair.bddl.name, horizon=pair.horizon, prompt=pair.prompt,
                                   renderer=renderer, vendor=vendor)
                elif op == "start":
                    name = OSC if v == OSC else JOINT
                    obs = pair.start(name)
                    decoder = Decoder(pair.chain, obs["robot0_joint_pos"], v, cal)
                    if any(not np.array_equal(a, obs[k]) for k, a in pair.first[name].items()):
                        raise ValueError("Restored observation differs from the paired opening")
                    robot = pair.envs[name].robots[0]
                    low, high = pair.envs[name].env.action_spec
                    payload = dict(obs=obs, controller=robot.controller.name, low=low, high=high,
                                   native_output=robot.controller.output_max, joint_names=robot.robot_joints)
                    t = 0
                elif op == "chunk":
                    if not 1 <= len(v) <= 5 or t >= pair.horizon:
                        raise ValueError("Invalid motor chunk")
                    actions, rewards = [], []
                    for canonical in v:
                        a = decoder.action(obs["robot0_joint_pos"], canonical)
                        obs, reward = pair.step(name, a)
                        t += 1
                        actions.append(a)
                        rewards.append(reward)
                        if reward or t == pair.horizon:
                            break
                    payload = dict(obs=obs, native=actions, rewards=rewards,
                                   success=bool(rewards[-1]), truncated=t == pair.horizon and not rewards[-1])
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
            if pair is not None:
                pair.close()


if __name__ == "__main__":
    run(int(sys.argv[1]), "--fixture" in sys.argv[2:])
