"""Pinned LIBERO interpreter bridge. Preserve the raw observation and physics at every control."""
from collections import defaultdict
from contextlib import closing
from multiprocessing import Pipe
from multiprocessing.connection import Connection
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from large_panel_protocol import require
from ur5e_sim_worker import SIM_PYTHON, UR5eSimulator, _probe_egl

PROMPT = "push the plate to the front of the stove"
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
PROPRIO = ("robot0_eef_pos", "robot0_eef_quat_xyzw", "robot0_gripper_qpos", "robot0_joint_pos", "robot0_joint_vel")


class Simulator(UR5eSimulator):
    def __init__(self, fixture):
        self._closed = False
        self.pipe, peer = Pipe()
        try:
            self.worker = subprocess.Popen(
                [str(SIM_PYTHON), "-Bu", __file__, str(peer.fileno()), str(int(fixture))],
                pass_fds=(peer.fileno(),), env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        finally:
            peer.close()


class Tape:
    def __init__(self):
        self.values = defaultdict(list)
        self.previous = {}

    def observe(self, observation, flat):
        for key in PROPRIO:
            self.values[key].append(observation[key].copy())
        self.values["physics"].append(np.array(flat, copy=True))
        for key in CAMERAS:
            frame = observation[key]
            last = self.previous.get(key)
            self.values[key].append(frame.copy() if last is None else np.bitwise_xor(last, frame))
            self.previous[key] = frame.copy()

    def finish(self, path):
        import h5py

        with h5py.File(path, "x") as f:
            f.attrs["camera_encoding"] = "frame[0] raw; frame[t] XOR frame[t-1] thereafter (lossless)"
            for k, values in self.values.items():
                array = np.stack(values)
                if k in CAMERAS:
                    f.create_dataset(k, data=array, chunks=(1, *array.shape[1:]), compression="lzf")
                else:
                    f.create_dataset(k, data=array, compression="lzf")


def reset(arm, row, g):
    previous, arm._states = arm._states, (np.asarray(row["donor_state"], np.float64),)
    try:
        packet = arm.reset(0, row["layout_seed"])
    finally:
        arm._states = previous
    p = packet["environment_reproducibility"]
    require(p.pop("released_row_fingerprint") == row["initial_state_sha256"] and p["scene_fingerprint"] == row["scene_hash"]
            and all(p[k] == row[k] for k in ("suite", "task_id", "source_pin", "bddl_sha256", "manifest_sha256"))
            and p["seed"] == p["perturbation_plan"]["seed"] == row["layout_seed"]
            and p["perturbation_plan"]["case_id"] == "BASE", "wrong generated placement or release")
    p.update(generated_row_fingerprint=row["initial_state_sha256"], bank_slot=p.pop("init_state_id"),
             case_id=row["index"], settled_admission=g.collision_admission(arm._env))
    identity = packet["model_identity"]
    require(identity["model_class"] == "MountedKinova3" and identity["controller"] == "OSC_POSE" and
            identity["gripper"] == "PandaGripper" and identity["joint_count"] == 7 and identity["control_freq"] == 20 and
            packet["worker_step_index"] == 0 and packet["max_steps"] == 300 and packet["language_instruction"] == PROMPT,
            "wrong native task, horizon or controller")
    return packet


def serve(fd, fixture):
    import generated_layouts as g
    from plan import seed

    with Connection(fd) as pipe:
        try:
            renderer = _probe_egl(fixture)
            require("llvmpipe" in renderer[1].lower() if fixture else renderer[0] == "NVIDIA Corporation", "renderer")
            with g.native_module().ArmEnv("Kinova3", "libero_goal", 5) as arm:
                while True:
                    op, value = pipe.recv()
                    if op == "fixture":
                        with closing(g.make_sampler()) as sampler:
                            state, row = g.sample_layout(sampler, seed("layout", -1), g.released_fingerprints())
                        row.update(index=-1, donor_state=state.tolist(), noise_seed=seed("noise", -1))
                        result = row
                    elif op == "reset":
                        packet = reset(arm, value, g)
                        tape = Tape()
                        raw = packet["observation"]
                        tape.observe(raw, arm._env.get_sim_state())
                        info = {k: packet[k] for k in ("environment_reproducibility", "model_identity", "max_steps")}
                        result = raw, dict(info, renderer=list(renderer))
                    elif op == "chunk":
                        require(value.dtype == np.float32 and value.shape == (5, 7), "invalid control chunk")
                        positions, fingers = [], []
                        for action in value:
                            packet = arm.step(action)
                            require(np.array_equal(action, packet["issued_action"]) and packet["language_instruction"] == PROMPT,
                                    "native command or prompt changed")
                            for k in ("issued_action", "reward", "success", "terminated", "truncated", "worker_step_index"):
                                tape.values[k].append(packet[k])
                            raw = packet["observation"]
                            tape.observe(raw, arm._env.get_sim_state())
                            positions.append(raw["robot0_eef_pos"].copy())
                            fingers.append(raw["robot0_gripper_qpos"].copy())
                            if packet["terminated"] or packet["truncated"]:
                                break
                        result = raw, positions, fingers, (packet["terminated"] or packet["truncated"])
                    elif op == "save":
                        tape.finish(Path(value))
                        result = True
                    else:
                        raise ValueError(op)
                    pipe.send((True, result))
        except EOFError:
            pass
        except BaseException:
            pipe.send((False, traceback.format_exc()))


if __name__ == "__main__":
    serve(int(sys.argv[1]), bool(int(sys.argv[2])))
