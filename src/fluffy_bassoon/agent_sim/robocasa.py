"""RoboCasa365's native Gym interface, with Human300 action conventions.

No model, reward shaping, success-based stopping or guessed IK is introduced.
Run in the pinned native RoboCasa runtime on an allocated Linux GPU.
"""
import importlib.metadata
import os
import random
import sys

import numpy as np

from .core import Observation, Transition


CAMERAS = ("video.robot0_agentview_left", "video.robot0_eye_in_hand",
           "video.robot0_agentview_right")
STATES = ("state.end_effector_position_relative", "state.end_effector_rotation_relative",
          "state.base_position", "state.base_rotation", "state.gripper_qpos")


class RoboCasaBackend:
    def __init__(self, *, split="pretrain", execution="baige"):
        if split not in ("pretrain", "target"):
            raise ValueError("Use an explicit native pretrain or target split")
        self.split, self.execution = split, execution
        self.env = None
        self.runtime = {}

    def describe(self):
        return dict(benchmark="RoboCasa365", split=self.split, action_dim=12,
            action_format="Human300 decoded commands before native convert_action",
            action_order=["arm_dx", "arm_dy", "arm_dz", "arm_drx", "arm_dry", "arm_drz",
                          "gripper_close", "base_0", "base_1", "base_2", "torso", "base_mode"],
            action_bounds=[-1, 1], binary_coordinates=[6, 11], binary_threshold=0.5,
            action_note="Arm commands are normalized controller deltas, not absolute metre targets. "
                        "Use the measured controller scale/frame below; no action clipping is applied.",
            camera_order=list(CAMERAS), state_order=list(STATES),
            quaternion_order="xyzw", success_termination=False, **self.runtime)

    def reset(self, task, seed):
        if sys.platform != "linux":
            raise RuntimeError("Run native simulation on allocated Linux resources")
        from fluffy_bassoon.gpu_execution import validate_execution
        validate_execution(self.execution, os.environ)
        if os.environ.get("MUJOCO_GL") != "egl" or "MUJOCO_EGL_DEVICE_ID" not in os.environ:
            raise RuntimeError("Bind EGL rendering to the probed allocation before starting")
        import gymnasium as gym
        import robocasa  # noqa: F401 -- registers native environments
        from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS
        from fluffy_bassoon.robocasa.pairing import stable_counter_regions

        if task not in ATOMIC_TASK_DATASETS:
            raise ValueError("This collector currently admits RoboCasa365 atomic tasks")
        self.close()
        stable_counter_regions()
        random.seed(seed)
        np.random.seed(seed)
        self.env = gym.make("robocasa/" + task, split=self.split, seed=seed)
        obs, _ = self.env.reset(seed=seed)
        native = self.env.unwrapped.env
        ctrl = native.robots[0].composite_controller.part_controllers["right"]
        if ctrl.input_type != "delta" or ctrl.input_ref_frame != "base":
            raise RuntimeError("Require the Human300 native delta/base-frame controller")
        if not native.ignore_done:
            raise RuntimeError("Native full-horizon collection requires ignore_done=True")
        self.runtime = dict(robocasa_version=importlib.metadata.version("robocasa"),
            registry_horizon=ATOMIC_TASK_DATASETS[task]["horizon"],
            control_freq=float(native.control_freq), controller=dict(
                input_type=ctrl.input_type, input_ref_frame=ctrl.input_ref_frame,
                input_min=np.asarray(ctrl.input_min).tolist(), input_max=np.asarray(ctrl.input_max).tolist(),
                output_min=np.asarray(ctrl.output_min).tolist(), output_max=np.asarray(ctrl.output_max).tolist()))
        self.initial = True
        return self._observation(obs)

    def _observation(self, raw):
        return Observation(images={k: np.asarray(raw[k]).copy() for k in CAMERAS},
            state={k: np.asarray(raw[k]).copy() for k in STATES},
            instruction=str(raw["annotation.human.task_description"]),
            sim_time=float(self.env.unwrapped.env.sim.data.time))

    def canonicalize(self, actions):
        if actions.ndim != 2 or actions.shape[1] != 12 or (np.abs(actions) > 1).any():
            raise ValueError("Require native [N,12] commands in [-1,1]; never silently clip")
        actions[:, [6, 11]] = (actions[:, [6, 11]] >= .5).astype(actions.dtype)
        return actions

    def step(self, action):
        from robocasa.utils.env_utils import convert_action

        obs, reward, terminated, truncated, info = self.env.step(convert_action(action))
        return Transition(self._observation(obs), action.copy(), float(reward),
                          bool(terminated), bool(truncated), bool(info["success"]), self.snapshot())

    def snapshot(self):
        native = self.env.unwrapped.env
        value = dict(sim_state=native.sim.get_state().flatten(),
                     body_names=list(native.sim.model.body_names),
                     body_positions=native.sim.data.body_xpos.copy(),
                     body_rotations=native.sim.data.body_xmat.copy())
        if self.initial:
            value.update(model_xml=native.sim.model.get_xml(), episode_metadata=native.get_ep_meta())
            self.initial = False
        return value

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
