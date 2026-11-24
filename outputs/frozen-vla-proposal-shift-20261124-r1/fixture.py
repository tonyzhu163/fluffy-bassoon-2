"""Deterministic CPU simulator, policy and packer for harness tests. Rewards are synthetic.

Patch the packer before a rollout:
    monkeypatch.setattr(ur5e_motor, "to_openpi_observation", fixture.pack)
"""
import numpy as np

from harness import TASK, array_digest, input_digests


def case(arm, layout):
    return dict(arm=arm, suite="libero_goal", task_id=3, variant="BASE",
                init_state_id=layout, seed=42710300 + layout)


def pack(raw, prompt):
    state = np.concatenate((raw["robot0_eef_pos"], raw["robot0_eef_quat_xyzw"], raw["robot0_gripper_qpos"]))
    return {"prompt": prompt, "observation/state": state, "observation/image": raw["agentview_image"],
            "observation/wrist_image": raw["robot0_eye_in_hand_image"]}


class Policy:
    """Each action column answers to one input: prompt, noise, camera, wrist camera, state."""

    def __init__(self):
        self.calls = []

    def infer(self, item, *, noise):
        self.calls.append(dict(prompt=item["prompt"], noise=array_digest(noise), inputs=input_digests(item)))
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:, 0] = .3
        actions[:, 1] = float(item["prompt"] != TASK)
        actions[:, 2] = np.clip(.1 * noise[0, 0], -.2, .2)
        actions[:, 3] = item["observation/image"].mean() / 255
        actions[:, 4] = item["observation/wrist_image"].mean() / 255
        actions[:, 5] = item["observation/state"][0]
        return {"actions": actions}


class Plant:
    """Odd layouts pay only after a control with column 1 above .25.
    Even layouts pay unless a control had column 2 below -.25."""

    def __init__(self):
        self.horizon = 20
        self.stop_at = None
        self.action_schema = dict(low=[-1.] * 7, high=[1.] * 7)
        self.corrupt = False
        self.resets = 0

    def observation(self):
        return dict(
            robot0_eef_pos=self.position.copy(), robot0_eef_quat_xyzw=np.array([0., 0., 0., 1.]),
            robot0_gripper_qpos=np.full(2, .001 * self.tick), robot0_joint_pos=np.full(self.joints, .01 * self.tick),
            agentview_image=np.full((8, 8, 3), self.tick, np.uint8),
            robot0_eye_in_hand_image=np.full((8, 8, 3), 3 * self.tick, np.uint8),
            # Audit-only keys that an adapter must never receive.
            robot0_joint_vel=np.zeros(self.joints), object_state=np.array([float(self.cued)]),
        )

    def call(self, op, value):
        if op == "create":
            self.tick, self.cued, self.spoiled, self.executed = 0, False, False, []
            self.resets += 1
            self.layout = value["init_state_id"]
            self.joints = 6 if value["arm"] == "UR5e" else 7
            self.position = np.zeros(3)
            physics = f"fixture-{value['arm']}-{self.layout}" + (f"-{self.resets}" if self.corrupt else "")
            raw = self.observation()
            return dict(max_steps=self.horizon), dict(
                observation=raw, anchor_eef_pos=raw["robot0_eef_pos"], language_instruction=TASK,
                action_schema=self.action_schema, agentview_64=raw["agentview_image"],
                environment_reproducibility=dict(scene_fingerprint="fixture", perturbation_plan_fingerprint="fixture",
                                                 initial_state_fingerprint=physics),
                model_identity=dict(arm=value["arm"], model_class="fixture", joint_count=self.joints),
                renderer_vendor="fixture", renderer="fixture", worker_step_index=0)
        assert op == "chunk"
        end = self.horizon if self.stop_at is None else min(self.horizon, self.stop_at)
        issued = np.array(value[:end - self.tick])
        self.executed.append(issued)
        self.cued |= bool((issued[:, 1] > .25).any())
        self.spoiled |= bool((issued[:, 2] < -.25).any())
        self.tick += len(issued)
        self.position = self.position + .01 * issued[:, :3].sum(axis=0)
        raw = self.observation()
        terminal = self.tick == end
        success = terminal and (self.cued if self.layout % 2 else not self.spoiled)
        return dict(observation=raw, steps=len(issued), rewards=[0.] * (len(issued) - 1) + [float(success)],
                    terminated=success, truncated=terminal and not success, success=success,
                    language_instruction=TASK, positions=[raw["robot0_eef_pos"]], agentview_64=raw["agentview_image"],
                    gripper_positions=[raw["robot0_gripper_qpos"]], joint_positions=[raw["robot0_joint_pos"]])
