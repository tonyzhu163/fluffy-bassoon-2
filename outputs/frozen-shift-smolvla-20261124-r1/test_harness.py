"""CPU fixture: the episode loop with a scripted motor and a toy arm. No simulator, no weights."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import harness  # noqa: E402


class ToyArm:
    """Succeeds once the summed first command component reaches `need`."""

    def __init__(self, need=60.0, max_steps=120, secret=0.0):
        self.need, self.max_steps, self.secret = need, max_steps, secret

    def _obs(self):
        return dict(agentview_image=np.full((4, 4, 3), self.t % 255, np.uint8),
                    robot0_eye_in_hand_image=np.zeros((4, 4, 3), np.uint8),
                    robot0_joint_pos=np.zeros(7, np.float32), robot0_joint_vel=np.ones(7, np.float32),
                    robot0_eef_pos=np.array([self.x, 0, 0], np.float32),
                    robot0_eef_quat_xyzw=np.array([0, 0, 0, 1], np.float32),
                    robot0_gripper_qpos=np.zeros(2, np.float32), object_state=np.array([self.secret]))

    def reset(self, init_state_id, seed):
        self.t, self.x = 0, 0.0
        return dict(observation=self._obs(), language_instruction="open the drawer", max_steps=self.max_steps,
                    environment_reproducibility=dict(scene_fingerprint=f"scene-{init_state_id}-{seed}"))

    def step(self, action):
        assert action.dtype == np.float32 and np.all(np.abs(action) <= 1)
        self.t += 1
        self.x += float(action[0])
        success = self.x >= self.need
        return dict(observation=self._obs(), reward=float(success), success=success, terminated=success,
                    truncated=self.t >= self.max_steps and not success)


class ScriptedMotor:
    """Chunk gain is the mean of the noise it was given; it records every call."""

    def __init__(self):
        self.calls = []

    def proposal(self, obs, prompt, noise, hooks=None, context=None):
        assert set(obs) == set(harness.PUBLIC)  # reward, joint velocity and object state never arrive
        state = np.r_[obs["robot0_eef_pos"], np.zeros(5)].astype(np.float32)
        if hooks is not None:
            state = hooks.state(state.copy(), obs, context)
        self.calls.append(dict(prompt=prompt, noise=noise.copy(), state=state, step=context["step"]))
        chunk = np.zeros((50, 7), np.float32)
        chunk[:, 0] = 1.5 if "fast" in prompt else float(np.sign(noise.mean()) or 1) * 0.5
        return chunk, dict(state=state.tolist(), state_digest=harness.digest(state),
                           agent_digest=harness.digest(obs["agentview_image"]), wrist_digest="w")


def test_untreated_episode_is_matched_across_arms_and_repeats():
    runs = [harness.run_episode(ToyArm(secret=s), ScriptedMotor(), 14, 7, 7) for s in (0.0, 1.0)]
    assert runs[0]["command_digest"] == runs[1]["command_digest"]
    assert [r["noise_digests"] for r in runs[0]["replans"]] == [r["noise_digests"] for r in runs[1]["replans"]]
    other = harness.run_episode(ToyArm(), ScriptedMotor(), 14, 7, 8)
    assert other["replans"][0]["noise_digests"] != runs[0]["replans"][0]["noise_digests"]
    assert runs[0]["scene_fingerprint"] == "scene-14-7" and runs[0]["adapter"] == "raw"


def test_timeout_is_reported_as_failure_with_full_horizon():
    episode = harness.run_episode(ToyArm(need=1e9, max_steps=120), ScriptedMotor(), 0, 1, 1)
    assert not episode["success"] and episode["truncated"] and episode["controls"] == 120
    assert len(episode["commands"]) == 120 and [r["step"] for r in episode["replans"]] == [0, 50, 100]


def test_commands_are_clipped_and_counted_and_success_stops_the_episode():
    class Fast(harness.Hooks):
        name = "fast"

        def prompt(self, prompt, obs, context):
            return prompt + " fast"

    episode = harness.run_episode(ToyArm(need=60), ScriptedMotor(), 0, 1, 1, Fast())
    assert episode["success"] and episode["controls"] == 60 and episode["clipped_controls"] == 60
    assert np.max(episode["commands"]) == 1.0 and episode["replans"][0]["prompt"].endswith("fast")


def test_every_hook_changes_what_the_motor_receives_or_executes():
    class All(harness.Hooks):
        name, proposals, replan = "all", 3, 5

        def observation(self, obs, context):
            return dict(obs, robot0_eef_pos=obs["robot0_eef_pos"] + 1)

        def state(self, state8, obs, context):
            return state8 * 2

        def noise(self, draw, context):
            return -np.abs(draw)

        def choose(self, proposals, obs, context):
            assert len(proposals) == 3 and len(context["commands"]) == context["step"]
            return 2

    motor = ScriptedMotor()
    episode = harness.run_episode(ToyArm(need=1e9, max_steps=12), motor, 0, 1, 1, All())
    assert [r["step"] for r in episode["replans"]] == [0, 5, 10] and episode["replans"][0]["chosen"] == 2
    assert len(episode["replans"][0]["noise_digests"]) == 3 and len(motor.calls) == 9
    assert motor.calls[0]["state"][0] == 2.0 and (motor.calls[0]["noise"] <= 0).all()
    assert np.allclose(np.asarray(episode["commands"])[:, 0], -0.5)


def test_bad_noise_shape_is_refused():
    class Bad(harness.Hooks):
        def noise(self, draw, context):
            return draw[:, :10]

    with pytest.raises(ValueError):
        harness.run_episode(ToyArm(), ScriptedMotor(), 0, 1, 1, Bad())
