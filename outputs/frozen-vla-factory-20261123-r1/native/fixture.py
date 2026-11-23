"""Execute the same paired driver and pinned motor episode loop with CPU test doubles."""
import argparse
import hashlib
import importlib.util
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from orbit import ARMS, OUTPUTS, command_tapes, corrections, panel, require, write_json
from rollout import paired_cell, shared_scale_grip, summarize

RUNNER_PIN = "ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c"
MOTOR_PATH = "outputs/libero-xemb-ur5e-sysid-20261120-r1/ur5e_motor.py"


def load_episode(tmp):
    path = tmp / "ur5e_motor.py"
    path.write_bytes(subprocess.check_output(["git", "-C", str(OUTPUTS.parent), "show", f"{RUNNER_PIN}:{MOTOR_PATH}"]))
    spec = importlib.util.spec_from_file_location("fixture_motor", path)
    motor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(motor)
    # Only substitute the observation packer: the OpenPI client / image resize library is a PFS asset.
    motor.to_openpi_observation = lambda raw, prompt: dict(
        prompt=prompt, **{"observation/image": raw["agentview_image"], "observation/state": raw["robot0_eef_pos"]})
    return motor.run_episode


class FixturePolicy:
    def __init__(self, commands):
        self.commands = commands
        self.calls = 0

    def infer(self, item, *, noise):
        self.calls += 1
        actions = self.commands[:10].copy()
        actions[:, :3] += item["observation/state"] * 0.01
        actions[:, :2] += noise[:, :2] * 0.001
        return {"actions": actions}


class FixtureSim:
    def call(self, op, value):
        image = np.zeros((64, 64, 3), np.uint8)
        if op == "create":
            self.step = 0
            self.position = np.zeros(3, np.float32)
            info = dict(initial_state_fingerprint=f"fixture-query-{value['seed']}", scene_fingerprint="fixture",
                        perturbation_plan_fingerprint="fixture")
            self.raw = dict(agentview_image=image, robot0_eef_pos=self.position,
                            robot0_joint_pos=np.zeros(6, np.float32), robot0_gripper_qpos=np.zeros(2, np.float32))
            return dict(max_steps=10), dict(observation=self.raw, anchor_eef_pos=self.position,
                    language_instruction="fixture", action_schema=dict(low=[-1.] * 7, high=[1.] * 7),
                    agentview_64=image, environment_reproducibility=info, model_identity=dict(arm="UR5e",
                    model_class="fixture", joint_count=6), worker_step_index=0, renderer="fixture", renderer_vendor="CPU")
        require(op == "chunk" and len(value) == 5 and np.isfinite(value).all(), "Wrong fixture chunk")
        self.step += len(value)
        self.position = self.position + np.sum(value[:, :3], axis=0) * 0.01
        self.raw = dict(self.raw, robot0_eef_pos=self.position)
        return dict(observation=self.raw, agentview_64=image, positions=[self.position],
                    gripper_positions=[self.raw["robot0_gripper_qpos"]], joint_positions=[self.raw["robot0_joint_pos"]],
                    steps=len(value), rewards=[0.] * len(value), terminated=False, truncated=self.step == 10,
                    success=False, language_instruction="fixture")


def check(tmp):
    episode = load_episode(tmp)
    policy = FixturePolicy(next(iter(command_tapes().values())))
    sim = FixtureSim()
    results, refs = [], {}
    rows = [r for r in panel() if r["angle"]]
    for row in rows:
        # The compact orbit receipts omit jaw motion. Materialize only the test-double NPZ with synthetic jaws.
        path = tmp / "support.npz"
        np.savez(path, issued_mean=row["issued"], delta_eef=row["delta_eef"], delta_jaw=np.zeros((3, 2)),
                 chunk_steps=[5, 5, 5], valid=np.ones(3, bool))
        row = dict(row, support=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        heads = corrections(row, *shared_scale_grip(row))
        results.append(paired_cell(episode, sim, policy, row, heads, refs, fixture=True))
    summary = summarize(results)
    require(policy.calls == 2 * summary["episodes"] == 192, "Expected two replans on each fixture trajectory")
    rejected = {}
    for field, message in (("opening", "Opening proposal mismatch"), ("reset", "Unmatched reset"),
                           ("terminal", "Incomplete native outcome")):
        calls = 0

        def corrupt(*args, **kwargs):
            nonlocal calls
            calls += 1
            if field == "reset" and calls == 2:
                check_reset = kwargs["check_reset"]
                kwargs["check_reset"] = lambda data: check_reset(dict(data, prompt="mismatch"))
            record = episode(*args, **kwargs)
            if calls == 2 and field == "opening":
                record["opening_proposal_sha256"] = "0" * 64
            if calls == 2 and field == "terminal":
                record["truncated"] = False
            return record

        try:
            paired_cell(corrupt, sim, policy, row, heads, {}, fixture=True)
        except ValueError as exc:
            require(message in str(exc), f"Unexpected qualification error: {exc}")
            rejected[field] = str(exc)
        else:
            raise AssertionError(f"Accepted bad {field}")
    return dict(type="CPU fixture only; synthetic policy and plant", pinned_motor_episode_source=RUNNER_PIN,
                cells=summary["cells"], arms=list(ARMS), accepted_fixture_episodes=summary["episodes"],
                full_cell_replans=192, negative_controls=rejected, new_native_episodes=0)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    with TemporaryDirectory() as tmp:
        write_json(cli.parse_args().output, check(Path(tmp)))
