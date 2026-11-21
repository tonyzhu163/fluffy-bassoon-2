"""One paired native development start. Run once per policy in a fresh process."""

import json
import os
from pathlib import Path
import sys

import numpy as np

import eval_stage2a_live_ur5e as campaign
import ur5e_motor as motor
import ur5e_sim_worker as sim

CASE = dict(suite="libero_goal", task_id=1, init_state_id=13, seed=20500101, variant="BASE")
HORIZONS = {"pi0": 50, "pi05": 10}
ARMS = ("Panda", "UR5e")


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n")
    tmp.replace(path)


class ProposalRecorder:
    """Observe unmodified inference results while exposing the original motor metadata."""

    def __init__(self, policy, horizon):
        self.policy = policy
        self._model = policy._model
        if int(self._model.action_horizon) != horizon:
            raise ValueError("Unexpected frozen policy horizon")
        self.proposals = []

    def infer(self, item, *, noise):
        result = self.policy.infer(item, noise=noise)
        if len(self.proposals) < 5:
            actions = np.asarray(result["actions"], dtype=np.float32)
            self.proposals.append(dict(full_sha256=motor.array_digest(actions),
                                       first_five_actions=actions[:5].tolist(),
                                       first_five_sha256=motor.array_digest(actions[:5])))
        return result


def check_reset(report, path, arm, policy, reset):
    key = f"{policy}/{arm}"
    # Retain the original hash as evidence. Only the shape-dependent noise bank
    # digest is excluded from the within-arm equality check.
    rows = report["resets"]
    rows[key] = reset
    errors = []
    if (reset["arm"] != arm or reset["worker_step_index"] != 0
            or reset["noise_seed"] != CASE["seed"] + motor.NOISE_SEED_OFFSET):
        errors.append("reset identity, step or seed")
    if policy == "pi05":
        prior = rows[f"pi0/{arm}"]
        fields = (reset.keys() | prior.keys()) - {"noise_sha256"}
        different = sorted(k for k in fields if k not in reset or k not in prior or reset[k] != prior[k])
        report["same_arm_mismatches"][arm] = different
        errors.extend(different)
    if arm == "UR5e":
        panda = rows[f"{policy}/Panda"]
        matched = reset["scene_fingerprint"] == panda["scene_fingerprint"] and reset["prompt"] == panda["prompt"]
        report["released_scene_and_prompt_match"][policy] = matched
        if not matched:
            errors.append("released scene or prompt")
    if errors:
        report.update(status="invalid_reset", failed_cell=key)
    save(path, report)
    if errors:
        raise RuntimeError(f"{key} reset mismatch: {errors}")


def main(policy_name, path):
    deps = Path(os.environ["SMOKE_DEPS"]).resolve() / "outputs"
    for module, directory in ((campaign, "xemb-icml-campaign-20261121-r1"),
                              (sim, "libero-xemb-ur5e-sysid-20261120-r1"),
                              (motor, "libero-xemb-ur5e-sysid-20261120-r1")):
        if Path(module.__file__).resolve() != deps / directory / (module.__name__ + ".py"):
            raise RuntimeError(f"Unexpected dependency path for {module.__name__}")
    if policy_name == "pi0":
        report = dict(status="partial", case=CASE, dependency_commit=os.environ["SMOKE_DEP_PIN"],
                      wrapper_commit=os.environ["SMOKE_WRAPPER_PIN"], resets={}, outcomes={},
                      same_arm_mismatches={}, released_scene_and_prompt_match={},
                      interpretation="One predetermined start per cell (n=1). A 0/1 or 1/1 is descriptive only.")
        save(path, report)
    else:
        report = json.loads(path.read_text())
    policy = campaign.load_frozen_policy(policy_name)
    horizon = HORIZONS[policy_name]
    with sim.UR5eSimulator() as simulator:
        for arm in ARMS:
            recorder = ProposalRecorder(policy, horizon)
            record, trace = motor.run_episode(
                simulator, recorder, dict(CASE, arm=arm), campaign.get_embodiment_spec("nom"),
                record_chunks=True, max_record_chunks=5,
                check_reset=lambda reset: check_reset(report, path, arm, policy_name, reset),
            )
            if not (record["terminated"] or record["truncated"]):
                raise RuntimeError("Native episode did not finish")
            if recorder.proposals[0]["full_sha256"] != record["opening_proposal_sha256"]:
                raise RuntimeError("Opening proposal disagrees with the pinned motor")
            row = {k: record[k] for k in ("success", "controls", "terminated", "truncated", "native_return",
                                          "prefix_xy_disp_15_mm", "prefix_xy_disp_30_mm", "opening_proposal_sha256")}
            row.update(horizon=horizon, execute=5, proposals=recorder.proposals,
                       nonzero_rewards=[[i + 1, r] for i, r in enumerate(record["native_rewards"]) if r],
                       first_chunks={k: trace[k][trace["valid"]].tolist() for k in
                                     ("issued_mean", "delta_eef", "delta_jaw", "delta_joint")})
            report["outcomes"][f"{policy_name}/{arm}"] = row
            save(path, report)
    if policy_name == "pi05":
        if set(report["outcomes"]) != {f"{p}/{a}" for a in ARMS for p in HORIZONS}:
            raise RuntimeError("Incomplete outcome panel")
        report["status"] = "complete"
        save(path, report)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
