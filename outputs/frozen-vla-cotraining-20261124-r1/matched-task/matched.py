"""Use the parent evaluator's native episode, parity checks and replay validator on the crossed panel."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
from unittest.mock import patch

import numpy as np
import rollout
import verify
from check_fit import inspect, sha
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from ur5e_motor import run_episode
from panel import MANIFEST, RELEASE, key, require, save
from ur5e_sim_worker import UR5eSimulator

HERE = Path(__file__).resolve().parent
MODES = ("original", "policy_only", "adapter_only", "joint", "joint_zero")
ARMS = ("Kinova3", "Panda")


def contract():
    data = json.loads((HERE / "contract.json").read_text())
    cases, tasks = [], {}
    for suite, task in data["tasks"].items():
        task_id = task["task_id"]
        # The pinned parent runner indexes its registered row bank by released_id minus 46.
        bank = {int(row) - 46: value for row, value in task["layouts"].items()}
        tasks[suite, task_id] = (task["seed_base"], task["bddl"]["max_steps"], task["native_prompt"], bank)
        for row in map(int, task["layouts"]):
            for arm in ARMS:
                cases.append(dict(suite=suite, task_id=task_id, init_state_id=row, variant="BASE",
                                  arm=arm, seed=task["seed_base"]+row))
    require(len(tasks) == 2 and len({key(c) for c in cases}) == len(cases) == 8, "eight crossed cases")
    return data, cases, tasks


def episode(sim, motor, case, mode, reference, fixture, task):
    # The held-out parent's episode admits only tabletop-mounted arms. Object uses the
    # released floor-placement factory. Keep its public Wire and native motor unchanged.
    mount = "OnTheGround" if case["suite"] == "libero_object" else "Mounted"
    for attempt in range(1, 4):
        wire = rollout.Wire(sim, motor, mode)

        def check(reset):
            env = wire.first["environment_reproducibility"]
            body = wire.first["model_identity"]
            require(all(env[k] == case[k] for k in ("suite", "task_id", "init_state_id", "seed"))
                    and env["source_pin"] == RELEASE and env["manifest_sha256"] == MANIFEST
                    and env["bddl_sha256"] == task["bddl"]["bddl_sha256"]
                    and env["released_row_fingerprint"] == task["layouts"][str(case["init_state_id"])]
                    and env["perturbation_plan"]["case_id"] == "BASE", "registered BASE row")
            require(reset["worker_step_index"] == 0 and reset["prompt"] == task["native_prompt"]
                    and reset["horizon"] == task["bddl"]["max_steps"] and body["arm"] == case["arm"]
                    and body["model_class"] == mount+case["arm"] and body["joint_count"] == 7
                    and body["control_freq"] == 20 and body["controller"] == "OSC_POSE"
                    and body["gripper"] == "PandaGripper", "registered native interface")
            require("llvmpipe" in reset["renderer"].lower() if fixture else reset["renderer_vendor"] == "NVIDIA Corporation",
                    "wrong renderer")
            if reference and (reset != reference["native"]["reset"] or env != reference["environment"]):
                raise rollout.UnmatchedReset("opening differs before first action")

        try:
            native = run_episode(wire, wire, case, get_embodiment_spec("nom"), check_reset=check,
                                 max_controls=15 if fixture else None, canonicalize_quat=False)
        except rollout.UnmatchedReset:
            if attempt == 3:
                raise
            continue
        first = wire.first
        raw = first["observation"]
        return dict(case=case, mode=mode, fixture=fixture, accepted_reset_attempt=attempt, native=native,
                    initial_xyz=raw["robot0_eef_pos"].tolist(), initial_fingers=raw["robot0_gripper_qpos"].tolist(),
                    environment=first["environment_reproducibility"], body=first["model_identity"],
                    parity=wire.parity_result, mechanism=wire.mechanism, calls=wire.calls)


def estimate(values):
    a = np.asarray(values, dtype=float)
    return dict(values=values, mean=float(a.mean()),
                leave_one_start_out_means=((a.sum()-a)/(len(a)-1)).tolist())


def summarize(out, cases, fixture):
    result, crossed = [], {}
    for c in cases:
        modes, ref, starts = {}, None, None
        for mode in MODES:
            v = json.loads((out/"episodes"/f"{key(c)}-{mode}.json").read_text())
            require(v["case"] == c and v["mode"] == mode, "wrong saved episode")
            modes[mode] = verify.validate(v, fixture)
            r, e = v["native"]["reset"], v["environment"]
            opening = [r, e, v["body"]]
            if ref is None:
                ref = opening
                starts = {k: e[k] for k in ("released_row_fingerprint", "scene_fingerprint", "bddl_sha256",
                                             "perturbation_plan_fingerprint")}
                starts.update({k:r[k] for k in ("prompt", "horizon", "noise_sha256", "action_schema")})
            require(ref == opening, "unpaired physical reset within arm")
        require(modes["joint"]["opening_proposal"] == modes["joint_zero"]["opening_proposal"]
                and modes["adapter_only"]["opening_proposal"] == modes["original"]["opening_proposal"],
                "empty-history control diverged")
        scene = (c["suite"], c["task_id"], c["init_state_id"])
        if scene in crossed:
            require(crossed[scene] == starts, "unpaired scene, prompt or noise across arms")
        crossed[scene] = starts
        result.append(dict(case=c, shared_start=starts, opening_physics=ref[0]["physics"], modes=modes))
    require(len(crossed) == 4 and len(list((out/"episodes").glob("*.json"))) == 40, "incomplete panel")
    summary = dict(fixture=fixture, physical_cases=len(cases), shared_starts=len(crossed), episodes=40, rows=result)
    if fixture:
        return summary
    contrasts = {}
    for field in ("success", "native_return"):
        contrasts[field] = {}
        for mode in (m for m in MODES if m != "joint"):
            arms = {}
            for arm in ARMS:
                deltas = [int(r["modes"]["joint"][field]) - int(r["modes"][mode][field])
                          for r in result if r["case"]["arm"] == arm]
                arms[arm] = estimate(deltas)
                if field == "success":
                    arms[arm].update(rescues=deltas.count(1), losses=deltas.count(-1), pairs=len(deltas))
            arms["kinova_minus_panda"] = estimate([k-p for k, p in zip(
                arms["Kinova3"]["values"], arms["Panda"]["values"], strict=True)])
            contrasts[field][mode] = arms
    summary["joint_minus_comparator"] = contrasts
    return summary


def run(args):
    registered, cases, tasks = contract()
    pin = subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "HEAD"], text=True).strip()
    require(pin == args.pin and sha(args.data_manifest) == registered["data_manifest"], "code or source manifest lock")
    metadata = dict(pin=pin, plan_sha256=sha(HERE/"PLAN.md"), contract_sha256=sha(HERE/"contract.json"),
                    host=socket.gethostname(), fixture=args.fixture, cpu_affinity=sorted(os.sched_getaffinity(0)),
                    source_manifest=registered["data_manifest"])
    if args.fixture:
        from fixture import Fixture
        require(len(metadata["cpu_affinity"]) <= 2 and os.environ["CUDA_VISIBLE_DEVICES"] == ""
                and os.environ["JAX_PLATFORMS"] == "cpu", "two-core CPU fixture")
        motor = Fixture()
    else:
        import jax
        from cotrain_policy import LockedPolicy
        require(len(jax.devices()) == 1 and jax.devices()[0].platform == "gpu", "one CUDA device")
        # Keep the shared fit-admission checker bound to the parent's held-out test-suite definitions.
        metadata["offline_fit"] = inspect(args.fit, args.fit_sha, args.data_manifest)
        receipt = json.loads((args.fit/"train-cache.json").read_text())
        expected = json.loads(args.data_manifest.read_text())["splits"]["train"]["examples"]
        require(receipt["examples_before_draws"] == receipt["source_dataset_rows"] == expected,
                "fit did not cover the complete registered train split")
        metadata["all_source_examples_fitted"] = expected
        motor = LockedPolicy.load(args.fit)
    save(args.out/"run.json", metadata)
    before = motor.digests()
    if not args.fixture:
        require(before["fit_file"] == args.fit_sha and before["resident_stock"] and before["resident_heads"]
                and before["normalization"], "resident fit or normalization")
    save(args.out/"frozen-before.json", before)
    (args.out/"episodes").mkdir()
    with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
        sim = UR5eSimulator(allow_mesa=args.fixture)
    with sim, patch.object(verify, "TASKS", tasks):
        for c in cases:
            ref = None
            sim.call("create", dict(c, embodiment="nom"))
            for mode in MODES:
                v = episode(sim, motor, c, mode, ref, args.fixture, registered["tasks"][c["suite"]])
                verify.validate(v, args.fixture)
                save(args.out/"episodes"/f"{key(c)}-{mode}.json", v)
                if mode == "original":
                    ref = v
                print(key(c), mode, v["native"]["controls"], flush=True)
        after = motor.digests()
        save(args.out/"frozen-after.json", after)
        require(before == after, "weights or normalization changed")
        summary = summarize(args.out, cases, args.fixture)
        save(args.out/("cpu-summary.json" if args.fixture else "native-summary.json"), summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("pin")
    p.add_argument("data_manifest", type=Path)
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--fit", type=Path)
    p.add_argument("--fit-sha")
    run(p.parse_args())
