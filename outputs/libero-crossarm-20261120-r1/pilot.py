"""Physical-arm pilot on pinned LIBERO BASE tasks, with a frozen JAX OpenPI motor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "libero-xemb-sysid-20261119-r1"))
from motor_iface import (NOISE_SEED_OFFSET, OPENPI_CKPT, SPLITS_PATH, array_digest,
                         load_frozen_pi05, public_digest, to_openpi_observation, write_json)
from runner_native import Simulator

SOURCES = ("Panda", "Sawyer")
TARGET = "UR5e"
FIT_TASKS = (("libero_goal", 4), ("libero_goal", 5))
DEV_TASKS = (("libero_goal", 1), ("libero_spatial", 1))
SUITES = {"libero_goal": 0, "libero_spatial": 1}
FIT_CONFIG = dict(train_bodies=SOURCES, reference_body="Panda", seed=41, pretrain_steps=300,
                  joint_steps=400, batch_size=32, support_size=16)
CONDITIONS = ("unadapted", "learned", "cleared", "wrong_body", "action_oracle")


def case(body, task, pose):
    suite, task_id = task
    seed = 20261120 + 10000 * SUITES[suite] + 100 * task_id + pose
    return dict(body=body, suite=suite, task_id=task_id, init_state_id=pose, seed=seed)


def lineage(c):
    return f"{c['suite']}-{c['task_id']}-i{c['init_state_id']}-s{c['seed']}"


def check_split():
    splits = json.loads(SPLITS_PATH.read_text())
    manifest = Path(os.environ["CROSSARM_RELEASE"]) / "assets/task_manifest.json"
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != splits["manifest_sha256"]:
        raise ValueError("Released task manifest differs from registered split")
    for name, tasks in (("train", FIT_TASKS), ("development", DEV_TASKS)):
        for suite, task_id in tasks:
            if task_id not in splits[name][suite] or task_id in splits["test"][suite]:
                raise ValueError(f"Invalid {name} task {suite}/{task_id}")
    return splits["manifest_sha256"]


def probe_sequence(seed, *, calibration=False):
    """Balanced signed OSC pulses for calibration, independent mixed commands for histories."""
    rng = np.random.default_rng(seed)
    if calibration:
        blocks = []
        for _ in range(2):
            axes = np.concatenate((np.eye(6), -np.eye(6)))[rng.permutation(12)]
            arm = axes * rng.uniform(0.3, 0.65, (12, 1))
            grip = rng.choice([-0.8, 0.8], size=(12, 1))
            blocks.append(np.column_stack((arm, grip)))
        actions = np.repeat(np.concatenate(blocks), 3, axis=0)
    else:
        n = FIT_CONFIG["support_size"] // 2
        arm = rng.uniform(-0.5, 0.5, (n, 6))
        grip = rng.choice([-0.8, 0.8], size=(n, 1))
        actions = np.repeat(np.column_stack((arm, grip)), 2, axis=0)
    return actions.astype(np.float32)


def save_trace(path, arrays):
    with path.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    path.with_suffix(".tmp").replace(path)


def require_match(path, expected, observed, what):
    if expected != observed:
        write_json(path / "mismatch.json", dict(field=what, expected=expected, observed=observed))
        raise ValueError(f"Unpaired {what} in {path}")


def validate_body(c, first):
    model = first["model_identity"]
    raw = first["observation"]
    joints = {"Panda": 7, "Sawyer": 7, "UR5e": 6}[c["body"]]
    if model["arm"] != c["body"] or model["joint_count"] != joints:
        raise ValueError(f"Unexpected simulator body for {c}: {model}")
    if model["gripper"] != "PandaGripper" or not model["gripper_xml_sha256"] or not model["xml_sha256"]:
        raise ValueError(f"Unexpected gripper or missing physical model identity on {c}")
    shapes = dict(agentview_image=(256, 256, 3), robot0_eye_in_hand_image=(256, 256, 3),
                  robot0_joint_pos=(joints,), robot0_joint_vel=(joints,), robot0_eef_pos=(3,),
                  robot0_eef_quat_xyzw=(4,), robot0_gripper_qpos=(2,))
    for key, shape in shapes.items():
        arr = np.asarray(raw[key])
        if arr.shape != shape or arr.dtype != (np.uint8 if key.endswith("_image") else np.float32):
            raise ValueError(f"Invalid reset observation {key} on {c}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Nonfinite reset observation {key} on {c}")
    physics = first["environment_reproducibility"]
    for key in ("scene_fingerprint", "initial_state_fingerprint", "released_row_fingerprint"):
        if not physics[key]:
            raise ValueError(f"Missing {key} on {c}")
    if any(physics[key] != c[key] for key in ("suite", "task_id", "init_state_id", "seed")):
        raise ValueError(f"Simulator reset differs from requested task or start on {c}")


def shared_scene(one, other):
    first, second = one["reset"], other["reset"]
    if (one["episode_id"] != other["episode_id"] or first["prompt"] != second["prompt"]
            or first["model"]["gripper_xml_sha256"] != second["model"]["gripper_xml_sha256"]
            or any(first["physics"][key] != second["physics"][key]
                   for key in ("scene_fingerprint", "released_row_fingerprint", "bddl_sha256"))):
        raise ValueError(f"Different released scene at {one['tape']} and {other['tape']}")


def check_bodies(out, records):
    groups = {}
    for record in records:
        groups.setdefault(record["episode_id"], {})[record["case"]["body"]] = record
    for group in groups.values():
        if set(group) != {*SOURCES, TARGET}:
            raise ValueError("Missing one of the three physical arms")
        if len({row["reset"]["model"]["xml_sha256"] for row in group.values()}) != 3:
            raise ValueError("Compiled arm identities alias across requested bodies")
        for record in group.values():
            shared_scene(group["Panda"], record)
    write_json(out / "calibration-resets.json", [
        dict(case=r["case"], model=r["reset"]["model"], physics=r["reset"]["physics"], tape=r["tape"])
        for r in records
    ])


def episode(sim, out, pool, c, condition, *, policy=None, probe=None, adapt=None,
            reference=None, history=None):
    """Run a fresh native episode. A prescribed probe may end at its registered practice budget."""
    if (policy is None) == (probe is None):
        raise ValueError("Supply one action source")
    path = out / pool / lineage(c) / c["body"] / condition
    path.mkdir(parents=True, exist_ok=False)
    first = sim.call("reset", c)
    validate_body(c, first)
    raw, prompt = first["observation"], first["language_instruction"]
    horizon = int(first["max_steps"])
    rng = np.random.default_rng(c["seed"] + NOISE_SEED_OFFSET)
    noise = rng.standard_normal(((horizon + 4) // 5, 10, 32)).astype(np.float32)
    reset = dict(physics=first["environment_reproducibility"], model=first["model_identity"],
                 public=public_digest(raw), prompt=prompt, horizon=horizon, renderer=first["renderer"],
                 noise_seed=c["seed"] + NOISE_SEED_OFFSET, noise_sha256=array_digest(noise))
    if reference is not None:
        require_match(path, reference["reset"], reset, "query reset and noise")
    write_json(path / "reset.json", reset)
    limit = horizon if probe is None else len(probe)
    if not 0 < limit <= horizon:
        raise ValueError("Practice budget exceeds task horizon")
    commands, motion, rewards, proposals = [], [], [], []
    controls = 0
    opening = None
    while controls < limit:
        count = min(5, limit - controls)
        if probe is None:
            item = to_openpi_observation(raw, prompt)
            proposal = np.asarray(policy.infer(item, noise=noise[controls // 5])["actions"], dtype=np.float32)
            if proposal.shape != (10, 7) or not np.isfinite(proposal).all():
                raise ValueError("Invalid frozen pi0.5 action proposal")
            if controls == 0:
                opening = array_digest(proposal)
                if reference is not None:
                    require_match(path, reference["opening_sha256"], opening, "first frozen proposal")
            canonical = np.clip(proposal[:count], -1, 1)
            issued = canonical if adapt is None else np.asarray(adapt(canonical), dtype=np.float32)
            if issued.shape != (count, 7) or not np.isfinite(issued).all() or np.max(np.abs(issued)) > 1:
                raise ValueError("Adapter returned an invalid normalized action")
        else:
            issued = probe[controls:controls + count]

        response = sim.call("chunk", issued)
        n = len(response["commands"])
        if probe is None:
            proposals.append(canonical[:n])
        commands.append(response["commands"])
        motion.append(response["motion"])
        rewards.extend(response["rewards"])
        controls += n
        raw = response["observation"]
        if response["terminated"] or response["truncated"]:
            break

    tape = dict(commands=np.concatenate(commands), motion=np.concatenate(motion),
                rewards=np.asarray(rewards, dtype=np.float64))
    if proposals:
        tape["proposals"] = np.concatenate(proposals)
    save_trace(path / "tape.npz", tape)
    record = dict(case=c, episode_id=lineage(c), condition=condition, pool=pool, reset=reset,
                  opening_sha256=opening, history=history, controls=controls,
                  native_return=float(sum(rewards)), native_success=response["success"],
                  terminated=response["terminated"], truncated=response["truncated"],
                  tape=str((path / "tape.npz").relative_to(out)))
    write_json(path / "episode.json", record)
    print(json.dumps(dict(pool=pool, body=c["body"], case=lineage(c), condition=condition,
                          controls=controls, native_success=response["success"])), flush=True)
    if probe is not None and controls != len(probe):
        raise ValueError("Native episode ended before completing the planned support or calibration budget")
    return record, tape


def data_row(record, tape, fields):
    return dict(body=record["case"]["body"], episode_id=record["episode_id"],
                **{field: tape[field] for field in fields})


def collect_probes(sim, out, pool, bodies, tasks, poses, *, calibration=False):
    rows, records = [], []
    for task in tasks:
        for pose in poses:
            for body in bodies:
                c = case(body, task, pose)
                probe = probe_sequence(c["seed"] + 1619, calibration=calibration)
                record, tape = episode(sim, out, pool, c, "prescribed", probe=probe)
                rows.append(data_row(record, tape, ("commands", "motion")))
                records.append(record)
    return rows, records


def support_ref(record, row):
    return dict(body=record["case"]["body"], tape=record["tape"],
                commands_sha256=array_digest(row["commands"]), motion_sha256=array_digest(row["motion"]))


def check_response(teacher, validations):
    """Can the fitted local forward response beat constant drift on fresh commands?"""
    target = teacher.diagnostics["target"]
    commands = np.concatenate([row["commands"][:, :6] for row in validations])
    observed = np.concatenate([row["motion"][:, :6] for row in validations])
    drift = np.asarray(target["drift_m_rad"])
    predicted = commands @ np.asarray(target["cartesian_gain"]) + drift
    model_mse = np.mean((predicted - observed) ** 2, axis=0)
    constant_mse = np.mean((drift - observed) ** 2, axis=0)
    return dict(episodes=[r["episode_id"] for r in validations], rows=len(commands),
                model_rmse_m_rad=np.sqrt(model_mse).tolist(), drift_rmse_m_rad=np.sqrt(constant_mse).tolist(),
                passed=bool(model_mse[:3].sum() < constant_mse[:3].sum()
                            and model_mse[3:].sum() < constant_mse[3:].sum()))


def tallies(pairs):
    counts = {}
    for condition in CONDITIONS:
        outcomes = [r["arms"][condition]["native_success"] for r in pairs]
        counts[condition] = dict(successes=sum(outcomes), trials=len(outcomes))
        for control in ("unadapted", "learned"):
            other = [r["arms"][control]["native_success"] for r in pairs]
            counts[condition][f"vs_{control}"] = dict(wins=sum(a and not b for a, b in zip(outcomes, other)),
                                                       losses=sum(b and not a for a, b in zip(outcomes, other)))
    return counts


def run(out):
    from learner import fit, fit_teacher, load

    train_poses = dict(calibration=[0, 1], pretrain=[2, 3, 4], joint_support=[5, 6], joint_queries=[7, 8])
    plan = dict(source_bodies=SOURCES, target_body=TARGET, fit_tasks=FIT_TASKS, development_tasks=DEV_TASKS,
                source_poses=train_poses, calibration_check_pose=9, admission_pose=10,
                evaluation_pairs=[[11, 13], [12, 14]],
                fit=FIT_CONFIG, checkpoint=str(OPENPI_CKPT), manifest_sha256=check_split(),
                conditions=CONDITIONS, primary_metric="native_success", native_variant="BASE")
    write_json(out / "config.json", plan)
    with Simulator() as sim:
        source_cal, source_records = collect_probes(sim, out, "calibration", SOURCES, FIT_TASKS,
                                                    train_poses["calibration"], calibration=True)
        target_cal, target_records = collect_probes(sim, out, "oracle_calibration", [TARGET], FIT_TASKS,
                                                    train_poses["calibration"], calibration=True)
        check_bodies(out, source_records + target_records)
        panda_cal = [r for r in source_cal if r["body"] == "Panda"]
        teachers = {body: fit_teacher(panda_cal, [r for r in source_cal if r["body"] == body])
                    for body in SOURCES}
        target_teacher = fit_teacher(panda_cal, target_cal)
        responses = {}
        for body, teacher in {**teachers, TARGET: target_teacher}.items():
            validation, _ = collect_probes(sim, out, "calibration_check", [body], FIT_TASKS,
                                            [plan["calibration_check_pose"]])
            responses[body] = dict(fit=teacher.diagnostics, validation=check_response(teacher, validation))
            write_json(out / "responses.json", responses)

        # These ceiling measurements constrain interpretation of the fixed, exploratory panel.
        policy = load_frozen_pi05()
        gate = dict(source={}, target={}, passed=False)
        for body in SOURCES:
            c = case(body, FIT_TASKS[0], plan["admission_pose"])
            baseline, _ = episode(sim, out, "source_gate", c, "unadapted", policy=policy)
            oracle, _ = episode(sim, out, "source_gate", c, "action_oracle", policy=policy,
                                adapt=teachers[body].act, reference=baseline)
            gate["source"][body] = dict(unadapted=baseline["native_success"], oracle=oracle["native_success"])
            write_json(out / "gate.json", gate)
        for task in DEV_TASKS:
            c = case(TARGET, task, plan["admission_pose"])
            oracle, _ = episode(sim, out, "target_gate", c, "action_oracle", policy=policy,
                                adapt=target_teacher.act)
            gate["target"][lineage(c)] = oracle["native_success"]
            write_json(out / "gate.json", gate)
        gate["passed"] = bool(all(r["validation"]["passed"] for r in responses.values())
                              and gate["source"]["Panda"]["unadapted"]
                              and all(r["oracle"] for r in gate["source"].values())
                              and any(gate["target"].values()))
        write_json(out / "gate.json", gate)

        pretrain, _ = collect_probes(sim, out, "pretrain", SOURCES, FIT_TASKS, train_poses["pretrain"])
        joint_support, _ = collect_probes(sim, out, "joint_support", SOURCES, FIT_TASKS, train_poses["joint_support"])
        joint_queries, banks = [], []
        for task in FIT_TASKS:
            for pose in train_poses["joint_queries"]:
                for body in SOURCES:
                    record, tape = episode(sim, out, "joint_queries", case(body, task, pose),
                                           "unadapted", policy=policy)
                    # Both source teachers label the identical bank, so proposal/body
                    # correlation cannot identify which inverse should be predicted.
                    bank = f"{record['episode_id']}-{body}"
                    banks.append(dict(bank=bank, source=record["case"], tape=record["tape"],
                                      label_bodies=SOURCES, proposals_sha256=array_digest(tape["proposals"])))
                    for label_body in SOURCES:
                        joint_queries.append(dict(body=label_body, episode_id=bank, proposals=tape["proposals"]))
        write_json(out / "proposal-banks.json", banks)
        model, report = fit(source_cal, pretrain, joint_support, joint_queries, **FIT_CONFIG)
        model_path = out / "reader-adapter.pt"
        model.save(model_path)
        write_json(out / "fit.json", report)
        frozen = load(model_path)

        pairs = []
        for task_index, task in enumerate(DEV_TASKS):
            for index, (support_pose, query_pose) in enumerate(plan["evaluation_pairs"]):
                donor_body = SOURCES[(index + task_index) % len(SOURCES)]
                supports, records = collect_probes(sim, out, "eval_support", (TARGET, donor_body),
                                                    [task], [support_pose])
                retained, donor = supports
                shared_scene(records[0], records[1])
                if not np.array_equal(retained["commands"], donor["commands"]):
                    raise ValueError("Different prescribed support commands across donor bodies")
                keep_ref = support_ref(records[0], retained)
                wrong_ref = support_ref(records[1], donor)
                controllers = (
                    ("unadapted", None, None),
                    ("learned", lambda p: frozen.act(p, retained["commands"], retained["motion"]), keep_ref),
                    ("cleared", frozen.act, None),
                    ("wrong_body", lambda p: frozen.act(p, donor["commands"], donor["motion"]), wrong_ref),
                    ("action_oracle", target_teacher.act, None),
                )
                query = case(TARGET, task, query_pose)
                arms = {}
                for name, adapt, history in controllers:
                    record, _ = episode(sim, out, "evaluation", query, name, policy=policy, adapt=adapt,
                                        reference=arms.get("unadapted"), history=history)
                    arms[name] = record
                pairs.append(dict(case=query, support=keep_ref, donor=wrong_ref, arms=arms))
                write_json(out / "pairs.json", pairs)
        return dict(status="complete", calibrated_control_passed=gate["passed"],
                    paired_cases=len(pairs), counts=tallies(pairs),
                    checkpoint_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    out = parser.parse_args().output
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = run(out)
    except Exception as err:
        write_json(out / "status.json", dict(status="failed", error=type(err).__name__, reason=str(err)))
        raise
    write_json(out / "status.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
