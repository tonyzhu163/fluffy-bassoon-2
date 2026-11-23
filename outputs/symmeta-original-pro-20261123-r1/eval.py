"""Matched native OpenPI / within-attempt SymMeta evaluation on original LIBERO-PRO."""
import argparse
from hashlib import sha256
import itertools
import json
import math
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
HORIZONS = dict(libero_goal=300, libero_spatial=220, libero_object=280, libero_10=520)
READER_SHA = "c12bace5cb391499d6abd7cc5767f94b1519de7d77940efb0cfcb71ca0f686a4"
POLICY_RECEIPT_SHA = "ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"
PINS = dict(ORIGINAL_PRO_ROOT="e235a970758ecd78ecbb585fca6b915a8d44d18d",
            ORIGINAL_PRO_ASSETS="3a33dc4cfbea7ff5eac6764e9c557500972b6259",
            OPENPI_SOURCE="215abfb217dbac7d5f1273282331b9b1866c0479",
            SYMMETA_SOURCE="ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c")


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def contract(path):
    manifest = json.loads(path.read_text())
    if (manifest["release_revision"], manifest["assets_revision"], manifest["openpi_revision"]) != (
            PINS["ORIGINAL_PRO_ROOT"], PINS["ORIGINAL_PRO_ASSETS"], PINS["OPENPI_SOURCE"]):
        raise ValueError("Wrong source revisions in case manifest")
    cases = manifest["cases"]
    if (len(cases) != manifest["case_count"] or not cases
            or len({c["case_id"] for c in cases}) != len(cases)):
        raise ValueError("Invalid case ledger")
    for c in cases:
        variant = c["suite"] + "_" + c["suffix"]
        if (c["max_steps"] != HORIZONS[c["suite"]] or c["suffix"] not in ("object", "swap", "lan", "task", "env")
                or c["task_name"] != manifest["full_task_order"][c["suite"]][c["task_id"]]
                or c["task_suite_name"] != variant or Path(c["task_name"]).name != c["task_name"]
                or c["bddl_path"] != f"bddl_files/{variant}/{c['task_name']}.bddl"
                or c["init_path"] != f"init_files/{variant}/{c['task_name']}.pruned_init"
                or not 0 <= c["init_index"] < c["init_shape"][0]):
            raise ValueError(f"Invalid case: {c}")
    return manifest


def dependencies():
    paths = {name: Path(os.environ[name]).resolve() for name in PINS}
    for name, pin in PINS.items():
        actual = subprocess.check_output(["git", "-C", paths[name], "rev-parse", "HEAD"], text=True).strip()
        if actual != pin:
            raise ValueError(f"Wrong {name} revision {actual}")
    old = paths["SYMMETA_SOURCE"]
    sys.path[:0] = list(map(str, (old / "src", old / "outputs/xemb-icml-campaign-20261121-r1",
                                  old / "outputs/libero-xemb-ur5e-sysid-20261120-r1",
                                  old / "outputs/libero-crossarm-20261120-r1")))
    import cv2
    import eval_reader as reader
    import ur5e_motor as motor

    cv2.setNumThreads(1)
    ckpt = Path(os.environ["SYMMETA_CHECKPOINT"])
    official = Path(os.environ["PI05_CHECKPOINT"])
    if file_sha(ckpt) != READER_SHA or file_sha(official / "READY.json") != POLICY_RECEIPT_SHA:
        raise ValueError("Unrecognized reader or policy weights receipt")
    if official.resolve() != motor.OPENPI_CKPT.resolve():
        raise ValueError("Historical pi05_libero loader and runtime disagree on weights path")
    model = reader.load_v2_checkpoint(ckpt)
    return motor, reader, model


def simulator():
    from ur5e_sim_worker import UR5eSimulator

    # Reuse the historical policy-side pipe client; replace only its Pro2 simulator child.
    class OriginalSimulator(UR5eSimulator):
        def __init__(self):
            self._closed = False
            self.pipe, peer = Pipe()
            try:
                self.worker = subprocess.Popen([os.environ["SIM_PYTHON"], "-Bu", str(HERE / "eval_sim.py"),
                                                str(peer.fileno())], pass_fds=(peer.fileno(),),
                                               env=dict(os.environ, PYTHONHASHSEED="0"))
            finally:
                peer.close()
    return OriginalSimulator()


def check_equal(label, first, second):
    if first != second:
        keys = sorted(set(first) | set(second)) if isinstance(first, dict) and isinstance(second, dict) else []
        different = [k for k in keys if first.get(k) != second.get(k)]
        raise ValueError(f"Unmatched {label}: {different or (first, second)}")


def policy_input(motor, obs, prompt):
    item = motor.to_openpi_observation(obs, prompt)
    # The historical client promotes the quaternion to float64 before the arithmetic.
    # Use the exact OpenPI examples/libero/main.py expression with the observation dtype.
    quat = obs["robot0_eef_quat_xyzw"].copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    axis_angle = np.zeros(3) if math.isclose(den, 0.0) else (quat[:3] * 2.0 * math.acos(quat[3])) / den
    item["observation/state"] = np.concatenate((obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"]))
    return item


def episode(sim, policy, motor, reader, reader_model, case, seed, noise, *, cpu, native=None):
    from fluffy_bassoon.xemb.embodiment import retarget_action_from_motor, retarget_observation_for_motor

    receipt, raw = sim.call("create", (case, seed, cpu))
    anchor = np.asarray(raw["robot0_eef_pos"], np.float32).copy()
    receipt["settled_public"] = motor.public_digest(raw)
    receipt["noise_sha256"] = motor.array_digest(noise)
    if native is not None:
        check_equal("initial conditions", native["reset"], receipt)
    reader_state = reader.OnlineReader(reader_model) if native is not None else None
    low, high = (np.asarray(receipt[k], np.float32) for k in ("low", "high"))
    records, rewards = [], []
    limit = 15 if cpu else case["max_steps"]
    done = False
    while not done and len(rewards) < limit:
        qi = len(records)
        estimate = reader_state.adapter if reader_state else reader.IDENTITY
        corrected = retarget_observation_for_motor(raw, anchor, estimate[0]) if estimate[0] else raw
        item = policy_input(motor, corrected, receipt["prompt"])
        before_input = motor.public_digest({k: v for k, v in item.items() if k != "prompt"})
        proposal = np.asarray(policy.infer(item, noise=noise[qi])["actions"])
        if (proposal.shape != (noise.shape[1], 7) or not np.issubdtype(proposal.dtype, np.floating)
                or not np.isfinite(proposal).all()):
            raise ValueError(f"Invalid proposal at query {qi}: {proposal.shape}")
        commands = proposal[:min(5, limit - len(rewards))]
        issued = commands if estimate == reader.IDENTITY else retarget_action_from_motor(
            np.clip(commands, low, high), *estimate, low, high)
        start = len(rewards)
        r = dict(query=qi, start=start, estimate=list(estimate), noise=motor.array_digest(noise[qi]),
                 input=before_input, proposal_dtype=proposal.dtype.str, issued_dtype=issued.dtype.str,
                 proposal=motor.array_digest(proposal), issued=motor.array_digest(issued))
        if native is not None and qi < 2:
            for field in ("start", "estimate", "noise", "input", "proposal", "issued"):
                check_equal(f"prefix query {qi}/{field}", native["queries"][qi][field], r[field])
        if native is not None and qi < len(native["queries"]):
            check_equal(f"noise at query {qi}", native["queries"][qi]["noise"], r["noise"])
        step = sim.call("chunk", issued)
        after, done = step["observation"], step["done"]
        n = len(step["rewards"])
        if not 1 <= n <= len(issued) or (n != len(issued) and not done):
            raise ValueError("Simulator stopped within a nonterminal chunk")
        rewards.extend(step["rewards"])
        delta = issued[:n] - commands[:n]
        r.update(count=n, commands=commands[:n].tolist(), executed=issued[:n].tolist(),
                 action_delta_l2=np.linalg.norm(delta, axis=1).tolist(),
                 controls_changed=int(np.any(delta != 0, axis=1).sum()),
                 eef_input_delta=float(np.linalg.norm(corrected["robot0_eef_pos"] - raw["robot0_eef_pos"])),
                 post_public=motor.public_digest(after), post_physics=step["physics"],
                 rewards=step["rewards"], done=done)
        if native is not None and qi < 2:
            for field in ("count", "post_public", "post_physics", "rewards", "done"):
                check_equal(f"prefix query {qi}/{field}", native["queries"][qi][field], r[field])
        if reader_state and not done and qi < reader.UPDATE_LIMIT and len(rewards) < case["max_steps"]:
            r["update"] = reader_state.observe(raw, after, issued[:n])
        records.append(r)
        raw = after
    return dict(reset=receipt, queries=records, controls=len(rewards), native_rewards=rewards,
                native_return=sum(rewards), success=done, horizon_exhausted=not done and len(rewards)==case["max_steps"])


def configuration(case, seed, shape, motor):
    key = sha256(f"{case['case_id']}|{seed}".encode()).digest()
    noise_seed = int.from_bytes(key[:8], byteorder="big") + motor.NOISE_SEED_OFFSET
    rng = np.random.default_rng(noise_seed)
    noise = rng.standard_normal(((case["max_steps"]+4)//5, *shape)).astype(np.float32)
    noise.setflags(write=False)
    return noise_seed, noise


def pair(case, env_seed, dest, policy, shape, deps, cpu):
    motor, reader, reader_model = deps
    noise_seed, noise = configuration(case, env_seed, shape, motor)
    output = {}
    for method in ("native", "symmeta"):
        with simulator() as sim:
            output[method] = episode(sim, policy, motor, reader, reader_model, case, env_seed, noise,
                                     cpu=cpu, native=output.get("native"))
        write(dest / f"{method}.json", output[method])
    a, b = output.values()
    updates = [q["update"] for q in b["queries"] if "update" in q]
    active = [q for q in b["queries"] if q["controls_changed"] or q["eef_input_delta"] > 0]
    action_delta = [value for q in b["queries"] for value in q["action_delta_l2"]]
    row = dict(case_id=case["case_id"], suite=case["suite"], task=case["task_name"], suffix=case["suffix"],
               init_index=case["init_index"], env_seed=env_seed, noise_seed=noise_seed,
               native_success=a["success"], symmeta_success=b["success"],
               difference=int(b["success"])-int(a["success"]), native_return=a["native_return"],
               symmeta_return=b["native_return"], native_controls=a["controls"], symmeta_controls=b["controls"],
               matched_prefix_controls=sum(q["count"] for q in b["queries"][:2]),
               matched_noise_queries=min(len(a["queries"]),len(b["queries"])),
               initially_solved=b["reset"]["initial_success"], settle_success=any(b["reset"]["settle_done"]),
               completed_before_adaptation=b["success"] and len(b["queries"]) <= 2,
               reader_updates=len(updates), eligible_yaw_updates=sum(u["eligible_yaw"] for u in updates),
               first_intervention_control=active[0]["start"] if active else None,
               first_changed_command_control=next((q["start"] + j for q in b["queries"]
                    for j, value in enumerate(q["action_delta_l2"]) if value != 0.), None),
               policy_prompt=b["reset"]["prompt"], bddl_prompt=b["reset"]["bddl_prompt"],
               bddl_differs_from_policy_prompt=b["reset"]["prompt"].casefold() != b["reset"]["bddl_prompt"].casefold(),
               selected_state_sha256=b["reset"]["selected_state"],
               bank_unique_states=b["reset"]["bank_unique_states"],
               selected_state_multiplicity=b["reset"]["selected_state_multiplicity"],
               changed_controls=sum(q["controls_changed"] for q in b["queries"]),
               total_action_delta_l2=float(np.sum(action_delta)), max_action_delta_l2=max(action_delta),
               max_abs_applied_yaw=max(abs(q["estimate"][0]) for q in b["queries"]),
               max_abs_applied_log_gain=max(abs(float(np.log(q["estimate"][1]))) for q in b["queries"]),
               flipped_grip_queries=sum(q["estimate"][2] < 0 for q in b["queries"]))
    write(dest / "pair.json", row)
    return row


def counts(rows):
    effects = [r["difference"] for r in rows]
    n = len(rows)
    native = sum(r["native_success"] for r in rows)
    symmeta = sum(r["symmeta_success"] for r in rows)
    return dict(pairs=n, native_successes=native, symmeta_successes=symmeta,
                paired_difference=sum(effects)/n, rescues=effects.count(1), regressions=effects.count(-1),
                both_success=native-effects.count(-1), both_failure=n-native-effects.count(1),
                native_floor=native==0, native_ceiling=native==n, no_discordance=not any(effects),
                initially_solved=sum(r["initially_solved"] for r in rows),
                successes_during_settle=sum(r["settle_success"] for r in rows),
                finished_in_prefix=sum(r["completed_before_adaptation"] for r in rows),
                repeated_state_in_bank=sum(r["selected_state_multiplicity"] > 1 for r in rows),
                bddl_differs_from_policy_prompt=sum(r["bddl_differs_from_policy_prompt"] for r in rows),
                no_eligible_yaw_update=sum(r["eligible_yaw_updates"]==0 for r in rows),
                no_intervention=sum(r["first_intervention_control"] is None for r in rows),
                changed_controls=sum(r["changed_controls"] for r in rows),
                symmeta_controls=sum(r["symmeta_controls"] for r in rows))


def summary(rows):
    by_suite = {s: [r for r in rows if r["suite"] == s] for s in sorted({r["suite"] for r in rows})}
    by_family = {s: counts([r for r in rows if r["suffix"] == s]) for s in sorted({r["suffix"] for r in rows})}
    # Enumerate the task-cluster bootstrap, resampling the two tasks per suite with replacement.
    suite_draws = []
    for group in by_suite.values():
        tasks = sorted({r["task"] for r in group})
        rates = [np.mean([r["difference"] for r in group if r["task"] == task]) for task in tasks]
        suite_draws.append([np.mean(draw) for draw in itertools.product(rates, repeat=len(tasks))])
    draws = np.mean(list(itertools.product(*suite_draws)), axis=1)
    return dict(observed_families=sorted(by_family), env_pairs=sum(r["suffix"] == "env" for r in rows),
                overall=counts(rows), by_suite={s:counts(g) for s,g in by_suite.items()},
                by_family=by_family, by_task={
                    f"{suite}/{task}": counts([r for r in group if r["task"] == task])
                    for suite, group in by_suite.items() for task in sorted({r["task"] for r in group})},
                task_cluster_bootstrap_95=np.quantile(draws,[.025,.975]).tolist(),
                bootstrap_draws=len(draws), rows=rows)


def progress(cases, rows):
    ids = [r["case_id"] for r in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= {c["case_id"] for c in cases}:
        raise ValueError("Invalid pair membership")
    missing = [c["case_id"] for c in cases if c["case_id"] not in ids]
    total = len(cases)
    net = sum(r["difference"] for r in rows)
    suffixes = sorted({c["suffix"] for c in cases})
    return dict(assigned_pairs=total, complete_pairs=len(rows), complete_case_ids=ids,
                missing_case_ids=missing, evaluated_families=suffixes,
                env_pairs_assigned=sum(c["suffix"] == "env" for c in cases),
                observed=counts(rows) if rows else None,
                full_panel_difference_bounds=[(net-len(missing))/total, (net+len(missing))/total])


class WiringPolicy:
    """CPU-only deterministic control fixture. Never used for benchmark outcomes."""
    def infer(self, item, *, noise):
        return {"actions": (noise[:, :7] * .15).astype(np.float32)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "cpu-check", "report"])
    parser.add_argument("--output", type=Path, required=True, help="Attempt directory. An eval subdirectory must be fresh.")
    parser.add_argument("--manifest", type=Path, default=HERE/"CASES.json")
    args = parser.parse_args()
    if args.mode == "report":
        dest = args.output / "eval"
        snapshot = json.loads((dest/"contract.json").read_text())
        cases = snapshot["manifest"]["cases"]
        rows = []
        for index, case in enumerate(cases):
            pair_path = dest / f"case-{index:03d}" / "pair.json"
            if pair_path.exists():
                row = json.loads(pair_path.read_text())
                check_equal("report case ID", case["case_id"], row["case_id"])
                rows.append(row)
        print(json.dumps(progress(cases, rows), indent=2))
        return
    manifest = contract(args.manifest)
    if file_sha(args.manifest) != file_sha(HERE/"CASES.json"):
        raise ValueError("Use the locked CASES.json or its byte-identical launcher copy")
    cpu = args.mode == "cpu-check"
    dest = args.output / ("eval-cpu" if cpu else "eval")
    dest.mkdir(parents=True, exist_ok=False)
    write(dest/"contract.json", dict(mode=args.mode, revisions=PINS, cases_sha256=file_sha(args.manifest),
                                     reader_sha256=READER_SHA, pi05_ready_sha256=POLICY_RECEIPT_SHA,
                                     reader_update_limit=24, native_prefix_controls=10,
                                     query_cadence=5, wait_controls=10, manifest=manifest))
    cases = manifest["cases"][:1] if cpu else manifest["cases"]
    if not cpu:
        write(dest/"progress.json", progress(cases, []))
    deps = dependencies()
    import jax
    if cpu:
        if jax.default_backend() != "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise RuntimeError("cpu-check requires JAX_PLATFORMS=cpu and CUDA_VISIBLE_DEVICES=''")
        policy, shape = WiringPolicy(), (10, 32)
    else:
        policy = deps[0].load_frozen_pi05()
        shape = (policy._model.action_horizon, policy._model.action_dim)
    rows = []
    for i, c in enumerate(cases):
        path = dest / f"case-{i:03d}"
        path.mkdir()
        try:
            row = pair(c, manifest["environment_seed"], path, policy, shape, deps, cpu=cpu)
        except Exception as exc:
            write(path/"failure.json", dict(case_id=c["case_id"], error=type(exc).__name__, detail=str(exc)))
            raise
        rows.append(row)
        if not cpu:
            write(dest/"progress.json", progress(cases, rows))
        print(json.dumps(dict(index=i, pairs=len(cases), **row)), flush=True)
    if cpu:
        if (rows[0]["matched_prefix_controls"] != 10 or rows[0]["reader_updates"] < 2
                or rows[0]["first_changed_command_control"] != 10):
            raise RuntimeError("CPU fixture failed to exercise the matched prefix and first corrected control")
        write(dest/"wiring.json", dict(fixture="synthetic policy; actual original simulator and seed-41 reader", row=rows[0]))
    else:
        write(dest/"summary.json", summary(rows))


if __name__ == "__main__":
    main()
