"""Paired native LIBERO evaluation for Stage-1 latent steering and Stage-2 history beliefs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from data import DEFAULT_CASES, parse_cases
from fluffy_bassoon.libero.history_encoder import (
    HistoryEncoderConfig,
    apply_history_encoder,
    decode_steering_latent,
    pack_control_tokens,
)
from motor import LatentMotor


CAMPAIGN = Path(__file__).resolve().parents[1] / "libero-meta-learning-20261117-r1"
PRO = Path("/mowenjie/fluffy-bassoon-2/sources/libero-pro2-e5500c69")
sys.path.insert(0, str(CAMPAIGN))
from reference import write
from simulator import Simulator


NOISE_SEED_OFFSET = 817117
PREFIX_CONTROLS = 40


def digest(array):
    array = np.asarray(array)
    prefix = repr((array.shape, array.dtype.str)).encode()
    return hashlib.sha256(prefix + array.tobytes()).hexdigest()


def public_digest(raw):
    return {key: digest(value) for key, value in raw.items()}


def has_variant(manifest, suite, task_id, variant):
    try:
        manifest.require(suite, task_id, variant)
        return True
    except Exception:
        return False


def native_cases(manifest, tasks, cases, start_ids, seed_base):
    for suite_index, (suite, task_ids) in enumerate(sorted(tasks.items())):
        for task in task_ids:
            for repeat, reset_id in enumerate(start_ids):
                seed = seed_base + 10_000 * suite_index + 100 * task + repeat
                for variant in cases:
                    if has_variant(manifest, suite, task, variant):
                        yield dict(
                            suite=suite,
                            task_id=task,
                            variant=variant,
                            init_state_id=reset_id,
                            seed=seed,
                        )


def rollout_episode(
    simulator,
    motor,
    case,
    prompt,
    *,
    latent=None,
    active=True,
    switch_after=None,
    switch_latent=None,
    save_tape=False,
    max_controls=None,
):
    """Run one paired episode reusing the persistent simulator and batched chunk IPC."""
    spec, initial = simulator.call("create", case)
    raw = initial["observation"]
    released_prompt = initial["language_instruction"]
    prompt = released_prompt if prompt is None else prompt
    reset = dict(
        physics=initial["environment_reproducibility"]["initial_state_fingerprint"],
        public=public_digest(raw),
        horizon=spec["max_steps"],
        prompt=prompt,
    )
    bounds = initial["action_schema"]
    low, high = np.asarray(bounds["low"], np.float32), np.asarray(bounds["high"], np.float32)
    rng = np.random.default_rng(case["seed"] + NOISE_SEED_OFFSET)
    initial_position = raw["robot0_eef_pos"].copy()
    issued, positions, gripper, rewards, waits = [], [], [], [], []
    step = None
    limit = reset["horizon"] if max_controls is None else min(reset["horizon"], max_controls)
    start = time.monotonic()
    done = False
    while not done and len(issued) < limit:
        noise = rng.standard_normal((10, 32)).astype(np.float32)
        cur_latent = switch_latent if switch_after is not None and len(issued) >= switch_after else latent
        proposal = np.asarray(motor.proposal(raw, prompt, noise, latent=cur_latent, active=active), np.float32)
        remaining = limit - len(issued)
        if switch_after is not None and len(issued) < switch_after:
            remaining = min(remaining, switch_after - len(issued))
        count = min(5, remaining)
        bounded = np.clip(proposal[:count], low, high)
        step = simulator.call("chunk", bounded)
        n = step["steps"]
        issued.extend(bounded[:n])
        positions.extend(step["positions"])
        gripper.extend(step["gripper_positions"])
        rewards.extend(step["rewards"])
        waits.extend([False] * n)
        raw = step["observation"]
        done = step["terminated"] or step["truncated"]

    record = dict(
        reset=reset,
        released_prompt=released_prompt,
        controls=len(issued),
        success=bool(step["success"]) if step is not None else False,
        terminated=bool(step["terminated"]) if step is not None else False,
        truncated=bool(step["truncated"]) if step is not None else False,
        seconds=round(time.monotonic() - start, 3),
    )
    if save_tape:
        tape = dict(
            issued=np.asarray(issued, dtype=np.float32),
            positions=np.asarray(positions, dtype=np.float32),
            initial_position=np.asarray(initial_position, dtype=np.float32),
            gripper=np.asarray(gripper, dtype=np.float32),
            rewards=np.asarray(rewards, dtype=np.float32),
            waits=np.asarray(waits, dtype=np.bool_),
            final_raw=raw,
            rng_state=rng.bit_generator.state,
            done=done,
        )
        return record, tape
    return record


def stage_adapter(simulator, manifest, checkpoint_path, output, panel, cases_override):
    motor = LatentMotor(checkpoint_path)
    cases = parse_cases(cases_override) if cases_override else motor.cases
    adapter_cases = list(motor.cases)
    splits = json.loads((CAMPAIGN / "SPLITS.json").read_text())
    source_tasks = {suite: task_ids[:1] for suite, task_ids in splits["train"].items()}
    dev_tasks = splits["development"]
    start_ids = (13, 19)
    if panel == "smoke":
        source_tasks = dev_tasks = {"libero_goal": [4]}
        start_ids = (13,)

    source_cases = list(native_cases(manifest, source_tasks, cases, (11,), 19601000))
    query_cases = list(native_cases(manifest, dev_tasks, cases, start_ids, 19701000))
    fixed_arms = {f"z_{c}": motor.condition_latent(adapter_cases.index(c)) for c in adapter_cases}

    def run_panel(case_list, arms_dict):
        prompts = {}
        rows = []
        for case in case_list:
            key = case["suite"], case["task_id"]
            outcomes = {}
            for arm_name, (latent, active) in arms_dict.items():
                if arm_name == "privileged_true_z":
                    latent = motor.condition_latent(adapter_cases.index(case["variant"]))
                    active = True
                outcomes[arm_name] = rollout_episode(
                    simulator, motor, case, prompts.get(key), latent=latent, active=active
                )
            first = next(iter(outcomes.values()))
            prompts.setdefault(key, first["reset"]["prompt"])
            if any(row["reset"] != first["reset"] for row in outcomes.values()):
                raise ValueError(f"Paired reset mismatch on {case}")
            rows.append(dict(case=case, arms=outcomes))
            print(json.dumps(dict(case=case, success={k: v["success"] for k, v in outcomes.items()})), flush=True)
        return rows

    source_arms = {name: (vec, True) for name, vec in fixed_arms.items()}
    source_rows = run_panel(source_cases, source_arms)
    fixed_counts = {name: sum(row["arms"][name]["success"] for row in source_rows) for name in fixed_arms}
    best_fixed_name = max(fixed_counts, key=fixed_counts.__getitem__)
    best_fixed_vec = fixed_arms[best_fixed_name]

    dev_arms = {"default": (None, True), "solver_zero": (best_fixed_vec, False)}
    dev_arms.update({name: (vec, True) for name, vec in fixed_arms.items()})
    dev_arms["privileged_true_z"] = (best_fixed_vec, True)
    dev_rows = run_panel(query_cases, dev_arms)

    summary_by_case = {}
    for variant in (*cases, "overall"):
        group = [r for r in dev_rows if variant == "overall" or r["case"]["variant"] == variant]
        if not group:
            continue
        succ = {arm: sum(r["arms"][arm]["success"] for r in group) for arm in dev_arms}
        succ["best_fixed_z"] = succ[best_fixed_name]
        summary_by_case[variant] = dict(trials=len(group), successes=succ)

    result = dict(
        stage="adapter",
        best_fixed_name=best_fixed_name,
        source_counts=fixed_counts,
        summary=summary_by_case,
        dev_rows=dev_rows,
    )
    write(output / "adapter-summary.json", result)
    print(json.dumps(dict(best_fixed_name=best_fixed_name, summary=summary_by_case), indent=2), flush=True)


def stage_collect_history(simulator, manifest, adapter_ckpt, output, panel):
    motor = LatentMotor(adapter_ckpt)
    cases = motor.cases
    splits = json.loads((CAMPAIGN / "SPLITS.json").read_text())
    train_tasks = splits["development_scaling"]["task_subsets"]["8"]
    dev_tasks = splits["development"]
    train_starts = (0, 3)
    dev_starts = (13, 19)
    if panel == "smoke":
        train_tasks = dev_tasks = {"libero_goal": [4]}
        train_starts = (0,)
        dev_starts = (13,)

    tapes_dir = output / "tapes"
    tapes_dir.mkdir(parents=True, exist_ok=True)
    trials = []

    def collect_split(split_name, tasks, starts, seed_base):
        prompts = {}
        for idx, case in enumerate(native_cases(manifest, tasks, cases, starts, seed_base)):
            key = case["suite"], case["task_id"]
            support_case = dict(case)
            query_case = dict(case, init_state_id=case["init_state_id"] + 1, seed=case["seed"] + 1)
            sup_rec, sup_tape = rollout_episode(
                simulator, motor, support_case, prompts.get(key), save_tape=True
            )
            prompts.setdefault(key, sup_rec["reset"]["prompt"])
            # Reset query pose, record query embedding at t=0 and after 40-control default prefix.
            _, q_init = simulator.call("create", query_case)
            q_reset_emb = motor.query_embedding(q_init["observation"], prompts[key])
            pfx_rec, pfx_tape = rollout_episode(
                simulator, motor, query_case, prompts[key], save_tape=True, max_controls=PREFIX_CONTROLS
            )
            q_pfx_emb = motor.query_embedding(pfx_tape["final_raw"], prompts[key])

            folder = tapes_dir / f"{split_name}-{idx:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                folder / "support.npz",
                issued=sup_tape["issued"],
                positions=sup_tape["positions"],
                initial_position=sup_tape["initial_position"],
                gripper=sup_tape["gripper"],
                rewards=sup_tape["rewards"],
                waits=sup_tape["waits"],
            )
            np.savez_compressed(
                folder / "prefix.npz",
                issued=pfx_tape["issued"],
                positions=pfx_tape["positions"],
                initial_position=pfx_tape["initial_position"],
                gripper=pfx_tape["gripper"],
                rewards=pfx_tape["rewards"],
                waits=pfx_tape["waits"],
            )
            np.save(folder / "query_reset.npy", q_reset_emb)
            np.save(folder / "query_prefix.npy", q_pfx_emb)
            trials.append(
                dict(
                    split=split_name,
                    suite=case["suite"],
                    task_id=case["task_id"],
                    variant=case["variant"],
                    condition_id=cases.index(case["variant"]),
                    support_case=support_case,
                    query_case=query_case,
                    support_success=sup_rec["success"],
                    support_path=f"tapes/{folder.name}/support.npz",
                    prefix_path=f"tapes/{folder.name}/prefix.npz",
                    query_reset_embedding=f"tapes/{folder.name}/query_reset.npy",
                    query_prefix_embedding=f"tapes/{folder.name}/query_prefix.npy",
                )
            )

    collect_split("train", train_tasks, train_starts, 19801000)
    collect_split("dev", dev_tasks, dev_starts, 19901000)
    write(output / "archive.json", dict(cases=list(cases), trials=trials))
    print(json.dumps(dict(collected=len(trials), cases=list(cases))), flush=True)


def stage_history_eval(simulator, manifest, adapter_ckpt, history_ckpt, archive_path, output):
    motor = LatentMotor(adapter_ckpt)
    cases = motor.cases
    h_state = serialization.msgpack_restore(history_ckpt.read_bytes())
    cfg = HistoryEncoderConfig(**h_state["encoder_config"])
    h_params = jax.tree.map(jnp.asarray, h_state["params"])
    cond_table = jnp.asarray(motor.params["condition_table"], dtype=jnp.float32)

    archive = json.loads(archive_path.read_text())
    archive_dir = archive_path.parent
    dev_trials = [t for t in archive["trials"] if t["split"] == "dev"]
    if not dev_trials:
        raise ValueError("Archive contains no dev trials")

    def predict_latent(support_tape, prefix_tape, q_emb, *, retain_support, retain_prefix, mode):
        if support_tape is not None:
            st, sv = pack_control_tokens(
                support_tape["issued"],
                support_tape["positions"],
                support_tape["initial_position"],
                support_tape["gripper"],
                support_tape["rewards"],
                support_tape["waits"],
                max_controls=cfg.max_support_controls,
                controls_per_token=cfg.controls_per_token,
            )
        else:
            st = np.zeros((cfg.support_token_count, cfg.token_input_dim), dtype=np.float32)
            sv = np.zeros((cfg.support_token_count,), dtype=bool)
        if prefix_tape is not None:
            pt, pv = pack_control_tokens(
                prefix_tape["issued"],
                prefix_tape["positions"],
                prefix_tape["initial_position"],
                prefix_tape["gripper"],
                prefix_tape["rewards"],
                prefix_tape["waits"],
                max_controls=cfg.max_prefix_controls,
                controls_per_token=cfg.controls_per_token,
            )
        else:
            pt = np.zeros((cfg.prefix_token_count, cfg.token_input_dim), dtype=np.float32)
            pv = np.zeros((cfg.prefix_token_count,), dtype=bool)
        logits, delta = apply_history_encoder(
            h_params,
            cfg,
            jnp.asarray(st)[None],
            jnp.asarray(sv)[None],
            jnp.asarray(pt)[None],
            jnp.asarray(pv)[None],
            jnp.asarray(q_emb)[None],
            retain_support=retain_support,
            retain_prefix=retain_prefix,
        )
        z = decode_steering_latent(logits, delta, cond_table, mode=mode)
        return np.asarray(z[0], dtype=np.float32), int(jnp.argmax(logits[0]))

    # Pre-index swapped donor support tapes from distinct conditions.
    rng = np.random.default_rng(20261119)
    rows = []
    for idx, trial in enumerate(dev_trials):
        q_case = trial["query_case"]
        prompt = None
        with np.load(archive_dir / trial["support_path"]) as s_npz:
            own_sup = {k: s_npz[k] for k in s_npz.files}
        others = [j for j, t in enumerate(dev_trials) if t["condition_id"] != trial["condition_id"]]
        donor_idx = int(others[rng.integers(len(others))]) if others else idx
        with np.load(archive_dir / dev_trials[donor_idx]["support_path"]) as d_npz:
            donor_sup = {k: d_npz[k] for k in d_npz.files}

        # Run default once from reset to get prompt and opening state.
        default_rec = rollout_episode(simulator, motor, q_case, None, latent=None)
        prompt = default_rec["reset"]["prompt"]
        q_reset_emb = np.load(archive_dir / trial["query_reset_embedding"])
        q_pfx_emb = np.load(archive_dir / trial["query_prefix_embedding"])
        with np.load(archive_dir / trial["prefix_path"]) as p_npz:
            pfx_tape = {k: p_npz[k] for k in p_npz.files}

        z_ret, pred_ret = predict_latent(own_sup, None, q_reset_emb, retain_support=True, retain_prefix=False, mode="prototype")
        z_soft, _ = predict_latent(own_sup, None, q_reset_emb, retain_support=True, retain_prefix=False, mode="soft")
        z_clr, pred_clr = predict_latent(own_sup, None, q_reset_emb, retain_support=False, retain_prefix=False, mode="prototype")
        z_don, pred_don = predict_latent(donor_sup, None, q_reset_emb, retain_support=True, retain_prefix=False, mode="prototype")
        z_pfx, pred_pfx = predict_latent(None, pfx_tape, q_pfx_emb, retain_support=False, retain_prefix=True, mode="prototype")
        z_true = motor.condition_latent(int(trial["condition_id"]))

        arms = {
            "default": default_rec,
            "history_retained": rollout_episode(simulator, motor, q_case, prompt, latent=z_ret),
            "history_retained_soft": rollout_episode(simulator, motor, q_case, prompt, latent=z_soft),
            "history_cleared": rollout_episode(simulator, motor, q_case, prompt, latent=z_clr),
            "history_swapped_donor": rollout_episode(simulator, motor, q_case, prompt, latent=z_don),
            "history_current_prefix_only": rollout_episode(
                simulator,
                motor,
                q_case,
                prompt,
                latent=None,
                switch_after=PREFIX_CONTROLS,
                switch_latent=z_pfx,
            ),
            "privileged_true_z": rollout_episode(simulator, motor, q_case, prompt, latent=z_true),
        }
        rows.append(
            dict(
                case=q_case,
                predictions=dict(retained=pred_ret, cleared=pred_clr, donor=pred_don, prefix=pred_pfx),
                arms=arms,
            )
        )

    totals = {arm: sum(r["arms"][arm]["success"] for r in rows) for arm in rows[0]["arms"]}
    summary = dict(trials=len(rows), successes=totals, rows=rows)
    write(output / "history-eval.json", summary)
    print(json.dumps(dict(trials=len(rows), successes=totals), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("adapter", "collect-history", "history"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Stage-1 adapter checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", choices=("smoke", "development"), default="development")
    parser.add_argument("--cases", type=str, default=None)
    parser.add_argument("--history-checkpoint", type=Path, default=None)
    parser.add_argument("--archive", type=Path, default=None)
    args = parser.parse_args()

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Expected one allocated GPU for evaluation")
    from libero_remote.environment.task_manifest import TaskManifest

    manifest = TaskManifest.load(str(PRO / "assets/task_manifest.json"))
    args.output.mkdir(parents=True, exist_ok=True)
    with Simulator() as simulator:
        if args.stage == "adapter":
            stage_adapter(simulator, manifest, args.checkpoint, args.output, args.panel, args.cases)
        elif args.stage == "collect-history":
            stage_collect_history(simulator, manifest, args.checkpoint, args.output, args.panel)
        elif args.stage == "history":
            if not args.history_checkpoint or not args.archive:
                parser.error("--stage history requires --history-checkpoint and --archive")
            stage_history_eval(simulator, manifest, args.checkpoint, args.history_checkpoint, args.archive, args.output)


if __name__ == "__main__":
    main()
