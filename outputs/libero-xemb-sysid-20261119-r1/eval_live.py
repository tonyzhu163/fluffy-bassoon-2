"""Live closed-loop frozen-π0.5 evaluation across all 8 dev tasks x 12 embodiments."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from fluffy_bassoon.xemb.embodiment import EVAL_REGIMES, get_embodiment_spec
from fluffy_bassoon.xemb.readers import (
    ReaderConfig, analytic_proprio_sysid, analytic_visual_yaw, image_motion, init_reader, predict_embodiment,
)
from collect import POC_EMBODIMENTS, POC_TASK, SUPPORT_BUDGET, issued_exposure, read_support, validate_archive
from train_readers import make_donor_permutation, offline_gate
from motor_iface import SPLITS_PATH, load_frozen_pi05, run_episode, write_json
from sim_worker import XEmbSimulator


ARMS = (
    "nominal",
    "privileged_oracle",
    "analytic_proprio_sysid",
    "joint_transformer_active",
    "factored_no_equiv_active",
    "ours_retained_active",
    "ours_retained_passive",
    "ours_cleared",
    "ours_swapped_donor",
    "ours_online_prefix",
)


def load_reader_checkpoint(directory, mode, model_type, seed, archive_sha, steps):
    path = directory / f"{mode}-{model_type}-seed{seed}.msgpack"
    saved = serialization.msgpack_restore(path.read_bytes())
    cfg = ReaderConfig(**saved["reader_config"])
    if (saved.get("input_schema"), saved.get("archive_sha256"), saved["mode"],
        cfg.model_type, saved["seed"], saved["step"]) != (
        "rgb-unit-motion-cm-v1", archive_sha, mode, model_type, seed, steps
    ):
        raise ValueError(f"Reader inputs, archive, fit identity or final step differ on {path}")
    # Flax state dictionaries encode lists under numeric string keys. Restore
    # against the architecture before traversing MLP layers or attention blocks.
    template = init_reader(jax.random.key(seed), cfg)
    params = serialization.from_state_dict(template, saved["params"])
    return jax.tree.map(jnp.asarray, params), cfg


def load_support_batch(archive_dir, row, mode):
    return {k: v[None] for k, v in read_support(archive_dir, row, mode).items()}


def select_panel(archive, audit, panel, init_id, archive_sha, seed):
    if (audit.get("complete") is not True or audit.get("archive_sha256") != archive_sha
            or audit.get("panel") != archive["panel"] or seed not in audit["seeds"]
            or audit.get("offline_gate_passed") is False):
        raise ValueError("A completed fit for this archive and seed is required")
    rows = [r for r in archive["trials"] if r["split"] == "dev"]
    if panel == "development":
        if (archive["panel"] != "full" or audit.get("offline_gate_passed") is not True
                or not offline_gate(audit, 20.0)[0]):
            raise ValueError("Full live evaluation requires the full archive and passed 20-degree offline gate")
    else:
        rows = [r for r in rows if (r["suite"], r["task_id"]) == POC_TASK and r["embodiment"] in POC_EMBODIMENTS]
        if len(rows) != len(POC_EMBODIMENTS):
            raise ValueError("Missing fixed small development panel")
        if panel == "smoke":
            rows = rows[:2]
    if init_id != 13 or any(r["init_state_id"] != 11 for r in rows):
        raise ValueError("Development uses support reset 11 and query reset 13")
    return rows


def query_seed(suites, suite, task_id, init_id):
    return 20400000 + 10_000 * sorted(suites).index(suite) + 100 * task_id + init_id


def check_query_reset(resets, key, support_physics, reset):
    if reset["physics"] == support_physics:
        raise ValueError(f"Query reset collided with support physics on {key}")
    if resets.setdefault(key, reset) != reset:
        raise ValueError(f"Unmatched query physics, public frame, noise or prompt on {key}")


def control_account(arm, support, donor, rec):
    active = arm not in ("nominal", "privileged_oracle", "ours_online_prefix", "ours_retained_passive")
    acquired = 0
    if active:
        acquired = (donor if arm == "ours_swapped_donor" else support)["active_controls"]
    elif arm == "ours_retained_passive":
        acquired = support["passive_controls"]
    prefix = min(SUPPORT_BUDGET, rec["controls"]) if arm == "ours_online_prefix" else 0
    return dict(support_acquired=acquired, support_observed=0 if arm == "ours_cleared" else min(acquired, SUPPORT_BUDGET),
                query_prefix=prefix, query_motor=rec["controls"] - prefix, query_limit=rec["reset"]["horizon"],
                total_executed=acquired + rec["controls"])


def infer_one(params, cfg: ReaderConfig, batch: dict):
    y, s, g = predict_embodiment(params, cfg, batch)
    return float(y[0]), float(s[0]), int(g[0])


def task_clustered_bootstrap(rows: list, arm_a: str, arm_b: str, seed: int = 20261119, draws: int = 2000):
    tasks = sorted({(r["suite"], r["task_id"]) for r in rows})
    by_task = {
        t: np.array(
            [
                int(r["arms"][arm_a]["success"]) - int(r["arms"][arm_b]["success"])
                for r in rows
                if (r["suite"], r["task_id"]) == t
            ],
            dtype=np.float64,
        )
        for t in tasks
    }
    diffs = np.concatenate([by_task[t] for t in tasks])
    wins = int(np.sum(diffs > 0))
    losses = int(np.sum(diffs < 0))
    ties = int(np.sum(diffs == 0))
    rng = np.random.default_rng(seed)
    n_tasks = len(tasks)
    if n_tasks == 1:
        return dict(mean_diff=float(diffs.mean()), ci95=None, wins=wins, losses=losses, ties=ties)
    boot = []
    for _ in range(draws):
        sampled = rng.integers(n_tasks, size=n_tasks)
        vals = np.concatenate([by_task[tasks[i]] for i in sampled])
        boot.append(float(vals.mean()))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(
        mean_diff=float(diffs.mean()),
        ci95=[float(lo), float(hi)],
        wins=wins,
        losses=losses,
        ties=ties,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-state-id", type=int, default=13)
    parser.add_argument("--seed-ckpt", type=int, default=41)
    parser.add_argument("--panel", choices=("smoke", "poc", "development"), default="development")
    args = parser.parse_args()

    archive_dir = args.archive.parent
    raw_archive = args.archive.read_bytes()
    archive_sha = hashlib.sha256(raw_archive).hexdigest()
    archive = json.loads(raw_archive)
    validate_archive(archive)
    audit = json.loads((args.train_dir / "offline-audit.json").read_text())
    trial_support_rows = select_panel(archive, audit, args.panel, args.init_state_id, archive_sha, args.seed_ckpt)
    if set(archive["modes"]) != {"active_probe", "passive_attempt1"}:
        parser.error("Live active/passive comparison needs both support modes")

    arms = list(ARMS)
    active_fit = audit["modes"]["active_probe"]
    active_gain = active_fit["nominal_gain"]
    visual = active_fit["analytic_visual_sysid"]
    if visual["available"]:
        arms.append("analytic_visual_sysid")
    readers = {}
    for arm, mode, model_type in (
        ("ours_retained_active", "active_probe", "factored_equiv"),
        ("ours_retained_passive", "passive_attempt1", "factored_equiv"),
        ("joint_transformer_active", "active_probe", "joint_transformer"),
        ("factored_no_equiv_active", "active_probe", "factored_no_equiv"),
    ):
        if arm in arms:
            readers[arm] = load_reader_checkpoint(args.train_dir / "checkpoints", mode, model_type,
                                                  args.seed_ckpt, archive_sha, audit["steps"])
    def online_reader(probe_trace):
        return infer_one(*readers["ours_retained_active"], probe_trace)

    suites = json.loads(SPLITS_PATH.read_text())["development"]
    donors = make_donor_permutation(trial_support_rows, np.random.default_rng(20261119))
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("live-progress.json", "live-eval-summary.json")):
        raise FileExistsError("Use a fresh live output directory")
    policy = load_frozen_pi05()
    rows = []
    prompts = {}
    query_resets = {}
    opening_proposals = {}
    with XEmbSimulator() as simulator:
        for idx, sup_row in enumerate(trial_support_rows):
            suite = sup_row["suite"]
            task_id = int(sup_row["task_id"])
            spec = get_embodiment_spec(sup_row["embodiment"])
            seed = query_seed(suites, suite, task_id, args.init_state_id)
            case = dict(
                suite=suite,
                task_id=task_id,
                variant="BASE",
                init_state_id=args.init_state_id,
                seed=seed,
            )

            act_batch = load_support_batch(archive_dir, sup_row, "active_probe")
            pas_batch = load_support_batch(archive_dir, sup_row, "passive_attempt1")
            clr_batch = dict(act_batch, valid=np.zeros_like(act_batch["valid"]))

            donor_row = trial_support_rows[donors[idx]]
            don_batch = load_support_batch(archive_dir, donor_row, "active_probe")

            issued = issued_exposure(act_batch)
            ya, sa, ga = analytic_proprio_sysid(
                issued,
                act_batch["delta_eef"],
                act_batch["delta_jaw"],
                act_batch["valid"],
                nominal_gain_m_per_cmd=active_gain,
            )
            preds = {
                "nominal": (0.0, 1.0, 1),
                "privileged_oracle": (spec.yaw_deg, spec.xy_scale, spec.gripper_sign),
                "analytic_proprio_sysid": (float(ya[0]), float(sa[0]), int(ga[0])),
                "ours_cleared": online_reader(clr_batch),
                "ours_swapped_donor": online_reader(don_batch),
            }
            for name, reader in readers.items():
                batch = pas_batch if name == "ours_retained_passive" else act_batch
                preds[name] = infer_one(*reader, batch)
            if visual["available"]:
                valid = act_batch["valid"]
                motion = image_motion(act_batch["img_before"], act_batch["img_after"], valid)
                yaw = analytic_visual_yaw(issued, motion, act_batch["delta_eef"][..., 2],
                                          valid, np.asarray(visual["projection"]))
                preds["analytic_visual_sysid"] = (float(yaw[0]), float(sa[0]), int(ga[0]))

            key = (suite, task_id)
            check_reset = partial(check_query_reset, query_resets, key, sup_row["physics"])
            arms_out = {}
            for arm_name in arms:
                prompt = prompts.get(key)
                if arm_name == "ours_online_prefix":
                    rec = run_episode(
                        simulator,
                        policy,
                        case,
                        spec,
                        prompt=prompt,
                        check_reset=check_reset,
                        online_probe_reader=online_reader,
                    )
                else:
                    yh, sh, gh = preds[arm_name]
                    rec = run_episode(
                        simulator,
                        policy,
                        case,
                        spec,
                        prompt=prompt,
                        check_reset=check_reset,
                        yaw_hat=yh,
                        scale_hat=sh,
                        grip_hat=gh,
                    )
                prompts.setdefault(key, rec["reset"]["prompt"])
                if arm_name != "ours_online_prefix":
                    opening = rec["opening_proposal_sha256"]
                    if not opening or opening_proposals.setdefault(key, opening) != opening:
                        raise ValueError(f"Unmatched opening frozen motor proposal on {case}/{arm_name}")
                rec["control_budget"] = control_account(arm_name, sup_row, donor_row, rec)
                arms_out[arm_name] = rec

            row = dict(
                index=idx,
                suite=suite,
                task_id=task_id,
                init_state_id=args.init_state_id,
                seed=seed,
                embodiment=spec.name,
                regime=spec.regime,
                true_latent=dict(yaw_deg=spec.yaw_deg, xy_scale=spec.xy_scale, gripper_sign=spec.gripper_sign),
                donor_embodiment=donor_row["embodiment"],
                arms=arms_out,
            )
            rows.append(row)
            write_json(args.output / "live-progress.json", rows)
            print(
                json.dumps(
                    dict(
                        done=len(rows),
                        total=len(trial_support_rows),
                        emb=spec.name,
                        succ={a: arms_out[a]["success"] for a in arms},
                    )
                ),
                flush=True,
            )

    by_regime = {}
    for regime in (*EVAL_REGIMES, "shifted", "overall"):
        if regime == "shifted":
            grp = [r for r in rows if r["embodiment"] != "fam_nom"]
        elif regime == "overall":
            grp = rows
        else:
            grp = [r for r in rows if r["regime"] == regime]
        if not grp:
            continue
        by_regime[regime] = dict(
            trials=len(grp),
            successes={a: sum(r["arms"][a]["success"] for r in grp) for a in arms},
            rates={a: sum(r["arms"][a]["success"] for r in grp) / len(grp) for a in arms},
        )

    contrasts = {}
    for main_arm in ("ours_retained_active", "ours_retained_passive", "ours_online_prefix"):
        if main_arm not in arms:
            continue
        contrasts[main_arm] = {
            comp: task_clustered_bootstrap(rows, main_arm, comp)
            for comp in arms
            if comp != main_arm
        }

    summary = dict(
        panel=args.panel, arms=arms, archive_sha256=archive_sha,
        visual_baseline_available=visual["available"], matched_support_cap=SUPPORT_BUDGET,
        seed_ckpt=args.seed_ckpt,
        trials=len(rows),
        by_regime=by_regime,
        paired_contrasts_overall=contrasts,
        rows=rows,
    )
    write_json(args.output / "live-eval-summary.json", summary)
    print(json.dumps(dict(trials=len(rows), by_regime=by_regime, contrasts=contrasts), indent=2), flush=True)


if __name__ == "__main__":
    main()
