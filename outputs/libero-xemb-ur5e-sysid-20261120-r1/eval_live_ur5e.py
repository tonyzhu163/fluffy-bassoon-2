"""Live closed-loop frozen-pi0.5 evaluation on held-out 6-DoF UR5e across 8 dev tasks x 6 embodiments."""

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
    ReaderConfig,
    analytic_proprio_sysid,
    analytic_visual_yaw,
    image_motion,
    init_reader,
    predict_embodiment,
)
from collect_ur5e import DEV_ARM, SUPPORT_BUDGET, issued_exposure, read_support, validate_archive
from gate1_ur5e import GATE1_EMBODIMENT_NAMES
from train_ur5e_readers import make_donor_permutation
from ur5e_motor import SPLITS_PATH, load_frozen_pi05, run_episode, write_json
from ur5e_sim_worker import UR5eSimulator


ARMS = (
    "nominal",
    "privileged_oracle",
    "analytic_proprio_sysid",
    "analytic_visual_sysid_active",
    "analytic_visual_sysid_passive",
    "joint_transformer_active",
    "factored_no_equiv_active",
    "ours_retained_active",
    "ours_retained_passive",
    "ours_cleared",
    "ours_swapped_donor",
)


def load_reader_checkpoint(directory: Path, mode: str, model_type: str, seed: int, archive_sha: str, steps: int):
    path = directory / f"{mode}-{model_type}-seed{seed}.msgpack"
    saved = serialization.msgpack_restore(path.read_bytes())
    cfg = ReaderConfig(**saved["reader_config"])
    if (saved.get("input_schema"), saved.get("archive_sha256"), saved["mode"], cfg.model_type, saved["seed"], saved["step"]) != (
        "rgb-unit-motion-cm-v1",
        archive_sha,
        mode,
        model_type,
        seed,
        steps,
    ):
        raise ValueError(f"Checkpoint metadata mismatch on {path}")
    template = init_reader(jax.random.key(seed), cfg)
    params = serialization.from_state_dict(template, saved["params"])
    return jax.tree.map(jnp.asarray, params), cfg


def load_support_batch(archive_dir: Path, row: dict, mode: str):
    return {k: v[None] for k, v in read_support(archive_dir, row, mode).items()}


def infer_one(params, cfg: ReaderConfig, batch: dict):
    y, s, g = predict_embodiment(params, cfg, batch)
    return float(y[0]), float(s[0]), int(g[0])


def check_query_reset(resets, key, support_physics, reset):
    if reset["physics"] == support_physics:
        raise ValueError(f"Query reset collided with support physics on {key}")
    if resets.setdefault(key, reset) != reset:
        raise ValueError(f"Unmatched query physics, public frame, noise or prompt on {key}")


def task_clustered_bootstrap(rows: list, arm_a: str, arm_b: str, seed: int = 20261120, draws: int = 2000):
    tasks = sorted({(r["suite"], r["task_id"]) for r in rows})
    by_task = {
        t: np.array(
            [int(r["arms"][arm_a]["success"]) - int(r["arms"][arm_b]["success"]) for r in rows if (r["suite"], r["task_id"]) == t],
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
    return dict(mean_diff=float(diffs.mean()), ci95=[float(lo), float(hi)], wins=wins, losses=losses, ties=ties)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init-state-id", type=int, default=13)
    parser.add_argument("--seed-ckpt", type=int, default=41)
    parser.add_argument("--panel", choices=("smoke", "development"), default="development")
    args = parser.parse_args()

    archive_dir = args.archive.parent
    raw_archive = args.archive.read_bytes()
    archive_sha = hashlib.sha256(raw_archive).hexdigest()
    archive = json.loads(raw_archive)
    validate_archive(archive)
    audit = json.loads((args.train_dir / "offline-audit.json").read_text())
    if audit.get("complete") is not True or audit.get("archive_sha256") != archive_sha:
        raise ValueError("Completed offline-audit.json matching archive SHA-256 is required")

    dev_rows = [r for r in archive["trials"] if r["split"] == "dev" and r["arm"] == DEV_ARM]
    if args.panel == "development":
        trial_rows = [r for r in dev_rows if r["embodiment"] in GATE1_EMBODIMENT_NAMES]
        if len(trial_rows) != 48:
            raise ValueError(f"Expected 48 UR5e dev trials (8 tasks x 6 embodiments), got {len(trial_rows)}")
    else:
        trial_rows = [r for r in dev_rows if (r["suite"], r["task_id"]) == ("libero_goal", 1) and r["embodiment"] in ("fam_nom", "comp_180_lo_inv")]

    active_fit = audit["modes"]["active_probe"]
    passive_fit = audit["modes"]["passive_attempt1"]
    active_gain = active_fit["nominal_gain"]
    passive_gain = passive_fit["nominal_gain"]
    vis_active = active_fit["analytic_visual_sysid"]
    vis_passive = passive_fit["analytic_visual_sysid"]

    readers = {}
    for arm_name, mode, model_type in (
        ("ours_retained_active", "active_probe", "factored_equiv"),
        ("ours_retained_passive", "passive_attempt1", "factored_equiv"),
        ("joint_transformer_active", "active_probe", "joint_transformer"),
        ("factored_no_equiv_active", "active_probe", "factored_no_equiv"),
    ):
        readers[arm_name] = load_reader_checkpoint(
            args.train_dir / "checkpoints", mode, model_type, args.seed_ckpt, archive_sha, audit["steps"]
        )

    suites = sorted(json.loads(SPLITS_PATH.read_text())["development"])
    donors = make_donor_permutation(trial_rows, np.random.default_rng(20261120))
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("live-progress.json", "live-eval-summary.json")):
        raise FileExistsError("Use a fresh live output directory")

    policy = load_frozen_pi05()
    rows = []
    prompts = {}
    query_resets = {}
    opening_proposals = {}

    with UR5eSimulator() as simulator:
        for idx, sup_row in enumerate(trial_rows):
            suite = sup_row["suite"]
            task_id = int(sup_row["task_id"])
            spec = get_embodiment_spec(sup_row["embodiment"])
            seed = 20800000 + 10_000 * suites.index(suite) + 100 * task_id + args.init_state_id
            case = dict(
                arm=DEV_ARM,
                suite=suite,
                task_id=task_id,
                variant="BASE",
                init_state_id=args.init_state_id,
                seed=seed,
            )

            act_batch = load_support_batch(archive_dir, sup_row, "active_probe")
            pas_batch = load_support_batch(archive_dir, sup_row, "passive_attempt1")
            clr_batch = dict(act_batch, valid=np.zeros_like(act_batch["valid"]))
            donor_row = trial_rows[donors[idx]]
            don_batch = load_support_batch(archive_dir, donor_row, "active_probe")

            act_iss = issued_exposure(act_batch)
            ya, sa, ga = analytic_proprio_sysid(
                act_iss, act_batch["delta_eef"], act_batch["delta_jaw"], act_batch["valid"], nominal_gain_m_per_cmd=active_gain
            )
            pas_iss = issued_exposure(pas_batch)
            yp, sp, gp = analytic_proprio_sysid(
                pas_iss, pas_batch["delta_eef"], pas_batch["delta_jaw"], pas_batch["valid"], nominal_gain_m_per_cmd=passive_gain
            )

            # Analytic visual baselines (active + passive).
            act_mot = image_motion(act_batch["img_before"], act_batch["img_after"], act_batch["valid"])
            yaw_va = (
                float(
                    analytic_visual_yaw(
                        act_iss, act_mot, act_batch["delta_eef"][..., 2], act_batch["valid"], np.asarray(vis_active["projection"])
                    )[0]
                )
                if vis_active.get("available")
                else 0.0
            )
            pas_mot = image_motion(pas_batch["img_before"], pas_batch["img_after"], pas_batch["valid"])
            yaw_vp = (
                float(
                    analytic_visual_yaw(
                        pas_iss, pas_mot, pas_batch["delta_eef"][..., 2], pas_batch["valid"], np.asarray(vis_passive["projection"])
                    )[0]
                )
                if vis_passive.get("available")
                else 0.0
            )

            preds = {
                "nominal": (0.0, 1.0, 1),
                "privileged_oracle": (spec.yaw_deg, spec.xy_scale, spec.gripper_sign),
                "analytic_proprio_sysid": (float(ya[0]), float(sa[0]), int(ga[0])),
                "analytic_visual_sysid_active": (yaw_va, float(sa[0]), int(ga[0])),
                "analytic_visual_sysid_passive": (yaw_vp, float(sp[0]), int(gp[0])),
                "ours_cleared": infer_one(*readers["ours_retained_active"], clr_batch),
                "ours_swapped_donor": infer_one(*readers["ours_retained_active"], don_batch),
            }
            for name, reader in readers.items():
                batch = pas_batch if name == "ours_retained_passive" else act_batch
                preds[name] = infer_one(*reader, batch)

            key = (suite, task_id)
            check_reset = partial(check_query_reset, query_resets, key, sup_row["physics"])
            arms_out = {}
            for arm_name in ARMS:
                prompt = prompts.get(key)
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
                opening = rec["opening_proposal_sha256"]
                if not opening or opening_proposals.setdefault(key, opening) != opening:
                    raise ValueError(f"Unmatched opening frozen motor proposal on {case}/{arm_name}")
                arms_out[arm_name] = rec

            row = dict(
                index=idx,
                arm=DEV_ARM,
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
                        total=len(trial_rows),
                        task=f"{suite}-{task_id}",
                        emb=spec.name,
                        succ={a: int(arms_out[a]["success"]) for a in ARMS},
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
            successes={a: sum(int(bool(r["arms"][a]["success"])) for r in grp) for a in ARMS},
            rates={a: sum(int(bool(r["arms"][a]["success"])) for r in grp) / len(grp) for a in ARMS},
        )

    contrasts = {}
    for main_arm in ("ours_retained_passive", "ours_retained_active"):
        contrasts[main_arm] = {
            comp: task_clustered_bootstrap(rows, main_arm, comp) for comp in ARMS if comp != main_arm
        }

    summary = dict(
        arm=DEV_ARM,
        panel=args.panel,
        arms=list(ARMS),
        archive_sha256=archive_sha,
        matched_support_cap=SUPPORT_BUDGET,
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
