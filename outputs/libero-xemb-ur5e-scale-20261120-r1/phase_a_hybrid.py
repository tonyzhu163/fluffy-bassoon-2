"""Phase A: Train 10-channel hybrid (rgb diff + luminance flow) SO(2)-equivariant reader and live-eval vs 24-task RGB & analytic visual flow on 96 UR5e queries."""

import argparse
from functools import partial
import hashlib
import json
from pathlib import Path
import sys

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fluffy_bassoon.xemb.embodiment import EVAL_REGIMES, get_embodiment_spec
from fluffy_bassoon.xemb.readers import (
    ReaderConfig,
    analytic_proprio_sysid,
    analytic_visual_yaw,
    fit_nominal_gain,
    image_motion,
    init_reader,
)
from collect_scale import validate_scale_archive
from collect_ur5e import DEV_ARM, TRAIN_ARMS, issued_exposure
from eval_live_ur5e import check_query_reset, infer_one, load_support_batch, task_clustered_bootstrap
from train_scale import train_one
from train_ur5e_readers import aggregate_seeds, audit_model, load_mode_split, make_donor_permutation
from ur5e_motor import SPLITS_PATH, load_frozen_pi05, run_episode, write_json
from ur5e_sim_worker import UR5eSimulator


PHASE_A_ARMS = (
    "privileged_oracle",
    "analytic_visual_sysid_passive",
    "ours_rgb_active",
    "ours_rgb_passive",
    "ours_hybrid_active",
    "ours_hybrid_passive",
)


def load_tagged_checkpoint(directory: Path, tag_prefix: str, mode: str, model_type: str, seed: int, archive_sha: str, steps: int):
    path = directory / f"{tag_prefix}{mode}-{model_type}-seed{seed}.msgpack"
    saved = serialization.msgpack_restore(path.read_bytes())
    cfg = ReaderConfig(**saved["reader_config"])
    if (saved.get("archive_sha256"), saved["mode"], cfg.model_type, saved["seed"], saved["step"]) != (
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


def cmd_train_hybrid(archive_path: Path, out_dir: Path, steps: int, batch_size: int, seeds: list[int], cnn_width: int, mlp_width: int):
    raw_archive = archive_path.read_bytes()
    archive_sha = hashlib.sha256(raw_archive).hexdigest()
    archive = json.loads(raw_archive)
    validate_scale_archive(archive)

    train_rows = [r for r in archive["trials"] if r["split"] == "train" and r["arm"] in TRAIN_ARMS]
    dev_rows = [r for r in archive["trials"] if r["split"] == "dev" and r["arm"] == DEV_ARM]
    donors = make_donor_permutation(dev_rows, np.random.default_rng(20261120))
    archive_dir = archive_path.parent
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    modes_out = {}
    for mode in ("active_probe", "passive_attempt1"):
        train_data = load_mode_split(archive_dir, train_rows, mode)
        dev_data = load_mode_split(archive_dir, dev_rows, mode)
        nom_mask = np.isclose(train_data["xy_scale"], 1.0)
        nominal_gain = fit_nominal_gain(
            train_data["issued_mean"][nom_mask], train_data["delta_eef"][nom_mask], train_data["valid"][nom_mask]
        )
        per_seed = []
        for seed in seeds:
            params, cfg, meta = train_one(
                mode,
                "factored_equiv",
                seed,
                train_data,
                steps,
                batch_size,
                nominal_gain,
                ckpt_dir,
                archive_sha,
                visual_mode="hybrid",
                cnn_width=cnn_width,
                mlp_width=mlp_width,
                tag_prefix="hybrid-",
            )
            audit = audit_model(params, cfg, dev_data, donors)
            per_seed.append(dict(meta, audit=audit))
        agg = aggregate_seeds(per_seed)
        modes_out[mode] = dict(nominal_gain=nominal_gain, seeds=per_seed, aggregate=agg)
        ov = agg["retained"]["overall"]
        br = agg["retained"]
        print(
            json.dumps(
                dict(
                    mode=mode,
                    model="factored_equiv_hybrid",
                    overall_yaw=round(ov["yaw_err_deg"]["mean"], 2),
                    overall_all=round(ov["all_correct"]["mean"], 3),
                    fam_yaw=round(br["familiar"]["yaw_err_deg"]["mean"], 2),
                    comp_yaw=round(br["compositional_ood"]["yaw_err_deg"]["mean"], 2),
                    cont_yaw=round(br["continuous_angle_ood"]["yaw_err_deg"]["mean"], 2),
                )
            ),
            flush=True,
        )

    write_json(
        out_dir / "hybrid-audit.json",
        dict(
            complete=True,
            archive_sha256=archive_sha,
            steps=steps,
            batch_size=batch_size,
            seeds=seeds,
            cnn_width=cnn_width,
            mlp_width=mlp_width,
            visual_mode="hybrid",
            modes=modes_out,
        ),
    )


def cmd_eval_shard(
    archive_path: Path,
    base_train_dir: Path,
    hybrid_train_dir: Path,
    output_dir: Path,
    shard_idx: int,
    num_shards: int,
    seed_ckpt: int,
):
    archive_dir = archive_path.parent
    raw_archive = archive_path.read_bytes()
    archive_sha = hashlib.sha256(raw_archive).hexdigest()
    archive = json.loads(raw_archive)
    validate_scale_archive(archive)
    audit = json.loads((base_train_dir / "offline-audit.json").read_text())
    hybrid_audit = json.loads((hybrid_train_dir / "hybrid-audit.json").read_text())
    if audit.get("archive_sha256") != archive_sha or hybrid_audit.get("archive_sha256") != archive_sha:
        raise ValueError("Archive SHA-256 mismatch")

    dev_rows = [r for r in archive["trials"] if r["split"] == "dev" and r["arm"] == DEV_ARM]
    dev_tasks = sorted({(r["suite"], int(r["task_id"])) for r in dev_rows})
    my_tasks = {t for idx, t in enumerate(dev_tasks) if idx % num_shards == shard_idx}
    my_indices = [i for i, r in enumerate(dev_rows) if (r["suite"], int(r["task_id"])) in my_tasks]

    passive_fit = audit["modes"]["passive_attempt1"]
    passive_gain = passive_fit["nominal_gain"]
    vis_passive = passive_fit["analytic_visual_sysid"]

    readers = {
        "ours_rgb_active": load_tagged_checkpoint(
            base_train_dir / "checkpoints", "rgb-", "active_probe", "factored_equiv", seed_ckpt, archive_sha, audit["steps"]
        ),
        "ours_rgb_passive": load_tagged_checkpoint(
            base_train_dir / "checkpoints", "rgb-", "passive_attempt1", "factored_equiv", seed_ckpt, archive_sha, audit["steps"]
        ),
        "ours_hybrid_active": load_tagged_checkpoint(
            hybrid_train_dir / "checkpoints", "hybrid-", "active_probe", "factored_equiv", seed_ckpt, archive_sha, hybrid_audit["steps"]
        ),
        "ours_hybrid_passive": load_tagged_checkpoint(
            hybrid_train_dir / "checkpoints", "hybrid-", "passive_attempt1", "factored_equiv", seed_ckpt, archive_sha, hybrid_audit["steps"]
        ),
    }

    suites = sorted(json.loads(SPLITS_PATH.read_text())["development"])
    shard_dir = output_dir / f"shard-{shard_idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    policy = load_frozen_pi05()

    rows = []
    prompts = {}
    query_resets = {}
    opening_proposals = {}
    init_state_id = 13

    with UR5eSimulator() as simulator:
        for local_i, global_idx in enumerate(my_indices):
            sup_row = dev_rows[global_idx]
            suite = sup_row["suite"]
            task_id = int(sup_row["task_id"])
            spec = get_embodiment_spec(sup_row["embodiment"])
            seed = 21800000 + 10_000 * suites.index(suite) + 100 * task_id + init_state_id
            case = dict(arm=DEV_ARM, suite=suite, task_id=task_id, variant="BASE", init_state_id=init_state_id, seed=seed)

            act_batch = load_support_batch(archive_dir, sup_row, "active_probe")
            pas_batch = load_support_batch(archive_dir, sup_row, "passive_attempt1")
            pas_iss = issued_exposure(pas_batch)
            _, sp, gp = analytic_proprio_sysid(
                pas_iss, pas_batch["delta_eef"], pas_batch["delta_jaw"], pas_batch["valid"], nominal_gain_m_per_cmd=passive_gain
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
                "privileged_oracle": (spec.yaw_deg, spec.xy_scale, spec.gripper_sign),
                "analytic_visual_sysid_passive": (yaw_vp, float(sp[0]), int(gp[0])),
                "ours_rgb_active": infer_one(*readers["ours_rgb_active"], act_batch),
                "ours_rgb_passive": infer_one(*readers["ours_rgb_passive"], pas_batch),
                "ours_hybrid_active": infer_one(*readers["ours_hybrid_active"], act_batch),
                "ours_hybrid_passive": infer_one(*readers["ours_hybrid_passive"], pas_batch),
            }

            key = (suite, task_id)
            check_reset = partial(check_query_reset, query_resets, key, sup_row["physics"])
            arms_out = {}
            for arm_name in PHASE_A_ARMS:
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
                nr = rec.pop("native_rewards", [])
                rec["native_rewards_len"] = len(nr)
                rec["native_rewards_nonzero"] = sum(1 for x in nr if float(x) != 0.0)
                prompts.setdefault(key, rec["reset"]["prompt"])
                opening = rec["opening_proposal_sha256"]
                if not opening or opening_proposals.setdefault(key, opening) != opening:
                    raise ValueError(f"Unmatched opening proposal on {case}/{arm_name}")
                arms_out[arm_name] = rec

            row = dict(
                index=global_idx,
                shard=shard_idx,
                arm=DEV_ARM,
                suite=suite,
                task_id=task_id,
                init_state_id=init_state_id,
                seed=seed,
                embodiment=spec.name,
                regime=spec.regime,
                true_latent=dict(yaw_deg=spec.yaw_deg, xy_scale=spec.xy_scale, gripper_sign=spec.gripper_sign),
                arms=arms_out,
            )
            rows.append(row)
            write_json(shard_dir / "live-progress.json", rows)
            print(
                json.dumps(
                    dict(
                        shard=shard_idx,
                        done=local_i + 1,
                        total=len(my_indices),
                        task=f"{suite}-{task_id}",
                        emb=spec.name,
                        succ={a: int(arms_out[a]["success"]) for a in PHASE_A_ARMS},
                    )
                ),
                flush=True,
            )

    write_json(shard_dir / "shard-live.json", dict(shard=shard_idx, archive_sha256=archive_sha, seed_ckpt=seed_ckpt, rows=rows))


def cmd_merge(output_dir: Path, num_shards: int):
    rows = []
    archive_sha = None
    seed_ckpt = None
    for s in range(num_shards):
        part = json.loads((output_dir / f"shard-{s}" / "shard-live.json").read_text())
        archive_sha = archive_sha or part["archive_sha256"]
        seed_ckpt = seed_ckpt or part["seed_ckpt"]
        rows.extend(part["rows"])
    rows.sort(key=lambda r: r["index"])
    if len(rows) != 96:
        raise ValueError(f"Expected 96 merged rows, got {len(rows)}")

    by_regime = {}
    for regime in (*EVAL_REGIMES, "shifted", "overall"):
        if regime == "overall":
            sub = rows
        elif regime == "shifted":
            sub = [r for r in rows if r["embodiment"] != "fam_nom"]
        else:
            sub = [r for r in rows if r["regime"] == regime]
        succ = {a: sum(int(r["arms"][a]["success"]) for r in sub) for a in PHASE_A_ARMS}
        by_regime[regime] = dict(trials=len(sub), successes=succ, rates={a: succ[a] / len(sub) for a in PHASE_A_ARMS})

    contrasts = {}
    for ours in ("ours_rgb_active", "ours_rgb_passive", "ours_hybrid_active", "ours_hybrid_passive"):
        contrasts[ours] = {
            other: task_clustered_bootstrap(rows, ours, other)
            for other in PHASE_A_ARMS
            if other != ours
        }

    summary = dict(
        arm=DEV_ARM,
        panel="phase_a_rgb_vs_hybrid_96",
        arms=list(PHASE_A_ARMS),
        archive_sha256=archive_sha,
        seed_ckpt=seed_ckpt,
        trials=len(rows),
        by_regime=by_regime,
        paired_contrasts_overall=contrasts,
        rows=rows,
    )
    write_json(output_dir / "phase-a-summary.json", summary)
    print(json.dumps(dict(overall=by_regime["overall"], ood_comp=by_regime["compositional_ood"], ood_cont=by_regime["continuous_angle_ood"]), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_tr = sub.add_parser("train-hybrid")
    p_tr.add_argument("--archive", type=Path, required=True)
    p_tr.add_argument("--out-dir", type=Path, required=True)
    p_tr.add_argument("--steps", type=int, default=3000)
    p_tr.add_argument("--batch-size", type=int, default=64)
    p_tr.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    p_tr.add_argument("--cnn-width", type=int, default=96)
    p_tr.add_argument("--mlp-width", type=int, default=192)

    p_ev = sub.add_parser("eval-shard")
    p_ev.add_argument("--archive", type=Path, required=True)
    p_ev.add_argument("--base-train-dir", type=Path, required=True)
    p_ev.add_argument("--hybrid-train-dir", type=Path, required=True)
    p_ev.add_argument("--output-dir", type=Path, required=True)
    p_ev.add_argument("--shard-idx", type=int, required=True)
    p_ev.add_argument("--num-shards", type=int, default=8)
    p_ev.add_argument("--seed-ckpt", type=int, default=41)

    p_mg = sub.add_parser("merge")
    p_mg.add_argument("--output-dir", type=Path, required=True)
    p_mg.add_argument("--num-shards", type=int, default=8)

    args = parser.parse_args()
    if args.cmd == "train-hybrid":
        cmd_train_hybrid(args.archive, args.out_dir, args.steps, args.batch_size, args.seeds, args.cnn_width, args.mlp_width)
    elif args.cmd == "eval-shard":
        cmd_eval_shard(args.archive, args.base_train_dir, args.hybrid_train_dir, args.output_dir, args.shard_idx, args.num_shards, args.seed_ckpt)
    else:
        cmd_merge(args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
