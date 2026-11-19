"""Live closed-loop frozen-π0.5 evaluation across all 8 dev tasks x 12 embodiments."""

import argparse
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from fluffy_bassoon.xemb.embodiment import EVAL_EMBODIMENTS, EVAL_REGIMES, get_embodiment_spec
from fluffy_bassoon.xemb.readers import ReaderConfig, analytic_proprio_sysid, predict_embodiment
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


def load_reader_checkpoint(path: Path):
    saved = serialization.msgpack_restore(path.read_bytes())
    params = jax.tree.map(jnp.asarray, saved["params"])
    cfg = ReaderConfig(**saved["reader_config"])
    return params, cfg, float(saved["nominal_gain"])


def load_support_batch(archive_dir: Path, row: dict, mode: str) -> dict:
    key_name = "active_path" if mode == "active_probe" else "passive_path"
    with np.load(archive_dir / row[key_name], allow_pickle=False) as npz:
        return dict(
            img_before=npz["img_before"][None].astype(np.float32) / 255.0,
            img_after=npz["img_after"][None].astype(np.float32) / 255.0,
            issued_mean=npz["issued_mean"][None].astype(np.float32),
            delta_eef=npz["delta_eef"][None].astype(np.float32),
            delta_jaw=npz["delta_jaw"][None].astype(np.float32),
            valid=npz["valid"][None].astype(bool),
            frame0_img=npz["frame0_img"][None].astype(np.float32) / 255.0,
            frame0_state=npz["frame0_state"][None].astype(np.float32),
        )


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
    parser.add_argument("--panel", choices=("smoke", "development"), default="development")
    args = parser.parse_args()

    archive = json.loads(args.archive.read_text())
    archive_dir = args.archive.parent
    dev_support_rows = [r for r in archive["trials"] if r["split"] == "dev"]

    ckpt_dir = args.train_dir / "checkpoints"
    j_params, j_cfg, active_gain = load_reader_checkpoint(
        ckpt_dir / f"active_probe-joint_transformer-seed{args.seed_ckpt}.msgpack"
    )
    fne_params, fne_cfg, _ = load_reader_checkpoint(
        ckpt_dir / f"active_probe-factored_no_equiv-seed{args.seed_ckpt}.msgpack"
    )
    ours_act_params, ours_act_cfg, _ = load_reader_checkpoint(
        ckpt_dir / f"active_probe-factored_equiv-seed{args.seed_ckpt}.msgpack"
    )
    ours_pas_params, ours_pas_cfg, _ = load_reader_checkpoint(
        ckpt_dir / f"passive_attempt1-factored_equiv-seed{args.seed_ckpt}.msgpack"
    )

    if args.panel == "smoke":
        trial_support_rows = [
            r for r in dev_support_rows if r["embodiment"] in ("fam_nom", "comp_180_lo_inv", "cont_45_nom")
        ][:2]
    else:
        trial_support_rows = dev_support_rows

    policy = load_frozen_pi05()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    prompts = {}

    def online_reader(probe_trace):
        return infer_one(ours_act_params, ours_act_cfg, probe_trace)

    with XEmbSimulator() as simulator:
        for idx, sup_row in enumerate(trial_support_rows):
            suite = sup_row["suite"]
            task_id = int(sup_row["task_id"])
            spec = get_embodiment_spec(sup_row["embodiment"])
            if int(sup_row["init_state_id"]) == args.init_state_id:
                raise ValueError("Support and query must use distinct init_state_id")
            suite_idx = sorted(json.loads(SPLITS_PATH.read_text())["development"].keys()).index(suite)
            query_seed = 20400000 + 10_000 * suite_idx + 100 * task_id + EVAL_EMBODIMENTS.index(spec)
            case = dict(
                suite=suite,
                task_id=task_id,
                variant="BASE",
                init_state_id=args.init_state_id,
                seed=query_seed,
            )

            act_batch = load_support_batch(archive_dir, sup_row, "active_probe")
            pas_batch = load_support_batch(archive_dir, sup_row, "passive_attempt1")
            clr_batch = dict(act_batch, valid=np.zeros_like(act_batch["valid"]))

            same_task_others = [
                r
                for r in dev_support_rows
                if r["suite"] == suite and int(r["task_id"]) == task_id and r["embodiment"] != spec.name
            ]
            donor_row = same_task_others[idx % len(same_task_others)] if same_task_others else sup_row
            don_batch = load_support_batch(archive_dir, donor_row, "active_probe")

            ya, sa, ga = analytic_proprio_sysid(
                act_batch["issued_mean"],
                act_batch["delta_eef"],
                act_batch["delta_jaw"],
                act_batch["valid"],
                nominal_gain_m_per_cmd=active_gain,
            )
            preds = {
                "nominal": (0.0, 1.0, 1),
                "privileged_oracle": (spec.yaw_deg, spec.xy_scale, spec.gripper_sign),
                "analytic_proprio_sysid": (float(ya[0]), float(sa[0]), int(ga[0])),
                "joint_transformer_active": infer_one(j_params, j_cfg, act_batch),
                "factored_no_equiv_active": infer_one(fne_params, fne_cfg, act_batch),
                "ours_retained_active": infer_one(ours_act_params, ours_act_cfg, act_batch),
                "ours_retained_passive": infer_one(ours_pas_params, ours_pas_cfg, pas_batch),
                "ours_cleared": infer_one(ours_act_params, ours_act_cfg, clr_batch),
                "ours_swapped_donor": infer_one(ours_act_params, ours_act_cfg, don_batch),
            }

            key = (suite, task_id)
            arms_out = {}
            for arm_name in ARMS:
                prompt = prompts.get(key)
                if arm_name == "ours_online_prefix":
                    rec = run_episode(
                        simulator,
                        policy,
                        case,
                        spec,
                        prompt=prompt,
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
                        yaw_hat=yh,
                        scale_hat=sh,
                        grip_hat=gh,
                    )
                prompts.setdefault(key, rec["reset"]["prompt"])
                if rec["reset"]["physics"] == sup_row["physics"]:
                    raise ValueError(f"Query reset collided with support physics on {case}")
                arms_out[arm_name] = rec

            first = arms_out[ARMS[0]]
            for arm_name in ARMS[1:]:
                if arms_out[arm_name]["reset"] != first["reset"]:
                    raise ValueError(f"Paired reset mismatch on {case} arm={arm_name}")

            row = dict(
                index=idx,
                suite=suite,
                task_id=task_id,
                init_state_id=args.init_state_id,
                seed=query_seed,
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
                        succ={a: arms_out[a]["success"] for a in ARMS},
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
            successes={a: sum(r["arms"][a]["success"] for r in grp) for a in ARMS},
            rates={a: sum(r["arms"][a]["success"] for r in grp) / len(grp) for a in ARMS},
        )

    contrasts = {}
    for main_arm in ("ours_retained_active", "ours_retained_passive", "ours_online_prefix"):
        contrasts[main_arm] = {
            comp: task_clustered_bootstrap(rows, main_arm, comp)
            for comp in ARMS
            if comp != main_arm
        }

    summary = dict(
        panel=args.panel,
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
