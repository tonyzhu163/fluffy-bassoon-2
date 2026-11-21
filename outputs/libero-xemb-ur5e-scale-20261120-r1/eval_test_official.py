"""Phase B: One-shot confirmatory 8-GPU evaluation on the strictly untouched 8 official LIBERO-Pro2.0 test tasks (SPLITS.json['test']) on held-out 6-DoF UR5e."""

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

from fluffy_bassoon.xemb.embodiment import EVAL_EMBODIMENTS, EVAL_REGIMES, get_embodiment_spec
from fluffy_bassoon.xemb.readers import (
    ReaderConfig,
    analytic_proprio_sysid,
    analytic_visual_yaw,
    image_motion,
    init_reader,
)
from collect_ur5e import CHUNK_SIZE, DEV_ARM, SUPPORT_BUDGET, issued_exposure, run_active_probe_episode
from eval_live_ur5e import check_query_reset, infer_one, task_clustered_bootstrap
from phase_a_hybrid import load_tagged_checkpoint
from train_ur5e_readers import make_donor_permutation
from ur5e_motor import SPLITS_PATH, load_frozen_pi05, run_episode, write_json
from ur5e_sim_worker import UR5eSimulator


TEST_ARMS = (
    "nominal",
    "privileged_oracle",
    "analytic_proprio_sysid",
    "analytic_visual_sysid_active",
    "analytic_visual_sysid_passive",
    "joint_transformer_active",
    "factored_no_equiv_active",
    "ours_lumaflow_active",
    "ours_rgb_active",
    "ours_rgb_passive",
    "ours_cleared",
    "ours_swapped_donor",
)


def build_test_plan():
    splits = json.loads(SPLITS_PATH.read_text())
    test_map = splits["test"]
    suites = sorted(test_map.keys())
    tasks = []
    for suite in suites:
        for tid in test_map[suite]:
            tasks.append((suite, int(tid)))
    if len(tasks) != 8:
        raise ValueError(f"Expected 8 official test tasks, got {len(tasks)}")
    plan = []
    idx = 0
    for suite, tid in tasks:
        sup_seed = 31800000 + 10_000 * suites.index(suite) + 100 * tid + 11
        for spec in EVAL_EMBODIMENTS:
            plan.append(
                dict(
                    index=idx,
                    split="test",
                    arm=DEV_ARM,
                    suite=suite,
                    task_id=tid,
                    init_state_id=11,
                    seed=sup_seed,
                    support_init_state_id=11,
                    query_init_state_id=13,
                    embodiment=spec.name,
                    regime=spec.regime,
                )
            )
            idx += 1
    return tasks, plan


def support_to_batch(tapes: dict, mode: str):
    t = tapes[mode]
    return dict(
        img_before=np.asarray(t["img_before"])[None],
        img_after=np.asarray(t["img_after"])[None],
        issued_mean=np.asarray(t["issued_mean"])[None],
        delta_eef=np.asarray(t["delta_eef"])[None],
        delta_jaw=np.asarray(t["delta_jaw"])[None],
        valid=np.asarray(t["valid"], dtype=bool)[None],
    )


def run_shard(train_dir: Path, output_dir: Path, shard_idx: int, num_shards: int, seed_ckpt: int, reuse_support_dir: Path | None = None):
    audit = json.loads((train_dir / "offline-audit.json").read_text())
    if audit.get("complete") is not True:
        raise ValueError("Completed offline-audit.json is required")
    archive_sha = audit["archive_sha256"]
    steps = int(audit["steps"])

    tasks, plan = build_test_plan()
    my_tasks = {t for i, t in enumerate(tasks) if i % num_shards == shard_idx}
    my_entries = [e for e in plan if (e["suite"], e["task_id"]) in my_tasks]
    if len(my_entries) != 12:
        raise ValueError(f"Shard {shard_idx}: expected 12 test queries, got {len(my_entries)}")

    # Build deterministic within-task donor permutation across the 12 embodiments (guaranteed different embodiment).
    donors_local = make_donor_permutation(my_entries, np.random.default_rng(20261121 + shard_idx))

    active_fit = audit["modes"]["active_probe"]
    passive_fit = audit["modes"]["passive_attempt1"]
    active_gain = active_fit["nominal_gain"]
    passive_gain = passive_fit["nominal_gain"]
    vis_active = active_fit["analytic_visual_sysid"]
    vis_passive = passive_fit["analytic_visual_sysid"]

    ckpt_dir = train_dir / "checkpoints"
    readers = {
        "joint_transformer_active": load_tagged_checkpoint(ckpt_dir, "", "active_probe", "joint_transformer", seed_ckpt, archive_sha, steps),
        "factored_no_equiv_active": load_tagged_checkpoint(ckpt_dir, "", "active_probe", "factored_no_equiv", seed_ckpt, archive_sha, steps),
        "ours_lumaflow_active": load_tagged_checkpoint(ckpt_dir, "", "active_probe", "factored_equiv", seed_ckpt, archive_sha, steps),
        "ours_rgb_active": load_tagged_checkpoint(ckpt_dir, "rgb-", "active_probe", "factored_equiv", seed_ckpt, archive_sha, steps),
        "ours_rgb_passive": load_tagged_checkpoint(ckpt_dir, "rgb-", "passive_attempt1", "factored_equiv", seed_ckpt, archive_sha, steps),
    }

    suites = sorted(json.loads(SPLITS_PATH.read_text())["test"].keys())
    shard_dir = output_dir / f"shard-{shard_idx}"
    support_dir = shard_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)
    policy = load_frozen_pi05()

    # Step 1: Reuse already-collected official test support tapes if provided, otherwise collect matched 15-control tapes on UR5e.
    support_rows = []
    prompts = {}
    max_passive_chunks = SUPPORT_BUDGET // CHUNK_SIZE
    reused_meta = (reuse_support_dir / f"shard-{shard_idx}" / "support-meta.json") if reuse_support_dir else None
    with UR5eSimulator() as simulator:
        if reused_meta and reused_meta.exists():
            support_rows = json.loads(reused_meta.read_text())
            if len(support_rows) != len(my_entries):
                raise ValueError(f"Reused support-meta length mismatch on shard {shard_idx}")
        else:
            for local_i, entry in enumerate(my_entries):
                suite = entry["suite"]
                task_id = int(entry["task_id"])
                spec = get_embodiment_spec(entry["embodiment"])
                sup_seed = 31800000 + 10_000 * suites.index(suite) + 100 * task_id + entry["support_init_state_id"]
                sup_case = dict(
                    arm=DEV_ARM,
                    suite=suite,
                    task_id=task_id,
                    variant="BASE",
                    init_state_id=entry["support_init_state_id"],
                    seed=sup_seed,
                )
                key = (suite, task_id)
                prompt = prompts.get(key)
                act_meta, act_trace = run_active_probe_episode(simulator, sup_case, spec, prompt=prompt)
                prompts.setdefault(key, act_meta["prompt"])
                prompt = prompts[key]
                pas_rec, pas_trace = run_episode(
                    simulator,
                    policy,
                    sup_case,
                    spec,
                    prompt=prompt,
                    yaw_hat=0.0,
                    scale_hat=1.0,
                    grip_hat=1,
                    max_controls=SUPPORT_BUDGET,
                    record_chunks=True,
                    max_record_chunks=max_passive_chunks,
                )
                act_npz = support_dir / f"{local_i:02d}-{spec.name}-active.npz"
                pas_npz = support_dir / f"{local_i:02d}-{spec.name}-passive.npz"
                np.savez_compressed(act_npz, **act_trace)
                np.savez_compressed(pas_npz, **pas_trace)
                support_rows.append(
                    dict(
                        embodiment=spec.name,
                        physics=act_meta["physics"],
                        active_npz=str(act_npz),
                        passive_npz=str(pas_npz),
                    )
                )
            write_json(shard_dir / "support-meta.json", support_rows)

        # Step 2: Live closed-loop evaluation on query reset (init_state_id=13) across all 12 arms.
        progress_path = shard_dir / "live-progress.json"
        if progress_path.exists():
            rows = json.loads(progress_path.read_text())
            print(f"RESUMING shard {shard_idx} from {len(rows)}/{len(my_entries)} completed queries", flush=True)
        else:
            rows = []
        done_indices = {int(r["index"]) for r in rows}
        query_resets = {}
        opening_proposals = {}
        for local_i, entry in enumerate(my_entries):
            if int(entry["index"]) in done_indices:
                continue
            sup_row = support_rows[local_i]
            donor_row = support_rows[donors_local[local_i]]
            suite = entry["suite"]
            task_id = int(entry["task_id"])
            spec = get_embodiment_spec(entry["embodiment"])
            init_state_id = entry["query_init_state_id"]
            q_seed = 41800000 + 10_000 * suites.index(suite) + 100 * task_id + init_state_id
            case = dict(arm=DEV_ARM, suite=suite, task_id=task_id, variant="BASE", init_state_id=init_state_id, seed=q_seed)

            def _load_npz(path_str: str):
                with np.load(path_str) as z:
                    d = {k: np.asarray(z[k])[None] for k in z.files}
                d["chunk_steps"] = np.where(d["valid"], CHUNK_SIZE, 0).astype(np.int32)
                return d

            act_batch = _load_npz(sup_row["active_npz"])
            pas_batch = _load_npz(sup_row["passive_npz"])
            don_batch = _load_npz(donor_row["active_npz"])
            clr_batch = dict(act_batch, valid=np.zeros_like(act_batch["valid"]))

            act_iss = issued_exposure(act_batch)
            ya, sa, ga = analytic_proprio_sysid(
                act_iss, act_batch["delta_eef"], act_batch["delta_jaw"], act_batch["valid"], nominal_gain_m_per_cmd=active_gain
            )
            pas_iss = issued_exposure(pas_batch)
            _, sp, gp = analytic_proprio_sysid(
                pas_iss, pas_batch["delta_eef"], pas_batch["delta_jaw"], pas_batch["valid"], nominal_gain_m_per_cmd=passive_gain
            )
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
                "joint_transformer_active": infer_one(*readers["joint_transformer_active"], act_batch),
                "factored_no_equiv_active": infer_one(*readers["factored_no_equiv_active"], act_batch),
                "ours_lumaflow_active": infer_one(*readers["ours_lumaflow_active"], act_batch),
                "ours_rgb_active": infer_one(*readers["ours_rgb_active"], act_batch),
                "ours_rgb_passive": infer_one(*readers["ours_rgb_passive"], pas_batch),
                "ours_cleared": infer_one(*readers["ours_rgb_active"], clr_batch),
                "ours_swapped_donor": infer_one(*readers["ours_rgb_active"], don_batch),
            }

            key = (suite, task_id)
            check_reset = partial(check_query_reset, query_resets, key, sup_row["physics"])
            arms_out = {}
            for arm_name in TEST_ARMS:
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
                index=entry["index"],
                shard=shard_idx,
                split="test",
                arm=DEV_ARM,
                suite=suite,
                task_id=task_id,
                init_state_id=init_state_id,
                seed=q_seed,
                embodiment=spec.name,
                regime=spec.regime,
                true_latent=dict(yaw_deg=spec.yaw_deg, xy_scale=spec.xy_scale, gripper_sign=spec.gripper_sign),
                donor_embodiment=donor_row["embodiment"],
                arms=arms_out,
            )
            rows.append(row)
            write_json(shard_dir / "live-progress.json", rows)
            print(
                json.dumps(
                    dict(
                        shard=shard_idx,
                        done=local_i + 1,
                        total=len(my_entries),
                        task=f"{suite}-{task_id}",
                        emb=spec.name,
                        succ={a: int(arms_out[a]["success"]) for a in TEST_ARMS},
                    )
                ),
                flush=True,
            )

    write_json(shard_dir / "shard-test.json", dict(shard=shard_idx, archive_sha256=archive_sha, seed_ckpt=seed_ckpt, rows=rows))


def merge_shards(output_dir: Path, num_shards: int):
    rows = []
    archive_sha = None
    seed_ckpt = None
    for s in range(num_shards):
        part = json.loads((output_dir / f"shard-{s}" / "shard-test.json").read_text())
        archive_sha = archive_sha or part["archive_sha256"]
        seed_ckpt = seed_ckpt or part["seed_ckpt"]
        rows.extend(part["rows"])
    rows.sort(key=lambda r: r["index"])
    if len(rows) != 96:
        raise ValueError(f"Expected 96 merged official test rows, got {len(rows)}")

    by_regime = {}
    for regime in (*EVAL_REGIMES, "ood_combined", "shifted", "overall"):
        if regime == "overall":
            sub = rows
        elif regime == "shifted":
            sub = [r for r in rows if r["embodiment"] != "fam_nom"]
        elif regime == "ood_combined":
            sub = [r for r in rows if r["regime"] in ("compositional_ood", "continuous_angle_ood")]
        else:
            sub = [r for r in rows if r["regime"] == regime]
        succ = {a: sum(int(r["arms"][a]["success"]) for r in sub) for a in TEST_ARMS}
        by_regime[regime] = dict(trials=len(sub), successes=succ, rates={a: succ[a] / len(sub) for a in TEST_ARMS})

    by_suite = {}
    for suite in sorted({r["suite"] for r in rows}):
        sub = [r for r in rows if r["suite"] == suite]
        succ = {a: sum(int(r["arms"][a]["success"]) for r in sub) for a in TEST_ARMS}
        by_suite[suite] = dict(trials=len(sub), successes=succ, rates={a: succ[a] / len(sub) for a in TEST_ARMS})

    contrasts = {}
    for ours in ("ours_rgb_active", "ours_rgb_passive", "ours_lumaflow_active"):
        contrasts[ours] = {other: task_clustered_bootstrap(rows, ours, other) for other in TEST_ARMS if other != ours}

    summary = dict(
        split="official_test",
        arm=DEV_ARM,
        panel="official_test_8tasks_12emb_96queries",
        arms=list(TEST_ARMS),
        train_archive_sha256=archive_sha,
        seed_ckpt=seed_ckpt,
        trials=len(rows),
        by_regime=by_regime,
        by_suite=by_suite,
        paired_contrasts_overall=contrasts,
        rows=rows,
    )
    write_json(output_dir / "test-eval-summary.json", summary)
    print(json.dumps(dict(overall=by_regime["overall"], ood=by_regime["ood_combined"]), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_ev = sub.add_parser("run-shard")
    p_ev.add_argument("--train-dir", type=Path, required=True)
    p_ev.add_argument("--output-dir", type=Path, required=True)
    p_ev.add_argument("--shard-idx", type=int, required=True)
    p_ev.add_argument("--num-shards", type=int, default=8)
    p_ev.add_argument("--seed-ckpt", type=int, default=41)
    p_ev.add_argument("--reuse-support-dir", type=Path, default=None)

    p_mg = sub.add_parser("merge")
    p_mg.add_argument("--output-dir", type=Path, required=True)
    p_mg.add_argument("--num-shards", type=int, default=8)

    args = parser.parse_args()
    if args.cmd == "run-shard":
        run_shard(args.train_dir, args.output_dir, args.shard_idx, args.num_shards, args.seed_ckpt, args.reuse_support_dir)
    else:
        merge_shards(args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
