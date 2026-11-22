"""3-Seed Passive V2 + Zero-Shot (Unshifted & Shifted) Official LIBERO-Pro2.0 Test Eval.

Reuses saved support traces from the locked 96-query official test panel for any
held-out robot (UR5e, IIWA, Kinova3) and frozen VLA (pi0.5, pi0), and evaluates:
  1. zero_shot_unshifted (n=8 tasks, nominal embodiment, no shift)
  2. zero_shot_shifted   (n=96 queries, shifted embodiment, no adaptation)
  3. ours_equiv_v2_passive_s42 / s43 (pairs with existing seed 41 -> 3-seed n=288)
  4. perceiver_icba_passive_s42 / s43 (pairs with existing seed 41 -> 3-seed n=288)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT_OUT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT_OUT / "libero-xemb-ur5e-sysid-20261120-r1"
SCALE_DIR = ROOT_OUT / "libero-xemb-ur5e-scale-20261120-r1"
CAMPAIGN_DIR = ROOT_OUT / "xemb-icml-campaign-20261121-r1"
for d in (BASE_DIR, SCALE_DIR, CAMPAIGN_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps
from fluffy_bassoon.xemb.embodiment import EVAL_REGIMES, get_embodiment_spec
from fluffy_bassoon.xemb.equiv_v2 import predict_embodiment_v2
from collect_ur5e import CHUNK_SIZE, DEV_ARM
from eval_live_ur5e import task_clustered_bootstrap
from eval_stage2a_live_ur5e import (
    load_frozen_policy,
    load_v2_checkpoint,
    run_paired_query,
)
from eval_test_official import build_test_plan
from ur5e_motor import run_episode, write_json
from ur5e_sim_worker import UR5eSimulator

ARMS = (
    "zero_shot_shifted",
    "ours_equiv_v2_passive_s42",
    "ours_equiv_v2_passive_s43",
    "perceiver_icba_passive_s42",
    "perceiver_icba_passive_s43",
)


def _load_npz(path_str: str):
    with np.load(path_str) as z:
        d = {k: np.asarray(z[k])[None] for k in z.files}
    d["chunk_steps"] = np.where(d["valid"], CHUNK_SIZE, 0).astype(np.int32)
    d["dis_flow"] = compute_dis_flow_maps(d["img_before"], d["img_after"], d["valid"])
    return d


def _pred(params, cfg, batch):
    yh, sh, gh = predict_embodiment_v2(params, cfg, batch)
    return float(yh[0]), float(sh[0]), int(gh[0])


def run_shard(
    ckpt_dir: Path,
    reuse_support_dir: Path,
    output_dir: Path,
    shard_idx: int,
    num_shards: int,
    policy_name: str,
    target_arm: str,
    reset_create_attempts: int = 3,
    warmup_create: bool = True,
):
    tasks, plan = build_test_plan()
    if num_shards != len(tasks):
        raise ValueError(f"Expected num_shards == {len(tasks)}, got {num_shards}")
    suites = sorted({s for s, _ in tasks})
    my_suite, my_tid = tasks[shard_idx]
    my_entries = [e for e in plan if e["suite"] == my_suite and int(e["task_id"]) == my_tid]
    canon_quat = policy_name == "pi0"

    shard_dir = output_dir / f"shard-{shard_idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    p_v2_42, c_v2_42, _ = load_v2_checkpoint(ckpt_dir / "equiv_v2_dis_dz-passive_attempt1-seed42.msgpack")
    p_v2_43, c_v2_43, _ = load_v2_checkpoint(ckpt_dir / "equiv_v2_dis_dz-passive_attempt1-seed43.msgpack")
    p_pc_42, c_pc_42, _ = load_v2_checkpoint(ckpt_dir / "perceiver_icba_dis-passive_attempt1-seed42.msgpack")
    p_pc_43, c_pc_43, _ = load_v2_checkpoint(ckpt_dir / "perceiver_icba_dis-passive_attempt1-seed43.msgpack")

    support_meta_path = reuse_support_dir / f"shard-{shard_idx}" / "support-meta.json"
    support_rows = json.loads(support_meta_path.read_text())
    if [r["embodiment"] for r in support_rows] != [e["embodiment"] for e in my_entries]:
        raise ValueError(f"Reused support-meta mismatch on shard {shard_idx}")

    policy = load_frozen_policy(policy_name)
    prompts = {}
    rows = []
    query_resets = {}

    with UR5eSimulator() as simulator:
        # 1. Original zero-shot unshifted baseline (nominal embodiment, no shift)
        nom_spec = get_embodiment_spec("nom")
        first_init = my_entries[0]["query_init_state_id"]
        nom_seed = 41800000 + 10_000 * suites.index(my_suite) + 100 * my_tid + first_init
        nom_case = dict(
            arm=target_arm, suite=my_suite, task_id=my_tid,
            variant="BASE", init_state_id=first_init, seed=nom_seed,
        )
        if warmup_create:
            simulator.call("create", dict(nom_case, embodiment=nom_spec.name))
        nom_rec = run_episode(
            simulator, policy, nom_case, nom_spec,
            yaw_hat=0.0, scale_hat=1.0, grip_hat=1, canonicalize_quat=canon_quat,
        )
        unshifted_row = dict(
            suite=my_suite, task_id=my_tid, init_state_id=first_init,
            arm=target_arm, policy=policy_name,
            success=bool(nom_rec["success"]), controls=int(nom_rec["controls"]),
        )
        print(
            f"[shard {shard_idx}] ZERO_SHOT_UNSHIFTED {target_arm} ({policy_name}) "
            f"{my_suite}-{my_tid}: success={int(unshifted_row['success'])} controls={unshifted_row['controls']}",
            flush=True,
        )

        # 2. Shifted queries: zero_shot_shifted + seeds 42 & 43 of Passive V2 and Perceiver-ICBA
        for local_i, entry in enumerate(my_entries):
            sup_row = support_rows[local_i]
            suite, tid = entry["suite"], int(entry["task_id"])
            spec = get_embodiment_spec(entry["embodiment"])
            init_state_id = entry["query_init_state_id"]
            q_seed = 41800000 + 10_000 * suites.index(suite) + 100 * tid + init_state_id
            case = dict(
                arm=target_arm, suite=suite, task_id=tid,
                variant="BASE", init_state_id=init_state_id, seed=q_seed,
            )

            pas_b = _load_npz(sup_row["passive_npz"])
            arm_preds = {
                "zero_shot_shifted": (0.0, 1.0, 1),
                "ours_equiv_v2_passive_s42": _pred(p_v2_42, c_v2_42, pas_b),
                "ours_equiv_v2_passive_s43": _pred(p_v2_43, c_v2_43, pas_b),
                "perceiver_icba_passive_s42": _pred(p_pc_42, c_pc_42, pas_b),
                "perceiver_icba_passive_s43": _pred(p_pc_43, c_pc_43, pas_b),
            }

            key = (suite, tid)
            q_key = (suite, tid, init_state_id)
            row = dict(
                index=int(entry["index"]), suite=suite, task_id=tid,
                embodiment=spec.name, regime=spec.regime,
                true_yaw=spec.yaw_deg, true_scale=spec.xy_scale, true_grip=spec.gripper_sign,
                arms={},
            )
            for arm_name in ARMS:
                yh, sh, gh = arm_preds[arm_name]
                rec = run_paired_query(
                    run_episode, simulator, policy, case, spec,
                    arm_name=arm_name, resets=query_resets, key=q_key,
                    support_physics=sup_row["physics"],
                    expected_prompt=prompts.get(key),
                    create_attempts=reset_create_attempts,
                    yaw_hat=yh, scale_hat=sh, grip_hat=gh,
                    canonicalize_quat=canon_quat,
                )
                prompts.setdefault(key, rec["released_prompt"])
                row["arms"][arm_name] = dict(
                    success=bool(rec["success"]), controls=int(rec["controls"]),
                    yaw_hat=round(yh, 2), scale_hat=round(sh, 3), grip_hat=int(gh),
                )
            rows.append(row)
            write_json(shard_dir / "live-progress.json", dict(unshifted=unshifted_row, rows=rows))
            s_str = " ".join(
                f"{a.replace('ours_equiv_v2_passive_', 'v2_').replace('perceiver_icba_passive_', 'pc_').replace('zero_shot_shifted', 'zs')}={int(row['arms'][a]['success'])}"
                for a in ARMS
            )
            print(f"[shard {shard_idx}] {len(rows)}/{len(my_entries)} {suite}-{tid} {spec.name}: {s_str}", flush=True)

    write_json(shard_dir / "live-shard.json", dict(shard_idx=shard_idx, unshifted=unshifted_row, rows=rows))


def _regime_mask(rows, reg: str):
    if reg == "overall":
        return rows
    if reg == "ood_combined":
        return [r for r in rows if r["regime"] in ("compositional_ood", "continuous_angle_ood")]
    return [r for r in rows if r["regime"] == reg]


def merge_shards(output_dir: Path, num_shards: int, target_arm: str, policy_name: str, seed41_dir: Path | None):
    unshifted_rows = []
    all_rows = []
    for s in range(num_shards):
        p = output_dir / f"shard-{s}" / "live-shard.json"
        data = json.loads(p.read_text())
        unshifted_rows.append(data["unshifted"])
        all_rows.extend(data["rows"])
    all_rows.sort(key=lambda r: int(r["index"]))

    # Optional merge with existing seed-41 Stage 2A rows
    s41_by_idx = {}
    if seed41_dir is not None and seed41_dir.exists():
        for s in range(num_shards):
            p41 = seed41_dir / f"shard-{s}" / "live-shard.json"
            if p41.exists():
                for r in json.loads(p41.read_text())["rows"]:
                    s41_by_idx[int(r["index"])] = r
        if len(s41_by_idx) == len(all_rows):
            for r in all_rows:
                r41 = s41_by_idx[int(r["index"])]
                r["arms"]["ours_equiv_v2_passive_s41"] = r41["arms"]["ours_equiv_v2_passive"]
                r["arms"]["perceiver_icba_passive_s41"] = r41["arms"]["perceiver_icba_passive"]
                for k in ("privileged_oracle", "analytic_visual_hs_passive", "reground_babbling"):
                    if k in r41["arms"]:
                        r["arms"][k] = r41["arms"][k]

    present_arms = sorted(all_rows[0]["arms"].keys()) if all_rows else []
    by_regime = {}
    for reg in (*EVAL_REGIMES, "ood_combined", "overall"):
        sub = _regime_mask(all_rows, reg)
        n = len(sub)
        sc = {a: sum(int(r["arms"][a]["success"]) for r in sub) for a in present_arms}
        by_regime[reg] = dict(trials=n, successes=sc, rates={a: round(sc[a] / max(n, 1), 4) for a in present_arms})

    # 3-seed mean +/- std summary
    multiseed = {}
    for base in ("ours_equiv_v2_passive", "perceiver_icba_passive"):
        seed_keys = [f"{base}_s{sd}" for sd in (41, 42, 43) if f"{base}_s{sd}" in present_arms]
        if seed_keys:
            multiseed[base] = {}
            for reg in (*EVAL_REGIMES, "ood_combined", "overall"):
                rates = [by_regime[reg]["rates"][k] for k in seed_keys]
                multiseed[base][reg] = dict(
                    seeds=seed_keys,
                    per_seed=rates,
                    mean=round(float(np.mean(rates)), 4),
                    std=round(float(np.std(rates, ddof=0)), 4),
                    total_trials=by_regime[reg]["trials"] * len(seed_keys),
                )

    unshifted_succ = sum(int(r["success"]) for r in unshifted_rows)
    summary = dict(
        split="official_test",
        arm=target_arm,
        policy=policy_name,
        trials_per_seed=len(all_rows),
        zero_shot_unshifted=dict(
            trials=len(unshifted_rows),
            successes=unshifted_succ,
            rate=round(unshifted_succ / max(len(unshifted_rows), 1), 4),
            rows=unshifted_rows,
        ),
        by_regime=by_regime,
        multiseed=multiseed,
    )
    write_json(output_dir / f"multiseed-zeroshot-{target_arm.lower()}-{policy_name}.json", summary)
    print(f"=== 3-SEED + ZERO-SHOT SUMMARY ({target_arm}, {policy_name}) ===", flush=True)
    print(json.dumps(dict(
        zero_shot_unshifted=dict(successes=unshifted_succ, trials=len(unshifted_rows), rate=summary["zero_shot_unshifted"]["rate"]),
        zero_shot_shifted_overall=by_regime["overall"]["rates"].get("zero_shot_shifted"),
        multiseed_overall={k: v["overall"] for k, v in multiseed.items()},
        multiseed_ood={k: v["ood_combined"] for k, v in multiseed.items()},
    ), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run-shard")
    p_run.add_argument("--ckpt-dir", type=Path, required=True)
    p_run.add_argument("--reuse-support-dir", type=Path, required=True)
    p_run.add_argument("--output-dir", type=Path, required=True)
    p_run.add_argument("--shard-idx", type=int, required=True)
    p_run.add_argument("--num-shards", type=int, default=8)
    p_run.add_argument("--policy", type=str, default="pi05", choices=("pi05", "pi0"))
    p_run.add_argument("--arm", type=str, default=DEV_ARM, choices=("UR5e", "IIWA", "Kinova3", "Panda"))

    p_m = sub.add_parser("merge")
    p_m.add_argument("--output-dir", type=Path, required=True)
    p_m.add_argument("--num-shards", type=int, default=8)
    p_m.add_argument("--policy", type=str, default="pi05", choices=("pi05", "pi0"))
    p_m.add_argument("--arm", type=str, default=DEV_ARM, choices=("UR5e", "IIWA", "Kinova3", "Panda"))
    p_m.add_argument("--seed41-dir", type=Path, default=None)

    args = parser.parse_args()
    if args.cmd == "run-shard":
        run_shard(
            args.ckpt_dir, args.reuse_support_dir, args.output_dir,
            args.shard_idx, args.num_shards, args.policy, args.arm,
        )
    else:
        merge_shards(args.output_dir, args.num_shards, args.arm, args.policy, args.seed41_dir)


if __name__ == "__main__":
    main()
