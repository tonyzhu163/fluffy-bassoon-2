"""Confirmatory 3-seed x 576-episode evaluation on the locked 8 official LIBERO-Pro2.0 test tasks (SPLITS.json['test']) on held-out 6-DoF UR5e.

16 shards = 8 official test tasks x 2 query init states (13, 14).
Per (task, init_state, shift):
- 4 seed-independent arms (192 episodes each across 16 shards):
  nominal, privileged_oracle, analytic_visual_sysid_passive, farneback_flow_passive
- 5 learned arms x 3 trained seeds (41, 42, 43) = 15 arms (576 independent episodes per method):
  ours_lumaflow_active, ours_rgb_passive, joint_transformer_active, rma_gru_active, factored_no_equiv_active
Total: 16 x 12 x 19 = 3,648 live closed-loop episodes.
"""

import argparse
from functools import partial
import json
from pathlib import Path
import sys

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fluffy_bassoon.xemb.embodiment import EVAL_EMBODIMENTS, EVAL_REGIMES, get_embodiment_spec
from fluffy_bassoon.xemb.readers import (
    analytic_proprio_sysid,
    analytic_visual_yaw,
    farneback_image_motion,
    image_motion,
)
from collect_ur5e import CHUNK_SIZE, DEV_ARM, issued_exposure
from eval_live_ur5e import check_query_reset, infer_one, task_clustered_bootstrap
from phase_a_hybrid import load_tagged_checkpoint
from ur5e_motor import SPLITS_PATH, load_frozen_pi05, run_episode, write_json
from ur5e_sim_worker import UR5eSimulator

SEEDS = (41, 42, 43)
QUERY_INIT_STATES = (13, 14)

SEED_INDEP_ARMS = (
    "nominal",
    "privileged_oracle",
    "analytic_visual_sysid_passive",
    "farneback_flow_passive",
)
LEARNED_ARMS = (
    "ours_lumaflow_active",
    "ours_rgb_passive",
    "joint_transformer_active",
    "rma_gru_active",
    "factored_no_equiv_active",
)
ALL_METHODS = SEED_INDEP_ARMS + LEARNED_ARMS


def build_16shard_plan():
    splits = json.loads(SPLITS_PATH.read_text())
    test_map = splits["test"]
    suites = sorted(test_map.keys())
    tasks = []
    for suite in suites:
        for tid in test_map[suite]:
            tasks.append((suite, int(tid)))
    if len(tasks) != 8:
        raise ValueError(f"Expected 8 official test tasks, got {len(tasks)}")
    shards = []
    idx = 0
    for q_init in QUERY_INIT_STATES:
        for task_slot, (suite, tid) in enumerate(tasks):
            entries = []
            for spec in EVAL_EMBODIMENTS:
                entries.append(
                    dict(
                        index=idx,
                        task_slot=task_slot,
                        suite=suite,
                        task_id=tid,
                        query_init_state_id=q_init,
                        embodiment=spec.name,
                        regime=spec.regime,
                    )
                )
                idx += 1
            shards.append((task_slot, q_init, suite, tid, entries))
    return suites, shards


def _load_npz(path_str: str):
    with np.load(path_str) as z:
        d = {k: np.asarray(z[k])[None] for k in z.files}
    d["chunk_steps"] = np.where(d["valid"], CHUNK_SIZE, 0).astype(np.int32)
    return d


def run_shard(
    train_dir: Path,
    baselines_dir: Path,
    reuse_support_dir: Path,
    output_dir: Path,
    shard_idx: int,
):
    audit = json.loads((train_dir / "offline-audit.json").read_text())
    archive_sha = audit["archive_sha256"]
    steps = int(audit["steps"])
    passive_fit = audit["modes"]["passive_attempt1"]
    passive_gain = passive_fit["nominal_gain"]
    vis_passive = passive_fit["analytic_visual_sysid"]

    fb_calib = json.loads((baselines_dir / "farneback-calib.json").read_text())
    fb_proj = np.asarray(fb_calib["projection"], dtype=np.float32)

    suites, shards = build_16shard_plan()
    task_slot, q_init, suite, task_id, my_entries = shards[shard_idx]

    # Support traces for all 8 tasks live at reuse_support_dir/shard-{0..7}/support-meta.json (init_state=11, disjoint from 13 & 14).
    sup_meta_path = reuse_support_dir / f"shard-{task_slot}" / "support-meta.json"
    support_rows = json.loads(sup_meta_path.read_text())
    if len(support_rows) != len(my_entries):
        raise ValueError(f"Support length mismatch on shard {shard_idx}")

    ckpt_dir = train_dir / "checkpoints"
    base_ckpt_dir = baselines_dir / "checkpoints"
    readers = {}
    for s in SEEDS:
        readers[("ours_lumaflow_active", s)] = load_tagged_checkpoint(ckpt_dir, "", "active_probe", "factored_equiv", s, archive_sha, steps)
        readers[("ours_rgb_passive", s)] = load_tagged_checkpoint(ckpt_dir, "rgb-", "passive_attempt1", "factored_equiv", s, archive_sha, steps)
        readers[("joint_transformer_active", s)] = load_tagged_checkpoint(ckpt_dir, "", "active_probe", "joint_transformer", s, archive_sha, steps)
        readers[("factored_no_equiv_active", s)] = load_tagged_checkpoint(ckpt_dir, "", "active_probe", "factored_no_equiv", s, archive_sha, steps)
        readers[("rma_gru_active", s)] = load_tagged_checkpoint(base_ckpt_dir, "", "active_probe", "rma_gru", s, archive_sha, steps)

    shard_dir = output_dir / f"shard-{shard_idx:02d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    progress_path = shard_dir / "live-progress.json"
    rows = json.loads(progress_path.read_text()) if progress_path.exists() else []
    done_indices = {int(r["index"]) for r in rows}
    if rows:
        print(f"RESUMING shard {shard_idx:02d} from {len(rows)}/{len(my_entries)} queries", flush=True)

    policy = load_frozen_pi05()
    prompts = {}
    with UR5eSimulator() as simulator:
        for local_i, entry in enumerate(my_entries):
            if int(entry["index"]) in done_indices:
                continue
            sup_row = support_rows[local_i]
            spec = get_embodiment_spec(entry["embodiment"])
            q_seed = 41800000 + 10_000 * suites.index(suite) + 100 * task_id + q_init
            case = dict(arm=DEV_ARM, suite=suite, task_id=task_id, variant="BASE", init_state_id=q_init, seed=q_seed)

            act_batch = _load_npz(sup_row["active_npz"])
            pas_batch = _load_npz(sup_row["passive_npz"])

            pas_iss = issued_exposure(pas_batch)
            _, sp, gp = analytic_proprio_sysid(
                pas_iss, pas_batch["delta_eef"], pas_batch["delta_jaw"], pas_batch["valid"], nominal_gain_m_per_cmd=passive_gain
            )
            pas_mot = image_motion(pas_batch["img_before"], pas_batch["img_after"], pas_batch["valid"])
            yaw_vp = float(
                analytic_visual_yaw(pas_iss, pas_mot, pas_batch["delta_eef"][..., 2], pas_batch["valid"], np.asarray(vis_passive["projection"]))[0]
            )
            fb_mot = farneback_image_motion(pas_batch["img_before"], pas_batch["img_after"], pas_batch["valid"])
            yaw_fb = float(analytic_visual_yaw(pas_iss, fb_mot, pas_batch["delta_eef"][..., 2], pas_batch["valid"], fb_proj)[0])

            preds = {
                "nominal": (0.0, 1.0, 1),
                "privileged_oracle": (spec.yaw_deg, spec.xy_scale, spec.gripper_sign),
                "analytic_visual_sysid_passive": (yaw_vp, float(sp[0]), int(gp[0])),
                "farneback_flow_passive": (yaw_fb, float(sp[0]), int(gp[0])),
            }
            arm_keys = list(SEED_INDEP_ARMS)
            for m in LEARNED_ARMS:
                batch_in = pas_batch if m == "ours_rgb_passive" else act_batch
                for s in SEEDS:
                    ak = f"{m}__s{s}"
                    preds[ak] = infer_one(*readers[(m, s)], batch_in)
                    arm_keys.append(ak)

            key = (suite, task_id, q_init)
            query_resets = {}
            opening_proposals = {}
            check_reset = partial(check_query_reset, query_resets, key, sup_row["physics"])
            arms_out = {}
            for ak in arm_keys:
                prompt = prompts.get((suite, task_id))
                yh, sh, gh = preds[ak]
                last_err = None
                for attempt_try in range(2):
                    try:
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
                        last_err = None
                        break
                    except ValueError as exc:
                        last_err = exc
                        if attempt_try == 0:
                            print(f"WARN retry reset on shard {shard_idx:02d}/{ak}: {exc}", flush=True)
                if last_err is not None:
                    raise last_err
                rec.pop("native_rewards", None)
                prompts.setdefault((suite, task_id), rec["reset"]["prompt"])
                opening = rec["opening_proposal_sha256"]
                if not opening or opening_proposals.setdefault(key, opening) != opening:
                    raise ValueError(f"Unmatched opening proposal on {case}/{ak}")
                arms_out[ak] = dict(success=bool(rec["success"]), yaw_hat=float(yh), scale_hat=float(sh), grip_hat=int(gh))

            row = dict(
                index=entry["index"],
                shard=shard_idx,
                suite=suite,
                task_id=task_id,
                init_state_id=q_init,
                embodiment=spec.name,
                regime=spec.regime,
                arms=arms_out,
            )
            rows.append(row)
            write_json(progress_path, rows)
            print(
                json.dumps(
                    dict(
                        shard=shard_idx,
                        done=len(rows),
                        total=len(my_entries),
                        task=f"{suite}-{task_id}@q{q_init}",
                        emb=spec.name,
                        oracle=int(arms_out["privileged_oracle"]["success"]),
                        ours_act=sum(int(arms_out[f"ours_lumaflow_active__s{s}"]["success"]) for s in SEEDS),
                        ours_pas=sum(int(arms_out[f"ours_rgb_passive__s{s}"]["success"]) for s in SEEDS),
                        jt=sum(int(arms_out[f"joint_transformer_active__s{s}"]["success"]) for s in SEEDS),
                        gru=sum(int(arms_out[f"rma_gru_active__s{s}"]["success"]) for s in SEEDS),
                        fb=int(arms_out["farneback_flow_passive"]["success"]),
                    )
                ),
                flush=True,
            )

    write_json(shard_dir / "shard-test.json", dict(shard=shard_idx, rows=rows))


def merge_shards(output_dir: Path, num_shards: int = 16):
    all_rows = []
    for s in range(num_shards):
        p = output_dir / f"shard-{s:02d}" / "shard-test.json"
        if not p.exists():
            raise FileNotFoundError(f"Missing shard {s:02d}: {p}")
        all_rows.extend(json.loads(p.read_text())["rows"])
    all_rows.sort(key=lambda r: int(r["index"]))

    # Expand to 576 seed-aligned episode records (192 queries x 3 seeds) for unified bootstrap & per-seed stats.
    expanded = []
    for r in all_rows:
        for s in SEEDS:
            arms_s = {}
            for m in SEED_INDEP_ARMS:
                arms_s[m] = r["arms"][m]["success"]
            for m in LEARNED_ARMS:
                arms_s[m] = r["arms"][f"{m}__s{s}"]["success"]
            expanded.append(
                dict(
                    suite=r["suite"],
                    task_id=r["task_id"],
                    init_state_id=r["init_state_id"],
                    embodiment=r["embodiment"],
                    regime=r["regime"],
                    seed=s,
                    arms=arms_s,
                )
            )

    def summarize_subset(sub):
        n = len(sub)
        succs = {m: sum(int(x["arms"][m]) for x in sub) for m in ALL_METHODS}
        rates = {m: round(succs[m] / max(n, 1), 4) for m in ALL_METHODS}
        per_seed = {}
        for m in ALL_METHODS:
            sr = []
            for s in SEEDS:
                ss = [x for x in sub if x["seed"] == s]
                sr.append(sum(int(x["arms"][m]) for x in ss) / max(len(ss), 1))
            per_seed[m] = dict(mean=round(float(np.mean(sr)), 4), std=round(float(np.std(sr)), 4), seeds=[round(v, 4) for v in sr])
        return dict(episodes=n, successes=succs, rates=rates, per_seed=per_seed)

    by_regime = {}
    for reg in (*EVAL_REGIMES, "ood_combined", "overall"):
        if reg == "overall":
            sub = expanded
        elif reg == "ood_combined":
            sub = [x for x in expanded if x["regime"] in ("compositional_ood", "continuous_angle_ood")]
        else:
            sub = [x for x in expanded if x["regime"] == reg]
        by_regime[reg] = summarize_subset(sub)

    by_suite = {}
    for suite in sorted({x["suite"] for x in expanded}):
        by_suite[suite] = summarize_subset([x for x in expanded if x["suite"] == suite])

    # Task-clustered 95% bootstrap CIs over all 576 episodes (clustered by the 8 official test tasks).
    task_rows = [dict(suite=x["suite"], task_id=x["task_id"], arms={m: dict(success=x["arms"][m]) for m in ALL_METHODS}) for x in expanded]
    contrasts = {}
    for focus in ("ours_lumaflow_active", "ours_rgb_passive"):
        contrasts[focus] = {}
        for other in ALL_METHODS:
            if other == focus:
                continue
            contrasts[focus][other] = task_clustered_bootstrap(task_rows, focus, other, n_boot=4000, seed=20261121)

    summary = dict(
        split="official_test_3seed_576",
        arm=DEV_ARM,
        seeds=list(SEEDS),
        query_init_states=list(QUERY_INIT_STATES),
        queries=len(all_rows),
        episodes_per_method=len(expanded),
        by_regime=by_regime,
        by_suite=by_suite,
        paired_contrasts_overall=contrasts,
    )
    write_json(output_dir / "official-test-3seed-576-summary.json", summary)
    print(json.dumps(dict(overall=by_regime["overall"]["rates"], ood=by_regime["ood_combined"]["rates"]), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run-shard")
    p_run.add_argument("--train-dir", type=Path, required=True)
    p_run.add_argument("--baselines-dir", type=Path, required=True)
    p_run.add_argument("--reuse-support-dir", type=Path, required=True)
    p_run.add_argument("--output-dir", type=Path, required=True)
    p_run.add_argument("--shard-idx", type=int, required=True)

    p_m = sub.add_parser("merge")
    p_m.add_argument("--output-dir", type=Path, required=True)
    p_m.add_argument("--num-shards", type=int, default=16)

    args = parser.parse_args()
    if args.cmd == "run-shard":
        run_shard(args.train_dir, args.baselines_dir, args.reuse_support_dir, args.output_dir, args.shard_idx)
    else:
        merge_shards(args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
