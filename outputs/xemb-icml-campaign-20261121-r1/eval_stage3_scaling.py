"""Stage 3: ICML Embodiment-Diversity Scaling Law (M in {2,4,7,16,32,64,128} x Model Capacity) & Targeted Ablations.

Collects real MuJoCo Panda+Sawyer train-split passive Attempt-1 trajectories across 128 continuous
source embodiments, sweeps embodiment diversity M and model capacity (3 seeds each), and evaluates
every trained reader on the locked 96/96 Official Held-Out 6-DoF UR5e Test Set (`gpu09test` tapes).
Also runs the Passive Support Horizon ablation (K in {1,2,3,4,6,8,12,16,20,24} chunks = 5..120 controls).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"
SCALE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-scale-20261120-r1"
HERE_DIR = Path(__file__).resolve().parent
for d in (BASE_DIR, SCALE_DIR, HERE_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from fluffy_bassoon.xemb.baselines import (
    compute_dis_flow_maps,
    dense_flow_image_motion,
    reground_babbling_sysid,
)
from fluffy_bassoon.xemb.embodiment import (
    EmbodimentSpec,
    TRAIN_EMBODIMENTS,
    circular_yaw_error_deg,
    get_embodiment_spec,
)
from fluffy_bassoon.xemb.equiv_v2 import (
    ReaderV2Config,
    init_reader_v2,
    predict_embodiment_v2,
    reader_loss_v2,
)
from fluffy_bassoon.xemb.readers import (
    analytic_proprio_sysid,
    analytic_visual_yaw,
    image_motion,
)
from collect_ur5e import CHUNK_SIZE, SUPPORT_BUDGET, issued_exposure
from eval_pro2_visual_stress import load_split_tasks
from eval_stage2a_live_ur5e import fit_source_calibrators, load_frozen_policy, load_v2_checkpoint
from eval_test_official import build_test_plan
from ur5e_motor import run_episode, write_json
from ur5e_sim_worker import UR5eSimulator


M_LEVELS = (2, 4, 7, 16, 32, 64, 128)
SCALING_SEEDS = (41, 42, 43)
HORIZON_KS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 24)


def build_128_embodiments() -> list[EmbodimentSpec]:
    """Deterministic pool of 128 source embodiments (first 7 = canonical TRAIN_EMBODIMENTS)."""
    specs = list(TRAIN_EMBODIMENTS)
    rng = np.random.default_rng(20261122)
    for i in range(7, 128):
        yaw = float(rng.uniform(-180.0, 180.0))
        scale = float(rng.uniform(0.60, 1.40))
        grip = int(rng.choice([-1, 1]))
        specs.append(EmbodimentSpec(f"cont_src_{i:03d}", round(yaw, 2), round(scale, 3), grip, "train_diverse"))
    return specs


def build_collection_plan() -> list[dict]:
    """1,488 real MuJoCo Panda+Sawyer train-split trials spanning 128 continuous embodiments."""
    train_tasks = load_split_tasks("train")
    embs = build_128_embodiments()
    plan = []
    idx = 0
    for e_i, spec in enumerate(embs):
        # First 7 get 36 trials each (252), 7..31 get 14 each (350), 32..127 get 9 each (864) -> 1,466 trials
        reps = 36 if e_i < 7 else (14 if e_i < 32 else 9)
        for r in range(reps):
            arm = "Panda" if (e_i + r) % 2 == 0 else "Sawyer"
            suite, tid = train_tasks[(e_i * 7 + r) % len(train_tasks)]
            init_state_id = (e_i + r) % 10
            plan.append(dict(
                index=idx,
                emb_idx=e_i,
                emb_name=spec.name,
                yaw_deg=spec.yaw_deg,
                xy_scale=spec.xy_scale,
                gripper_sign=spec.gripper_sign,
                arm=arm,
                suite=suite,
                task_id=tid,
                init_state_id=init_state_id,
            ))
            idx += 1
    return plan


def collect_diverse_shard(out_dir: Path, shard_idx: int, num_shards: int):
    plan = build_collection_plan()
    my_plan = [p for p in plan if p["index"] % num_shards == shard_idx]
    suites = sorted({p["suite"] for p in plan})
    shard_dir = out_dir / "diverse_shards" / f"shard-{shard_idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    policy = load_frozen_policy("pi05")
    traces = []
    meta = []
    t0 = time.monotonic()

    with UR5eSimulator() as simulator:
        for i, item in enumerate(my_plan):
            spec = EmbodimentSpec(item["emb_name"], item["yaw_deg"], item["xy_scale"], item["gripper_sign"], "train_diverse")
            seed = 91000000 + 100_000 * (0 if item["arm"] == "Panda" else 1) + 10_000 * suites.index(item["suite"]) + 100 * item["task_id"] + item["init_state_id"]
            case = dict(arm=item["arm"], suite=item["suite"], task_id=item["task_id"], variant="BASE", init_state_id=item["init_state_id"], seed=seed)
            # Reuse simulator create with nominal wire and apply spec directly via run_episode on custom EmbodimentSpec!
            # Wait: ur5e_sim_worker looks up get_embodiment_spec(spec.name) — so for continuous random embodiments,
            # we collect directly using run_episode_custom_spec below!
            tr = _collect_passive_custom(simulator, policy, case, spec)
            tr["emb_idx"] = np.int32(item["emb_idx"])
            traces.append(tr)
            meta.append(item)
            if (i + 1) % 25 == 0 or i + 1 == len(my_plan):
                print(f"[stage3 collect s{shard_idx}] {i+1}/{len(my_plan)} ({time.monotonic()-t0:.1f}s)", flush=True)

    keys = ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid", "yaw_deg", "xy_scale", "gripper_sign", "emb_idx")
    stacked = {k: np.stack([t[k] for t in traces], axis=0) for k in keys}
    stacked["dis_flow"] = compute_dis_flow_maps(stacked["img_before"], stacked["img_after"], stacked["valid"])
    np.savez_compressed(shard_dir / "diverse.npz", **stacked)
    write_json(shard_dir / "meta.json", meta)
    print(f"[stage3 collect s{shard_idx}] DONE {len(meta)} trials in {time.monotonic()-t0:.1f}s", flush=True)


def _collect_passive_custom(simulator, policy, case: dict, spec: EmbodimentSpec) -> dict:
    """Collect a 120-control (24-chunk) passive Attempt-1 trace for arbitrary continuous EmbodimentSpec."""
    from fluffy_bassoon.xemb.embodiment import apply_embodiment_action, jaw_width_from_qpos, rot2, to_local_proprio
    from ur5e_motor import NOISE_SEED_OFFSET, to_openpi_observation

    sim_spec, initial = simulator.call("create", dict(case, embodiment="nom"))
    raw = initial["observation"]
    anchor = np.asarray(initial["anchor_eef_pos"], dtype=np.float32)
    prompt = initial["language_instruction"]
    horizon = min(int(sim_spec["max_steps"]), SUPPORT_BUDGET)
    low = np.asarray(initial["action_schema"]["low"], dtype=np.float32)
    high = np.asarray(initial["action_schema"]["high"], dtype=np.float32)

    rng = np.random.default_rng(int(case["seed"]) + NOISE_SEED_OFFSET)
    noise_bank = rng.standard_normal(((horizon + 4) // 5, 10, 32)).astype(np.float32)

    max_k = 24
    img_b = np.zeros((max_k, 64, 64, 3), dtype=np.uint8)
    img_a = np.zeros((max_k, 64, 64, 3), dtype=np.uint8)
    iss_m = np.zeros((max_k, 7), dtype=np.float32)
    deef_loc = np.zeros((max_k, 3), dtype=np.float32)
    djaw = np.zeros((max_k, 2), dtype=np.float32)
    valid = np.zeros((max_k,), dtype=bool)

    r_inv = rot2(-spec.yaw_deg)
    cur_img = np.asarray(initial["agentview_64"], dtype=np.uint8)
    cur_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32).copy()
    cur_jaw = float(jaw_width_from_qpos(raw["robot0_gripper_qpos"]))

    controls = 0
    chunk_idx = 0
    done = False
    while not done and controls < horizon and chunk_idx < max_k:
        local_raw = raw if spec.yaw_deg == 0 else to_local_proprio(raw, anchor, spec)
        item = to_openpi_observation(local_raw, prompt)
        proposal = np.asarray(policy.infer(item, noise=noise_bank[controls // 5])["actions"], dtype=np.float32)
        count = min(5, horizon - controls)
        issued = np.clip(proposal[:count], low, high).astype(np.float32)
        phys = apply_embodiment_action(issued, spec, low, high)
        step = simulator.call("chunk", phys)
        n = int(step["steps"])
        controls += n
        nxt_img = np.asarray(step["agentview_64"], dtype=np.uint8)
        nxt_pos = np.asarray(step["positions"][-1], dtype=np.float32)
        nxt_jaw = float(jaw_width_from_qpos(step["gripper_positions"][-1]))
        img_b[chunk_idx] = cur_img
        img_a[chunk_idx] = nxt_img
        iss_m[chunk_idx] = issued[:n].mean(axis=0)
        dp_world = nxt_pos - cur_pos
        dp_loc = dp_world.copy()
        dp_loc[:2] = r_inv @ dp_world[:2]
        deef_loc[chunk_idx] = dp_loc
        djaw[chunk_idx] = np.array([nxt_jaw - cur_jaw, nxt_jaw], dtype=np.float32)
        valid[chunk_idx] = True
        cur_img, cur_pos, cur_jaw = nxt_img, nxt_pos, nxt_jaw
        raw = step["observation"]
        done = bool(step["terminated"] or step["truncated"])
        chunk_idx += 1

    return dict(
        img_before=img_b,
        img_after=img_a,
        issued_mean=iss_m,
        delta_eef=deef_loc,
        delta_jaw=djaw,
        valid=valid,
        yaw_deg=np.float32(spec.yaw_deg),
        xy_scale=np.float32(spec.xy_scale),
        gripper_sign=np.int32(spec.gripper_sign),
        emb_idx=np.int32(0),  # overwritten by caller via index lookup? wait: set directly below!
    )


def load_ur5e_official_test_batch(reuse_support_dir: Path) -> tuple[dict, list[dict]]:
    """Load the locked 96/96 Official Held-Out 6-DoF UR5e passive support tapes from gpu09test."""
    tasks, plan = build_test_plan()
    rows_meta = []
    batches = []
    for s_idx, (suite, tid) in enumerate(tasks):
        my_entries = [e for e in plan if e["suite"] == suite and int(e["task_id"]) == tid]
        sup_meta = json.loads((reuse_support_dir / f"shard-{s_idx}" / "support-meta.json").read_text())
        for entry, sup in zip(my_entries, sup_meta, strict=True):
            spec = get_embodiment_spec(entry["embodiment"])
            with np.load(sup["passive_npz"]) as z:
                d = {k: np.asarray(z[k]) for k in z.files}
            d["yaw_deg"] = np.float32(spec.yaw_deg)
            d["xy_scale"] = np.float32(spec.xy_scale)
            d["gripper_sign"] = np.int32(spec.gripper_sign)
            batches.append(d)
            rows_meta.append(dict(index=entry["index"], suite=suite, task_id=tid, embodiment=spec.name, regime=spec.regime,
                                  yaw_deg=spec.yaw_deg, xy_scale=spec.xy_scale, gripper_sign=spec.gripper_sign))
    keys = ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid", "yaw_deg", "xy_scale", "gripper_sign")
    test_b = {k: np.stack([b[k] for b in batches], axis=0) for k in keys}
    test_b["dis_flow"] = compute_dis_flow_maps(test_b["img_before"], test_b["img_after"], test_b["valid"])
    return test_b, rows_meta


def score_predictions(yh, sh, gh, meta: list[dict]) -> dict:
    out = {}
    for reg in ("familiar", "compositional_ood", "continuous_angle_ood", "ood_combined", "overall"):
        if reg == "overall":
            idxs = list(range(len(meta)))
        elif reg == "ood_combined":
            idxs = [i for i, m in enumerate(meta) if m["regime"] in ("compositional_ood", "continuous_angle_ood")]
        else:
            idxs = [i for i, m in enumerate(meta) if m["regime"] == reg]
        yerrs = [circular_yaw_error_deg(float(yh[i]), meta[i]["yaw_deg"]) for i in idxs]
        s_ok = [abs(math.log(max(float(sh[i]), 1e-4)) - math.log(meta[i]["xy_scale"])) <= 0.25 for i in idxs]
        g_ok = [int(gh[i]) == meta[i]["gripper_sign"] for i in idxs]
        all_ok = [(yerrs[j] <= 20.0 and s_ok[j] and g_ok[j]) for j in range(len(idxs))]
        out[reg] = dict(
            n=len(idxs),
            yaw_mae_deg=round(float(np.mean(yerrs)), 2),
            all_correct_pct=round(100.0 * float(np.mean(all_ok)), 2),
        )
    return out


SCALING_MODELS = (
    ("equiv_v2_dis_dz", ReaderV2Config(model_type="factored_equiv_v2", visual_mode="dis_lumaflow", max_chunks=24, cnn_width=96, mlp_width=128, use_dz_parallax=True)),
    ("equiv_flow_homog", ReaderV2Config(model_type="equiv_flow_homography", visual_mode="dis_lumaflow", max_chunks=24, cnn_width=96, use_dz_parallax=False)),
    ("factored_no_equiv_v2", ReaderV2Config(model_type="factored_no_equiv_v2", visual_mode="dis_lumaflow", max_chunks=24, cnn_width=96, mlp_width=128)),
    ("perceiver_icba_w192", ReaderV2Config(model_type="perceiver_icba", visual_mode="dis_lumaflow", max_chunks=24, cnn_width=96, perceiver_latents=8, perceiver_width=192, perceiver_layers=3, perceiver_heads=4)),
    ("perceiver_icba_w384", ReaderV2Config(model_type="perceiver_icba", visual_mode="dis_lumaflow", max_chunks=24, cnn_width=128, perceiver_latents=16, perceiver_width=384, perceiver_layers=5, perceiver_heads=8)),
)


def run_scaling_worker(out_dir: Path, reuse_support_dir: Path, worker_idx: int, num_workers: int, steps: int = 2500):
    import jax
    import optax

    parts = []
    emb_ids = []
    for s in range(8):
        p = out_dir / "diverse_shards" / f"shard-{s}"
        with np.load(p / "diverse.npz") as z:
            parts.append({k: np.asarray(z[k]) for k in z.files})
        meta_s = json.loads((p / "meta.json").read_text())
        emb_ids.extend([int(m["emb_idx"]) for m in meta_s])
    train_all = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    emb_ids_arr = np.asarray(emb_ids, dtype=np.int32)
    test_b, test_meta = load_ur5e_official_test_batch(reuse_support_dir)

    jobs = [(m_name, cfg, M, seed) for m_name, cfg in SCALING_MODELS for M in M_LEVELS for seed in SCALING_SEEDS]
    my_jobs = [j for i, j in enumerate(jobs) if i % num_workers == worker_idx]

    results = []
    for m_name, cfg, M, seed in my_jobs:
        t0 = time.monotonic()
        pool_idx = np.where(emb_ids_arr < M)[0]
        params = init_reader_v2(jax.random.key(seed), cfg)
        sched = optax.cosine_decay_schedule(1.5e-3, steps, alpha=0.05)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
        opt_state = opt.init(params)

        @jax.jit
        def step_fn(p, st, b):
            (loss, _), grads = jax.value_and_grad(lambda pp: reader_loss_v2(pp, cfg, b), has_aux=True)(p)
            updates, st = opt.update(grads, st, p)
            return optax.apply_updates(p, updates), st, loss

        rng = np.random.default_rng(seed * 10007 + M * 97)
        for _ in range(steps):
            sel = rng.choice(pool_idx, size=32, replace=True)
            batch = {k: train_all[k][sel] for k in ("img_before", "img_after", "dis_flow", "issued_mean", "delta_eef", "delta_jaw", "valid", "yaw_deg", "xy_scale", "gripper_sign")}
            params, opt_state, _ = step_fn(params, opt_state, batch)

        yh, sh, gh = predict_embodiment_v2(params, cfg, test_b)
        metrics = score_predictions(yh, sh, gh, test_meta)
        rec = dict(model=m_name, M=M, seed=seed, pool_size=int(len(pool_idx)), metrics=metrics, seconds=round(time.monotonic() - t0, 1))
        results.append(rec)
        print(f"[scale w{worker_idx}] {m_name} M={M:3d} s={seed}: ood_yaw={metrics['ood_combined']['yaw_mae_deg']:5.2f}d ood_ok={metrics['ood_combined']['all_correct_pct']:5.1f}% ({rec['seconds']}s)", flush=True)

    write_json(out_dir / "scaling_workers" / f"worker-{worker_idx}.json", results)


def run_horizon_ablation(out_dir: Path, train_archive: Path, ckpt_dir: Path, homog_ckpt_dir: Path, reuse_support_dir: Path):
    """Support Horizon ablation: K in {1,2,3,4,6,8,12,16,20,24} chunks (5..120 passive Attempt-1 controls)."""
    test_b, test_meta = load_ur5e_official_test_batch(reuse_support_dir)
    cals = fit_source_calibrators(train_archive)

    horizon_out = {}
    for K in HORIZON_KS:
        b_k = dict(test_b)
        v_mask = test_b["valid"].copy()
        v_mask[:, K:] = False
        b_k["valid"] = v_mask
        b_k["chunk_steps"] = np.where(v_mask, CHUNK_SIZE, 0).astype(np.int32)
        iss = issued_exposure(b_k)

        entry = {}
        # Classical baselines at horizon K
        _, sp, gp = analytic_proprio_sysid(iss, b_k["delta_eef"], b_k["delta_jaw"], v_mask, nominal_gain_m_per_cmd=cals["passive_attempt1"]["nom_gain"])
        hs_mot = image_motion(b_k["img_before"], b_k["img_after"], v_mask)
        y_hs = analytic_visual_yaw(iss, hs_mot, b_k["delta_eef"][..., 2], v_mask, cals["passive_attempt1"]["hs_proj"])
        dis_mot = dense_flow_image_motion(b_k["img_before"], b_k["img_after"], v_mask, method="dis")
        y_dis = analytic_visual_yaw(iss, dis_mot, b_k["delta_eef"][..., 2], v_mask, cals["passive_attempt1"]["dis_proj"])
        yr, sr, gr = reground_babbling_sysid(iss, b_k["delta_eef"], b_k["delta_jaw"], v_mask, nominal_gain_m_per_cmd=cals["passive_attempt1"]["nom_gain"])

        entry["analytic_visual_hs"] = score_predictions(y_hs, sp, gp, test_meta)
        entry["dense_flow_dis"] = score_predictions(y_dis, sp, gp, test_meta)
        entry["reground_babbling"] = score_predictions(yr, sr, gr, test_meta)

        # Learned 3-seed checkpoints at horizon K
        for label, cdir, prefix in (
            ("equiv_v2_dis_dz", ckpt_dir, "equiv_v2_dis_dz-passive_attempt1"),
            ("equiv_v2_dis_nodz", ckpt_dir, "equiv_v2_dis_nodz-passive_attempt1"),
            ("equiv_flow_homog", homog_ckpt_dir, "equiv_flow_homog_nodz-passive_attempt1"),
            ("perceiver_icba", ckpt_dir, "perceiver_icba_dis-passive_attempt1"),
        ):
            seed_scores = []
            for seed in SCALING_SEEDS:
                p, c, _ = load_v2_checkpoint(cdir / f"{prefix}-seed{seed}.msgpack")
                yh, sh, gh = predict_embodiment_v2(p, c, b_k)
                seed_scores.append(score_predictions(yh, sh, gh, test_meta))
            agg = {}
            for reg in ("familiar", "compositional_ood", "continuous_angle_ood", "ood_combined", "overall"):
                agg[reg] = dict(
                    yaw_mae_deg=round(float(np.mean([s[reg]["yaw_mae_deg"] for s in seed_scores])), 2),
                    yaw_mae_std=round(float(np.std([s[reg]["yaw_mae_deg"] for s in seed_scores])), 2),
                    all_correct_pct=round(float(np.mean([s[reg]["all_correct_pct"] for s in seed_scores])), 2),
                )
            entry[label] = agg
        horizon_out[str(K * 5)] = entry
        print(
            f"[horizon {K*5:3d} ctrl] equiv_v2={entry['equiv_v2_dis_dz']['ood_combined']['yaw_mae_deg']:5.2f}d "
            f"homog={entry['equiv_flow_homog']['ood_combined']['yaw_mae_deg']:5.2f}d "
            f"hs={entry['analytic_visual_hs']['ood_combined']['yaw_mae_deg']:5.2f}d "
            f"perceiver={entry['perceiver_icba']['ood_combined']['yaw_mae_deg']:5.2f}d",
            flush=True,
        )

    write_json(out_dir / "ablation-horizon-ur5e.json", horizon_out)


def merge_scaling(out_dir: Path, num_workers: int):
    all_rows = []
    for w in range(num_workers):
        all_rows.extend(json.loads((out_dir / "scaling_workers" / f"worker-{w}.json").read_text()))

    table = {}
    for m_name, _ in SCALING_MODELS:
        table[m_name] = {}
        for M in M_LEVELS:
            sub = [r for r in all_rows if r["model"] == m_name and r["M"] == M]
            if not sub:
                continue
            table[m_name][str(M)] = {
                reg: dict(
                    yaw_mae_mean=round(float(np.mean([r["metrics"][reg]["yaw_mae_deg"] for r in sub])), 2),
                    yaw_mae_std=round(float(np.std([r["metrics"][reg]["yaw_mae_deg"] for r in sub])), 2),
                    all_ok_mean=round(float(np.mean([r["metrics"][reg]["all_correct_pct"] for r in sub])), 2),
                    all_ok_std=round(float(np.std([r["metrics"][reg]["all_correct_pct"] for r in sub])), 2),
                )
                for reg in ("familiar", "compositional_ood", "continuous_angle_ood", "ood_combined", "overall")
            }

    write_json(out_dir / "stage3-scaling-summary.json", dict(m_levels=list(M_LEVELS), seeds=list(SCALING_SEEDS), table=table, raw=all_rows))
    print("=== STAGE 3 EMBODIMENT-DIVERSITY SCALING LAW (HELD-OUT UR5e OOD) ===", flush=True)
    for m_name in table:
        line = " ".join(f"M={M}:{table[m_name][str(M)]['ood_combined']['yaw_mae_mean']:.1f}d({table[m_name][str(M)]['ood_combined']['all_ok_mean']:.0f}%)" for M in M_LEVELS if str(M) in table[m_name])
        print(f"  {m_name:22s}: {line}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("collect-diverse"); p1.add_argument("--out-dir", type=Path, required=True); p1.add_argument("--shard-idx", type=int, required=True); p1.add_argument("--num-shards", type=int, default=8)
    p2 = sub.add_parser("run-scaling"); p2.add_argument("--out-dir", type=Path, required=True); p2.add_argument("--reuse-support-dir", type=Path, required=True); p2.add_argument("--worker-idx", type=int, required=True); p2.add_argument("--num-workers", type=int, default=8); p2.add_argument("--steps", type=int, default=2500)
    p3 = sub.add_parser("run-horizon"); p3.add_argument("--out-dir", type=Path, required=True); p3.add_argument("--train-archive", type=Path, required=True); p3.add_argument("--ckpt-dir", type=Path, required=True); p3.add_argument("--homog-ckpt-dir", type=Path, required=True); p3.add_argument("--reuse-support-dir", type=Path, required=True)
    p4 = sub.add_parser("merge"); p4.add_argument("--out-dir", type=Path, required=True); p4.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    if args.cmd == "collect-diverse":
        collect_diverse_shard(args.out_dir, args.shard_idx, args.num_shards)
    elif args.cmd == "run-scaling":
        run_scaling_worker(args.out_dir, args.reuse_support_dir, args.worker_idx, args.num_workers, steps=args.steps)
    elif args.cmd == "run-horizon":
        run_horizon_ablation(args.out_dir, args.train_archive, args.ckpt_dir, args.homog_ckpt_dir, args.reuse_support_dir)
    else:
        merge_scaling(args.out_dir, args.num_workers)


if __name__ == "__main__":
    main()
