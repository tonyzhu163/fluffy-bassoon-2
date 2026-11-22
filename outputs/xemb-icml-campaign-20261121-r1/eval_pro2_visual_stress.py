"""Stage 2C: Competitive LIBERO-Pro2.0 Visual & Camera Perturbation Stress-Test on Held-Out 6-DoF UR5e.

Trains Camera-Conditioned EquivFlow-Procrustes (`equiv_flow_cam_homog`), Equiv-V2 (`equiv_v2_pro2`),
and Perceiver-ICBA (`perceiver_icba_pro2`) strictly on source 7-DoF Panda + Sawyer train tasks across
BASE + S13/S14/S15/S16/S19, then evaluates ZERO-SHOT on held-out 6-DoF UR5e official test tasks across
all 5 released LIBERO-Pro2.0 visual/camera perturbations (S13 recolor, S14 dark lighting, S15 camera
extrinsics, S16 camera FOV, S19 sensor noise) in BOTH offline calibration and live frozen pi0.5 rollouts.
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
    fit_dense_flow_projection,
    reground_babbling_sysid,
)
from fluffy_bassoon.xemb.embodiment import (
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
    fit_nominal_gain,
    fit_visual_projection,
    image_motion,
)
from collect_ur5e import CHUNK_SIZE, SUPPORT_BUDGET, issued_exposure, run_active_probe_episode
from eval_live_ur5e import check_query_reset, task_clustered_bootstrap
from eval_stage2a_live_ur5e import fit_source_calibrators, load_frozen_policy, load_v2_checkpoint, run_paired_query
from ur5e_motor import SPLITS_PATH, run_episode, write_json
from ur5e_sim_worker import UR5eSimulator


TRAIN_VARIANTS = ("BASE", "S13", "S14", "S15", "S16", "S19")
TEST_VARIANTS = ("S13", "S14", "S15", "S16", "S19")
STRESS_EMBS = (
    "fam_nom",
    "fam_yaw90",
    "comp_yaw180_s07_inv",
    "comp_yaw270_s13_inv",
    "cont_45_s085",
    "cont_135_s115_inv",
)
STRESS_ARMS = (
    "privileged_oracle",
    "nominal",
    "ours_equiv_v2_passive_base",
    "ours_equiv_cam_homog_active",
    "ours_equiv_v2_pro2_active",
    "analytic_visual_hs_passive",
    "dense_flow_dis_active",
    "reground_babbling",
    "perceiver_icba_pro2_active",
)


def load_split_tasks(split_name: str) -> list[tuple[str, int]]:
    splits = json.loads(SPLITS_PATH.read_text())
    return [(str(s), int(t)) for s, t in splits[split_name]]


def collect_train_shard(out_dir: Path, shard_idx: int, num_shards: int):
    train_tasks = load_split_tasks("train")
    suites = sorted({s for s, _ in train_tasks})
    my_tasks = [t for i, t in enumerate(train_tasks) if i % num_shards == shard_idx]
    shard_dir = out_dir / "train_shards" / f"shard-{shard_idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    meta = []
    t0 = time.monotonic()
    with UR5eSimulator() as simulator:
        for arm in ("Panda", "Sawyer"):
            for suite, tid in my_tasks:
                for v_i, variant in enumerate(TRAIN_VARIANTS):
                    init_state_id = (tid + v_i) % 8
                    seed = 71000000 + 100_000 * ("Panda", "Sawyer").index(arm) + 10_000 * suites.index(suite) + 100 * tid + 10 * v_i + init_state_id
                    for spec in TRAIN_EMBODIMENTS:
                        case = dict(arm=arm, suite=suite, task_id=tid, variant=variant, init_state_id=init_state_id, seed=seed)
                        _, tr = run_active_probe_episode(simulator, case, spec)
                        tr["yaw_deg"] = np.float32(spec.yaw_deg)
                        tr["xy_scale"] = np.float32(spec.xy_scale)
                        tr["gripper_sign"] = np.int32(spec.gripper_sign)
                        traces.append(tr)
                        meta.append(dict(arm=arm, suite=suite, task_id=tid, variant=variant, emb=spec.name))

    keys = ("img_before", "img_after", "frame0_img", "issued_mean", "delta_eef", "delta_jaw", "valid", "yaw_deg", "xy_scale", "gripper_sign")
    stacked = {k: np.stack([t[k] for t in traces], axis=0) for k in keys}
    stacked["dis_flow"] = compute_dis_flow_maps(stacked["img_before"], stacked["img_after"], stacked["valid"])
    np.savez_compressed(shard_dir / "train.npz", **stacked)
    write_json(shard_dir / "meta.json", meta)
    print(f"[collect-train shard {shard_idx}] saved {len(meta)} multi-variant probes in {time.monotonic()-t0:.1f}s", flush=True)


def train_pro2_readers(out_dir: Path, num_shards: int, steps: int = 3000):
    import jax
    import jax.numpy as jnp
    import optax
    from flax import serialization

    parts = []
    all_meta = []
    for s in range(num_shards):
        p = out_dir / "train_shards" / f"shard-{s}"
        with np.load(p / "train.npz") as z:
            parts.append({k: np.asarray(z[k]) for k in z.files})
        all_meta.extend(json.loads((p / "meta.json").read_text()))
    data = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    n = len(all_meta)
    print(f"[train-readers] loaded {n} Panda+Sawyer multi-variant probes across {TRAIN_VARIANTS}", flush=True)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("equiv_flow_cam_homog", ReaderV2Config(model_type="equiv_flow_cam_homog", visual_mode="dis_lumaflow", max_chunks=3, cnn_width=96, use_dz_parallax=True)),
        ("equiv_v2_pro2", ReaderV2Config(model_type="factored_equiv_v2", visual_mode="dis_lumaflow", max_chunks=3, cnn_width=96, mlp_width=128, use_dz_parallax=True)),
        ("perceiver_icba_pro2", ReaderV2Config(model_type="perceiver_icba", visual_mode="dis_lumaflow", max_chunks=3, cnn_width=96, perceiver_latents=8, perceiver_width=192, perceiver_layers=3, perceiver_heads=4)),
    ]

    for name, cfg in specs:
        for seed in (41, 42, 43):
            t0 = time.monotonic()
            params = init_reader_v2(jax.random.key(seed), cfg)
            sched = optax.cosine_decay_schedule(1.5e-3, steps, alpha=0.05)
            opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
            opt_state = opt.init(params)

            @jax.jit
            def step_fn(p, st, b):
                (loss, aux), grads = jax.value_and_grad(lambda pp: reader_loss_v2(pp, cfg, b), has_aux=True)(p)
                updates, st = opt.update(grads, st, p)
                p = optax.apply_updates(p, updates)
                return p, st, loss

            rng = np.random.default_rng(seed * 1009)
            last_loss = 0.0
            for _ in range(steps):
                idx = rng.integers(n, size=64)
                batch = {k: data[k][idx] for k in data}
                params, opt_state, loss = step_fn(params, opt_state, batch)
                last_loss = float(loss)

            blob = dict(
                name=name,
                seed=seed,
                steps=steps,
                final_loss=round(last_loss, 5),
                reader_v2_config=dict(
                    model_type=cfg.model_type, visual_mode=cfg.visual_mode, max_chunks=cfg.max_chunks,
                    cnn_width=cfg.cnn_width, mlp_width=cfg.mlp_width, use_dz_parallax=cfg.use_dz_parallax,
                    perceiver_latents=cfg.perceiver_latents, perceiver_width=cfg.perceiver_width,
                    perceiver_layers=cfg.perceiver_layers, perceiver_heads=cfg.perceiver_heads,
                ),
                params=serialization.to_state_dict(jax.device_get(params)),
            )
            (ckpt_dir / f"{name}-seed{seed}.msgpack").write_bytes(serialization.msgpack_serialize(blob))
            print(f"[train-readers] {name} seed={seed} loss={last_loss:.4f} ({time.monotonic()-t0:.1f}s)", flush=True)


def eval_test_shard(
    out_dir: Path,
    train_archive: Path,
    base_ckpt_dir: Path,
    shard_idx: int,
    num_shards: int,
    seed_ckpt: int = 41,
):
    test_tasks = load_split_tasks("test")
    if num_shards != len(test_tasks):
        raise ValueError(f"Expected num_shards == {len(test_tasks)}, got {num_shards}")
    suites = sorted({s for s, _ in test_tasks})
    suite, tid = test_tasks[shard_idx]

    shard_dir = out_dir / "test_shards" / f"shard-{shard_idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    cals = fit_source_calibrators(train_archive)
    p_base_pas, c_base_pas, _ = load_v2_checkpoint(base_ckpt_dir / f"equiv_v2_dis_dz-passive_attempt1-seed{seed_ckpt}.msgpack")
    p_cam, c_cam, _ = load_v2_checkpoint(out_dir / "checkpoints" / f"equiv_flow_cam_homog-seed{seed_ckpt}.msgpack")
    p_v2p, c_v2p, _ = load_v2_checkpoint(out_dir / "checkpoints" / f"equiv_v2_pro2-seed{seed_ckpt}.msgpack")
    p_percp, c_percp, _ = load_v2_checkpoint(out_dir / "checkpoints" / f"perceiver_icba_pro2-seed{seed_ckpt}.msgpack")

    policy = load_frozen_policy("pi05")
    rows = []

    def _pred(params, cfg, batch):
        yh, sh, gh = predict_embodiment_v2(params, cfg, batch)
        return float(yh[0]), float(sh[0]), int(gh[0])

    with UR5eSimulator() as simulator:
        for v_i, variant in enumerate(TEST_VARIANTS):
            query_resets = {}
            prompt_ref = None
            for e_i, emb_name in enumerate(STRESS_EMBS):
                spec = get_embodiment_spec(emb_name)
                sup_seed = 72000000 + 10_000 * suites.index(suite) + 100 * tid + 10 * v_i + 11
                q_seed = 73000000 + 10_000 * suites.index(suite) + 100 * tid + 10 * v_i + 13
                sup_case = dict(arm="UR5e", suite=suite, task_id=tid, variant=variant, init_state_id=11, seed=sup_seed)
                q_case = dict(arm="UR5e", suite=suite, task_id=tid, variant=variant, init_state_id=13, seed=q_seed)

                if v_i == 0 and e_i == 0:
                    simulator.call("create", dict(q_case, embodiment=spec.name))

                act_meta, act_trace = run_active_probe_episode(simulator, sup_case, spec, prompt=prompt_ref)
                prompt_ref = prompt_ref or act_meta["prompt"]
                _, pas_trace = run_episode(
                    simulator, policy, sup_case, spec, prompt=prompt_ref,
                    yaw_hat=0.0, scale_hat=1.0, grip_hat=1, max_controls=SUPPORT_BUDGET,
                    record_chunks=True, max_record_chunks=24,
                )

                def _pack(tr):
                    d = {k: np.asarray(v)[None] for k, v in tr.items()}
                    d["chunk_steps"] = np.where(d["valid"], CHUNK_SIZE, 0).astype(np.int32)
                    d["dis_flow"] = compute_dis_flow_maps(d["img_before"], d["img_after"], d["valid"])
                    return d

                act_b = _pack(act_trace)
                pas_b = _pack(pas_trace)
                act_iss = issued_exposure(act_b)
                pas_iss = issued_exposure(pas_b)

                _, sa, ga = analytic_proprio_sysid(act_iss, act_b["delta_eef"], act_b["delta_jaw"], act_b["valid"], nominal_gain_m_per_cmd=cals["active_probe"]["nom_gain"])
                _, sp, gp = analytic_proprio_sysid(pas_iss, pas_b["delta_eef"], pas_b["delta_jaw"], pas_b["valid"], nominal_gain_m_per_cmd=cals["passive_attempt1"]["nom_gain"])
                yr, sr, gr = reground_babbling_sysid(pas_iss, pas_b["delta_eef"], pas_b["delta_jaw"], pas_b["valid"], nominal_gain_m_per_cmd=cals["passive_attempt1"]["nom_gain"])

                pas_hs = image_motion(pas_b["img_before"], pas_b["img_after"], pas_b["valid"])
                yaw_hs_p = float(analytic_visual_yaw(pas_iss, pas_hs, pas_b["delta_eef"][..., 2], pas_b["valid"], cals["passive_attempt1"]["hs_proj"])[0])
                act_dis = dense_flow_image_motion(act_b["img_before"], act_b["img_after"], act_b["valid"], method="dis")
                yaw_dis_a = float(analytic_visual_yaw(act_iss, act_dis, act_b["delta_eef"][..., 2], act_b["valid"], cals["active_probe"]["dis_proj"])[0])

                preds = {
                    "privileged_oracle": (spec.yaw_deg, spec.xy_scale, spec.gripper_sign),
                    "nominal": (0.0, 1.0, 1),
                    "ours_equiv_v2_passive_base": _pred(p_base_pas, c_base_pas, pas_b),
                    "ours_equiv_cam_homog_active": _pred(p_cam, c_cam, act_b),
                    "ours_equiv_v2_pro2_active": _pred(p_v2p, c_v2p, act_b),
                    "analytic_visual_hs_passive": (yaw_hs_p, float(sp[0]), int(gp[0])),
                    "dense_flow_dis_active": (yaw_dis_a, float(sa[0]), int(ga[0])),
                    "reground_babbling": (float(yr[0]), float(sr[0]), int(gr[0])),
                    "perceiver_icba_pro2_active": _pred(p_percp, c_percp, act_b),
                }

                row = dict(
                    shard_idx=shard_idx, suite=suite, task_id=tid, variant=variant,
                    embodiment=spec.name, regime=spec.regime,
                    true_yaw=spec.yaw_deg, true_scale=spec.xy_scale, true_grip=spec.gripper_sign,
                    arms={},
                )
                q_key = (suite, tid, variant, 13)
                for arm_name in STRESS_ARMS:
                    yh, sh, gh = preds[arm_name]
                    rec = run_paired_query(
                        run_episode, simulator, policy, q_case, spec,
                        arm_name=arm_name, resets=query_resets, key=q_key,
                        support_physics=act_meta["physics"], expected_prompt=prompt_ref,
                        create_attempts=3, yaw_hat=yh, scale_hat=sh, grip_hat=gh,
                    )
                    yerr = float(circular_yaw_error_deg(yh, spec.yaw_deg))
                    row["arms"][arm_name] = dict(
                        success=bool(rec["success"]),
                        controls=int(rec["controls"]),
                        yaw_hat=round(yh, 2),
                        yaw_err_deg=round(yerr, 2),
                        scale_hat=round(sh, 3),
                        grip_hat=int(gh),
                    )
                rows.append(row)
                write_json(shard_dir / "progress.json", rows)
                s_str = " ".join(f"{a[:11]}={int(row['arms'][a]['success'])}({row['arms'][a]['yaw_err_deg']:.0f}d)" for a in STRESS_ARMS[:6])
                print(f"[pro2 s{shard_idx}] {len(rows)}/{len(TEST_VARIANTS)*len(STRESS_EMBS)} {variant} {spec.name}: {s_str}", flush=True)

    write_json(shard_dir / "shard.json", dict(shard_idx=shard_idx, rows=rows))


def merge_results(out_dir: Path, num_shards: int):
    all_rows = []
    for s in range(num_shards):
        p = out_dir / "test_shards" / f"shard-{s}" / "shard.json"
        all_rows.extend(json.loads(p.read_text())["rows"])

    by_variant = {}
    for var in (*TEST_VARIANTS, "overall"):
        sub = all_rows if var == "overall" else [r for r in all_rows if r["variant"] == var]
        n = len(sub)
        sc = {a: sum(int(r["arms"][a]["success"]) for r in sub) for a in STRESS_ARMS}
        yerr = {a: round(float(np.mean([r["arms"][a]["yaw_err_deg"] for r in sub])), 2) for a in STRESS_ARMS}
        by_variant[var] = dict(
            trials=n,
            successes=sc,
            rates={a: round(sc[a] / max(n, 1), 4) for a in STRESS_ARMS},
            mean_yaw_err_deg=yerr,
        )

    contrasts = {}
    for hero in ("ours_equiv_v2_passive_base", "ours_equiv_cam_homog_active", "ours_equiv_v2_pro2_active"):
        contrasts[hero] = {
            comp: task_clustered_bootstrap(all_rows, hero, comp, seed=20261122)
            for comp in STRESS_ARMS if comp != hero
        }

    summary = dict(
        benchmark="LIBERO-Pro2.0-Visual-Camera-Stress",
        target_arm="UR5e",
        policy="pi05",
        variants=list(TEST_VARIANTS),
        trials=len(all_rows),
        by_variant=by_variant,
        paired_contrasts_overall=contrasts,
    )
    write_json(out_dir / "pro2-stress-summary.json", summary)
    print("=== LIBERO-PRO2.0 STRESS SUMMARY (UR5e, pi0.5) ===", flush=True)
    print(json.dumps(by_variant, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("collect-train")
    p1.add_argument("--out-dir", type=Path, required=True)
    p1.add_argument("--shard-idx", type=int, required=True)
    p1.add_argument("--num-shards", type=int, default=8)

    p2 = sub.add_parser("train-readers")
    p2.add_argument("--out-dir", type=Path, required=True)
    p2.add_argument("--num-shards", type=int, default=8)
    p2.add_argument("--steps", type=int, default=3000)

    p3 = sub.add_parser("eval-test")
    p3.add_argument("--out-dir", type=Path, required=True)
    p3.add_argument("--train-archive", type=Path, required=True)
    p3.add_argument("--base-ckpt-dir", type=Path, required=True)
    p3.add_argument("--shard-idx", type=int, required=True)
    p3.add_argument("--num-shards", type=int, default=8)
    p3.add_argument("--seed-ckpt", type=int, default=41)

    p4 = sub.add_parser("merge")
    p4.add_argument("--out-dir", type=Path, required=True)
    p4.add_argument("--num-shards", type=int, default=8)

    args = parser.parse_args()
    if args.cmd == "collect-train":
        collect_train_shard(args.out_dir, args.shard_idx, args.num_shards)
    elif args.cmd == "train-readers":
        train_pro2_readers(args.out_dir, args.num_shards, steps=args.steps)
    elif args.cmd == "eval-test":
        eval_test_shard(args.out_dir, args.train_archive, args.base_ckpt_dir, args.shard_idx, args.num_shards, seed_ckpt=args.seed_ckpt)
    else:
        merge_results(args.out_dir, args.num_shards)


if __name__ == "__main__":
    main()
