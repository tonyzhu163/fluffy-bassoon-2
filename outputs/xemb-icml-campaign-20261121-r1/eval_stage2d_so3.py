"""Stage 2D: Full 3D SO(3) Command-Frame Rotation + Anisotropic Per-Axis Gain (gx, gy, gz) + Gripper Polarity.

Trains strictly on 7-DoF Panda + Sawyer (24 train tasks, 10 discrete axis-aligned 3D transforms),
then evaluates ZERO-SHOT on held-out 6-DoF UR5e (8 official test tasks, 12 3D transforms including
continuous oblique Euler rotations + unequal per-axis gains + inverted gripper) in BOTH offline 3D
geodesic error and live closed-loop frozen pi0.5 rollouts.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

BASE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"
SCALE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-scale-20261120-r1"
HERE_DIR = Path(__file__).resolve().parent
for d in (BASE_DIR, SCALE_DIR, HERE_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps
from fluffy_bassoon.xemb.embodiment import ACTIVE_PROBE_BURSTS, jaw_width_from_qpos
from fluffy_bassoon.xemb.equiv_v2 import (
    ReaderV2Config,
    init_reader_v2,
    predict_embodiment_so3,
    reader_loss_v2,
)
from eval_live_ur5e import task_clustered_bootstrap
from eval_pro2_visual_stress import load_split_tasks
from eval_stage2a_live_ur5e import load_frozen_policy
from ur5e_motor import NOISE_SEED_OFFSET, to_openpi_observation, write_json
from ur5e_sim_worker import UR5eSimulator


@dataclass(frozen=True)
class SO3Spec:
    name: str
    euler_zyx_deg: tuple[float, float, float]
    xyz_scale: tuple[float, float, float]
    gripper_sign: int
    regime: str

    def rot_mat(self) -> np.ndarray:
        return Rotation.from_euler("ZYX", self.euler_zyx_deg, degrees=True).as_matrix().astype(np.float32)

    def scale_vec(self) -> np.ndarray:
        return np.asarray(self.xyz_scale, dtype=np.float32)


# 10 discrete single-factor / axis-aligned 3D transforms for training (ZERO oblique compound 3D rotations!)
TRAIN_SO3 = (
    SO3Spec("tr_nom", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1, "train"),
    SO3Spec("tr_yaw90", (90.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1, "train"),
    SO3Spec("tr_yaw180", (180.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1, "train"),
    SO3Spec("tr_yaw270", (-90.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1, "train"),
    SO3Spec("tr_pitch35", (0.0, 35.0, 0.0), (1.0, 1.0, 1.0), 1, "train"),
    SO3Spec("tr_pitchm35", (0.0, -35.0, 0.0), (1.0, 1.0, 1.0), 1, "train"),
    SO3Spec("tr_roll35", (0.0, 0.0, 35.0), (1.0, 1.0, 1.0), 1, "train"),
    SO3Spec("tr_rollm35", (0.0, 0.0, -35.0), (1.0, 1.0, 1.0), 1, "train"),
    SO3Spec("tr_aniso_a", (0.0, 0.0, 0.0), (0.70, 1.30, 0.80), -1, "train"),
    SO3Spec("tr_aniso_b", (0.0, 0.0, 0.0), (1.30, 0.70, 1.25), 1, "train"),
)

# 12 held-out test 3D transforms (4 familiar, 4 compositional OOD, 4 continuous oblique SO(3) + unequal gains)
EVAL_SO3 = (
    SO3Spec("fam_nom", (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1, "familiar"),
    SO3Spec("fam_yaw90", (90.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1, "familiar"),
    SO3Spec("fam_pitch35", (0.0, 35.0, 0.0), (1.0, 1.0, 1.0), 1, "familiar"),
    SO3Spec("fam_rollm35", (0.0, 0.0, -35.0), (1.0, 1.0, 1.0), 1, "familiar"),
    SO3Spec("comp_y180_p35_aniso_inv", (180.0, 35.0, 0.0), (0.75, 1.25, 0.85), -1, "compositional_ood"),
    SO3Spec("comp_y270_rm35_aniso", (-90.0, 0.0, -35.0), (1.25, 0.75, 1.20), 1, "compositional_ood"),
    SO3Spec("comp_y90_p35_r35_inv", (90.0, 35.0, 35.0), (0.80, 0.80, 1.30), -1, "compositional_ood"),
    SO3Spec("comp_pm35_r35_aniso_inv", (0.0, -35.0, 35.0), (1.30, 0.70, 0.80), -1, "compositional_ood"),
    SO3Spec("cont_47_m31_23_aniso_inv", (47.0, -31.0, 23.0), (0.65, 1.25, 1.40), -1, "continuous_so3_ood"),
    SO3Spec("cont_m135_25_m40_aniso", (-135.0, 25.0, -40.0), (1.20, 0.75, 1.15), 1, "continuous_so3_ood"),
    SO3Spec("cont_115_m28_32_inv", (115.0, -28.0, 32.0), (0.85, 1.15, 0.75), -1, "continuous_so3_ood"),
    SO3Spec("cont_m62_38_m24_aniso", (-62.0, 38.0, -24.0), (1.15, 0.85, 1.25), 1, "continuous_so3_ood"),
)

# 6-chunk (30-control) rich 3D excitation probe spanning +/-X, +/-Y, +/-Z and both gripper states
PROBE_3D_BURSTS = (
    np.tile(np.array([0.80, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32), (5, 1)),
    np.tile(np.array([-0.80, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (5, 1)),
    np.tile(np.array([0.0, 0.80, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32), (5, 1)),
    np.tile(np.array([0.0, -0.80, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (5, 1)),
    np.tile(np.array([0.0, 0.0, 0.75, 0.0, 0.0, 0.0, -1.0], dtype=np.float32), (5, 1)),
    np.tile(np.array([0.0, 0.0, -0.75, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (5, 1)),
)

SO3_ARMS = (
    "privileged_oracle",
    "nominal",
    "ours_equiv_so3",
    "reground_babbling_3d",
    "ours_so3_swapped",
)


def geodesic_deg(r_hat: np.ndarray, r_true: np.ndarray) -> float:
    tr = float(np.trace(r_hat.T @ r_true))
    c = max(-1.0, min(1.0, 0.5 * (tr - 1.0)))
    return float(math.degrees(math.acos(c)))


def apply_so3_physical(issued: np.ndarray, spec: SO3Spec) -> np.ndarray:
    """Physical hidden embodiment: v_world_xyz = R @ (diag(scale) @ u_xyz), grip = s * u_6."""
    r_mat = spec.rot_mat()
    sc = spec.scale_vec()
    out = np.asarray(issued, dtype=np.float32).copy()
    out[..., :3] = np.einsum("ij,...j->...i", r_mat, out[..., :3] * sc)
    out[..., 6] = float(spec.gripper_sign) * out[..., 6]
    return np.clip(out, -1.0, 1.0)


def invert_so3_adapter(canonical: np.ndarray, r_hat: np.ndarray, scale_hat: np.ndarray, grip_hat: int) -> np.ndarray:
    """Inverse adapter u = D^{-1} R^T v_canonical, grip = g_hat * v_6."""
    out = np.asarray(canonical, dtype=np.float32).copy()
    sc = np.clip(np.asarray(scale_hat, dtype=np.float32), 0.25, 4.0)
    out[..., :3] = np.einsum("ji,...j->...i", r_hat, out[..., :3]) / sc
    out[..., 6] = float(grip_hat) * out[..., 6]
    return np.clip(out, -1.0, 1.0)


def collect_so3_probe(simulator, case: dict, spec: SO3Spec):
    """Run the 6-chunk (30-control) 3D probe under hidden SO(3) + anisotropic scale + grip polarity."""
    _, initial = simulator.call("create", dict(case, embodiment="nom"))
    raw = initial["observation"]
    r_mat = spec.rot_mat()
    k = len(PROBE_3D_BURSTS)
    img_b = np.zeros((k, 64, 64, 3), dtype=np.uint8)
    img_a = np.zeros((k, 64, 64, 3), dtype=np.uint8)
    iss_m = np.zeros((k, 7), dtype=np.float32)
    deef_local = np.zeros((k, 3), dtype=np.float32)
    djaw = np.zeros((k, 2), dtype=np.float32)
    valid = np.zeros((k,), dtype=bool)

    cur_img = np.asarray(initial["agentview_64"], dtype=np.uint8)
    cur_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32).copy()
    cur_jaw = float(jaw_width_from_qpos(raw["robot0_gripper_qpos"]))

    for idx, burst in enumerate(PROBE_3D_BURSTS):
        phys = apply_so3_physical(burst, spec)
        step = simulator.call("chunk", phys)
        n = int(step["steps"])
        nxt_img = np.asarray(step["agentview_64"], dtype=np.uint8)
        nxt_pos = np.asarray(step["positions"][-1], dtype=np.float32)
        nxt_jaw = float(jaw_width_from_qpos(step["gripper_positions"][-1]))
        img_b[idx] = cur_img
        img_a[idx] = nxt_img
        iss_m[idx] = burst[:n].mean(axis=0)
        # Local proprioception rotated into robot's local command frame R^T delta_p_world (strictly rotation-blind!)
        deef_local[idx] = r_mat.T @ (nxt_pos - cur_pos)
        djaw[idx] = np.array([nxt_jaw - cur_jaw, nxt_jaw], dtype=np.float32)
        valid[idx] = True
        cur_img, cur_pos, cur_jaw = nxt_img, nxt_pos, nxt_jaw
        if step["terminated"] or step["truncated"]:
            break

    return dict(
        img_before=img_b,
        img_after=img_a,
        issued_mean=iss_m,
        delta_eef=deef_local,
        delta_jaw=djaw,
        valid=valid,
        rot_mat=r_mat,
        xyz_scale=spec.scale_vec(),
        gripper_sign=np.int32(spec.gripper_sign),
    ), dict(prompt=initial["language_instruction"], physics=initial["environment_reproducibility"]["initial_state_fingerprint"])


def run_so3_live_episode(simulator, policy, case: dict, spec: SO3Spec, r_hat: np.ndarray, scale_hat: np.ndarray, grip_hat: int):
    sim_spec, initial = simulator.call("create", dict(case, embodiment="nom"))
    raw = initial["observation"]
    prompt = initial["language_instruction"]
    horizon = int(sim_spec["max_steps"])
    rng = np.random.default_rng(int(case["seed"]) + NOISE_SEED_OFFSET)
    noise_bank = rng.standard_normal(((horizon + 4) // 5, 10, 32)).astype(np.float32)

    controls = 0
    done = False
    step = None
    while not done and controls < horizon:
        item = to_openpi_observation(raw, prompt)
        noise = noise_bank[controls // 5]
        proposal = np.asarray(policy.infer(item, noise=noise)["actions"], dtype=np.float32)
        count = min(5, horizon - controls)
        canonical = np.clip(proposal[:count], -1.0, 1.0)
        issued = invert_so3_adapter(canonical, r_hat, scale_hat, grip_hat)
        phys = apply_so3_physical(issued, spec)
        step = simulator.call("chunk", phys)
        controls += int(step["steps"])
        raw = step["observation"]
        done = bool(step["terminated"] or step["truncated"])
    return dict(success=bool(step["success"]) if step is not None else False, controls=controls)


def collect_train_shard(out_dir: Path, shard_idx: int, num_shards: int):
    train_tasks = load_split_tasks("train")
    suites = sorted({s for s, _ in train_tasks})
    my_tasks = [t for i, t in enumerate(train_tasks) if i % num_shards == shard_idx]
    shard_dir = out_dir / "train_shards" / f"shard-{shard_idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    traces = []
    t0 = time.monotonic()
    with UR5eSimulator() as simulator:
        for arm in ("Panda", "Sawyer"):
            for suite, tid in my_tasks:
                for st_i in range(2):
                    init_state_id = (tid + st_i * 3) % 8
                    seed = 81000000 + 100_000 * ("Panda", "Sawyer").index(arm) + 10_000 * suites.index(suite) + 100 * tid + init_state_id
                    for spec in TRAIN_SO3:
                        case = dict(arm=arm, suite=suite, task_id=tid, variant="BASE", init_state_id=init_state_id, seed=seed)
                        tr, _ = collect_so3_probe(simulator, case, spec)
                        traces.append(tr)

    keys = tuple(traces[0].keys())
    stacked = {k: np.stack([t[k] for t in traces], axis=0) for k in keys}
    stacked["dis_flow"] = compute_dis_flow_maps(stacked["img_before"], stacked["img_after"], stacked["valid"])
    np.savez_compressed(shard_dir / "train.npz", **stacked)
    print(f"[so3 collect-train s{shard_idx}] saved {len(traces)} 3D probes in {time.monotonic()-t0:.1f}s", flush=True)


def train_so3_reader(out_dir: Path, num_shards: int, steps: int = 3000):
    import jax
    import optax
    from flax import serialization

    parts = []
    for s in range(num_shards):
        with np.load(out_dir / "train_shards" / f"shard-{s}" / "train.npz") as z:
            parts.append({k: np.asarray(z[k]) for k in z.files})
    data = {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
    n = len(data["gripper_sign"])
    print(f"[so3 train] loaded {n} Panda+Sawyer 3D probes", flush=True)

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = ReaderV2Config(model_type="factored_equiv_so3", visual_mode="dis_lumaflow", max_chunks=6, cnn_width=96, mlp_width=128)
    for seed in (41, 42, 43):
        t0 = time.monotonic()
        params = init_reader_v2(jax.random.key(seed), cfg)
        sched = optax.cosine_decay_schedule(1.5e-3, steps, alpha=0.05)
        opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
        opt_state = opt.init(params)

        @jax.jit
        def step_fn(p, st, b):
            (loss, _), grads = jax.value_and_grad(lambda pp: reader_loss_v2(pp, cfg, b), has_aux=True)(p)
            updates, st = opt.update(grads, st, p)
            return optax.apply_updates(p, updates), st, loss

        rng = np.random.default_rng(seed * 2029)
        last_loss = 0.0
        for _ in range(steps):
            idx = rng.integers(n, size=64)
            batch = {k: data[k][idx] for k in data}
            params, opt_state, loss = step_fn(params, opt_state, batch)
            last_loss = float(loss)

        blob = dict(
            seed=seed, steps=steps, final_loss=round(last_loss, 5),
            reader_v2_config=dict(
                model_type=cfg.model_type, visual_mode=cfg.visual_mode, max_chunks=cfg.max_chunks,
                cnn_width=cfg.cnn_width, mlp_width=cfg.mlp_width, use_dz_parallax=cfg.use_dz_parallax,
            ),
            params=serialization.to_state_dict(jax.device_get(params)),
        )
        (ckpt_dir / f"factored_equiv_so3-seed{seed}.msgpack").write_bytes(serialization.msgpack_serialize(blob))
        print(f"[so3 train] seed={seed} loss={last_loss:.4f} ({time.monotonic()-t0:.1f}s)", flush=True)


def eval_test_shard(out_dir: Path, shard_idx: int, num_shards: int, seed_ckpt: int = 41):
    import jax
    from flax import serialization

    test_tasks = load_split_tasks("test")
    suites = sorted({s for s, _ in test_tasks})
    suite, tid = test_tasks[shard_idx]
    shard_dir = out_dir / "test_shards" / f"shard-{shard_idx}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    meta = serialization.msgpack_restore((out_dir / "checkpoints" / f"factored_equiv_so3-seed{seed_ckpt}.msgpack").read_bytes())
    cd = meta["reader_v2_config"]
    cfg = ReaderV2Config(model_type=cd["model_type"], visual_mode=cd["visual_mode"], max_chunks=cd["max_chunks"], cnn_width=cd["cnn_width"], mlp_width=cd["mlp_width"])
    params = serialization.from_state_dict(init_reader_v2(jax.random.key(0), cfg), meta["params"])
    policy = load_frozen_policy("pi05")

    rows = []
    probes = []
    with UR5eSimulator() as simulator:
        for spec in EVAL_SO3:
            sup_seed = 82000000 + 10_000 * suites.index(suite) + 100 * tid + 11
            sup_case = dict(arm="UR5e", suite=suite, task_id=tid, variant="BASE", init_state_id=11, seed=sup_seed)
            tr, _ = collect_so3_probe(simulator, sup_case, spec)
            b = {k: np.asarray(v)[None] for k, v in tr.items()}
            b["dis_flow"] = compute_dis_flow_maps(b["img_before"], b["img_after"], b["valid"])
            probes.append(b)

        for i, spec in enumerate(EVAL_SO3):
            b = probes[i]
            don_b = probes[(i + 5) % len(EVAL_SO3)]
            r_hat, s_hat, g_hat = predict_embodiment_so3(params, cfg, b)
            rd_hat, sd_hat, gd_hat = predict_embodiment_so3(params, cfg, don_b)
            # ReGround-style 3D proprio babbling: recovers per-axis scale & grip, blind to rotated local frame (R_hat = I)
            rg_scale = np.clip(np.linalg.norm(b["delta_eef"][0], axis=0) / np.maximum(np.linalg.norm(b["issued_mean"][0, :, :3], axis=0) * 0.12, 1e-4), 0.25, 4.0)
            rg_grip = int(g_hat[0])

            preds = {
                "privileged_oracle": (spec.rot_mat(), spec.scale_vec(), spec.gripper_sign),
                "nominal": (np.eye(3, dtype=np.float32), np.ones(3, dtype=np.float32), 1),
                "ours_equiv_so3": (r_hat[0], s_hat[0], int(g_hat[0])),
                "reground_babbling_3d": (np.eye(3, dtype=np.float32), rg_scale.astype(np.float32), rg_grip),
                "ours_so3_swapped": (rd_hat[0], sd_hat[0], int(gd_hat[0])),
            }
            q_seed = 83000000 + 10_000 * suites.index(suite) + 100 * tid + 13
            q_case = dict(arm="UR5e", suite=suite, task_id=tid, variant="BASE", init_state_id=13, seed=q_seed)
            row = dict(shard_idx=shard_idx, suite=suite, task_id=tid, embodiment=spec.name, regime=spec.regime, arms={})
            r_true = spec.rot_mat()
            s_true = spec.scale_vec()
            for arm_name in SO3_ARMS:
                rh, sh, gh = preds[arm_name]
                rec = run_so3_live_episode(simulator, policy, q_case, spec, rh, sh, gh)
                row["arms"][arm_name] = dict(
                    success=bool(rec["success"]),
                    controls=int(rec["controls"]),
                    geodesic_err_deg=round(geodesic_deg(rh, r_true), 2),
                    scale_mae=round(float(np.mean(np.abs(sh - s_true))), 3),
                    grip_ok=bool(int(gh) == spec.gripper_sign),
                )
            rows.append(row)
            write_json(shard_dir / "progress.json", rows)
            s_str = " ".join(f"{a[:10]}={int(row['arms'][a]['success'])}({row['arms'][a]['geodesic_err_deg']:.0f}d)" for a in SO3_ARMS)
            print(f"[so3 s{shard_idx}] {len(rows)}/{len(EVAL_SO3)} {spec.name}: {s_str}", flush=True)

    write_json(shard_dir / "shard.json", dict(shard_idx=shard_idx, rows=rows))


def merge_so3(out_dir: Path, num_shards: int):
    all_rows = []
    for s in range(num_shards):
        all_rows.extend(json.loads((out_dir / "test_shards" / f"shard-{s}" / "shard.json").read_text())["rows"])

    by_regime = {}
    for reg in ("familiar", "compositional_ood", "continuous_so3_ood", "ood_combined", "overall"):
        if reg == "overall":
            sub = all_rows
        elif reg == "ood_combined":
            sub = [r for r in all_rows if r["regime"] in ("compositional_ood", "continuous_so3_ood")]
        else:
            sub = [r for r in all_rows if r["regime"] == reg]
        n = len(sub)
        sc = {a: sum(int(r["arms"][a]["success"]) for r in sub) for a in SO3_ARMS}
        gerr = {a: round(float(np.mean([r["arms"][a]["geodesic_err_deg"] for r in sub])), 2) for a in SO3_ARMS}
        smae = {a: round(float(np.mean([r["arms"][a]["scale_mae"] for r in sub])), 3) for a in SO3_ARMS}
        by_regime[reg] = dict(trials=n, successes=sc, rates={a: round(sc[a] / max(n, 1), 4) for a in SO3_ARMS}, mean_geodesic_deg=gerr, mean_scale_mae=smae)

    contrasts = {comp: task_clustered_bootstrap(all_rows, "ours_equiv_so3", comp, seed=20261122) for comp in SO3_ARMS if comp != "ours_equiv_so3"}
    summary = dict(benchmark="LIBERO-SO3-Anisotropic-3D", target_arm="UR5e", policy="pi05", trials=len(all_rows), by_regime=by_regime, paired_contrasts_overall=contrasts)
    write_json(out_dir / "stage2d-so3-summary.json", summary)
    print("=== STAGE 2D SO(3) + ANISOTROPIC 3D SUMMARY (UR5e, 96/96) ===", flush=True)
    print(json.dumps(dict(overall=by_regime["overall"], ood=by_regime["ood_combined"]), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("collect-train"); p1.add_argument("--out-dir", type=Path, required=True); p1.add_argument("--shard-idx", type=int, required=True); p1.add_argument("--num-shards", type=int, default=8)
    p2 = sub.add_parser("train-reader"); p2.add_argument("--out-dir", type=Path, required=True); p2.add_argument("--num-shards", type=int, default=8); p2.add_argument("--steps", type=int, default=3000)
    p3 = sub.add_parser("eval-test"); p3.add_argument("--out-dir", type=Path, required=True); p3.add_argument("--shard-idx", type=int, required=True); p3.add_argument("--num-shards", type=int, default=8); p3.add_argument("--seed-ckpt", type=int, default=41)
    p4 = sub.add_parser("merge"); p4.add_argument("--out-dir", type=Path, required=True); p4.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()
    if args.cmd == "collect-train":
        collect_train_shard(args.out_dir, args.shard_idx, args.num_shards)
    elif args.cmd == "train-reader":
        train_so3_reader(args.out_dir, args.num_shards, steps=args.steps)
    elif args.cmd == "eval-test":
        eval_test_shard(args.out_dir, args.shard_idx, args.num_shards, seed_ckpt=args.seed_ckpt)
    else:
        merge_so3(args.out_dir, args.num_shards)


if __name__ == "__main__":
    main()
