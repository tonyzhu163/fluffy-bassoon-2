"""Few-shot target-robot fine-tuning (SFT) baseline vs. zero-shot unshifted/shifted frozen pi0.5 on 6-DoF UR5e.

Evaluates on the locked 8 official LIBERO-Pro2.0 test tasks (SPLITS.json['test']) across 4 unseen shifted
6-DoF UR5e embodiments (2 compositional OOD + 2 continuous-angle OOD = 32 held-out test queries):
1. Zero-shot unshifted baseline (frozen pi0.5 on aligned 6-DoF UR5e, fam_nom)
2. Zero-shot shifted baseline (frozen pi0.5 unadapted on the 4 unseen shifted UR5e embodiments)
3. Few-shot target-robot SFT (Residual MLP Action Adapter & Linear Action-Head SFT) trained on K in {1, 3, 5, 10}
   real successful UR5e demonstrations (collected from disjoint init states), evaluated both on:
   - Same-shift held-out query reset (init_state_id=13)
   - Cross-shift generalization (K=10 adapter evaluated on a different unseen shift on UR5e)
"""

import argparse
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

BASE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fluffy_bassoon.xemb.embodiment import (
    EmbodimentSpec,
    get_embodiment_spec,
    retarget_action_from_motor,
    retarget_observation_for_motor,
)
from collect_ur5e import DEV_ARM
from ur5e_motor import SPLITS_PATH, load_frozen_pi05, to_openpi_observation, write_json
from ur5e_sim_worker import UR5eSimulator

TARGET_SHIFTS = (
    "comp_180_lo_inv",
    "comp_m90_hi_inv",
    "cont_135_075_inv",
    "cont_m45_075",
)
SHOTS = (1, 3, 10)
CHUNK_EXEC = 5


def build_official_test_tasks():
    splits = json.loads(SPLITS_PATH.read_text())
    test_map = splits["test"]
    suites = sorted(test_map.keys())
    tasks = []
    for suite in suites:
        for tid in test_map[suite]:
            tasks.append((suite, int(tid)))
    return suites, tasks


def collect_oracle_demo(simulator, policy, suite: str, task_id: int, init_state_id: int, seed: int, spec: EmbodimentSpec):
    """Collect one real successful oracle demonstration on shifted 6-DoF UR5e: pairs of (a_vla, state, a_target_issued)."""
    case = dict(arm=DEV_ARM, suite=suite, task_id=task_id, variant="BASE", init_state_id=init_state_id, seed=seed)
    reset = simulator.reset(case)
    raw = reset["raw"]
    prompt = reset["prompt"]
    max_steps = int(reset["max_steps"])

    a_vlas, states, a_targets = [], [], []
    step = 0
    done = False
    success = False
    while step < max_steps and not done:
        motor_obs = retarget_observation_for_motor(raw, spec)
        out = policy.infer(to_openpi_observation(motor_obs, prompt))
        chunk = np.asarray(out["actions"], dtype=np.float32)[:CHUNK_EXEC]
        state_vec = np.concatenate((motor_obs["robot0_eef_pos"], motor_obs["robot0_gripper_qpos"])).astype(np.float32)
        for act in chunk:
            if step >= max_steps or done:
                break
            # Exact oracle issued action on target robot
            target_issued, phys = retarget_action_from_motor(
                act, spec, yaw_hat=spec.yaw_deg, scale_hat=spec.xy_scale, grip_hat=spec.gripper_sign
            )
            a_vlas.append(act.astype(np.float32))
            states.append(state_vec)
            a_targets.append(target_issued.astype(np.float32))
            res = simulator.step(phys)
            raw = res["raw"]
            done = bool(res["done"])
            success = bool(res["success"])
            step += 1
    return success, np.asarray(a_vlas, dtype=np.float32), np.asarray(states, dtype=np.float32), np.asarray(a_targets, dtype=np.float32)


def train_mlp_adapter(a_vla: np.ndarray, states: np.ndarray, a_target: np.ndarray, steps: int = 500, seed: int = 41):
    """Fine-tune a 2-layer residual MLP action adapter f(a_vla, state) -> a_issued via BC MSE."""
    x_all = np.concatenate((a_vla, states), axis=-1)
    d_in = x_all.shape[-1]
    key = jax.random.key(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    init = jax.nn.initializers.lecun_normal()
    params = dict(
        w1=init(k1, (d_in, 128), jnp.float32),
        b1=jnp.zeros((128,), jnp.float32),
        w2=init(k2, (128, 128), jnp.float32),
        b2=jnp.zeros((128,), jnp.float32),
        w3=jnp.zeros((128, 7), jnp.float32),
        b3=jnp.zeros((7,), jnp.float32),
    )

    def forward(p, x, a0):
        h = jax.nn.silu(x @ p["w1"] + p["b1"])
        h = jax.nn.silu(h @ p["w2"] + p["b2"])
        return jnp.clip(a0 + h @ p["w3"] + p["b3"], -1.0, 1.0)

    opt = optax.adamw(optax.cosine_decay_schedule(3e-3, steps, alpha=0.05), weight_decay=1e-4)
    opt_state = opt.init(params)

    @jax.jit
    def update(p, os_st, xb, ab, yb):
        def loss_fn(pp):
            pred = forward(pp, xb, ab)
            return jnp.mean(jnp.square(pred - yb))
        loss, grads = jax.value_and_grad(loss_fn)(p)
        upd, os_st = opt.update(grads, os_st, p)
        return optax.apply_updates(p, upd), os_st, loss

    rng = np.random.default_rng(seed)
    n = len(x_all)
    bs = min(128, n)
    for _ in range(steps):
        idx = rng.integers(n, size=bs)
        params, opt_state, _ = update(params, opt_state, jnp.asarray(x_all[idx]), jnp.asarray(a_vla[idx]), jnp.asarray(a_target[idx]))

    def predict_fn(act: np.ndarray, state_vec: np.ndarray) -> np.ndarray:
        x = jnp.asarray(np.concatenate((act, state_vec), axis=-1)[None])
        a0 = jnp.asarray(act[None])
        return np.asarray(forward(params, x, a0)[0], dtype=np.float32)

    return predict_fn


def train_linear_adapter(a_vla: np.ndarray, a_target: np.ndarray):
    """Exact ridge-regularized linear action-head SFT: a_issued = W @ a_vla + b."""
    x = np.concatenate((a_vla, np.ones((len(a_vla), 1), dtype=np.float32)), axis=-1)
    reg = 1e-3 * np.eye(x.shape[1], dtype=np.float32)
    w = np.linalg.solve(x.T @ x + reg, x.T @ a_target)

    def predict_fn(act: np.ndarray, _state_vec: np.ndarray) -> np.ndarray:
        v = np.concatenate((act, np.ones((1,), dtype=np.float32)))
        return np.clip(v @ w, -1.0, 1.0).astype(np.float32)

    return predict_fn


def eval_with_adapter(simulator, policy, suite: str, task_id: int, init_state_id: int, seed: int, spec: EmbodimentSpec, adapter_fn=None) -> bool:
    case = dict(arm=DEV_ARM, suite=suite, task_id=task_id, variant="BASE", init_state_id=init_state_id, seed=seed)
    reset = simulator.reset(case)
    raw = reset["raw"]
    prompt = reset["prompt"]
    max_steps = int(reset["max_steps"])

    step = 0
    done = False
    success = False
    while step < max_steps and not done:
        motor_obs = retarget_observation_for_motor(raw, spec)
        out = policy.infer(to_openpi_observation(motor_obs, prompt))
        chunk = np.asarray(out["actions"], dtype=np.float32)[:CHUNK_EXEC]
        state_vec = np.concatenate((motor_obs["robot0_eef_pos"], motor_obs["robot0_gripper_qpos"])).astype(np.float32)
        for act in chunk:
            if step >= max_steps or done:
                break
            issued = adapter_fn(act.astype(np.float32), state_vec) if adapter_fn is not None else act.astype(np.float32)
            # Apply hidden physical shift directly to the issued action (identity adapter on top)
            _, phys = retarget_action_from_motor(issued, spec, yaw_hat=0.0, scale_hat=1.0, grip_hat=1)
            res = simulator.step(phys)
            raw = res["raw"]
            done = bool(res["done"])
            success = bool(res["success"])
            step += 1
    return success


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    t0 = time.monotonic()
    suites, tasks = build_official_test_tasks()
    policy = load_frozen_pi05()

    with UR5eSimulator() as simulator:
        # 1. Original zero-shot unshifted baseline on 6-DoF UR5e (fam_nom, init_state=13)
        nom_spec = get_embodiment_spec("fam_nom")
        zs_unshifted = []
        for suite, tid in tasks:
            q_seed = 41800000 + 10_000 * suites.index(suite) + 100 * tid + 13
            ok = eval_with_adapter(simulator, policy, suite, tid, 13, q_seed, nom_spec, adapter_fn=None)
            zs_unshifted.append(int(ok))
        print(f"ZERO-SHOT UNSHIFTED UR5e (fam_nom): {sum(zs_unshifted)}/{len(zs_unshifted)} ({100*np.mean(zs_unshifted):.1f}%)", flush=True)

        # 2. Few-shot SFT on 4 unseen shifted UR5e embodiments
        zs_shifted = []
        mlp_by_k = {k: [] for k in SHOTS}
        lin_by_k = {k: [] for k in SHOTS}
        cross_shift_mlp = []
        cross_shift_lin = []

        for sh_idx, sh_name in enumerate(TARGET_SHIFTS):
            spec = get_embodiment_spec(sh_name)
            other_spec = get_embodiment_spec(TARGET_SHIFTS[(sh_idx + 1) % len(TARGET_SHIFTS)])

            # Collect up to 10 successful oracle demos from disjoint init states (0..9) cycling across the 8 tasks
            demos = []
            for d_i in range(24):
                if len(demos) >= max(SHOTS):
                    break
                suite, tid = tasks[d_i % len(tasks)]
                init_id = d_i // len(tasks)
                d_seed = 51800000 + 10_000 * suites.index(suite) + 100 * tid + init_id
                ok, av, st, at = collect_oracle_demo(simulator, policy, suite, tid, init_id, d_seed, spec)
                if ok and len(av) > 10:
                    demos.append((av, st, at))
            print(f"[{sh_name}] Collected {len(demos)} successful target-UR5e demos ({sum(len(x[0]) for x in demos)} transitions)", flush=True)

            # Evaluate zero-shot shifted (0 demos)
            for suite, tid in tasks:
                q_seed = 41800000 + 10_000 * suites.index(suite) + 100 * tid + 13
                zs_shifted.append(int(eval_with_adapter(simulator, policy, suite, tid, 13, q_seed, spec, adapter_fn=None)))

            # Train & eval K in {1, 3, 5, 10}
            for k in SHOTS:
                use_demos = demos[: min(k, len(demos))]
                av_cat = np.concatenate([x[0] for x in use_demos], axis=0)
                st_cat = np.concatenate([x[1] for x in use_demos], axis=0)
                at_cat = np.concatenate([x[2] for x in use_demos], axis=0)

                mlp_fn = train_mlp_adapter(av_cat, st_cat, at_cat, steps=500, seed=41 + sh_idx)
                lin_fn = train_linear_adapter(av_cat, at_cat)

                for suite, tid in tasks:
                    q_seed = 41800000 + 10_000 * suites.index(suite) + 100 * tid + 13
                    mlp_by_k[k].append(int(eval_with_adapter(simulator, policy, suite, tid, 13, q_seed, spec, adapter_fn=mlp_fn)))
                    lin_by_k[k].append(int(eval_with_adapter(simulator, policy, suite, tid, 13, q_seed, spec, adapter_fn=lin_fn)))
                    if k == max(SHOTS):
                        cross_shift_mlp.append(int(eval_with_adapter(simulator, policy, suite, tid, 13, q_seed, other_spec, adapter_fn=mlp_fn)))
                        cross_shift_lin.append(int(eval_with_adapter(simulator, policy, suite, tid, 13, q_seed, other_spec, adapter_fn=lin_fn)))

            print(
                f"[{sh_name}] done | MLP(K=1,3,5,10)={[sum(mlp_by_k[k][-8:]) for k in SHOTS]} | Lin(K=1,3,5,10)={[sum(lin_by_k[k][-8:]) for k in SHOTS]}",
                flush=True,
            )

    summary = dict(
        arm=DEV_ARM,
        shifts=list(TARGET_SHIFTS),
        n_tasks=len(tasks),
        n_queries_per_arm=len(zs_shifted),
        zero_shot_unshifted=dict(successes=sum(zs_unshifted), trials=len(zs_unshifted), rate=round(float(np.mean(zs_unshifted)), 4)),
        zero_shot_shifted=dict(successes=sum(zs_shifted), trials=len(zs_shifted), rate=round(float(np.mean(zs_shifted)), 4)),
        sft_mlp_same_shift={str(k): dict(successes=sum(mlp_by_k[k]), trials=len(mlp_by_k[k]), rate=round(float(np.mean(mlp_by_k[k])), 4)) for k in SHOTS},
        sft_linear_same_shift={str(k): dict(successes=sum(lin_by_k[k]), trials=len(lin_by_k[k]), rate=round(float(np.mean(lin_by_k[k])), 4)) for k in SHOTS},
        sft_mlp_k10_cross_shift=dict(successes=sum(cross_shift_mlp), trials=len(cross_shift_mlp), rate=round(float(np.mean(cross_shift_mlp)), 4)),
        sft_linear_k10_cross_shift=dict(successes=sum(cross_shift_lin), trials=len(cross_shift_lin), rate=round(float(np.mean(cross_shift_lin)), 4)),
        seconds=round(time.monotonic() - t0, 1),
    )
    write_json(args.output, summary)
    print("=== FEW-SHOT TARGET-UR5e SFT SUMMARY ===", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
