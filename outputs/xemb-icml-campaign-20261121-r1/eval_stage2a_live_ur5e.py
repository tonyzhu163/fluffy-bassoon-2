"""Stage 2A: Locked 96/96 Official Test Live Closed-Loop Eval on Held-Out 6-DoF UR5e (Frozen pi0.5).

Evaluates our upgraded V2 equivariant readers (EquivFlow-Procrustes active & passive,
Equiv-V2 DIS+dz passive, AND zero-probe zero-reset Online Streaming Attempt-1!) head-to-head
against the complete competitive baseline suite:
- privileged_oracle
- analytic_visual_hs_passive (Horn-Schunck)
- dense_flow_dis_active / dense_flow_dis_passive (OpenCV DIS + Procrustes)
- broyden_uvs_dis_active (classical online Broyden UVS Jacobian)
- reground_babbling (Siddique et al., Nov 2026)
- perceiver_icba_passive (Yang, Finn & Sadigh–style Perceiver-IO monolithic baseline)
- ours_v2_cleared & ours_v2_swapped (causal controls)
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
for d in (BASE_DIR, SCALE_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from fluffy_bassoon.xemb.baselines import (
    broyden_uvs_yaw,
    compute_dis_flow_maps,
    dense_flow_image_motion,
    fit_dense_flow_projection,
    reground_babbling_sysid,
)
from fluffy_bassoon.xemb.embodiment import (
    EVAL_REGIMES,
    get_embodiment_spec,
    jaw_width_from_qpos,
    retarget_action_from_motor,
    retarget_observation_for_motor,
)
from fluffy_bassoon.xemb.equiv_v2 import (
    ReaderV2Config,
    StreamingProcrustesEstimator,
    apply_reader_v2,
    extract_visual_vectors_v2,
    init_reader_v2,
    predict_embodiment_v2,
)
from fluffy_bassoon.xemb.readers import (
    analytic_proprio_sysid,
    analytic_visual_yaw,
    fit_nominal_gain,
    fit_visual_projection,
    image_motion,
)
from collect_ur5e import CHUNK_SIZE, DEV_ARM, SUPPORT_BUDGET, issued_exposure, run_active_probe_episode
from eval_live_ur5e import check_query_reset, task_clustered_bootstrap
from eval_test_official import build_test_plan
from train_ur5e_readers import load_mode_split, make_donor_permutation
from ur5e_motor import (
    NOISE_SEED_OFFSET,
    array_digest,
    load_frozen_pi05,
    public_digest,
    run_episode,
    to_openpi_observation,
    write_json,
)
from ur5e_sim_worker import UR5eSimulator


STAGE2A_ARMS = (
    "privileged_oracle",
    "ours_equiv_v2_passive",
    "ours_equiv_homog_active",
    "ours_equiv_homog_passive",
    "ours_equiv_v2_online",
    "analytic_visual_hs_passive",
    "dense_flow_dis_active",
    "dense_flow_dis_passive",
    "broyden_uvs_dis_active",
    "reground_babbling",
    "perceiver_icba_passive",
    "ours_v2_swapped",
)


def load_v2_checkpoint(ckpt_path: Path):
    from flax import serialization
    import jax

    raw = ckpt_path.read_bytes()
    meta = serialization.msgpack_restore(raw)
    cfg_dict = meta["reader_v2_config"]
    cfg = ReaderV2Config(
        model_type=str(cfg_dict["model_type"]),
        visual_mode=str(cfg_dict["visual_mode"]),
        max_chunks=int(cfg_dict["max_chunks"]),
        cnn_width=int(cfg_dict["cnn_width"]),
        mlp_width=int(cfg_dict["mlp_width"]),
        use_dz_parallax=bool(cfg_dict["use_dz_parallax"]),
        perceiver_latents=int(cfg_dict.get("perceiver_latents", 8)),
        perceiver_width=int(cfg_dict.get("perceiver_width", 192)),
        perceiver_layers=int(cfg_dict.get("perceiver_layers", 3)),
        perceiver_heads=int(cfg_dict.get("perceiver_heads", 4)),
    )
    template = init_reader_v2(jax.random.key(0), cfg)
    params = serialization.from_state_dict(template, meta["params"])
    return params, cfg, meta


_STREAMING_JIT_CACHE: dict = {}


def _get_streaming_step_fn(reader_cfg: ReaderV2Config):
    if reader_cfg not in _STREAMING_JIT_CACHE:
        import jax

        @jax.jit
        def _step(params, batch):
            v_planar, vis_w = extract_visual_vectors_v2(params, reader_cfg, batch)
            _, _, log_s, g_logit = apply_reader_v2(params, reader_cfg, batch)
            return v_planar, vis_w, log_s, g_logit

        _STREAMING_JIT_CACHE[reader_cfg] = _step
    return _STREAMING_JIT_CACHE[reader_cfg]


def run_online_streaming_episode(
    simulator,
    policy,
    case: dict,
    spec,
    *,
    prompt: str,
    reader_params,
    reader_cfg: ReaderV2Config,
    max_streaming_chunks: int = 24,
    check_reset=None,
):
    """Zero-probe, zero-reset Attempt-1 online streaming adaptation using StreamingProcrustesEstimator.

    Starts at nominal identity `(0 deg, 1.0, +1)` and after every executed 5-control chunk
    computes DIS optical flow + direction-blind visual motion vector + scale/grip estimates
    and updates the O(1) sufficient statistics live during the episode.
    """
    sim_spec, initial = simulator.call("create", dict(case, embodiment=spec.name))
    raw = initial["observation"]
    anchor = np.asarray(initial["anchor_eef_pos"], dtype=np.float32)
    released_prompt = initial["language_instruction"]
    if prompt is not None and prompt != released_prompt:
        raise ValueError(f"Expected paired prompt {prompt!r}, got {released_prompt!r}")
    prompt = released_prompt
    horizon = int(sim_spec["max_steps"])
    bounds = initial["action_schema"]
    low = np.asarray(bounds["low"], dtype=np.float32)
    high = np.asarray(bounds["high"], dtype=np.float32)

    model_obj = getattr(policy, "_model", None)
    action_horizon = int(getattr(model_obj, "action_horizon", 10))
    action_dim = int(getattr(model_obj, "action_dim", 32))
    rng = np.random.default_rng(int(case["seed"]) + NOISE_SEED_OFFSET)
    noise_bank = rng.standard_normal(((horizon + 4) // 5, action_horizon, action_dim)).astype(np.float32)
    frame0_img = initial["agentview_64"]
    first_item = to_openpi_observation(raw, prompt)
    frame0_state = first_item["observation/state"].astype(np.float32)
    repro = initial["environment_reproducibility"]
    model_id = initial["model_identity"]

    reset_info = dict(
        arm=str(model_id["arm"]),
        model_class=str(model_id["model_class"]),
        joint_count=int(model_id["joint_count"]),
        scene_fingerprint=repro["scene_fingerprint"],
        physics=repro["initial_state_fingerprint"],
        perturbation_plan=repro["perturbation_plan_fingerprint"],
        public=public_digest(raw),
        motor_frame0={k: array_digest(v) for k, v in first_item.items() if k != "prompt"},
        reader_frame0=dict(image=array_digest(frame0_img), state=array_digest(frame0_state)),
        worker_step_index=initial["worker_step_index"],
        action_schema=initial["action_schema"],
        renderer=initial["renderer"],
        renderer_vendor=initial["renderer_vendor"],
        noise_seed=int(case["seed"]) + NOISE_SEED_OFFSET,
        noise_sha256=array_digest(noise_bank),
        horizon=horizon,
        prompt=released_prompt,
    )
    if check_reset is not None:
        check_reset(reset_info)

    est = StreamingProcrustesEstimator(prior_weight=0.05)
    cur_yaw, cur_scale, cur_grip = 0.0, 1.0, 1
    cur_img = frame0_img
    cur_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32).copy()
    cur_jaw = float(jaw_width_from_qpos(raw["robot0_gripper_qpos"]))
    controls = 0
    chunk_idx = 0
    done = False
    step = None
    rewards = []
    started = time.monotonic()

    while not done and controls < horizon:
        if controls == 0:
            item = first_item
        else:
            motor_obs = raw if cur_yaw == 0 else retarget_observation_for_motor(raw, anchor, cur_yaw)
            item = to_openpi_observation(motor_obs, prompt)
        noise = noise_bank[controls // 5]
        proposal = np.asarray(policy.infer(item, noise=noise)["actions"], dtype=np.float32)
        if proposal.shape != (action_horizon, 7) or not np.isfinite(proposal).all():
            raise ValueError(f"Invalid policy proposal shape {proposal.shape} on {case} / {spec.name}")
        count = min(5, horizon - controls)
        canonical = np.clip(proposal[:count], low, high)
        if (cur_yaw, cur_scale, cur_grip) == (0.0, 1.0, 1):
            issued = canonical
        else:
            issued = retarget_action_from_motor(canonical, cur_yaw, cur_scale, cur_grip, low, high)
        bounded = np.clip(issued, low, high).astype(np.float32)

        step = simulator.call("chunk", bounded)
        n = int(step["steps"])
        controls += n
        rewards.extend(step["rewards"])
        next_img = np.asarray(step["agentview_64"], dtype=np.uint8)
        next_pos = np.asarray(step["positions"][-1], dtype=np.float32)
        next_jaw = float(jaw_width_from_qpos(step["gripper_positions"][-1]))
        raw = step["observation"]
        done = bool(step["terminated"] or step["truncated"])

        if not done and chunk_idx < max_streaming_chunks:
            iss_mean = bounded[:n].mean(axis=0)
            deef = next_pos - cur_pos
            djaw = np.array([next_jaw - cur_jaw, next_jaw], dtype=np.float32)
            # Single-chunk batch [1, 1, ...]
            ib = cur_img[None, None]
            ia = next_img[None, None]
            val = np.ones((1, 1), dtype=bool)
            single = dict(
                img_before=ib,
                img_after=ia,
                dis_flow=compute_dis_flow_maps(ib, ia, val),
                issued_mean=iss_mean[None, None],
                delta_eef=deef[None, None],
                delta_jaw=djaw[None, None],
                valid=val,
            )
            v_planar, vis_w, log_s, g_logit = _get_streaming_step_fn(reader_cfg)(reader_params, single)
            est.update(
                iss_mean[:2],
                np.asarray(v_planar)[0, 0],
                float(np.asarray(vis_w)[0, 0]),
                log_scale_est=float(np.asarray(log_s)[0]),
                scale_weight=float(np.linalg.norm(iss_mean[:3]) + 0.05),
                grip_logit_est=float(np.asarray(g_logit)[0]),
                grip_weight=float(abs(iss_mean[6]) + 0.05),
            )
            if chunk_idx >= 1:
                cur_yaw, cur_scale, cur_grip = est.estimate()

        cur_img, cur_pos, cur_jaw = next_img, next_pos, next_jaw
        chunk_idx += 1

    success = bool(step["success"]) if step is not None else False
    return dict(
        success=success,
        controls=controls,
        final_yaw_hat=round(cur_yaw, 2),
        final_scale_hat=round(cur_scale, 3),
        final_grip_hat=int(cur_grip),
        seconds=round(time.monotonic() - started, 2),
    )


def fit_source_calibrators(train_archive_path: Path):
    archive = json.loads(train_archive_path.read_text())
    archive_dir = train_archive_path.parent
    cals = {}
    clean_hs_proj = None
    clean_dis_proj = None
    for mode in ("active_probe", "passive_attempt1"):
        tr_data, tr_rows = load_mode_split(archive_dir, archive["trials"], mode, "train")
        nom_idx = np.array([i for i, r in enumerate(tr_rows) if r["embodiment"] == "nom"], dtype=np.int64)
        nom_gain = fit_nominal_gain(tr_data["issued_mean"][nom_idx], tr_data["delta_eef"][nom_idx], tr_data["valid"][nom_idx])
        if mode == "active_probe":
            tr_hs = image_motion(tr_data["img_before"][nom_idx], tr_data["img_after"][nom_idx], tr_data["valid"][nom_idx])
            clean_hs_proj = fit_visual_projection(tr_hs, tr_data["delta_eef"][nom_idx], tr_data["valid"][nom_idx])
            clean_dis_proj = fit_dense_flow_projection(
                tr_data["img_before"][nom_idx], tr_data["img_after"][nom_idx],
                tr_data["delta_eef"][nom_idx], tr_data["valid"][nom_idx], method="dis",
            )
        cals[mode] = dict(nom_gain=float(nom_gain), hs_proj=clean_hs_proj, dis_proj=clean_dis_proj)
    return cals


PI0_CKPT = Path("/mowenjie/fluffy-bassoon-2/inputs/pi0-libero-openpi-g1748456809412170-20261029-r1")


def load_frozen_policy(policy_name: str):
    if policy_name == "pi05":
        return load_frozen_pi05()
    if policy_name == "pi0":
        import jax
        from openpi.policies import policy_config
        from openpi.training import config

        if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
            raise RuntimeError("Expected exactly one allocated GPU for frozen OpenPI pi0")
        return policy_config.create_trained_policy(config.get_config("pi0_libero"), str(PI0_CKPT))
    raise ValueError(f"Unknown policy_name {policy_name!r}; expected 'pi05' or 'pi0'")


def reset_differences(expected, observed):
    differences = {}
    for key in sorted(expected.keys() | observed.keys()):
        if key not in expected or key not in observed:
            differences[key] = dict(expected=expected.get(key, "<missing>"), observed=observed.get(key, "<missing>"))
        elif isinstance(expected[key], dict) and isinstance(observed[key], dict):
            nested = reset_differences(expected[key], observed[key])
            if nested:
                differences[key] = nested
        elif expected[key] != observed[key]:
            differences[key] = dict(expected=expected[key], observed=observed[key])
    return differences


def run_paired_query(
    episode, simulator, policy, case, spec, *,
    arm_name, resets, key, support_physics, expected_prompt, create_attempts, **kwargs,
):
    # Only the pre-inference reset callback can raise this local exception.
    class RejectedCreate(ValueError):
        pass

    def check(reset):
        expected = resets.get(key)
        try:
            if expected_prompt is not None and reset["prompt"] != expected_prompt:
                raise ValueError(f"Expected paired prompt {expected_prompt!r}")
            check_query_reset(resets, key, support_physics, reset)
        except ValueError as exc:
            diff = reset_differences(expected, reset) if expected is not None else {}
            if expected_prompt is not None and reset["prompt"] != expected_prompt:
                diff["prompt"] = dict(expected=expected_prompt, observed=reset["prompt"])
            if reset["physics"] == support_physics:
                diff["support_collision"] = dict(support_physics=support_physics, query_physics=reset["physics"])
            print(
                f"Rejected query create {attempt}/{create_attempts} {key} {spec.name} {arm_name}: "
                f"{exc}; differences={json.dumps(diff, sort_keys=True)}",
                flush=True,
            )
            raise RejectedCreate(str(exc)) from exc
        if expected is None:
            print(f"Paired reset reference {key}: {json.dumps(reset, sort_keys=True)}", flush=True)

    for attempt in range(1, create_attempts + 1):
        try:
            # Both runners use the native released prompt. Check its equality in
            # the callback alongside the other fingerprints, before any action.
            return episode(simulator, policy, case, spec, prompt=None, check_reset=check, **kwargs)
        except RejectedCreate:
            if attempt == create_attempts:
                raise


def run_shard(
    train_archive: Path,
    ckpt_dir: Path,
    homog_ckpt_dir: Path,
    output_dir: Path,
    shard_idx: int,
    num_shards: int,
    seed_ckpt: int,
    reuse_support_dir: Path | None,
    policy_name: str = "pi05",
    reset_create_attempts: int = 1,
    warmup_create: bool = False,
):
    tasks, plan = build_test_plan()
    if num_shards != len(tasks):
        raise ValueError(f"Expected num_shards == {len(tasks)}, got {num_shards}")
    if not 0 <= shard_idx < num_shards or not 1 <= reset_create_attempts <= 3:
        raise ValueError("Invalid logical shard index or reset create limit")
    suites = sorted({s for s, _ in tasks})
    my_suite, my_tid = tasks[shard_idx]
    my_entries = [e for e in plan if e["suite"] == my_suite and int(e["task_id"]) == my_tid]
    donors_local = make_donor_permutation(my_entries, np.random.default_rng(20261121 + shard_idx))

    shard_dir = output_dir / f"shard-{shard_idx}"
    # Existing progress has no durable reset reference or partial-arm ledger.
    # Re-entering it could replay an outcome or change the paired reference.
    shard_dir.mkdir(parents=True, exist_ok=False)
    support_dir = shard_dir / "support"
    support_dir.mkdir(parents=True, exist_ok=True)

    cals = fit_source_calibrators(train_archive)
    p_v2_pas, c_v2_pas, _ = load_v2_checkpoint(ckpt_dir / f"equiv_v2_dis_dz-passive_attempt1-seed{seed_ckpt}.msgpack")
    p_homog_act, c_homog_act, _ = load_v2_checkpoint(homog_ckpt_dir / f"equiv_flow_homog_nodz-active_probe-seed{seed_ckpt}.msgpack")
    p_homog_pas, c_homog_pas, _ = load_v2_checkpoint(homog_ckpt_dir / f"equiv_flow_homog_nodz-passive_attempt1-seed{seed_ckpt}.msgpack")
    p_perc_pas, c_perc_pas, _ = load_v2_checkpoint(ckpt_dir / f"perceiver_icba_dis-passive_attempt1-seed{seed_ckpt}.msgpack")

    policy = load_frozen_policy(policy_name)
    prompts = {}

    reused_meta = (reuse_support_dir / f"shard-{shard_idx}" / "support-meta.json") if reuse_support_dir else None
    with UR5eSimulator() as simulator:
        if reused_meta is not None:
            support_rows = json.loads(reused_meta.read_text())
            if [r["embodiment"] for r in support_rows] != [e["embodiment"] for e in my_entries]:
                raise ValueError(f"Reused support-meta mismatch on shard {shard_idx}")
        else:
            support_rows = []
            for local_i, entry in enumerate(my_entries):
                suite, tid = entry["suite"], int(entry["task_id"])
                spec = get_embodiment_spec(entry["embodiment"])
                sup_seed = 31800000 + 10_000 * suites.index(suite) + 100 * tid + entry["support_init_state_id"]
                sup_case = dict(arm=DEV_ARM, suite=suite, task_id=tid, variant="BASE", init_state_id=entry["support_init_state_id"], seed=sup_seed)
                key = (suite, tid)
                act_meta, act_trace = run_active_probe_episode(simulator, sup_case, spec, prompt=prompts.get(key))
                prompts.setdefault(key, act_meta["prompt"])
                _, pas_trace = run_episode(
                    simulator, policy, sup_case, spec, prompt=prompts[key],
                    yaw_hat=0.0, scale_hat=1.0, grip_hat=1, max_controls=SUPPORT_BUDGET,
                    record_chunks=True, max_record_chunks=24,
                )
                act_npz = support_dir / f"{local_i:02d}-{spec.name}-active.npz"
                pas_npz = support_dir / f"{local_i:02d}-{spec.name}-passive.npz"
                np.savez_compressed(act_npz, **act_trace)
                np.savez_compressed(pas_npz, **pas_trace)
                support_rows.append(dict(embodiment=spec.name, physics=act_meta["physics"], active_npz=str(act_npz), passive_npz=str(pas_npz)))
            write_json(shard_dir / "support-meta.json", support_rows)

        progress_path = shard_dir / "live-progress.json"
        rows = []
        query_resets = {}

        for local_i, entry in enumerate(my_entries):
            sup_row = support_rows[local_i]
            donor_row = support_rows[donors_local[local_i]]
            suite, tid = entry["suite"], int(entry["task_id"])
            spec = get_embodiment_spec(entry["embodiment"])
            init_state_id = entry["query_init_state_id"]
            q_seed = 41800000 + 10_000 * suites.index(suite) + 100 * tid + init_state_id
            case = dict(arm=DEV_ARM, suite=suite, task_id=tid, variant="BASE", init_state_id=init_state_id, seed=q_seed)
            if warmup_create and local_i == 0:
                warmup_case = dict(case, embodiment=spec.name)
                print(f"Unscored query create (zero actions): {warmup_case}", flush=True)
                simulator.call("create", warmup_case)

            def _load_npz(path_str: str):
                with np.load(path_str) as z:
                    d = {k: np.asarray(z[k])[None] for k in z.files}
                d["chunk_steps"] = np.where(d["valid"], CHUNK_SIZE, 0).astype(np.int32)
                d["dis_flow"] = compute_dis_flow_maps(d["img_before"], d["img_after"], d["valid"])
                return d

            act_b = _load_npz(sup_row["active_npz"])
            pas_b = _load_npz(sup_row["passive_npz"])
            don_b = _load_npz(donor_row["passive_npz"])

            act_iss = issued_exposure(act_b)
            pas_iss = issued_exposure(pas_b)
            _, sa, ga = analytic_proprio_sysid(act_iss, act_b["delta_eef"], act_b["delta_jaw"], act_b["valid"], nominal_gain_m_per_cmd=cals["active_probe"]["nom_gain"])
            _, sp, gp = analytic_proprio_sysid(pas_iss, pas_b["delta_eef"], pas_b["delta_jaw"], pas_b["valid"], nominal_gain_m_per_cmd=cals["passive_attempt1"]["nom_gain"])
            yr, sr, gr = reground_babbling_sysid(pas_iss, pas_b["delta_eef"], pas_b["delta_jaw"], pas_b["valid"], nominal_gain_m_per_cmd=cals["passive_attempt1"]["nom_gain"])

            pas_hs = image_motion(pas_b["img_before"], pas_b["img_after"], pas_b["valid"])
            yaw_hs_p = float(analytic_visual_yaw(pas_iss, pas_hs, pas_b["delta_eef"][..., 2], pas_b["valid"], cals["passive_attempt1"]["hs_proj"])[0])
            act_dis = dense_flow_image_motion(act_b["img_before"], act_b["img_after"], act_b["valid"], method="dis")
            pas_dis = dense_flow_image_motion(pas_b["img_before"], pas_b["img_after"], pas_b["valid"], method="dis")
            yaw_dis_a = float(analytic_visual_yaw(act_iss, act_dis, act_b["delta_eef"][..., 2], act_b["valid"], cals["active_probe"]["dis_proj"])[0])
            yaw_dis_p = float(analytic_visual_yaw(pas_iss, pas_dis, pas_b["delta_eef"][..., 2], pas_b["valid"], cals["passive_attempt1"]["dis_proj"])[0])
            yaw_broy_a = float(broyden_uvs_yaw(act_iss, act_dis, act_b["delta_eef"][..., 2], act_b["valid"], cals["active_probe"]["dis_proj"])[0])

            def _pred(params, cfg, batch):
                yh, sh, gh = predict_embodiment_v2(params, cfg, batch)
                return float(yh[0]), float(sh[0]), int(gh[0])

            arm_preds = {
                "privileged_oracle": (spec.yaw_deg, spec.xy_scale, spec.gripper_sign),
                "ours_equiv_v2_passive": _pred(p_v2_pas, c_v2_pas, pas_b),
                "ours_equiv_homog_active": _pred(p_homog_act, c_homog_act, act_b),
                "ours_equiv_homog_passive": _pred(p_homog_pas, c_homog_pas, pas_b),
                "analytic_visual_hs_passive": (yaw_hs_p, float(sp[0]), int(gp[0])),
                "dense_flow_dis_active": (yaw_dis_a, float(sa[0]), int(ga[0])),
                "dense_flow_dis_passive": (yaw_dis_p, float(sp[0]), int(gp[0])),
                "broyden_uvs_dis_active": (yaw_broy_a, float(sa[0]), int(ga[0])),
                "reground_babbling": (float(yr[0]), float(sr[0]), int(gr[0])),
                "perceiver_icba_passive": _pred(p_perc_pas, c_perc_pas, pas_b),
                "ours_v2_swapped": _pred(p_v2_pas, c_v2_pas, don_b),
            }

            key = (suite, tid)
            row = dict(
                index=int(entry["index"]), suite=suite, task_id=tid,
                embodiment=spec.name, regime=spec.regime,
                true_yaw=spec.yaw_deg, true_scale=spec.xy_scale, true_grip=spec.gripper_sign,
                arms={},
            )
            q_key = (suite, tid, init_state_id)
            for arm_name in STAGE2A_ARMS:
                pairing = dict(
                    arm_name=arm_name, resets=query_resets, key=q_key, support_physics=sup_row["physics"],
                    expected_prompt=prompts.get(key), create_attempts=reset_create_attempts,
                )
                if arm_name == "ours_equiv_v2_online":
                    rec = run_paired_query(
                        run_online_streaming_episode, simulator, policy, case, spec,
                        reader_params=p_v2_pas, reader_cfg=c_v2_pas, **pairing,
                    )
                    row["arms"][arm_name] = rec
                else:
                    yh, sh, gh = arm_preds[arm_name]
                    rec = run_paired_query(
                        run_episode, simulator, policy, case, spec,
                        yaw_hat=yh, scale_hat=sh, grip_hat=gh, **pairing,
                    )
                    prompts.setdefault(key, rec["released_prompt"])
                    row["arms"][arm_name] = dict(
                        success=bool(rec["success"]), controls=int(rec["controls"]),
                        yaw_hat=round(yh, 2), scale_hat=round(sh, 3), grip_hat=int(gh),
                    )
            rows.append(row)
            write_json(progress_path, rows)
            s_str = " ".join(f"{a[:10]}={int(row['arms'][a]['success'])}" for a in STAGE2A_ARMS)
            print(f"[shard {shard_idx}] {len(rows)}/{len(my_entries)} {suite}-{tid} {spec.name}: {s_str}", flush=True)

    write_json(shard_dir / "live-shard.json", dict(shard_idx=shard_idx, rows=rows))


def unique_json_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key {key!r} in shard results")
        obj[key] = value
    return obj


def validate_arm_results(row):
    arms = row["arms"]
    if not isinstance(arms, dict) or arms.keys() != set(STAGE2A_ARMS):
        raise ValueError(f"Index {row['index']} has the wrong arm set")
    for arm, result in arms.items():
        label = f"index {row['index']} arm {arm}"
        if not isinstance(result, dict) or type(result.get("success")) is not bool:
            raise ValueError(f"Invalid success for {label}")
        if type(result.get("controls")) is not int or result["controls"] <= 0:
            raise ValueError(f"Invalid controls for {label}")
        fields = ("yaw_hat", "scale_hat", "grip_hat")
        if arm == "ours_equiv_v2_online":
            fields = ("final_yaw_hat", "final_scale_hat", "final_grip_hat", "seconds")
        for field in fields:
            value = result.get(field)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"Missing or invalid {field} for {label}")


def merge_shards(output_dir: Path, num_shards: int):
    tasks, plan = build_test_plan()
    if num_shards != len(tasks) or len(set(tasks)) != 8:
        raise ValueError("Merge requires all eight distinct task shards")
    if len(plan) != 96 or {e["index"] for e in plan} != set(range(96)):
        raise ValueError("Merge requires all eight shards of the locked 96-query plan")
    expected_dirs = {f"shard-{s}" for s in range(num_shards)}
    extra_dirs = {p.name for p in output_dir.glob("shard-*") if p.is_dir()} - expected_dirs
    if extra_dirs:
        raise ValueError(f"Unexpected shard directories: {sorted(extra_dirs)}")
    all_rows = []
    for s, task in enumerate(tasks):
        expected = {e["index"]: e for e in plan if (e["suite"], e["task_id"]) == task}
        p = output_dir / f"shard-{s}" / "live-shard.json"
        if p.exists():
            data = json.loads(p.read_text(), object_pairs_hook=unique_json_keys)
            if type(data["shard_idx"]) is not int or data["shard_idx"] != s:
                raise ValueError(f"Wrong shard_idx in {p}")
            rows = data["rows"]
        else:
            p = p.with_name("live-progress.json")
            rows = json.loads(p.read_text(), object_pairs_hook=unique_json_keys)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list of rows in {p}")
        for row in rows:
            idx = row["index"]
            if type(idx) is not int or idx not in expected:
                raise ValueError(f"Duplicate or unexpected index {idx!r} in {p}")
            entry = expected.pop(idx)
            for field in ("suite", "task_id", "embodiment", "regime"):
                if type(row.get(field)) is not type(entry[field]) or row[field] != entry[field]:
                    raise ValueError(f"Index {idx} in {p}: expected {field}={entry[field]!r}, got {row.get(field)!r}")
            validate_arm_results(row)
        if expected:
            raise ValueError(f"Missing indices {sorted(expected)} in {p}")
        all_rows.extend(rows)
    all_rows.sort(key=lambda r: int(r["index"]))

    by_regime = {}
    for reg in (*EVAL_REGIMES, "ood_combined", "shifted", "overall"):
        if reg == "overall":
            sub = all_rows
        elif reg == "ood_combined":
            sub = [r for r in all_rows if r["regime"] in ("compositional_ood", "continuous_angle_ood")]
        elif reg == "shifted":
            sub = [r for r in all_rows if r["embodiment"] != "fam_nom"]
        else:
            sub = [r for r in all_rows if r["regime"] == reg]
        n = len(sub)
        sc = {a: sum(int(r["arms"][a]["success"]) for r in sub) for a in STAGE2A_ARMS}
        by_regime[reg] = dict(trials=n, successes=sc, rates={a: round(sc[a] / max(n, 1), 4) for a in STAGE2A_ARMS})

    contrasts = {}
    for hero in ("ours_equiv_v2_passive", "ours_equiv_homog_active", "ours_equiv_homog_passive", "ours_equiv_v2_online"):
        contrasts[hero] = {}
        for comp in STAGE2A_ARMS:
            if comp == hero:
                continue
            contrasts[hero][comp] = task_clustered_bootstrap(all_rows, hero, comp, seed=20261121)

    summary = dict(split="official_test", arm="UR5e", trials=len(all_rows), arms=list(STAGE2A_ARMS), by_regime=by_regime, paired_contrasts_overall=contrasts)
    write_json(output_dir / "stage2a-summary.json", summary)
    print("=== STAGE 2A OFFICIAL TEST SUMMARY (96/96) ===", flush=True)
    print(json.dumps(dict(overall=by_regime["overall"], ood=by_regime["ood_combined"]), indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run-shard")
    p_run.add_argument("--train-archive", type=Path, required=True)
    p_run.add_argument("--ckpt-dir", type=Path, required=True)
    p_run.add_argument("--homog-ckpt-dir", type=Path, required=True)
    p_run.add_argument("--output-dir", type=Path, required=True)
    p_run.add_argument("--shard-idx", type=int, required=True)
    p_run.add_argument("--num-shards", type=int, default=8)
    p_run.add_argument("--seed-ckpt", type=int, default=41)
    p_run.add_argument("--reuse-support-dir", type=Path, default=None)
    p_run.add_argument("--policy", type=str, default="pi05", choices=("pi05", "pi0"))
    p_run.add_argument("--reset-create-attempts", type=int, choices=(1, 2, 3), default=1,
                       help="Maximum native query creates per arm, including the first")
    p_run.add_argument("--warmup-create", action="store_true",
                       help="Discard one native create of the first query before establishing its paired reference")

    p_m = sub.add_parser("merge")
    p_m.add_argument("--output-dir", type=Path, required=True)
    p_m.add_argument("--num-shards", type=int, default=8)

    args = parser.parse_args()
    if args.cmd == "run-shard":
        run_shard(args.train_archive, args.ckpt_dir, args.homog_ckpt_dir, args.output_dir, args.shard_idx,
                  args.num_shards, args.seed_ckpt, args.reuse_support_dir, policy_name=args.policy,
                  reset_create_attempts=args.reset_create_attempts, warmup_create=args.warmup_create)
    else:
        merge_shards(args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
