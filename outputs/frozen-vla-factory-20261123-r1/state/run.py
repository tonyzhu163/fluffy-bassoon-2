"""Real CPU SmolVLA inference on held-out simulated UR5e and IIWA recordings."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.video_utils import decode_video_frames
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import processor_smolvla  # register saved pipeline step
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
sys.path.insert(0, str(OUTPUTS.parent / "src"))
sys.path.insert(0, str(OUTPUTS / "symmeta-camera-command-20261123-r1"))
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec, retarget_observation_for_motor, to_local_proprio
from world_control import align

ARMS = ("ur5e", "iiwa")
TASKS = range(2, 10)
WEIGHT_SHA = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def source_pose(pose):
    """Apply the archive's downward-gripper quaternion hemisphere convention in rotation-vector coordinates."""
    p = pose.copy()
    rot = p[3:6]
    angle = np.linalg.norm(rot)
    if rot[0] < 0 and abs(np.cos(angle / 2)) < 0.85:
        rot *= 1 - 2 * np.pi / angle
    return p


def encode_joints(q):
    # Same vectorization for each interface. Missing trailing encoders are zero radians.
    q = np.pad(q, (0, 7 - len(q))).astype(np.float64)
    return np.concatenate((np.sin(q), np.cos(q)))


def joint_fit(data):
    features, poses, episodes = [], [], {}
    for arm, frames in data.items():
        for task in (0, 1):
            selected = frames.loc[frames.task_index == task].episode_index.unique()[:3]
            episodes[f"{arm}/{task}"] = [int(x) for x in selected]
            for ep in selected:
                traj = frames[frames.episode_index == ep]
                for ix in np.linspace(0, len(traj) - 1, 24, dtype=int):
                    row = traj.iloc[ix]
                    features.append(encode_joints(row["observation.states.joint_state"]))
                    poses.append(source_pose(row["observation.states.ee_state"]))
    return np.stack(features), np.stack(poses), episodes


def local_pose(q, support):
    """Local affine ridge, predicting TCP from the pooled paired calibration set."""
    x, y = support
    query = encode_joints(q)
    nearest = np.argsort(np.linalg.norm(x - query, axis=1), kind="stable")[:32]
    x, y = x[nearest], y[nearest]
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    coef = np.linalg.solve(xc.T @ xc + 1e-3 * np.eye(x.shape[1]), xc.T @ yc)
    return y.mean(axis=0) + (query - x.mean(axis=0)) @ coef


def yaw_support():
    # The retrieved case specifies which deployment support to supply. Estimation
    # takes only observed controls and FK displacement, never an arm or angle label.
    archive = OUTPUTS / "symmeta-fk-crossarm-20261123-r1/evidence"
    experiments = (("gpu08pi0fix", 1), ("gpu09iiwa", 11))
    estimates = []
    for panel, index in experiments:
        with gzip.open(archive / f"{panel}-cases.json.gz", "rt") as f:
            row = json.load(f)["rows"][index]
        assert row["suite"] == "libero_10" and row["task_id"] == 5 and row["fk_available"]
        fit = align(row["issued_xy"], row["world_delta_xy_m"])
        estimates.append(dict(panel=panel, index=index, interface=row["embodiment"],
                              support_sha256=row["support_sha256"], yaw_hat=fit["yaw_hat_deg"]))
    return estimates


def observations(arm_path, episode, row):
    images = {}
    for key, motor_key in (("image", "image"), ("wrist_image", "image2")):
        group = f"videos/observation.images.{key}"
        chunk = episode[f"{group}/chunk_index"]
        file = episode[f"{group}/file_index"]
        ts = float(episode[f"{group}/from_timestamp"] + row.timestamp)
        path = arm_path / group / f"chunk-{chunk:03d}/file-{file:03d}.mp4"
        images[f"observation.images.{motor_key}"] = decode_video_frames(
            path, [ts], tolerance_s=0.005, backend="pyav"
        )[0]
    assert all(v.shape == (3, 256, 256) and v.std() > 0 for v in images.values())
    return images


def build_policy(root):
    path = root / "ckpt"
    with (path / "model.safetensors").open("rb") as f:
        assert hashlib.file_digest(f, "sha256").hexdigest() == WEIGHT_SHA
    cfg = PreTrainedConfig.from_pretrained(path, cli_overrides=["--device=cpu"])
    cfg.vlm_model_name = str(root / "vlm")
    cfg.load_vlm_weights = False  # materialize architecture, then load the entire fine-tuned policy strictly
    policy = SmolVLAPolicy.from_pretrained(path, config=cfg, strict=True).float().eval().requires_grad_(False)
    pre, post = make_pre_post_processors(cfg, str(path), preprocessor_overrides={
        "device_processor": {"device": "cpu"}, "tokenizer_processor": {"tokenizer_name": cfg.vlm_model_name}
    })
    assert all(not m.training for m in policy.modules()) and all(not p.requires_grad for p in policy.parameters())
    assert all(p.device.type == "cpu" for p in policy.parameters())
    stats = load_file(path / "policy_preprocessor_step_5_normalizer_processor.safetensors")
    assert stats["observation.state.min"][3] > 0  # recorded source X rotation-vector component
    return policy, pre, post, stats["observation.state.mean"].numpy()


def state_inputs(row, anchor, support, yaw):
    logged = row["observation.state"]
    fingers = row["observation.states.gripper_state"]
    np.testing.assert_array_equal(logged, np.r_[row["observation.states.ee_state"], fingers])
    known = source_pose(logged)
    q = row["observation.states.joint_state"]
    states = {"known": known, "reported_tcp": logged, "raw_joint_slots": np.r_[q[:6], fingers],
              "joint_to_pose": np.r_[local_pose(q, support), fingers].astype(np.float32)}
    for calibration in yaw:
        spec = get_embodiment_spec(calibration["interface"])
        public = to_local_proprio({"robot0_eef_pos": known[:3]}, anchor, spec)
        canonical = retarget_observation_for_motor(public, anchor, calibration["yaw_hat"])
        oracle = retarget_observation_for_motor(public, anchor, spec.yaw_deg)
        np.testing.assert_allclose(oracle["robot0_eef_pos"], known[:3], rtol=0, atol=1e-6)
        states[f"raw_{spec.name}"] = np.r_[public["robot0_eef_pos"], known[3:]]
        states[f"fit_{spec.name}"] = np.r_[canonical["robot0_eef_pos"], known[3:]]
    return states


def rmse(a, b):
    return float(np.sqrt(np.mean(np.square(a.astype(np.float64) - b))))


@torch.inference_mode()
def run(root):
    policy, pre, post, mean = build_policy(root)
    table_path = "data/chunk-000/file-000.parquet"
    data = {arm: pd.read_parquet(root / arm / table_path) for arm in ARMS}
    x, y, fit_episodes = joint_fit(data)
    yaw = yaw_support()
    rows = []
    for arm in ARMS:
        episodes = pd.read_parquet(root / arm / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
        frames = data[arm]
        for task in TASKS:
            ep = int(frames.loc[frames.task_index == task, "episode_index"].min())
            traj = frames[frames.episode_index == ep].sort_values("frame_index")
            row = traj.iloc[len(traj) // 2]
            episode = episodes.loc[ep]
            assert episode["length"] == len(traj) and row.frame_index == len(traj) // 2
            prompt, = episode.tasks
            images = observations(root / arm, episode, row)
            states = state_inputs(row, traj.iloc[0]["observation.state"][:3], (x, y), yaw)
            states["drop"] = mean
            seed = 817117 + task
            noise = torch.randn(1, 50, 32, generator=torch.Generator().manual_seed(seed))
            outputs = {}
            for mode, state in states.items():
                policy.reset()
                batch = pre({**images, "observation.state": torch.from_numpy(state.copy()), "task": str(prompt)})
                if mode == "drop":
                    assert torch.count_nonzero(batch["observation.state"]) == 0
                assert policy.prepare_state(batch).shape == (1, 32)
                assert torch.count_nonzero(policy.prepare_state(batch)[:, 8:]) == 0
                action = post(policy.predict_action_chunk(batch, noise=noise.clone()))
                assert action.shape == (1, 50, 7) and torch.isfinite(action).all()
                outputs[mode] = action[0].numpy()
                if not rows and mode == "known":
                    policy.reset()
                    repeat = post(policy.predict_action_chunk(batch, noise=noise.clone()))
                    assert torch.equal(action, repeat)
            reference = outputs["known"]
            assert np.all(reference.std(axis=0) > 0)
            bounded = {m: a[:5].clip(-1, 1) for m, a in outputs.items()}
            target = states["joint_to_pose"]
            rows.append(dict(arm=arm, task=int(task), prompt=str(prompt), episode=ep, frame=int(row.frame_index),
                             noise_seed=seed, noise_sha=digest(noise), images={k: digest(v) for k,v in images.items()},
                             known_chunk_sha=digest(torch.from_numpy(reference)), known_first=reference[0].tolist(),
                             known_temporal_std=reference.std(axis=0).tolist(),
                             position_error_m=float(np.linalg.norm(states["known"][:3] - target[:3])),
                             rotation_vector_error_rad=float(np.linalg.norm(states["known"][3:6] - target[3:6])),
                             first5_bounded_rmse={m: rmse(a, bounded["known"]) for m,a in bounded.items()},
                             full50_unbounded_rmse={m: rmse(a, reference) for m,a in outputs.items()}))
            print(arm, task, rows[-1]["first5_bounded_rmse"], flush=True)
    return dict(weights_sha256=WEIGHT_SHA, params=sum(p.numel() for p in policy.parameters()), trainable=0,
                eval=True, device="cpu", dtype="float32", deterministic_equal_repeat=True, noise_shape=[1,50,32],
                action_shape=[1,50,7], flow_steps=policy.config.num_steps, fit_episodes=fit_episodes,
                n_fit_rows=len(y), fit_yaw=yaw, rows=rows)


def summarize(rows):
    metrics = pd.DataFrame([r["first5_bounded_rmse"] for r in rows])
    metrics["task"] = [r["task"] for r in rows]
    paired = metrics.groupby("task").mean()
    rng = np.random.default_rng(915)
    draws = rng.integers(len(paired), size=(10000, len(paired)))
    def interval(series):
        values = series.to_numpy()
        return [float(values.mean()), *np.quantile(values[draws].mean(axis=1), [0.025, 0.975]).tolist()]
    result = {name: interval(series) for name,series in paired.items()}
    for raw, fitted in [("raw_joint_slots", "joint_to_pose"), ("drop", "joint_to_pose"),
                        ("raw_fam_yaw90", "fit_fam_yaw90"), ("raw_cont_m45_075", "fit_cont_m45_075")]:
        result[f"{raw}_minus_{fitted}"] = interval(paired[raw] - paired[fitted])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(117)
    torch.set_num_threads(6)
    results = run(args.assets.resolve())
    results["task_mean_bootstrap95"] = summarize(results["rows"])
    args.out.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
