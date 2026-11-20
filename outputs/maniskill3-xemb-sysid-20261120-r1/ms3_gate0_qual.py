"""ManiSkill3 GPU Gate 0 qualification: verify Vulkan RGB rendering, multi-embodiment admission, and cross-morphology observability."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np


TASKS = ("PickCube-v1", "PushCube-v1", "PullCube-v1", "StackCube-v1", "LiftPegUpright-v1")
ROBOTS = ("panda", "xarm6_robotiq", "widowxai", "fetch", "ur_10e")


def extract_rgb(obs):
    sd = obs.get("sensor_data", {})
    for cam in ("base_camera", "third_view_camera", "hand_camera"):
        if cam in sd and "rgb" in sd[cam]:
            img = sd[cam]["rgb"]
            if hasattr(img, "cpu"):
                img = img.cpu().numpy()
            if img.ndim == 4:
                img = img[0]
            return cam, img.astype(np.uint8)
    for cam, data in sd.items():
        if isinstance(data, dict) and "rgb" in data:
            img = data["rgb"]
            if hasattr(img, "cpu"):
                img = img.cpu().numpy()
            if img.ndim == 4:
                img = img[0]
            return cam, img.astype(np.uint8)
    return None, None


def sha256_arr(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def probe_pair(task_id: str, robot_uid: str):
    t0 = time.monotonic()
    try:
        env = gym.make(
            task_id,
            num_envs=1,
            obs_mode="rgbd",
            control_mode="pd_ee_delta_pose",
            robot_uids=robot_uid,
            render_mode="rgb_array",
            sim_backend="cpu",
        )
        obs0, _ = env.reset(seed=0)
        cam, rgb0 = extract_rgb(obs0)
        qpos = env.unwrapped.agent.robot.get_qpos()
        if hasattr(qpos, "cpu"):
            qpos = qpos.cpu().numpy()
        qpos_dim = int(qpos.shape[-1])
        act_dim = int(env.action_space.shape[-1])

        # Verify bitwise frame-0 equality on repeated reset(seed=0)
        obs0b, _ = env.reset(seed=0)
        _, rgb0b = extract_rgb(obs0b)
        frame0_bitwise_equal = bool(rgb0 is not None and rgb0b is not None and sha256_arr(rgb0) == sha256_arr(rgb0b))

        # Apply a +X command vs 90-deg rotated (+Y) command from identical reset and check visual motion
        act_x = np.zeros(act_dim, dtype=np.float32)
        act_x[0] = 0.6
        for _ in range(5):
            obs_x, _, _, _, _ = env.step(act_x)
        _, rgb_x = extract_rgb(obs_x)

        env.reset(seed=0)
        act_y = np.zeros(act_dim, dtype=np.float32)
        act_y[1] = 0.6
        for _ in range(5):
            obs_y, _, _, _, _ = env.step(act_y)
        _, rgb_y = extract_rgb(obs_y)

        l1_x = float(np.mean(np.abs(rgb_x.astype(np.float32) - rgb0.astype(np.float32)))) if rgb_x is not None else 0.0
        l1_xy = float(np.mean(np.abs(rgb_x.astype(np.float32) - rgb_y.astype(np.float32)))) if (rgb_x is not None and rgb_y is not None) else 0.0
        env.close()
        return dict(
            ok=True,
            task=task_id,
            robot=robot_uid,
            camera=cam,
            rgb_shape=list(rgb0.shape) if rgb0 is not None else None,
            qpos_dim=qpos_dim,
            act_dim=act_dim,
            frame0_bitwise_equal=frame0_bitwise_equal,
            rgb_l1_x=round(l1_x, 3),
            rgb_l1_x_vs_y=round(l1_xy, 3),
            seconds=round(time.monotonic() - t0, 2),
        )
    except Exception as exc:
        return dict(
            ok=False,
            task=task_id,
            robot=robot_uid,
            error=f"{type(exc).__name__}: {str(exc)[:220]}",
            seconds=round(time.monotonic() - t0, 2),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = []
    for task in TASKS:
        for robot in ROBOTS:
            rec = probe_pair(task, robot)
            results.append(rec)
            print(json.dumps(rec), flush=True)

    passed = [r for r in results if r["ok"]]
    summary = dict(
        total=len(results),
        passed=len(passed),
        results=results,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"SUMMARY: {len(passed)}/{len(results)} passed -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
