"""Phase C: ManiSkill3 Cross-Morphology Visuomotor SysID (7-DoF Panda+Fetch -> held-out 6-DoF xarm6_robotiq).

Runs Gate 0 (frame-0 equality + proprio blindness + 7!=6 joint-dim mismatch),
Gate 1 (oracle vs nominal gap on held-out 6-DoF xarm6_robotiq),
collection on 7-DoF Panda+Fetch (train) and 6-DoF xarm6_robotiq (eval),
3-seed training of factored_equiv, factored_no_equiv, and joint_transformer (PyTorch),
and full Direction 3 offline + live closed-loop audit on 6-DoF xarm6_robotiq.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fluffy_bassoon.xemb.embodiment import EVAL_EMBODIMENTS, EVAL_REGIMES, TRAIN_EMBODIMENTS, EmbodimentSpec, get_embodiment_spec
from fluffy_bassoon.xemb.readers import (
    analytic_proprio_sysid,
    analytic_visual_yaw,
    fit_nominal_gain,
    fit_visual_projection,
    image_motion,
)

TASKS = ("PickCube-v1", "PushCube-v1", "PullCube-v1")
TRAIN_ROBOTS = ("panda", "fetch")
TARGET_ROBOT = "xarm6_robotiq"
PROBE_BURSTS = (
    np.array([0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    np.array([-0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    np.array([0.0, 0.75, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    np.array([0.0, -0.75, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.55, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
)
BURST_STEPS = 4


def rot2d(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def circular_yaw_err(pred_deg: float, true_deg: float) -> float:
    d = (pred_deg - true_deg + 180.0) % 360.0 - 180.0
    return abs(d)


def extract_rgb64(obs) -> np.ndarray:
    sd = obs.get("sensor_data", {})
    for cam in ("base_camera", "third_view_camera", "hand_camera"):
        if cam in sd and "rgb" in sd[cam]:
            img = sd[cam]["rgb"]
            if hasattr(img, "cpu"):
                img = img.cpu().numpy()
            if img.ndim == 4:
                img = img[0]
            return cv2.resize(img.astype(np.uint8), (64, 64), interpolation=cv2.INTER_AREA)
    raise RuntimeError(f"No RGB camera in obs sensor_data keys={list(sd.keys())}")


def get_eef_pos(env) -> np.ndarray:
    p = env.unwrapped.agent.tcp.pose.p
    if hasattr(p, "cpu"):
        p = p.cpu().numpy()
    p = np.asarray(p, dtype=np.float32)
    return p[0] if p.ndim == 2 else p


def get_jaw_width(env) -> float:
    q = env.unwrapped.agent.robot.get_qpos()
    if hasattr(q, "cpu"):
        q = q.cpu().numpy()
    q = np.asarray(q, dtype=np.float32).ravel()
    # Last 2 qpos entries reflect finger opening across panda/fetch/xarm6_robotiq
    return float(np.mean(np.abs(q[-2:])))


def format_env_action(env, cmd7: np.ndarray) -> np.ndarray:
    act_dim = int(env.action_space.shape[-1])
    if act_dim == 7:
        return np.clip(cmd7, -1.0, 1.0).astype(np.float32)
    # Fetch has 12 dims: arm(6) + gripper(1) + body/head/base(5) -> keep base/head static
    out = np.zeros(act_dim, dtype=np.float32)
    out[:7] = np.clip(cmd7, -1.0, 1.0)
    return out


def apply_embodiment_and_adapter(
    cmd7: np.ndarray,
    spec: EmbodimentSpec,
    yaw_hat: float = 0.0,
    scale_hat: float = 1.0,
    grip_hat: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply inverse adapter (yaw_hat, scale_hat, grip_hat) to produce issued cmd, then hidden physical embodiment."""
    issued = cmd7.copy()
    inv_r = rot2d(-yaw_hat)
    issued[:2] = (inv_r @ issued[:2]) / max(float(scale_hat), 0.25)
    issued[6] = float(grip_hat) * issued[6]
    issued = np.clip(issued, -1.0, 1.0)

    phys = issued.copy()
    fwd_r = rot2d(spec.yaw_deg)
    phys[:2] = float(spec.xy_scale) * (fwd_r @ phys[:2])
    phys[6] = float(spec.gripper_sign) * phys[6]
    phys = np.clip(phys, -1.0, 1.0)
    return issued, phys


def frozen_task_policy(env, task_id: str, step_idx: int) -> np.ndarray:
    """100% frozen canonical pd_ee_delta_pose controller shared across Panda, Fetch, and xArm6."""
    unw = env.unwrapped
    eef = get_eef_pos(env)
    cmd = np.zeros(7, dtype=np.float32)

    def _np3(x):
        if hasattr(x, "cpu"):
            x = x.cpu().numpy()
        x = np.asarray(x, dtype=np.float32)
        return x[0] if x.ndim == 2 else x

    obj = _np3(unw.obj.pose.p) if hasattr(unw, "obj") else _np3(unw.cube.pose.p)
    goal = _np3(unw.goal_site.pose.p) if hasattr(unw, "goal_site") else _np3(unw.goal_region.pose.p)

    if task_id == "PickCube-v1":
        if step_idx < 18:
            target = obj + np.array([0.0, 0.0, 0.06], dtype=np.float32)
            grip = 1.0  # open
        elif step_idx < 34:
            target = obj + np.array([0.0, 0.0, 0.005], dtype=np.float32)
            grip = 1.0
        elif step_idx < 46:
            target = obj + np.array([0.0, 0.0, 0.005], dtype=np.float32)
            grip = -1.0  # close
        else:
            target = goal
            grip = -1.0
        delta = np.clip((target - eef) * 14.0, -0.9, 0.9)
        cmd[:3] = delta
        cmd[6] = grip
    elif task_id in ("PushCube-v1", "PullCube-v1"):
        direction = goal[:2] - obj[:2]
        norm = float(np.linalg.norm(direction)) + 1e-6
        u = direction / norm
        behind = np.array([obj[0] - 0.075 * u[0], obj[1] - 0.075 * u[1], obj[2] + 0.015], dtype=np.float32)
        if step_idx < 20:
            target = behind + np.array([0.0, 0.0, 0.06], dtype=np.float32)
        elif step_idx < 34:
            target = behind
        else:
            target = np.array([goal[0], goal[1], obj[2] + 0.015], dtype=np.float32)
        delta = np.clip((target - eef) * 14.0, -0.9, 0.9)
        cmd[:3] = delta
        cmd[6] = -1.0  # closed finger for pushing/pulling
    return cmd


def collect_probe_trace(env, spec: EmbodimentSpec, seed: int):
    obs, _ = env.reset(seed=seed)
    cur_img = extract_rgb64(obs)
    cur_pos = get_eef_pos(env)
    cur_jaw = get_jaw_width(env)
    f0_sha = hashlib.sha256(np.ascontiguousarray(cur_img).tobytes()).hexdigest()

    K = len(PROBE_BURSTS)
    img_b = np.zeros((K, 64, 64, 3), dtype=np.uint8)
    img_a = np.zeros((K, 64, 64, 3), dtype=np.uint8)
    iss_m = np.zeros((K, 7), dtype=np.float32)
    deef_local = np.zeros((K, 3), dtype=np.float32)
    djaw = np.zeros((K, 2), dtype=np.float32)
    valid = np.ones((K,), dtype=bool)
    inv_r = rot2d(-spec.yaw_deg)

    for k, burst in enumerate(PROBE_BURSTS):
        issued, phys = apply_embodiment_and_adapter(burst, spec)
        env_act = format_env_action(env, phys)
        for _ in range(BURST_STEPS):
            obs, _, _, _, _ = env.step(env_act)
        nxt_img = extract_rgb64(obs)
        nxt_pos = get_eef_pos(env)
        nxt_jaw = get_jaw_width(env)

        dp_world = nxt_pos - cur_pos
        dp_loc = dp_world.copy()
        dp_loc[:2] = inv_r @ dp_world[:2]

        img_b[k] = cur_img
        img_a[k] = nxt_img
        iss_m[k] = issued
        deef_local[k] = dp_loc
        djaw[k] = np.array([nxt_jaw - cur_jaw, nxt_jaw], dtype=np.float32)
        cur_img, cur_pos, cur_jaw = nxt_img, nxt_pos, nxt_jaw

    return dict(
        img_before=img_b,
        img_after=img_a,
        issued_mean=iss_m,
        delta_eef=deef_local,
        delta_jaw=djaw,
        valid=valid,
        frame0_sha256=f0_sha,
    )


def run_live_episode(env, task_id: str, spec: EmbodimentSpec, seed: int, yaw_hat=0.0, scale_hat=1.0, grip_hat=1, horizon=90) -> bool:
    env.reset(seed=seed)
    success = False
    for t in range(horizon):
        cmd = frozen_task_policy(env, task_id, t)
        _, phys = apply_embodiment_and_adapter(cmd, spec, yaw_hat=yaw_hat, scale_hat=scale_hat, grip_hat=grip_hat)
        _, _, term, trunc, info = env.step(format_env_action(env, phys))
        if bool(info.get("success", False)):
            success = True
            break
        if bool(term) or bool(trunc):
            break
    return success


class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 8, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width // 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.pool = nn.Linear(width, 1)

    def forward(self, x):
        h = self.net(x)  # [N, C, H, W]
        n, c, gh, gw = h.shape
        flat = h.permute(0, 2, 3, 1).reshape(n, gh * gw, c)
        w = torch.softmax(self.pool(flat), dim=1)
        return (w * flat).sum(dim=1)


class MS3Reader(nn.Module):
    """PyTorch mirror of our JAX readers: factored_equiv, factored_no_equiv, joint_transformer."""

    def __init__(self, model_type: str, width: int = 96):
        super().__init__()
        self.model_type = model_type
        self.cnn = SmallCNN(8, width)
        if model_type == "joint_transformer":
            self.in_proj = nn.Linear(width + 12, 128)
            enc_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True, activation="gelu")
            self.tx = nn.TransformerEncoder(enc_layer, num_layers=2)
            self.head = nn.Linear(128, 4)
        else:
            in_d = width if model_type == "factored_equiv" else width + 2
            self.vis_head = nn.Sequential(nn.Linear(in_d, 128), nn.SiLU(), nn.Linear(128, 3))
            self.scale_head = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 2))
            self.grip_head = nn.Sequential(nn.Linear(4, 48), nn.SiLU(), nn.Linear(48, 2))

    def _vis_feat(self, img_b, img_a, valid):
        b, k, h, w, _ = img_b.shape
        ys = torch.linspace(-1.0, 1.0, h, device=img_b.device)
        xs = torch.linspace(-1.0, 1.0, w, device=img_b.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack((xx, yy), dim=-1)[None, None].expand(b, k, h, w, 2)
        m = valid[:, :, None, None, None].float()
        ib = (img_b.float() / 255.0) * m
        ia = (img_a.float() / 255.0) * m
        x = torch.cat((ia - ib, ib, coords), dim=-1).permute(0, 1, 4, 2, 3).reshape(b * k, 8, h, w)
        return self.cnn(x).reshape(b, k, -1)

    def forward(self, batch):
        valid = batch["valid"]
        v = self._vis_feat(batch["img_before"], batch["img_after"], valid)
        iss = batch["issued_mean"]
        deef = batch["delta_eef"]
        djaw = batch["delta_jaw"]
        has_valid = valid.any(dim=1)
        w = valid.float()

        if self.model_type == "joint_transformer":
            tok = torch.cat((v, iss, deef, djaw), dim=-1)
            h = self.tx(self.in_proj(tok))
            pooled = (h * w.unsqueeze(-1)).sum(dim=1) / w.sum(dim=1, keepdim=True).clamp(min=1.0)
            out = self.head(pooled)
            c, s = out[:, 0], out[:, 1]
            norm = torch.sqrt(c * c + s * s + 1e-8)
            cos_h = torch.where(has_valid, c / norm, torch.ones_like(c))
            sin_h = torch.where(has_valid, s / norm, torch.zeros_like(s))
            log_s = torch.where(has_valid, out[:, 2].clamp(-1.5, 1.5), torch.zeros_like(c))
            grip_l = torch.where(has_valid, out[:, 3], torch.zeros_like(c))
            return cos_h, sin_h, log_s, grip_l

        if self.model_type == "factored_equiv":
            out = self.vis_head(v)
            vec = out[..., :2]
            wt = F.softplus(out[..., 2]) * w
            ax, ay = iss[..., 0], iss[..., 1]
            vx, vy = vec[..., 0], vec[..., 1]
            dot = (wt * (ax * vx + ay * vy)).sum(dim=-1)
            cross = (wt * (ax * vy - ay * vx)).sum(dim=-1)
            norm = torch.sqrt(dot * dot + cross * cross + 1e-8)
            cos_h = torch.where(has_valid, dot / norm, torch.ones_like(dot))
            sin_h = torch.where(has_valid, cross / norm, torch.zeros_like(dot))
        else:
            out = self.vis_head(torch.cat((v, iss[..., :2]), dim=-1))
            wt = F.softplus(out[..., 2]) * w
            c = (wt * out[..., 0]).sum(dim=-1)
            s = (wt * out[..., 1]).sum(dim=-1)
            norm = torch.sqrt(c * c + s * s + 1e-8)
            cos_h = torch.where(has_valid, c / norm, torch.ones_like(c))
            sin_h = torch.where(has_valid, s / norm, torch.zeros_like(c))

        # Factored scale & grip heads
        s_in = torch.cat((iss[..., :2], deef), dim=-1)
        s_out = self.scale_head(s_in)
        sw = F.softplus(s_out[..., 1]) * w
        log_s = (sw * s_out[..., 0]).sum(dim=-1) / sw.sum(dim=-1).clamp(min=1e-6)
        log_s = torch.where(has_valid, log_s.clamp(-1.5, 1.5), torch.zeros_like(log_s))

        g_in = torch.cat((iss[..., 6:7], djaw, iss[..., 6:7] * djaw[..., 0:1]), dim=-1)
        g_out = self.grip_head(g_in)
        gw = F.softplus(g_out[..., 1]) * w
        grip_l = (gw * g_out[..., 0]).sum(dim=-1) / gw.sum(dim=-1).clamp(min=1e-6)
        grip_l = torch.where(has_valid, grip_l, torch.zeros_like(grip_l))
        return cos_h, sin_h, log_s, grip_l


def train_reader(model_type: str, train_batch: dict, seed: int, steps: int = 1200, batch_size: int = 32, device: str = "cuda"):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MS3Reader(model_type).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-5)
    n = len(train_batch["yaw_deg"])
    rng = np.random.default_rng(seed)

    tensors = {k: torch.as_tensor(v, device=device) for k, v in train_batch.items()}
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(n, size=batch_size), device=device)
        b = {k: v[idx] for k, v in tensors.items()}
        cos_h, sin_h, log_s, grip_l = model(b)
        rad = torch.deg2rad(b["yaw_deg"])
        yaw_loss = ((cos_h - torch.cos(rad)) ** 2 + (sin_h - torch.sin(rad)) ** 2).mean()
        scale_loss = ((log_s - torch.log(b["xy_scale"])) ** 2).mean()
        grip_t = (b["gripper_sign"] > 0).float()
        grip_loss = F.binary_cross_entropy_with_logits(grip_l, grip_t)
        loss = yaw_loss + scale_loss + grip_loss
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
    model.eval()
    return model


@torch.no_grad()
def predict_reader(model: MS3Reader, batch: dict, device: str = "cuda"):
    b = {k: torch.as_tensor(v, device=device) for k, v in batch.items() if k in ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid")}
    cos_h, sin_h, log_s, grip_l = model(b)
    yaw = torch.rad2deg(torch.atan2(sin_h, cos_h)).cpu().numpy()
    scale = torch.exp(log_s).cpu().numpy()
    grip = np.where(grip_l.cpu().numpy() >= 0.0, 1, -1).astype(np.int32)
    return yaw, scale, grip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.monotonic()
    # 1. Collect training traces strictly on 7-DoF Panda + Fetch (3 tasks x 2 robots x 12 seeds x 7 train embodiments = 504 trials)
    train_lists = {k: [] for k in ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid", "yaw_deg", "xy_scale", "gripper_sign")}
    for task in TASKS:
        for robot in TRAIN_ROBOTS:
            env = gym.make(task, num_envs=1, obs_mode="rgbd", control_mode="pd_ee_delta_pose", robot_uids=robot, render_mode="rgb_array", sim_backend="cpu")
            for seed in range(12):
                for spec in TRAIN_EMBODIMENTS:
                    tr = collect_probe_trace(env, spec, seed=100 * TASKS.index(task) + seed)
                    for k in ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid"):
                        train_lists[k].append(tr[k])
                    train_lists["yaw_deg"].append(np.float32(spec.yaw_deg))
                    train_lists["xy_scale"].append(np.float32(spec.xy_scale))
                    train_lists["gripper_sign"].append(np.int32(spec.gripper_sign))
            env.close()
    train_batch = {k: np.stack(v, axis=0) for k, v in train_lists.items()}
    print(f"COLLECTED TRAIN (7-DoF Panda+Fetch): {len(train_batch['yaw_deg'])} trials in {time.monotonic()-t0:.1f}s", flush=True)

    # Fit analytic visual projection & nominal gain strictly on 7-DoF train traces
    nom_m = np.isclose(train_batch["xy_scale"], 1.0) & np.isclose(train_batch["yaw_deg"], 0.0)
    nom_gain = fit_nominal_gain(train_batch["issued_mean"][nom_m], train_batch["delta_eef"][nom_m], train_batch["valid"][nom_m])
    tr_mot = image_motion(train_batch["img_before"][nom_m], train_batch["img_after"][nom_m], train_batch["valid"][nom_m])
    vis_proj = fit_visual_projection(tr_mot, train_batch["delta_eef"][nom_m], train_batch["valid"][nom_m])

    # 2. Collect held-out 6-DoF xarm6_robotiq eval traces (3 tasks x 4 seeds x 12 eval embodiments = 144 trials) + Gate 0 check
    eval_lists = {k: [] for k in ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid", "yaw_deg", "xy_scale", "gripper_sign")}
    eval_meta = []
    frame0_ok = True
    target_qpos_dim = None
    for task in TASKS:
        env = gym.make(task, num_envs=1, obs_mode="rgbd", control_mode="pd_ee_delta_pose", robot_uids=TARGET_ROBOT, render_mode="rgb_array", sim_backend="cpu")
        for seed in (100, 101, 102, 103):
            f0_ref = None
            for spec in EVAL_EMBODIMENTS:
                tr = collect_probe_trace(env, spec, seed=500 * TASKS.index(task) + seed)
                if target_qpos_dim is None:
                    q = env.unwrapped.agent.robot.get_qpos()
                    target_qpos_dim = int(q.shape[-1])
                if f0_ref is None:
                    f0_ref = tr["frame0_sha256"]
                elif tr["frame0_sha256"] != f0_ref:
                    frame0_ok = False
                for k in ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid"):
                    eval_lists[k].append(tr[k])
                eval_lists["yaw_deg"].append(np.float32(spec.yaw_deg))
                eval_lists["xy_scale"].append(np.float32(spec.xy_scale))
                eval_lists["gripper_sign"].append(np.int32(spec.gripper_sign))
                eval_meta.append(dict(task=task, seed=500 * TASKS.index(task) + seed, emb=spec.name, regime=spec.regime))
        env.close()
    eval_batch = {k: np.stack(v, axis=0) for k, v in eval_lists.items()}
    print(f"COLLECTED HELD-OUT 6-DoF {TARGET_ROBOT}: {len(eval_meta)} trials (f0_eq={frame0_ok}, qpos_dim={target_qpos_dim})", flush=True)

    # Analytic baselines on held-out 6-DoF xarm6_robotiq
    yp_arr, sp_arr, gp_arr = analytic_proprio_sysid(eval_batch["issued_mean"], eval_batch["delta_eef"], eval_batch["delta_jaw"], eval_batch["valid"], nominal_gain_m_per_cmd=nom_gain)
    ev_mot = image_motion(eval_batch["img_before"], eval_batch["img_after"], eval_batch["valid"])
    yv_arr = analytic_visual_yaw(eval_batch["issued_mean"], ev_mot, eval_batch["delta_eef"][..., 2], eval_batch["valid"], vis_proj)

    def summarize_offline(y_pred):
        out = {}
        for reg in (*EVAL_REGIMES, "overall"):
            idx = [i for i, m in enumerate(eval_meta) if reg == "overall" or m["regime"] == reg]
            errs = [circular_yaw_err(float(y_pred[i]), float(eval_batch["yaw_deg"][i])) for i in idx]
            out[reg] = round(float(np.mean(errs)), 2)
        return out

    offline = {
        "analytic_proprio_sysid": summarize_offline(yp_arr),
        "analytic_visual_sysid": summarize_offline(yv_arr),
    }

    # 3. Train 3 seeds of each neural architecture strictly on 7-DoF Panda+Fetch and audit on 6-DoF xarm6_robotiq
    seed41_preds = {}
    for mtype in ("joint_transformer", "factored_no_equiv", "factored_equiv"):
        per_seed = []
        for s in (41, 42, 43):
            model = train_reader(mtype, train_batch, seed=s, steps=args.steps, device=device)
            yh, sh, gh = predict_reader(model, eval_batch, device=device)
            per_seed.append(summarize_offline(yh))
            if s == 41:
                seed41_preds[mtype] = (yh, sh, gh)
                if mtype == "factored_equiv":
                    clr = dict(eval_batch, valid=np.zeros_like(eval_batch["valid"]))
                    seed41_preds["ours_cleared"] = predict_reader(model, clr, device=device)
                    perm = np.roll(np.arange(len(eval_meta)), 1)
                    swp = {k: eval_batch[k][perm] for k in ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid")}
                    seed41_preds["ours_swapped_donor"] = predict_reader(model, swp, device=device)
        offline[mtype] = {reg: round(float(np.mean([d[reg] for d in per_seed])), 2) for reg in (*EVAL_REGIMES, "overall")}
        print(f"OFFLINE {mtype:20s}: {offline[mtype]}", flush=True)

    # 4. Live closed-loop evaluation on held-out physical 6-DoF xarm6_robotiq (first 2 seeds per task x 12 emb = 72 queries x 9 arms = 648 episodes)
    ARMS = (
        "privileged_oracle",
        "analytic_visual_sysid",
        "ours_factored_equiv",
        "factored_no_equiv",
        "joint_transformer",
        "analytic_proprio_sysid",
        "nominal",
        "ours_cleared",
        "ours_swapped_donor",
    )
    live_rows = []
    for task in TASKS:
        env = gym.make(task, num_envs=1, obs_mode="state", control_mode="pd_ee_delta_pose", robot_uids=TARGET_ROBOT, sim_backend="cpu")
        for i, m in enumerate(eval_meta):
            if m["task"] != task or (m["seed"] % 500) >= 102:
                continue
            spec = get_embodiment_spec(m["emb"])
            q_seed = m["seed"] + 50
            preds = {
                "privileged_oracle": (spec.yaw_deg, spec.xy_scale, spec.gripper_sign),
                "analytic_visual_sysid": (float(yv_arr[i]), float(sp_arr[i]), int(gp_arr[i])),
                "ours_factored_equiv": (float(seed41_preds["factored_equiv"][0][i]), float(seed41_preds["factored_equiv"][1][i]), int(seed41_preds["factored_equiv"][2][i])),
                "factored_no_equiv": (float(seed41_preds["factored_no_equiv"][0][i]), float(seed41_preds["factored_no_equiv"][1][i]), int(seed41_preds["factored_no_equiv"][2][i])),
                "joint_transformer": (float(seed41_preds["joint_transformer"][0][i]), float(seed41_preds["joint_transformer"][1][i]), int(seed41_preds["joint_transformer"][2][i])),
                "analytic_proprio_sysid": (float(yp_arr[i]), float(sp_arr[i]), int(gp_arr[i])),
                "nominal": (0.0, 1.0, 1),
                "ours_cleared": (float(seed41_preds["ours_cleared"][0][i]), float(seed41_preds["ours_cleared"][1][i]), int(seed41_preds["ours_cleared"][2][i])),
                "ours_swapped_donor": (float(seed41_preds["ours_swapped_donor"][0][i]), float(seed41_preds["ours_swapped_donor"][1][i]), int(seed41_preds["ours_swapped_donor"][2][i])),
            }
            succ = {}
            for a in ARMS:
                yh, sh, gh = preds[a]
                succ[a] = int(run_live_episode(env, task, spec, seed=q_seed, yaw_hat=yh, scale_hat=sh, grip_hat=gh))
            live_rows.append(dict(task=task, seed=q_seed, emb=spec.name, regime=spec.regime, succ=succ))
        env.close()

    by_regime = {}
    for reg in (*EVAL_REGIMES, "ood_combined", "overall"):
        if reg == "overall":
            sub = live_rows
        elif reg == "ood_combined":
            sub = [r for r in live_rows if r["regime"] in ("compositional_ood", "continuous_angle_ood")]
        else:
            sub = [r for r in live_rows if r["regime"] == reg]
        n = len(sub)
        sc = {a: sum(r["succ"][a] for r in sub) for a in ARMS}
        by_regime[reg] = dict(trials=n, successes=sc, rates={a: round(sc[a] / max(n, 1), 4) for a in ARMS})

    summary = dict(
        simulator="ManiSkill3-SAPIEN",
        train_robots=list(TRAIN_ROBOTS),
        target_robot=TARGET_ROBOT,
        target_qpos_dim=target_qpos_dim,
        tasks=list(TASKS),
        gate0_frame0_bitwise_equal=frame0_ok,
        offline_yaw_err_deg=offline,
        live_by_regime=by_regime,
        seconds=round(time.monotonic() - t0, 1),
    )
    (args.out_dir / "ms3-xemb-summary.json").write_text(json.dumps(summary, indent=2))
    print("=== MANISKILL3 FINAL SUMMARY ===", flush=True)
    print(json.dumps(dict(offline=offline, overall=by_regime["overall"], ood=by_regime["ood_combined"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
