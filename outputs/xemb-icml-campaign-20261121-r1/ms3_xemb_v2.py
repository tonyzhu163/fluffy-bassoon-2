"""Stage 1 ManiSkill3 V2 Gate: DIS-flow + local_dz parallax + Perceiver-ICBA + ReGround/Broyden/DIS baselines.

Evaluates 7-DoF Panda + Fetch -> held-out physical 6-DoF xarm6_robotiq (plus probes ur_10e / widowx250s
compatibility) across PickCube-v1, PushCube-v1, PullCube-v1 in SAPIEN / PhysX, comparing:
- privileged_oracle, nominal, ours_cleared, ours_swapped_donor
- analytic_proprio_sysid, reground_babbling
- analytic_visual_hs, dense_flow_dis_procrustes, broyden_uvs_dis
- joint_transformer, perceiver_icba, factored_no_equiv_v2
- factored_equiv_v2 (dis_lumaflow + local_dz parallax cancellation)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import cv2
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MS3_V1_DIR = Path(__file__).resolve().parents[1] / "maniskill3-xemb-sysid-20261120-r1"
if str(MS3_V1_DIR) not in sys.path:
    sys.path.insert(0, str(MS3_V1_DIR))

from fluffy_bassoon.xemb.baselines import (
    broyden_uvs_yaw,
    compute_dis_flow_maps,
    dense_flow_image_motion,
    fit_dense_flow_projection,
    reground_babbling_sysid,
)
from fluffy_bassoon.xemb.embodiment import EVAL_EMBODIMENTS, EVAL_REGIMES, TRAIN_EMBODIMENTS, get_embodiment_spec
from fluffy_bassoon.xemb.readers import (
    analytic_proprio_sysid,
    analytic_visual_yaw,
    fit_nominal_gain,
    fit_visual_projection,
    image_motion,
)
from ms3_xemb_exp import (
    TARGET_ROBOT,
    TASKS,
    TRAIN_ROBOTS,
    circular_yaw_err,
    collect_probe_trace,
    run_live_episode,
)


def vis_tensors_v2(img_b: torch.Tensor, img_a: torch.Tensor, dis_flow: torch.Tensor) -> torch.Tensor:
    """Build 9-channel appearance-invariant [fx, fy, ||f||, dt, ||dRGB||, gx, gy, x, y] tensor [N, 9, H, W]."""
    b = img_b.float() / 255.0
    a = img_a.float() / 255.0
    diff = a - b
    lb = b.mean(dim=-1, keepdim=True)
    la = a.mean(dim=-1, keepdim=True)
    dt = la - lb
    mag = torch.sqrt((diff * diff).mean(dim=-1, keepdim=True) + 1e-6)
    avg = 0.5 * (lb + la)
    gx = F.pad(0.5 * (avg[:, :, 2:, :] - avg[:, :, :-2, :]), (0, 0, 1, 1))
    gy = F.pad(0.5 * (avg[:, 2:, :, :] - avg[:, :-2, :, :]), (0, 0, 0, 0, 1, 1))
    fmag = torch.sqrt((dis_flow * dis_flow).sum(dim=-1, keepdim=True) + 1e-6)
    n, h, w, _ = b.shape
    ys = torch.linspace(-1.0, 1.0, h, device=b.device)
    xs = torch.linspace(-1.0, 1.0, w, device=b.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(n, -1, -1, -1)
    x = torch.cat((dis_flow, fmag, dt, mag, gx, gy, coords), dim=-1)
    return x.permute(0, 3, 1, 2).contiguous()


class ResAttnCNN(nn.Module):
    def __init__(self, in_ch: int = 9, width: int = 96):
        super().__init__()
        w1 = max(24, width // 2)
        self.c1 = nn.Conv2d(in_ch, w1, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(w1, width, 3, stride=2, padding=1)
        self.r1 = nn.Conv2d(width, width, 3, stride=1, padding=1)
        self.r2 = nn.Conv2d(width, width, 3, stride=1, padding=1)
        self.c3 = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.pool = nn.Linear(width, 2)
        self.proj = nn.Linear(2 * width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.c1(x))
        h = F.silu(self.c2(h))
        res = self.r2(F.silu(self.r1(h)))
        h = F.silu(h + res)
        h = F.silu(self.c3(h))
        n, c, gh, gw = h.shape
        flat = h.permute(0, 2, 3, 1).reshape(n, gh * gw, c)
        w = torch.softmax(self.pool(flat), dim=1)  # [N, HW, 2]
        pooled = torch.einsum("nsh,nsc->nhc", w, flat).reshape(n, 2 * c)
        return F.silu(self.proj(pooled))


class MS3ReaderV2(nn.Module):
    def __init__(self, model_type: str, width: int = 96, mlp_w: int = 192, use_dz_parallax: bool = True):
        super().__init__()
        self.model_type = model_type
        self.use_dz_parallax = use_dz_parallax
        self.cnn = ResAttnCNN(9, width)
        self.scale_head = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
        self.grip_head = nn.Sequential(nn.Linear(4, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 2))

        if model_type == "factored_equiv_v2":
            out_d = 5 if use_dz_parallax else 3
            self.vis_vec = nn.Sequential(nn.Linear(width, mlp_w), nn.SiLU(), nn.Linear(mlp_w, mlp_w // 2), nn.SiLU(), nn.Linear(mlp_w // 2, out_d))
            self.global_uz = nn.Parameter(torch.zeros(2))
        elif model_type == "factored_no_equiv_v2":
            self.vis_head = nn.Sequential(nn.Linear(width + 3, mlp_w), nn.SiLU(), nn.Linear(mlp_w, mlp_w // 2), nn.SiLU(), nn.Linear(mlp_w // 2, 3))
        elif model_type == "joint_transformer":
            self.tok_in = nn.Linear(width + 12, 192)
            enc = nn.TransformerEncoderLayer(d_model=192, nhead=4, dim_feedforward=512, activation="gelu", batch_first=True)
            self.tx = nn.TransformerEncoder(enc, num_layers=3)
            self.out_head = nn.Linear(192, 4)
        elif model_type == "perceiver_icba":
            self.tok_in = nn.Linear(width + 12, 192)
            self.latents = nn.Parameter(torch.randn(1, 8, 192) * 0.02)
            self.cross_attn = nn.MultiheadAttention(192, 4, batch_first=True)
            self.cross_ln_q = nn.LayerNorm(192)
            self.cross_ln_kv = nn.LayerNorm(192)
            self.cross_ff = nn.Sequential(nn.LayerNorm(192), nn.Linear(192, 512), nn.SiLU(), nn.Linear(512, 192))
            enc = nn.TransformerEncoderLayer(d_model=192, nhead=4, dim_feedforward=512, activation="gelu", batch_first=True)
            self.self_tx = nn.TransformerEncoder(enc, num_layers=3)
            self.out_head = nn.Linear(192, 4)
        else:
            raise ValueError(model_type)

    def forward(self, img_b, img_a, dis_flow, issued, deef, djaw, valid):
        bsz, k = valid.shape
        has_valid = valid.any(dim=1)
        x = vis_tensors_v2(img_b.reshape(bsz * k, 64, 64, 3), img_a.reshape(bsz * k, 64, 64, 3), dis_flow.reshape(bsz * k, 64, 64, 2))
        vis = self.cnn(x).reshape(bsz, k, -1)
        deef_cm = 100.0 * deef
        djaw_cm = 100.0 * djaw
        mask_f = valid.unsqueeze(-1).float()

        if self.model_type in ("joint_transformer", "perceiver_icba"):
            tok = torch.cat((vis, issued, deef_cm, djaw_cm), dim=-1) * mask_f
            h = self.tok_in(tok)
            if self.model_type == "joint_transformer":
                h = self.tx(h) * mask_f
                pooled = h.sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
            else:
                z = self.latents.expand(bsz, -1, -1)
                kpm = ~valid
                all_masked = ~has_valid
                safe_kpm = torch.where(all_masked.unsqueeze(1), torch.zeros_like(kpm), kpm)
                att, _ = self.cross_attn(self.cross_ln_q(z), self.cross_ln_kv(h), self.cross_ln_kv(h), key_padding_mask=safe_kpm)
                z = z + torch.where(has_valid[:, None, None], att, torch.zeros_like(att))
                z = z + self.cross_ff(z)
                z = self.self_tx(z)
                pooled = z.mean(dim=1)
            raw = self.out_head(pooled)
            norm = torch.sqrt(raw[:, 0] ** 2 + raw[:, 1] ** 2 + 1e-8)
            c = torch.where(has_valid, raw[:, 0] / norm, torch.ones_like(norm))
            s = torch.where(has_valid, raw[:, 1] / norm, torch.zeros_like(norm))
            ls = torch.where(has_valid, raw[:, 2], torch.zeros_like(norm))
            gl = torch.where(has_valid, raw[:, 3], torch.full_like(norm, 10.0))
            return c, s, ls, gl

        # Factored scale + grip heads
        iss_xyz = issued[..., :3]
        deef_xyz = deef_cm[..., :3]
        sf = torch.stack((iss_xyz, deef_xyz, iss_xyz.abs(), deef_xyz.abs(), torch.log(deef_xyz.abs() + 1e-4) - torch.log(iss_xyz.abs() + 1e-4)), dim=-1)
        so = self.scale_head(sf)
        sw = torch.where(valid.unsqueeze(-1), F.softplus(so[..., 1]) + 1e-4, torch.zeros_like(so[..., 1]))
        ls = torch.where(has_valid, (sw * so[..., 0]).sum(dim=(1, 2)) / sw.sum(dim=(1, 2)).clamp(min=1e-6), torch.zeros(bsz, device=issued.device))

        ig = issued[..., 6:7]
        gf = torch.cat((ig, djaw_cm, -ig * djaw_cm[..., 0:1]), dim=-1)
        go = self.grip_head(gf)
        gw = torch.where(valid, F.softplus(go[..., 1]) + 1e-4, torch.zeros_like(go[..., 1]))
        gl = torch.where(has_valid, (gw * go[..., 0]).sum(dim=1) / gw.sum(dim=1).clamp(min=1e-6), torch.full((bsz,), 10.0, device=issued.device))

        if self.model_type == "factored_equiv_v2":
            vo = self.vis_vec(vis)
            v_raw = vo[..., :2]
            w = torch.where(valid, F.softplus(vo[..., 2]) + 1e-4, torch.zeros_like(vo[..., 2]))
            if self.use_dz_parallax:
                dz = torch.where(valid, deef_cm[..., 2], torch.zeros_like(deef_cm[..., 2]))
                uz = vo[..., 3:5] + self.global_uz[None, None, :]
                v = v_raw - dz.unsqueeze(-1) * uz
            else:
                v = v_raw
            ax, ay = issued[..., 0], issued[..., 1]
            vx, vy = v[..., 0], v[..., 1]
            dot = (w * (ax * vx + ay * vy)).sum(dim=1)
            cross = (w * (ax * vy - ay * vx)).sum(dim=1)
        else:
            vo = self.vis_head(torch.cat((vis, issued[..., :2], deef_cm[..., 2:3]), dim=-1))
            w = torch.where(valid, F.softplus(vo[..., 2]) + 1e-4, torch.zeros_like(vo[..., 2]))
            dot = (w * vo[..., 0]).sum(dim=1)
            cross = (w * vo[..., 1]).sum(dim=1)

        norm = torch.sqrt(dot * dot + cross * cross + 1e-8)
        c = torch.where(has_valid, dot / norm, torch.ones_like(norm))
        s = torch.where(has_valid, cross / norm, torch.zeros_like(norm))
        return c, s, ls, gl


def train_and_predict(model_type: str, train_b: dict, eval_b: dict, *, seed: int, steps: int, device, use_dz_parallax: bool = True):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MS3ReaderV2(model_type, use_dz_parallax=use_dz_parallax).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-5)

    n = len(train_b["yaw_deg"])
    t_all = {k: torch.as_tensor(v, device=device) for k, v in train_b.items()}
    rad_all = torch.deg2rad(t_all["yaw_deg"])
    ct_all, st_all = torch.cos(rad_all), torch.sin(rad_all)
    ls_all = torch.log(t_all["xy_scale"])
    gt_all = (t_all["gripper_sign"] > 0).float()

    model.train()
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(n, size=64), device=device)
        c, s, ls, gl = model(
            t_all["img_before"][idx], t_all["img_after"][idx], t_all["dis_flow"][idx],
            t_all["issued_mean"][idx], t_all["delta_eef"][idx], t_all["delta_jaw"][idx], t_all["valid"][idx],
        )
        loss = (1.0 - (c * ct_all[idx] + s * st_all[idx])).mean() + F.mse_loss(ls, ls_all[idx]) + F.binary_cross_entropy_with_logits(gl, gt_all[idx])
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

    @torch.no_grad()
    def _pred(batch):
        model.eval()
        c, s, ls, gl = model(
            torch.as_tensor(batch["img_before"], device=device),
            torch.as_tensor(batch["img_after"], device=device),
            torch.as_tensor(batch["dis_flow"], device=device),
            torch.as_tensor(batch["issued_mean"], device=device),
            torch.as_tensor(batch["delta_eef"], device=device),
            torch.as_tensor(batch["delta_jaw"], device=device),
            torch.as_tensor(batch["valid"], device=device),
        )
        y = np.degrees(np.arctan2(s.cpu().numpy(), c.cpu().numpy())).astype(np.float32)
        sc = np.exp(np.clip(ls.cpu().numpy(), math.log(0.25), math.log(4.0))).astype(np.float32)
        gr = np.where(gl.cpu().numpy() >= 0.0, 1, -1).astype(np.int32)
        return y, sc, gr

    return _pred(eval_b), _pred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.monotonic()

    # 1. Collect training probes on 7-DoF Panda + Fetch (12 seeds/task x 3 tasks x 2 robots x 7 emb = 504 trials)
    train_traces = []
    train_meta = []
    for robot in TRAIN_ROBOTS:
        for task in TASKS:
            env = gym.make(task, num_envs=1, obs_mode="rgbd", control_mode="pd_ee_delta_pose", robot_uids=robot, sim_backend="cpu")
            for s_idx in range(12):
                seed = 1000 * TRAIN_ROBOTS.index(robot) + 100 * TASKS.index(task) + s_idx
                for spec in TRAIN_EMBODIMENTS:
                    tr = collect_probe_trace(env, spec, seed)
                    tr["yaw_deg"] = np.float32(spec.yaw_deg)
                    tr["xy_scale"] = np.float32(spec.xy_scale)
                    tr["gripper_sign"] = np.int32(spec.gripper_sign)
                    train_traces.append(tr)
                    train_meta.append(dict(robot=robot, task=task, seed=seed, emb=spec.name, regime=spec.regime))
            env.close()

    # 2. Collect held-out physical 6-DoF xarm6_robotiq probes (4 seeds/task x 3 tasks x 12 emb = 144 trials)
    eval_traces = []
    eval_meta = []
    f0_ok = True
    for task in TASKS:
        env = gym.make(task, num_envs=1, obs_mode="rgbd", control_mode="pd_ee_delta_pose", robot_uids=TARGET_ROBOT, sim_backend="cpu")
        for s_idx in range(4):
            seed = 500 * TASKS.index(task) + 100 + s_idx
            shas = set()
            for spec in EVAL_EMBODIMENTS:
                tr = collect_probe_trace(env, spec, seed)
                tr["yaw_deg"] = np.float32(spec.yaw_deg)
                tr["xy_scale"] = np.float32(spec.xy_scale)
                tr["gripper_sign"] = np.int32(spec.gripper_sign)
                shas.add(tr["frame0_sha256"])
                eval_traces.append(tr)
                eval_meta.append(dict(robot=TARGET_ROBOT, task=task, seed=seed, emb=spec.name, regime=spec.regime))
            if len(shas) != 1:
                f0_ok = False
        env.close()

    def stack(traces):
        keys = ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid", "yaw_deg", "xy_scale", "gripper_sign")
        d = {k: np.stack([t[k] for t in traces], axis=0) for k in keys}
        d["dis_flow"] = compute_dis_flow_maps(d["img_before"], d["img_after"], d["valid"])
        return d

    train_b = stack(train_traces)
    eval_b = stack(eval_traces)
    print(f"Collected MS3 train={len(train_traces)} eval={len(eval_traces)} f0_eq={f0_ok} ({time.monotonic()-t0:.1f}s)", flush=True)

    nom_idx = np.array([i for i, m in enumerate(train_meta) if m["emb"] == "nom"], dtype=np.int64)
    nom_gain = fit_nominal_gain(train_b["issued_mean"][nom_idx], train_b["delta_eef"][nom_idx], train_b["valid"][nom_idx])

    # Analytic + competitive baselines
    yp, sp, gp = analytic_proprio_sysid(eval_b["issued_mean"], eval_b["delta_eef"], eval_b["delta_jaw"], eval_b["valid"], nominal_gain_m_per_cmd=nom_gain)
    yr, sr, gr = reground_babbling_sysid(eval_b["issued_mean"], eval_b["delta_eef"], eval_b["delta_jaw"], eval_b["valid"], nominal_gain_m_per_cmd=nom_gain)
    hs_proj = fit_visual_projection(image_motion(train_b["img_before"][nom_idx], train_b["img_after"][nom_idx], train_b["valid"][nom_idx]), train_b["delta_eef"][nom_idx], train_b["valid"][nom_idx])
    ev_hs = image_motion(eval_b["img_before"], eval_b["img_after"], eval_b["valid"])
    y_hs = analytic_visual_yaw(eval_b["issued_mean"], ev_hs, eval_b["delta_eef"][..., 2], eval_b["valid"], hs_proj)

    dis_proj = fit_dense_flow_projection(train_b["img_before"][nom_idx], train_b["img_after"][nom_idx], train_b["delta_eef"][nom_idx], train_b["valid"][nom_idx], method="dis")
    ev_dis = dense_flow_image_motion(eval_b["img_before"], eval_b["img_after"], eval_b["valid"], method="dis")
    y_dis = analytic_visual_yaw(eval_b["issued_mean"], ev_dis, eval_b["delta_eef"][..., 2], eval_b["valid"], dis_proj)
    y_broy = broyden_uvs_yaw(eval_b["issued_mean"], ev_dis, eval_b["delta_eef"][..., 2], eval_b["valid"], dis_proj)

    def summarize_offline(y_pred):
        out = {}
        for reg in (*EVAL_REGIMES, "ood_combined", "overall"):
            if reg == "overall":
                idx = list(range(len(eval_meta)))
            elif reg == "ood_combined":
                idx = [i for i, m in enumerate(eval_meta) if m["regime"] in ("compositional_ood", "continuous_angle_ood")]
            else:
                idx = [i for i, m in enumerate(eval_meta) if m["regime"] == reg]
            out[reg] = round(float(np.mean([circular_yaw_err(float(y_pred[i]), float(eval_b["yaw_deg"][i])) for i in idx])), 2)
        return out

    offline = {
        "analytic_proprio_sysid": summarize_offline(yp),
        "reground_babbling": summarize_offline(yr),
        "analytic_visual_hs": summarize_offline(y_hs),
        "dense_flow_dis_procrustes": summarize_offline(y_dis),
        "broyden_uvs_dis": summarize_offline(y_broy),
    }
    for k, v in offline.items():
        print(f"OFFLINE {k:26s}: {v}", flush=True)

    # Train 3 seeds of V2 neural models
    seed41 = {}
    specs = [
        ("factored_equiv_v2_dz", "factored_equiv_v2", True),
        ("factored_equiv_v2_nodz", "factored_equiv_v2", False),
        ("factored_no_equiv_v2", "factored_no_equiv_v2", True),
        ("joint_transformer", "joint_transformer", True),
        ("perceiver_icba", "perceiver_icba", True),
    ]
    for tag, mtype, use_dz in specs:
        per_seed = []
        for s in (41, 42, 43):
            (yh, sh, gh), pred_fn = train_and_predict(mtype, train_b, eval_b, seed=s, steps=args.steps, device=device, use_dz_parallax=use_dz)
            per_seed.append(summarize_offline(yh))
            if s == 41:
                seed41[tag] = (yh, sh, gh)
                if tag == "factored_equiv_v2_dz":
                    clr = dict(eval_b, valid=np.zeros_like(eval_b["valid"]))
                    seed41["ours_cleared"] = pred_fn(clr)
                    perm = np.roll(np.arange(len(eval_meta)), 1)
                    swp = {k: eval_b[k][perm] for k in ("img_before", "img_after", "dis_flow", "issued_mean", "delta_eef", "delta_jaw", "valid")}
                    seed41["ours_swapped_donor"] = pred_fn(swp)
        offline[tag] = {reg: round(float(np.mean([d[reg] for d in per_seed])), 2) for reg in (*EVAL_REGIMES, "ood_combined", "overall")}
        print(f"OFFLINE {tag:26s}: {offline[tag]}", flush=True)

    # Live closed-loop eval on held-out 6-DoF xarm6_robotiq (72 queries x 12 arms = 864 episodes)
    ARMS = (
        "privileged_oracle",
        "ours_equiv_v2_dz",
        "ours_equiv_v2_nodz",
        "dense_flow_dis_procrustes",
        "broyden_uvs_dis",
        "analytic_visual_hs",
        "factored_no_equiv_v2",
        "perceiver_icba",
        "joint_transformer",
        "reground_babbling",
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
                "ours_equiv_v2_dz": (float(seed41["factored_equiv_v2_dz"][0][i]), float(seed41["factored_equiv_v2_dz"][1][i]), int(seed41["factored_equiv_v2_dz"][2][i])),
                "ours_equiv_v2_nodz": (float(seed41["factored_equiv_v2_nodz"][0][i]), float(seed41["factored_equiv_v2_nodz"][1][i]), int(seed41["factored_equiv_v2_nodz"][2][i])),
                "dense_flow_dis_procrustes": (float(y_dis[i]), float(sp[i]), int(gp[i])),
                "broyden_uvs_dis": (float(y_broy[i]), float(sp[i]), int(gp[i])),
                "analytic_visual_hs": (float(y_hs[i]), float(sp[i]), int(gp[i])),
                "factored_no_equiv_v2": (float(seed41["factored_no_equiv_v2"][0][i]), float(seed41["factored_no_equiv_v2"][1][i]), int(seed41["factored_no_equiv_v2"][2][i])),
                "perceiver_icba": (float(seed41["perceiver_icba"][0][i]), float(seed41["perceiver_icba"][1][i]), int(seed41["perceiver_icba"][2][i])),
                "joint_transformer": (float(seed41["joint_transformer"][0][i]), float(seed41["joint_transformer"][1][i]), int(seed41["joint_transformer"][2][i])),
                "reground_babbling": (float(yr[i]), float(sr[i]), int(gr[i])),
                "nominal": (0.0, 1.0, 1),
                "ours_cleared": (float(seed41["ours_cleared"][0][i]), float(seed41["ours_cleared"][1][i]), int(seed41["ours_cleared"][2][i])),
                "ours_swapped_donor": (float(seed41["ours_swapped_donor"][0][i]), float(seed41["ours_swapped_donor"][1][i]), int(seed41["ours_swapped_donor"][2][i])),
            }
            succ = {a: int(run_live_episode(env, task, spec, seed=q_seed, yaw_hat=preds[a][0], scale_hat=preds[a][1], grip_hat=preds[a][2])) for a in ARMS}
            live_rows.append(dict(task=task, seed=q_seed, emb=spec.name, regime=spec.regime, succ=succ))
        env.close()

    by_regime = {}
    for reg in (*EVAL_REGIMES, "ood_combined", "overall"):
        sub = live_rows if reg == "overall" else ([r for r in live_rows if r["regime"] in ("compositional_ood", "continuous_angle_ood")] if reg == "ood_combined" else [r for r in live_rows if r["regime"] == reg])
        n = len(sub)
        sc = {a: sum(r["succ"][a] for r in sub) for a in ARMS}
        by_regime[reg] = dict(trials=n, successes=sc, rates={a: round(sc[a] / max(n, 1), 4) for a in ARMS})

    summary = dict(
        simulator="ManiSkill3-SAPIEN",
        target_robot=TARGET_ROBOT,
        gate0_frame0_bitwise_equal=f0_ok,
        offline_yaw_err_deg=offline,
        live_by_regime=by_regime,
        seconds=round(time.monotonic() - t0, 1),
    )
    (args.out_dir / "ms3-v2-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("=== MANISKILL3 V2 SUMMARY ===", flush=True)
    print(json.dumps(dict(offline=offline, overall=by_regime["overall"], ood=by_regime["ood_combined"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
