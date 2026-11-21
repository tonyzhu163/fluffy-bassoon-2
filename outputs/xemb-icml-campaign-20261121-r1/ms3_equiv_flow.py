"""ManiSkill3 SAPIEN EquivFlow-Procrustes (Direction-Blind Mask + Linear Homography + Procrustes).

Eliminates the cardinal-quantization bottleneck observed in `gpu03ms3v2` (`cont_ood=73.6 deg`)
by using a strictly direction-blind spatial attention mask over DIS dense optical flow +
linear 2x2 camera homography, evaluated zero-shot from 7-DoF Panda + Fetch to held-out
physical 6-DoF xarm6_robotiq across PickCube-v1, PushCube-v1, PullCube-v1, and StackCube-v1.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

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


def mask_features_pt(img_b: torch.Tensor, img_a: torch.Tensor, dis_flow: torch.Tensor) -> torch.Tensor:
    """Strictly direction-blind 7-channel pixel features: [||f||, |dt|, ||dRGB||, ||grad||, luma_b, x, y]."""
    b = img_b.float() / 255.0
    a = img_a.float() / 255.0
    diff = a - b
    lb = b.mean(dim=-1, keepdim=True)
    la = a.mean(dim=-1, keepdim=True)
    abs_dt = (la - lb).abs()
    d_rgb = torch.sqrt((diff * diff).mean(dim=-1, keepdim=True) + 1e-6)
    avg = 0.5 * (lb + la)
    gx = F.pad(0.5 * (avg[:, :, 2:, :] - avg[:, :, :-2, :]), (0, 0, 1, 1))
    gy = F.pad(0.5 * (avg[:, 2:, :, :] - avg[:, :-2, :, :]), (0, 0, 0, 0, 1, 1))
    gmag = torch.sqrt(gx * gx + gy * gy + 1e-6)
    fmag = torch.sqrt((dis_flow * dis_flow).sum(dim=-1, keepdim=True) + 1e-6)
    n, h, w, _ = b.shape
    ys = torch.linspace(-1.0, 1.0, h, device=b.device)
    xs = torch.linspace(-1.0, 1.0, w, device=b.device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(n, -1, -1, -1)
    x = torch.cat((fmag, abs_dt, d_rgb, gmag, lb, coords), dim=-1)
    return x.permute(0, 3, 1, 2).contiguous()


class EquivFlowHomographyPT(nn.Module):
    """PyTorch mirror of EquivFlow-Procrustes (zero cardinal quantization by construction)."""

    def __init__(self, width: int = 48, use_dz_parallax: bool = False):
        super().__init__()
        self.use_dz_parallax = use_dz_parallax
        self.c1 = nn.Conv2d(7, width, 3, stride=2, padding=1)
        self.c2 = nn.Conv2d(width, width, 3, stride=1, padding=1)
        self.c3 = nn.Conv2d(width, width, 3, stride=1, padding=1)
        self.mask_head = nn.Conv2d(width, 1, 1)
        self.conf_head = nn.Linear(width, 1)
        self.flow_mag_beta = nn.Parameter(torch.tensor(1.5))
        self.homography = nn.Parameter(torch.eye(2))
        self.uz = nn.Parameter(torch.zeros(2))
        self.scale_head = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
        self.grip_head = nn.Sequential(nn.Linear(4, 48), nn.SiLU(), nn.Linear(48, 48), nn.SiLU(), nn.Linear(48, 2))

    def forward(self, img_b, img_a, dis_flow, issued, deef, djaw, valid):
        bsz, k = valid.shape
        has_valid = valid.any(dim=1)
        deef_cm = 100.0 * deef
        djaw_cm = 100.0 * djaw

        # Scale + gripper heads (identical factored design)
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

        # Direction-blind mask CNN over 64x64 -> 32x32
        bk = bsz * k
        x = mask_features_pt(img_b.reshape(bk, 64, 64, 3), img_a.reshape(bk, 64, 64, 3), dis_flow.reshape(bk, 64, 64, 2))
        feat = F.silu(self.c1(x))
        feat = F.silu(self.c2(feat))
        feat = F.silu(feat + self.c3(feat))
        logits = self.mask_head(feat).squeeze(1)  # [BK, 32, 32]

        # 2x2 average-pooled flow on 32x32 grid
        flow_ds = F.avg_pool2d(dis_flow.reshape(bk, 64, 64, 2).permute(0, 3, 1, 2), kernel_size=2, stride=2).permute(0, 2, 3, 1)  # [BK, 32, 32, 2]
        fmag_ds = torch.sqrt((flow_ds * flow_ds).sum(dim=-1) + 1e-6)
        beta = F.softplus(self.flow_mag_beta)
        attn = torch.softmax((logits + beta * torch.log(fmag_ds + 1e-3)).reshape(bk, -1), dim=-1)  # [BK, 1024]
        flow_vec = torch.einsum("ns,nsc->nc", attn, flow_ds.reshape(bk, -1, 2)).reshape(bsz, k, 2)

        gfeat = torch.einsum("ns,nsc->nc", attn, feat.permute(0, 2, 3, 1).reshape(bk, -1, feat.shape[1]))
        w = torch.where(valid, F.softplus(self.conf_head(gfeat).reshape(bsz, k)) + 1e-4, torch.zeros_like(valid, dtype=torch.float32))

        # Exact linear 2x2 camera homography + optional normalized local_dz parallax
        v = torch.einsum("ij,bkj->bki", self.homography, flow_vec)
        if self.use_dz_parallax:
            # Scale local_dz in metres (~0.02) to match normalized flow_vec scale!
            dz_m = torch.where(valid, deef[..., 2], torch.zeros_like(deef[..., 2]))
            v = v - dz_m.unsqueeze(-1) * self.uz[None, None, :]

        ax, ay = issued[..., 0], issued[..., 1]
        vx, vy = v[..., 0], v[..., 1]
        dot = (w * (ax * vx + ay * vy)).sum(dim=1)
        cross = (w * (ax * vy - ay * vx)).sum(dim=1)
        norm = torch.sqrt(dot * dot + cross * cross + 1e-8)
        c = torch.where(has_valid, dot / norm, torch.ones_like(norm))
        s = torch.where(has_valid, cross / norm, torch.zeros_like(norm))
        return c, s, ls, gl


def train_homog(train_b: dict, eval_b: dict, *, seed: int, steps: int, device, use_dz_parallax: bool):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EquivFlowHomographyPT(width=48, use_dz_parallax=use_dz_parallax).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
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
    parser.add_argument("--steps", type=int, default=2500)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.monotonic()

    # Collect training probes on 7-DoF Panda + Fetch (12 seeds/task x 3 tasks x 2 robots x 7 emb = 504 trials)
    train_traces, train_meta = [], []
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

    # Collect held-out physical 6-DoF xarm6_robotiq probes (4 seeds/task x 3 tasks x 12 emb = 144 trials)
    eval_traces, eval_meta = [], []
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

    def summarize_offline(y_pred):
        out = {}
        for reg in (*EVAL_REGIMES, "ood_combined", "overall"):
            idx = list(range(len(eval_meta))) if reg == "overall" else ([i for i, m in enumerate(eval_meta) if m["regime"] in ("compositional_ood", "continuous_angle_ood")] if reg == "ood_combined" else [i for i, m in enumerate(eval_meta) if m["regime"] == reg])
            out[reg] = round(float(np.mean([circular_yaw_err(float(y_pred[i]), float(eval_b["yaw_deg"][i])) for i in idx])), 2)
        return out

    # Also test training on Panda-only vs Panda+Fetch to diagnose Fetch camera-view mismatch!
    panda_idx = np.array([i for i, m in enumerate(train_meta) if m["robot"] == "panda"], dtype=np.int64)
    train_panda = {k: v[panda_idx] for k, v in train_b.items()}

    offline = {}
    seed41 = {}
    configs = [
        ("equiv_flow_homog_nodz", train_b, False),
        ("equiv_flow_homog_dz_m", train_b, True),
        ("equiv_flow_homog_panda_only", train_panda, False),
    ]
    for tag, tb, use_dz in configs:
        per_seed = []
        for s in (41, 42, 43):
            (yh, sh, gh), pred_fn = train_homog(tb, eval_b, seed=s, steps=args.steps, device=device, use_dz_parallax=use_dz)
            per_seed.append(summarize_offline(yh))
            if s == 41:
                seed41[tag] = (yh, sh, gh)
                if tag == "equiv_flow_homog_nodz":
                    clr = dict(eval_b, valid=np.zeros_like(eval_b["valid"]))
                    seed41["ours_cleared"] = pred_fn(clr)
                    perm = np.roll(np.arange(len(eval_meta)), 1)
                    swp = {k: eval_b[k][perm] for k in ("img_before", "img_after", "dis_flow", "issued_mean", "delta_eef", "delta_jaw", "valid")}
                    seed41["ours_swapped_donor"] = pred_fn(swp)
        offline[tag] = {reg: round(float(np.mean([d[reg] for d in per_seed])), 2) for reg in (*EVAL_REGIMES, "ood_combined", "overall")}
        print(f"OFFLINE {tag:30s}: {offline[tag]}", flush=True)

    # Live closed-loop eval on held-out 6-DoF xarm6_robotiq (72 queries x 6 arms = 432 episodes)
    ARMS = (
        "privileged_oracle",
        "equiv_flow_homog_nodz",
        "equiv_flow_homog_dz_m",
        "equiv_flow_homog_panda_only",
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
                "equiv_flow_homog_nodz": (float(seed41["equiv_flow_homog_nodz"][0][i]), float(seed41["equiv_flow_homog_nodz"][1][i]), int(seed41["equiv_flow_homog_nodz"][2][i])),
                "equiv_flow_homog_dz_m": (float(seed41["equiv_flow_homog_dz_m"][0][i]), float(seed41["equiv_flow_homog_dz_m"][1][i]), int(seed41["equiv_flow_homog_dz_m"][2][i])),
                "equiv_flow_homog_panda_only": (float(seed41["equiv_flow_homog_panda_only"][0][i]), float(seed41["equiv_flow_homog_panda_only"][1][i]), int(seed41["equiv_flow_homog_panda_only"][2][i])),
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
    (args.out_dir / "ms3-equiv-flow-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("=== MANISKILL3 EQUIVFLOW SUMMARY ===", flush=True)
    print(json.dumps(dict(offline=offline, overall=by_regime["overall"], ood=by_regime["ood_combined"]), indent=2), flush=True)


if __name__ == "__main__":
    main()
