# ICML Cross-Embodiment Visuomotor SysID Campaign Plan

**Run ID:** `xemb-icml-campaign-20261121-r1`  
**Target Venue:** ICML (extended abstract due in ~2 weeks)  
**Approved Compute Envelope:** `500–1,000` GPU-hours over ~7 days (staged with hard go/no-go gates; target spend `~300–500` GPUh on Baige `NVIDIA RPBZZZ6` Pro6K).

---

## 1. Literature Review, Novelty Audit & Baseline Check

### Direct Competing Papers (Verified Nov 2026)

| Prior Work | Base Policy | Morphology Shift | Target Data / Adaptation Mechanism | Key Limitation vs. Ours |
| --- | --- | --- | --- | --- |
| **Yang, Finn & Sadigh** (*Cross-Controller Transfer via In-Context Behavior Adaptation*, Oct 2026, `arXiv:2610.39280v2`) | Jointly retrained $\pi_{0.5}$ | Same arm (WidowX) | Monolithic DINOv2 + Perceiver encoding 3 target expert demos (`128` transitions) | Requires VLA retraining + target expert demos; monolithic sequence encoder memorizes factor grids & collapses on compositional/continuous OOD (`84.4% → 3.1%`). *(Note: fix citation from "Sherry Yang 2025" to "Jonathan Yang, Chelsea Finn, Dorsa Sadigh 2026"!)* |
| **ReGround** (*Siddique, Li & Cao*, Nov 2026, `arXiv:2611.20244`, SPAIS @ CoRL) | **100% Frozen VLA** ($\pi_{0.5}$ on LIBERO) | Same arm actuation shift (gain, remounting, latency) | Task-agnostic **motor babbling (`1,000` transitions)** to fit target dynamics model + source-body rollout tracking | **Direct competitor!** Needs `1,000` target transitions + per-target optimization, and relies on proprioceptive state mapping — which is **structurally yaw-blind** under uncalibrated rotated local frames (`90°` error) and fails across joint-DoF changes (`7 != 6`). |
| **PACE** (*Liu et al.*, Oct 2026, `arXiv:2610.19141`) | Retrained collective policy | Multi-arm (5 source → UR5e/KUKA) | Two-stage predictive dynamics latent on Cartesian states | Requires training base policy from scratch; proprioceptive/Cartesian state space only. |
| **Mirage** (RSS 2024) & **RoVi-Aug** (CoRL 2024) | Frozen / co-trained policy | Visual cross-arm transfer | Cross-painting / ControlNet using **known target URDF + calibrated camera extrinsics** | Assumes known target kinematics, known URDF, and calibrated extrinsics; cannot identify unknown command-frame rotation, gain, or inverted gripper. |
| **Align-Then-Steer / ATE** (ICLR 2026) & **X-VLA** | Pretrained VLA | Cross-embodiment | VAE latent alignment / soft prompts via **target demonstrations + fine-tuning** | Requires target-robot demonstrations and gradient updates; not zero-shot self-practice. |
| **Classical Visual Servoing / Optical Flow + Procrustes** (Horn–Schunck, Farneback/DIS, RAFT + Broyden UVS / Orthogonal Procrustes) | **100% Frozen VLA** | Cross-morphology visual motion | Hand-crafted or pretrained dense optical flow + source-fitted camera projection + Procrustes / Broyden Jacobian | **Strongest non-learned competitor!** Competitive on clean fixed cameras (`65.6%` on LIBERO `BASE`), **but collapses** under oblique/textured cameras (`1.4%` on ManiSkill3 SAPIEN), uncalibrated camera shifts (`LIBERO-Pro2.0 S15/S16`, `LIBERO-Plus` viewpoints), clutter/lighting/noise (`S13/S14/S19`), and 3D $SO(3)$ motion. |

### Honest Novelty Position (What We Can & Cannot Claim)
1. **What we DO NOT claim:** We do not claim to be the first to wrap a frozen VLA under actuation shift (`ReGround` does this via 1,000-step proprioceptive babbling), nor the first to condition a VLA on controller history (`Yang, Finn & Sadigh`).
2. **What IS genuinely novel and high-impact for ICML:**
   - **Empirical & Theoretical Diagnosis of Monolithic In-Context Adaptation:** Proving (with formal propositions + a full embodiment-diversity & architecture scaling law $M \in \{7, 16, 32, 64, 128, \text{cont}\}$ across Transformers and Perceiver-IO) *why* monolithic sequence encoders memorize discrete factor grids and suffer exponential sample-complexity penalties compared to group-equivariant factorization.
   - **Equivariant Visual-Correspondence SysID (`EqProcrustes-V2`):** Combining **action-blind visual motion extraction** (with `local_dz` vertical parallax cancellation and appearance-invariant / pretrained flow-attention features), **differentiable $SO(2)/SO(3)$ weighted Procrustes alignment**, and **online $O(1)$ streaming sufficient-statistic updates** — enabling zero-shot, zero-weight-update adaptation of 100% frozen VLAs (`pi0.5` AND `pi0`) across unseen morphologies (`7-DoF → 6-DoF`), uncalibrated/perturbed cameras (`LIBERO-Pro2.0` `S15`/`S16`/`S13`/`S19` & `LIBERO-Plus`), and independent physics engines (`MuJoCo` + `SAPIEN`), decisively beating **both** monolithic Transformers/Perceivers **and** classical optical flow / RAFT+Procrustes / ReGround babbling in closed loop.

---

## 2. Competitive Benchmark Suite & Mandatory Baselines

### Competitive Benchmarks (Headline Tables)
1. **Benchmark 1 — LIBERO-Pro2.0 Cross-Morphology + Official Perturbation Stress-Test (MuJoCo, 2 Frozen VLAs: $\pi_{0.5}$ & $\pi_0$):**
   - **1A (Multi-Morphology Zero-Shot Transfer):** Trained strictly on 7-DoF `Panda` + `Sawyer` (24 tasks) → zero-shot test on **3+ held-out physical morphologies** (`6-DoF UR5e`, `7-DoF IIWA`, `7-DoF Kinova3`/`Jaco`) across the 8 locked test tasks (`Spatial`, `Object`, `Goal`, `Long/10`) under **$SO(2)$** and **full $SO(3)$ + anisotropic gain + jaw polarity**.
   - **1B (Real Competitive Visual/Camera Perturbations):** Cross-morphology `UR5e` evaluated *simultaneously* under **official `LIBERO-Pro2.0` perturbations** (`S15` camera extrinsics, `S16` camera intrinsics/FOV, `S13`/`S14`/`S22` appearance/lighting, `S19` sensor blur/noise) + **`LIBERO-Plus`** viewpoint/lighting/noise variants — directly proving where classical fixed-projection optical flow breaks and learned equivariant visual SysID dominates.
   - **1C (Policy-Agnostic Check):** Exact same frozen reader wrapping **$\pi_0$ (`pi0_libero`)** alongside **$\pi_{0.5}$ (`pi05_libero`)**.
2. **Benchmark 2 — ManiSkill3 Cross-Morphology (SAPIEN / PhysX):**
   - Trained strictly on 7-DoF `Panda` + `Fetch` across **≥5 tasks** (`PickCube-v1`, `PushCube-v1`, `PullCube-v1`, `StackCube-v1`, `PokeCube-v1`) → zero-shot closed-loop evaluation on **≥2 held-out physical morphologies** (`6-DoF xArm6-Robotiq` + `6-DoF UR10e` / `WidowX-250s`).
3. **Benchmark 3 (Gate-0 Qualified) — RoboCasa365 Multi-Camera Kitchens (`Human300 π0.5`):**
   - Room-scale kitchens with realistic distractors & multiple camera views under unseen embodiment/command-frame shifts with frozen `pi05_pretrain_human300`.

### Mandatory Baseline Matrix (Zero Strawmen)
1. `nominal` (unadapted frozen VLA) & `privileged_oracle` (exact inverse upper bound).
2. `analytic_proprio_sysid` (proves yaw blindness).
3. `reground_babbling` (Siddique et al. Nov 2026 style target motor babbling at matched `15` controls AND privileged `1,000` controls).
4. `analytic_visual_sysid` (Horn–Schunck brightness constancy + Procrustes, active & passive).
5. `dense_flow_procrustes` (OpenCV DIS / `torchvision` RAFT-Small + source camera projection + Procrustes).
6. `broyden_uvs` (online Broyden uncalibrated visual-motor Jacobian estimation).
7. `joint_transformer` & `perceiver_icba` (Yang, Finn & Sadigh–style Perceiver with frozen SigLIP2/CNN features, scaled across $M \in \{7, 16, 32, 64, 128, \text{cont}\}$).
8. `factored_no_equiv` (isolates exact Procrustes group equivariance).
9. Causal controls: `frame0_probe`, `ours_cleared`, `ours_swapped_donor`, active vs. passive vs. **online streaming (Attempt 1)**.

---

## 3. Staged Execution Schedule & Go/No-Go Gates (~350–500 GPUh Total)

- **Stage 0 (CPU Gate 0, `0 GPUh`, Tonight):** Qualify `IIWA`/`Kinova3`/`Jaco` on LIBERO, `S15`/`S16`/`S13`/`S19` with `UR5e`, `pi0_libero` loading, `raft_small`/DIS availability, ManiSkill3 `ur_10e`/`widowx250s` + 5 tasks, and RoboCasa365 feasibility.
- **Stage 1 (Method V2 Dev Gate, `~20 GPUh`):** Build & test `factored_equiv_v2` (`local_dz` parallax cancellation + multi-scale/DIS-flow or pretrained visual motion features + online streaming Procrustes) + all new baselines (`reground_babbling`, `dense_flow_procrustes`, `broyden_uvs`, `perceiver_icba`) on LIBERO `UR5e` dev (`96` queries) and ManiSkill3 `xArm6`.
  - **Go/No-Go Gate 1:** `factored_equiv_v2` must **beat `analytic_visual_sysid_passive` (`65.6%`)** on LIBERO (`>= 72%` dev) AND reach **`>= 50%` (`< 12°` err)** on ManiSkill3 `xArm6` before scaling!
- **Stage 2 (Headline Multi-Benchmark + Multi-Morphology + Multi-VLA + `LIBERO-Pro2.0` Stress + $SO(3)$, `~180–240 GPUh`):**
  - Full locked LIBERO test across `UR5e`, `IIWA`, `Kinova3` × $\pi_{0.5}$ & $\pi_0$, clean + `S15/S16/S13/S19` camera/visual shifts + full $SO(3)$ + anisotropic gain.
  - Full ManiSkill3 multi-robot (`xArm6`, `UR10e`/`WidowX`) × 5 tasks.
  - RoboCasa365 kitchen cross-embodiment transfer (if Stage 0 passes).
- **Stage 3 (Targeted Ablations & Scaling Laws, `~120–180 GPUh`, ONLY after Headline Locks):**
  - Embodiment-diversity scaling law ($M \in \{7, 16, 32, 64, 128, \text{cont}\}$) for `joint_transformer` & `perceiver_icba` vs. $M \in \{2, 4, 7\}$ for `factored_equiv_v2` (3 seeds).
  - Component ablation (`local_dz`, visual features, streaming chunks $K$, active vs. passive vs. online).
  - Complete causal audit (`frame0`, `cleared`, `swapped`) + 3-seed task-clustered 95% bootstrap CIs.
