# In-Context Visuomotor Embodied System Identification on Frozen π0.5 (`libero-xemb-sysid-20261119-r1`)

Registered plan, November 19, 2026. Workstream combining **Direction 1 (In-Context Embodied System Identification for Frozen VLAs)** and **Direction 3 (Diagnostic Audit: Memorization vs. True Latent ID)** on top of frozen JAX OpenPI π0.5 and LIBERO-Pro2.0 `BASE` tasks.

## Hypotheses and Falsification Criteria

1. **Frame-0 Unidentifiability (`Frame0Probe`):**
   - Because local proprioception is anchored at `t = 0` (`to_local_proprio`), frame-0 RGB and robot state are bitwise identical across all embodiments. A frame-0 probe must perform at chance on all three factors (mean circular yaw error ~90°, scale relative error at prior mean, gripper polarity accuracy ~50%).
2. **Proprioceptive Blindness (`AnalyticProprioSysID`):**
   - Local proprioception shares the robot's uncalibrated base frame, so `delta_eef_local` is invariant to hand-eye yaw $\theta$. Analytic proprioceptive SysID recovers `xy_scale` and `gripper_sign` accurately, while remaining blind to $\theta$ (~90° error).
3. **Embodiment Memorization vs. Compositional & Continuous Generalization:**
   - A monolithic black-box history encoder (`JointVisuoMotorTransformer`) fits familiar single-factor training embodiments, but collapses on **Compositional OOD** (unseen combinations of familiar factor values) and **Continuous Angle OOD** (unseen oblique yaw angles).
   - A channel-factorized non-equivariant reader (`FactoredNoEquiv`) improves Compositional OOD on familiar discrete angles, while degrading on unseen continuous angles.
   - Our `FactoredEquivVisuoMotor` reader (differentiable SO(2) Procrustes contraction between CoordConv visual-delta motion vectors and issued XY commands, plus factored per-axis scale and gripper heads) generalizes across **Familiar**, **Compositional OOD**, and **Continuous Angle OOD** across 3 fitting seeds `(41, 42, 43)` with `< 15°` yaw error and `> 90%` gripper accuracy, trained strictly from public interaction history without privileged world-frame labels.
4. **Live Closed-Loop Native Realizability (Gate 1) & Adaptation (Gate 2):**
   - **Gate 1:** On 8 held-out development tasks × 6 representative embodiments (48 paired trials), `privileged_oracle` must achieve `>= 75%` overall native success and `>= +50 pp` over `nominal` frozen π0.5 without BASE regression.
   - **Gate 2 (Live POC):** On all 8 held-out development tasks × 12 evaluation embodiments (96 paired trials, reset `init_state_id=13`), `FactoredEquivVisuoMotor` (`ours_retained_active`, `ours_retained_passive`, and `ours_online_prefix`) must outperform `nominal`, `analytic_proprio_sysid`, `joint_transformer`, `factored_no_equiv`, `ours_cleared`, and `ours_swapped_donor` on native task success.

## Embodiment Splits

- **Training (7 single-factor + nominal embodiments; zero multi-factor combinations, zero oblique angles):**
  - `nom`: `(0.0°, 1.0, +1)`
  - `yaw90`: `(90.0°, 1.0, +1)`
  - `yaw180`: `(180.0°, 1.0, +1)`
  - `yawm90`: `(-90.0°, 1.0, +1)`
  - `scale_lo`: `(0.0°, 0.65, +1)`
  - `scale_hi`: `(0.0°, 1.55, +1)`
  - `grip_inv`: `(0.0°, 1.0, -1)`
- **Evaluation (12 embodiments across 3 regimes on 8 held-out development tasks; official 8 test tasks untouched):**
  - `familiar` (4): `(0°, 1.0, +1)`, `(90°, 1.0, +1)`, `(180°, 1.0, +1)`, `(0°, 1.0, -1)`
  - `compositional_ood` (4): `(180°, 0.65, -1)`, `(-90°, 1.55, -1)`, `(90°, 1.55, -1)`, `(-90°, 0.65, -1)`
  - `continuous_angle_ood` (4): `(45°, 1.0, +1)`, `(135°, 0.75, -1)`, `(-135°, 1.35, -1)`, `(-45°, 0.75, +1)`

## Task Split and Budget

- **Fitting tasks:** 8-task nested subset from `outputs/libero-meta-learning-20261117-r1/SPLITS.json` (`development_scaling.task_subsets["8"]`), 4 resets `(0, 1, 2, 3)` × 7 embodiments = 224 training trials.
- **Development evaluation tasks:** 8 held-out development tasks from `SPLITS.json["development"]`, support reset `11` (distinct from query reset `13`) × 12 embodiments = 96 development trials.
- **Official test tasks:** Untouched in this POC.
- **Budget ceiling:** `<= 12 GPU-hours` on Baige Distributed Training (1× NVIDIA RTX PRO 6000D `RPBZZZ6`).
