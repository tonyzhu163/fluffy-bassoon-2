# Phase 3: Held-Out Physical 6-DoF UR5e Cross-Morphology Visuomotor SysID (`libero-xemb-ur5e-sysid-20261120-r1`)

Registered plan, November 20, 2026. Evaluates zero-shot inference-time embodied system identification on a **held-out physical 6-DoF `UR5e`** arm after training readers strictly on physical **7-DoF `Panda` + `Sawyer`** single-factor conventions, using frozen JAX OpenPI $\pi_{0.5}$ and native LIBERO-Pro2.0 `BASE` rewards. Official 8-task test split (`SPLITS.json["test"]`) remains strictly untouched.

## Research Question and Counterexample Closure

Can a 90°-equivariant factored visuomotor history reader (`factored_equiv`, CoordConv visual motion vector + differentiable SO(2) Procrustes contraction) trained **only** on 7-DoF `Panda` + `Sawyer` single-factor conventions generalize zero-shot to an unseen 6-DoF `UR5e` under withheld multi-factor compositions and unseen continuous angles — in both `active_probe` and zero-probe `passive_attempt1` modes — while closing every prior counterexample simultaneously?

| Baseline / Counterexample | Why it fails on Held-Out 6-DoF `UR5e` |
| --- | --- |
| `frame0_probe` | Bitwise-identical frame-0 RGB and proprioception across all conventions on `UR5e` -> chance. |
| `analytic_proprio_sysid` | Local wrist displacement is yaw-invariant -> blind to yaw (`~90°`). |
| `analytic_joint_jacobian` (Phase 1 counterexample) | Source arms (`Panda`/`Sawyer`) are 7-DoF; target `UR5e` is 6-DoF (`7 != 6`, distinct kinematics, zero nominal `UR5e` traces). Both strict transfer (inapplicable), zero-padded transfer, and self-calibrated local Jacobian fail. |
| `analytic_visual_sysid` | Tested honestly on both `active_probe` and `passive_attempt1`; global brightness-constancy pixel flow degrades on natural closed-loop `passive_attempt1` trajectories. |
| `joint_transformer` | Memorizes training `(arm, single-factor)` tuples; collapses on withheld compositions and unseen continuous angles on `UR5e`. |
| `factored_no_equiv` | Ablating SO(2) Procrustes degrades continuous-angle OOD and natural `passive_attempt1` rollouts. |
| `ours_cleared` / `ours_swapped_donor` | Causal history controls proving live closed-loop rescue genuinely reads matched `UR5e` interaction history. |

## Splits and Simulator Interface

- **Simulator:** Reuses the qualified multi-arm MuJoCo wrapper [`outputs/libero-crossarm-20261120-r1/arm_env.py`](../libero-crossarm-20261120-r1/arm_env.py) (`Mounted*` / `OnTheGround*` `Panda`, `Sawyer`, `UR5e`, common `PandaGripper`, verified `112/112` admission in `cpuprobe01`).
- **Fitting tasks:** 8 tasks from `SPLITS.json["development_scaling"]["task_subsets"]["8"]` on `Panda` + `Sawyer` (7-DoF) ONLY (`UR5e` strictly excluded; single-factor `TRAIN_EMBODIMENTS` only).
- **Development tasks:** All 8 tasks from `SPLITS.json["development"]` on held-out `UR5e` (6-DoF), support reset `11` -> query reset `13`, `EVAL_EMBODIMENTS` (familiar + withheld compositional OOD + continuous-angle OOD).
- **Official test tasks:** Strictly untouched.

## Falsification Gates and Budget

1. **Gate 0 (CPU, 0 GPUh — `gate0_cpu_ur5e.py`):**
   - All 8 dev tasks on `UR5e` verify `joint_count == 6`, bitwise frame-0 equality across embodiments, yaw-invariance of local `delta_eef[:2]` (`< 1e-3 m`), non-trivial probe excitation (`sum_eef_mm > 15`, `max_jaw_mm > 10`, `mean_rgb_l1 > 1.0`), and `7 != 6` joint-Jacobian mismatch against 7-DoF `Panda`/`Sawyer`.
2. **Gate 1 (Baige Pro6K Distributed Training, ~0.4 GPUh — `gate1_ur5e.py`):**
   - 8 dev tasks × 6 embodiments (`fam_nom`, `fam_yaw90`, `comp_180_lo_inv`, `comp_m90_hi_inv`, `cont_45_nom`, `cont_135_075_inv`) = 48 paired trials (96 episodes) on held-out 6-DoF `UR5e`.
   - Go criteria: `shifted_oracle_rate >= 0.70` (`>= 28/40`), `overall_oracle_rate >= 0.70` (`>= 34/48`), `shifted_gain >= +0.50` (`>= +50 pp`), `fam_nom` nominal `>= 6/8`, and `no_base_regression`.
3. **Gate 2 (Baige Pro6K Distributed Training, ~1.8–2.5 GPUh, contingent on Gate 1):**
   - Full `Panda + Sawyer` training collection + `UR5e` dev collection (`active_probe` + `passive_attempt1`), 3-seed training `(41, 42, 43)`, complete Direction 3 offline audit, and multi-task live paired closed-loop evaluation on held-out 6-DoF `UR5e`.
- **Budget ceiling:** `<= 7.5` remaining GPU-hours within the `8.0` GPU-hour Phase 2+3 cap (`0.447` GPUh spent in Phase 2).
