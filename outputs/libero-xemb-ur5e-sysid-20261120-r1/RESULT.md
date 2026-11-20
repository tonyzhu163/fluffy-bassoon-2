# Held-Out 6-DoF UR5e Cross-Morphology Visuomotor SysID — Phase 3 Baseline POC

**Run ID:** `libero-xemb-ur5e-sysid-20261120-r1`  
**Status:** Complete (`exit 0` across CPU admission, Gate 0, Pro6K Gate 1, Stage B1 collect+train, and Stage B2 live closed-loop eval).  
**Total Scheduler Compute:** `2.321` Pro6K GPU-hours (`0.320` Gate 1 + `0.503` Stage B1 + `1.498` Stage B2).  
**Official Test Split (`SPLITS.json["test"]`):** Strictly untouched.

## 1. Research Question & Differentiation

Can a lightweight in-context visuomotor reader trained **exclusively on 7-DoF robots** (`Panda` + `Sawyer`) identify hidden continuous command-frame rotations ($\theta \in [-180^\circ, 180^\circ)$), gain, and jaw polarity **zero-shot on a held-out physical 6-DoF robot (`UR5e`)**, enabling a **100% frozen off-the-shelf VLA ($\pi_{0.5}$)** to recover closed-loop manipulation without VLA retraining, expert demonstrations, or target-arm calibration?

- **Why external vision is strictly required:** `UR5e` has 6 joints vs. 7 on `Panda`/`Sawyer` (`7 != 6`), so source-fitted joint-angle Jacobians are inapplicable (`joint_dim_mismatch`), and rotated-frame proprioception is blind to the hidden command-frame rotation (`analytic_proprio_sysid` error `85.33°–90.9°`).
- **Why factored SO(2)-equivariance matters:** Monolithic sequence readers (`joint_transformer`) memorize the training factor grid (`6.3°` familiar error) and collapse completely on unseen factor compositions (`106.0°`, `0/16` live success) and unseen continuous rotations (`82.9°`, `1/16` live success).

## 2. Gate 0 & Gate 1 Verification (Held-Out 6-DoF `UR5e`)

- **CPU Admission (`cpuprobe01`, 0 GPUh):** `112 / 112` passed across all 8 dev tasks (`Panda`/`Sawyer`/`UR5e`, resets `11, 13`) and 8 fitting tasks (`Panda`/`Sawyer`, resets `0..3`). Evidence: [admission.json](evidence/admission.json).
- **CPU Gate 0 (`cpugate02`, `6c16d9f8`, `exit 0`, all 7 predicates `true`):** 8/8 dev tasks verified physical 6-DoF `UR5e`, bitwise frame-0 equality across regimes, `analytic_proprio_sysid` `85.33°`, `Panda` joint Jacobian `11.17°` vs. `UR5e` `joint_dim_mismatch:7!=6` (`66.01°` zero-padded / `85.33°` self-cal local). Evidence: [gate0-summary.json](evidence/gate0-summary.json).
- **Pro6K Gate 1 (`job-34nf57gggb4z`, `6c16d9f8`, `0.320` GPUh, `exit 0`, 0 reset mismatches):** 48 paired trials (96 episodes) across 8 dev tasks × 6 regimes on `UR5e`:
  - **Shifted (`n=40`):** `privileged_oracle` **`37 / 40 (92.5%)`** vs. `nominal` **`0 / 40 (0.0%)`** (**`+92.5 pp`**).
  - **Overall (`n=48`):** `privileged_oracle` **`44 / 48 (91.7%)`** vs. `nominal` **`7 / 48 (14.6%)`**. Evidence: [gate1-summary-compact.json](evidence/gate1-summary-compact.json).

## 3. Stage B1 Offline Audit (96 Held-Out 6-DoF `UR5e` Trials, 3 Seeds)

Collected `320` paired support trials (`224` 7-DoF `Panda`+`Sawyer` train, `96` held-out 6-DoF `UR5e` dev) and trained 24 3-seed readers (`job-2aymxz7y0alu`, `d639787a`, `0.503` GPUh, `exit 0`). Evidence: [offline-audit-compact.json](evidence/offline-audit-compact.json).

| Reader / Baseline | `active_probe` Yaw Err (`all_correct`) | `passive_attempt1` Yaw Err (`all_correct`) | Notes |
| --- | ---: | ---: | --- |
| `factored_equiv` (Ours, raw RGB) | **`18.4°` (`52.4%`)** | `52.9°` (`25.0%`) | Active: `11.1°` fam / `22.2°` comp / `22.0°` cont; `90.0°` cleared / `92.9°` swapped |
| `analytic_visual_sysid` (grayscale flow) | `19.0°` (`47.9%`) | **`11.9°` (`70.8%`)** | Appearance-invariant grayscale optical flow (`10.9°` clean proj) |
| `factored_no_equiv` | `57.9°` (`27.1%`) | `65.0°` | Loses SO(2) equivariance |
| `joint_transformer` (monolithic) | `65.0°` (`28.1%`) | `70.1°` | Memorizes familiar (`6.3°`), collapses on comp (`106.0°`) & cont (`82.9°`) |
| `analytic_joint_jacobian` | `87.6°` pad / `90.9°` self-cal | `76.1°` pad / `89.3°` self-cal | Inapplicable (`7 != 6` joint DoF mismatch) |
| `analytic_proprio_sysid` | `90.9°` (`16.7%`) | `89.3°` | Blind to command-frame yaw |
| `frame0_probe` | `90.0°` (`0.0%`) | `90.0°` (`0.0%`) | Confirms zero frame-0 leakage |

## 4. Stage B2 Live Closed-Loop Evaluation (48 Paired `UR5e` Queries × 11 Arms = 528 Episodes)

Evaluated on `job-omd2j5x1wj0f` (`62854a88`, `89m 52s = 1.498` GPUh, `exit 0`) across 8 dev tasks × 6 regimes (16 familiar, 16 compositional OOD, 16 continuous-angle OOD). Evidence: [live-eval-summary-compact.json](evidence/live-eval-summary-compact.json).

| Arm | Familiar (`n=16`) | Comp OOD (`n=16`) | Cont OOD (`n=16`) | **Overall (`n=48`)** |
| --- | ---: | ---: | ---: | ---: |
| `privileged_oracle` | 12/16 (75.0%) | 13/16 (81.2%) | 14/16 (87.5%) | **39/48 (81.2%)** |
| `analytic_visual_sysid_passive` | 12/16 (75.0%) | 11/16 (68.8%) | 10/16 (62.5%) | **33/48 (68.8%)** |
| **`ours_retained_active` (`factored_equiv`, raw RGB)** | **10/16 (62.5%)** | **7/16 (43.8%)** | **6/16 (37.5%)** | **23/48 (47.9%)** |
| `analytic_visual_sysid_active` | 7/16 (43.8%) | 9/16 (56.2%) | 1/16 (6.2%) | 17/48 (35.4%) |
| `factored_no_equiv_active` | 5/16 (31.2%) | 6/16 (37.5%) | 3/16 (18.8%) | 14/48 (29.2%) |
| `ours_retained_passive` (`factored_equiv`, raw RGB) | 6/16 (37.5%) | 3/16 (18.8%) | 5/16 (31.2%) | 14/48 (29.2%) |
| `joint_transformer_active` (monolithic) | 7/16 (43.8%) | **0/16 (0.0%)** | **1/16 (6.2%)** | 8/48 (16.7%) |
| `nominal` (frozen $\pi_{0.5}$) | 6/16 (37.5%) | 0/16 (0.0%) | 1/16 (6.2%) | 7/48 (14.6%) |
| `ours_cleared` (zero-context control) | 6/16 (37.5%) | 0/16 (0.0%) | 1/16 (6.2%) | 7/48 (14.6%) |
| `analytic_proprio_sysid` | 6/16 (37.5%) | 0/16 (0.0%) | 0/16 (0.0%) | 6/48 (12.5%) |
| `ours_swapped_donor` (wrong-context control) | 0/16 (0.0%) | 0/16 (0.0%) | 0/16 (0.0%) | 0/48 (0.0%) |

### Key Paired Bootstrap Contrasts (`ours_retained_active`, `n=48`)
- vs. `nominal`: **`+33.3 pp`** (`95% CI [+22.9, +43.8]`, `17W / 1L / 30T`)
- vs. `analytic_proprio_sysid`: **`+35.4 pp`** (`95% CI [+22.9, +47.9]`, `18W / 1L / 29T`)
- vs. `joint_transformer_active`: **`+31.2 pp`** (`95% CI [+22.9, +39.6]`, `19W / 4L / 25T`)
- vs. `factored_no_equiv_active`: **`+18.8 pp`** (`95% CI [+6.2, +31.2]`, `12W / 3L / 33T`)
- vs. `analytic_visual_sysid_active`: **`+12.5 pp`** (`95% CI [+4.2, +22.9]`, `10W / 4L / 34T`)
- vs. `ours_cleared`: **`+33.3 pp`** (`95% CI [+22.9, +43.8]`, `17W / 1L / 30T`)
- vs. `ours_swapped_donor`: **`+47.9 pp`** (`95% CI [+33.3, +62.5]`, `23W / 0L / 25T`)

## 5. Conclusions & Follow-Up

1. **Positive Zero-Shot Cross-Morphology POC Established:** Even with raw RGB and only 8 training tasks (`224` trials), our SO(2)-equivariant factored reader beats monolithic `joint_transformer` by `+31.2 pp`, `factored_no_equiv` by `+18.8 pp`, `analytic_proprio_sysid` by `+35.4 pp`, and `nominal` by `+33.3 pp` on live closed-loop `UR5e` manipulation, with every 95% CI strictly above zero and exact causal collapse (`ours_cleared = 14.6%`, `ours_swapped_donor = 0.0%`).
2. **Remaining Bottleneck & Direct Follow-Up:** Color-invariant grayscale optical flow (`analytic_visual_sysid_passive`, `68.8%`) outperformed raw-RGB `factored_equiv` (`47.9%` active / `29.2%` passive) because raw-RGB features overfit to `Panda`/`Sawyer` arm appearance when transferring to blue/silver `UR5e`. This directly motivates the **8× Pro6K scaled follow-up** (`libero-xemb-ur5e-scale-20261120-r1`, `job-6prgb2hkbe8c`), which replaces raw RGB with RGB-permutation-invariant `luminance_flow` and scales training 6× across all 24 fitting tasks (`1,344` train trials).
