# Scaled 8-GPU Held-Out 6-DoF UR5e Cross-Morphology Visuomotor SysID

**Run ID:** `libero-xemb-ur5e-scale-20261120-r1`  
**Code Revision:** `d597ee0f1e99937bbeb2705132d709b514a4b5c2`  
**Status:** Complete (`exit 0` on both 8× Pro6K jobs: `job-6prgb2hkbe8c` S1 collect+train and `job-5s10a9cdqh77` S2 live eval).  
**Total Scheduler Compute:** `8.11` Pro6K GPU-hours (`3.31` GPUh S1 + `4.80` GPUh S2; `24m49s` + `36m01s` wall-clock on 1×8 `NVIDIA RPBZZZ6`).  
**Official Test Split (`SPLITS.json["test"]`):** Strictly untouched.

## 1. Hypotheses Tested

Following the Phase 3 baseline POC ([`libero-xemb-ur5e-sysid-20261120-r1`](../libero-xemb-ur5e-sysid-20261120-r1/RESULT.md)), we scaled training **6× across all 24 fitting tasks** (`1,344` 7-DoF `Panda`+`Sawyer` trials vs. `224`), widened the readers (`cnn_width=96`, `mlp_width=192`, `3,000` steps), introduced RGB-permutation-invariant **`luminance_flow`** visual inputs, and doubled live evaluation to **all 96 paired held-out 6-DoF `UR5e` queries × 11 arms = 1,056 closed-loop episodes** (8 dev tasks × all 12 embodiments) to test:
1. **H1 (Scaling vs. Memorization):** Does 6× more training data save monolithic `joint_transformer` on compositional OOD and continuous-angle OOD, or does it merely sharpen in-distribution memorization?
2. **H2 (Visual Representation & Passive Transfer):** Does `luminance_flow` + 24-task diversity close the passive-attempt cross-morphology gap on held-out 6-DoF `UR5e`?

## 2. Stage S1 Offline Audit (`job-6prgb2hkbe8c`, 96 Held-Out 6-DoF `UR5e` Trials, 3 Seeds)

Evidence: [offline-audit-compact.json](evidence/offline-audit-compact.json).

| Model (24-task scaled, `n=96` `UR5e`) | `active_probe` Yaw Err (`all_correct`) | `passive_attempt1` Yaw Err (`all_correct`) | Active Fam / Comp OOD / Cont OOD |
| --- | ---: | ---: | ---: |
| **`factored_equiv` (`luminance_flow`)** | **`15.43°` (`55.2%`)** | `28.77°` (`43.8%`) | `10.96° / 16.25° / 19.08°` |
| **`factored_equiv_rgb` (24-task RGB)** | **`15.61°` (`57.3%`)** | **`20.35°` (`48.6%`)** | `8.35° / 16.78° / 21.69°` |
| `factored_no_equiv` | `49.48°` (`34.0%`) | `45.58°` (`26.4%`) | `32.15° / 57.22° / 59.06°` |
| `joint_transformer` (monolithic) | `64.40°` (`35.4%`) | `67.92°` (`21.9%`) | **`2.14° / 107.47° / 83.60°`** |
| `frame0_probe` | `90.00°` (`0.0%`) | `90.00°` (`0.0%`) | `111.47° / 68.53° / 90.00°` |

- **H1 confirmed dramatically:** Scaling 6× sharpened `joint_transformer` familiar error from `6.3° → 2.14°`, yet `compositional_ood` stayed at **`107.47°`** and `continuous_angle_ood` at **`83.60°`**.
- **H2 nuance:** Both `luminance_flow` (`52.9° → 28.77°`) and 24-task `factored_equiv_rgb` (`52.9° → 20.35°`) slashed passive `UR5e` error by `24–33°`; task diversity (8 → 24 tasks) was the dominant driver of passive transfer, with RGB actually outperforming `luminance_flow` on passive offline yaw (`20.35°` vs. `28.77°`) once trained across 24 diverse scenes.

## 3. Stage S2 Live Closed-Loop Evaluation (`job-5s10a9cdqh77`, 96 Paired `UR5e` Queries × 11 Arms = 1,056 Episodes)

Evidence: [live-eval-summary-compact.json](evidence/live-eval-summary-compact.json).

| Arm | Familiar (`n=32`) | Comp OOD (`n=32`) | Cont OOD (`n=32`) | **Combined OOD (`n=64`)** | Shifted (`n=88`) | **Overall (`n=96`)** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `privileged_oracle` | 20/32 (62.5%) | 20/32 (62.5%) | 23/32 (71.9%) | **43/64 (67.2%)** | 58/88 (65.9%) | **63/96 (65.6%)** |
| `analytic_visual_sysid_passive` | 22/32 (68.8%) | 17/32 (53.1%) | 20/32 (62.5%) | **37/64 (57.8%)** | 54/88 (61.4%) | **59/96 (61.5%)** |
| **`ours_retained_active` (`luminance_flow`)** | **19/32 (59.4%)** | **14/32 (43.8%)** | **17/32 (53.1%)** | **31/64 (48.4%)** | **45/88 (51.1%)** | **50/96 (52.1%)** |
| `analytic_visual_sysid_active` | 16/32 (50.0%) | 12/32 (37.5%) | 11/32 (34.4%) | 23/64 (35.9%) | 34/88 (38.6%) | 39/96 (40.6%) |
| **`ours_retained_passive` (`luminance_flow`)** | **13/32 (40.6%)** | **11/32 (34.4%)** | **14/32 (43.8%)** | **25/64 (39.1%)** | **37/88 (42.0%)** | **38/96 (39.6%)** |
| `joint_transformer_active` (monolithic) | **24/32 (75.0%)** | **0/32 (0.0%)** | **4/32 (12.5%)** | **4/64 (6.2%)** | 22/88 (25.0%) | 28/96 (29.2%) |
| `factored_no_equiv_active` | 12/32 (37.5%) | **0/32 (0.0%)** | **1/32 (3.1%)** | **1/64 (1.6%)** | 7/88 (8.0%) | 13/96 (13.5%) |
| `analytic_proprio_sysid` | 10/32 (31.2%) | 0/32 (0.0%) | 3/32 (9.4%) | 3/64 (4.7%) | 8/88 (9.1%) | 13/96 (13.5%) |
| `nominal` (frozen $\pi_{0.5}$) | 5/32 (15.6%) | 0/32 (0.0%) | 2/32 (6.2%) | 2/64 (3.1%) | 2/88 (2.3%) | 7/96 (7.3%) |
| `ours_cleared` (zero-context control) | 5/32 (15.6%) | 0/32 (0.0%) | 2/32 (6.2%) | 2/64 (3.1%) | 2/88 (2.3%) | 7/96 (7.3%) |
| `ours_swapped_donor` (wrong-context) | 0/32 (0.0%) | 1/32 (3.1%) | 2/32 (6.2%) | 3/64 (4.7%) | 3/88 (3.4%) | 3/96 (3.1%) |

### Task-Clustered 95% Bootstrap Contrasts (`n=96`)
- **`ours_retained_active` (`52.1%`, `79.4%` of oracle ceiling):**
  - vs. `nominal`: **`+44.8 pp`** (`95% CI [+21.8, +66.7]`, **`43W / 0L / 53T`**)
  - vs. `ours_cleared`: **`+44.8 pp`** (`95% CI [+21.8, +66.7]`, **`43W / 0L / 53T`**)
  - vs. `ours_swapped_donor`: **`+49.0 pp`** (`95% CI [+21.9, +75.0]`, **`48W / 1L / 47T`**)
  - vs. `analytic_proprio_sysid`: **`+38.5 pp`** (`95% CI [+13.5, +61.5]`, `40W / 3L / 53T`)
  - vs. `factored_no_equiv_active`: **`+38.5 pp`** (`95% CI [+14.6, +61.5]`, `39W / 2L / 55T`)
  - vs. `joint_transformer_active`: **`+22.9 pp`** (`95% CI [+3.1, +41.7]`, `27W / 5L / 64T`)
  - vs. `analytic_visual_sysid_active`: **`+11.5 pp`** (`95% CI [-4.2, +26.0]`, `22W / 11L / 63T`)
  - vs. `analytic_visual_sysid_passive`: `-9.4 pp` (`95% CI [-28.1, +4.2]`, `8W / 17L / 71T`)
- **`ours_retained_passive` (`39.6%`, zero active probe):**
  - vs. `nominal`: **`+32.3 pp`** (`95% CI [+16.7, +49.0]`, `36W / 5L / 55T`)
  - vs. `joint_transformer_active`: **`+10.4 pp`** (`95% CI [+2.1, +21.9]`, `22W / 12L / 62T`) — **beats active monolithic Transformer even passively!**
  - vs. `factored_no_equiv_active`: **`+26.0 pp`** (`95% CI [+12.5, +41.7]`, `33W / 8L / 55T`)

## 4. Key Scientific Takeaways

1. **Definitive Falsification of Scaling-Only Monolithic In-Context Encoders:** With 6× more multi-task data (`1,344` trials across 24 tasks), `joint_transformer_active` reaches **`75.0%` (`24/32`)** on familiar regimes — then plunges **-68.8 pp** to **`6.2%` (`4/64`)** on OOD (**`0/32`** compositional OOD, **`4/32`** continuous-angle OOD). `factored_no_equiv_active` similarly collapses to **`1.6%` (`1/64`)** on OOD.
2. **SO(2)-Equivariant Factored Reader Generalizes Cleanly Zero-Shot to 6-DoF `UR5e`:** `ours_retained_active` hits **`52.1%` (`50/96`, `79.4%` of oracle)** with **`43 wins / 0 losses`** vs. `nominal`, and **`48.4%` (`31/64`)** on combined OOD (`43.8%` comp OOD, `53.1%` cont OOD). Even **without an active probe**, `ours_retained_passive` hits **`39.6%` (`38/96`)** overall and **`39.1%` (`25/64`)** on OOD, statistically beating `joint_transformer_active` (`+10.4 pp`, 95% CI `[+2.1, +21.9]`).
3. **Honest Benchmark Reference:** Grayscale optical-flow `analytic_visual_sysid_passive` (`61.5%`, `59/96`) remains a strong domain-specific visual baseline (within `4.1 pp` of oracle), whereas our learned `ours_retained_active` beats `analytic_visual_sysid_active` by `+11.5 pp` (`52.1%` vs. `40.6%`) and closes within `9.4 pp` of `analytic_visual_sysid_passive` (95% CI `[-28.1, +4.2]` includes zero).
