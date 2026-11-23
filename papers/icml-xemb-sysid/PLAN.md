# ICML Paper Plan — In-Context Embodied System Identification for Frozen VLAs

**Location:** `sticks-and-stones/papers/icml-xemb-sysid/`  
**Target Venue:** ICML (8-page main text + unlimited references/appendix)  
**Primary Verified Evidence Repo:** `fluffy-bassoon-2` (`libero-xemb-ur5e-sysid-20261120-r1`, `libero-xemb-ur5e-scale-20261120-r1`)

---

## 1. Working Title & Core Positioning

**Working Title:**  
*In-Context Embodied System Identification for Frozen VLAs: Why Monolithic Sequence Encoders Memorize and How SO(2)-Equivariant Factorization Transfers Zero-Shot Across Morphologies*

### Core Thesis
Off-the-shelf Vision-Language-Action models (VLAs, e.g., frozen $\pi_{0.5}$) already possess strong manipulation competence, yet collapse (`7.3%` overall; `2.3%` under shift) when deployed on an unseen physical robot morphology (`7-DoF Panda/Sawyer` $\rightarrow$ held-out `6-DoF UR5e`) with unknown command-frame rotation, gain, or jaw polarity. We show that:
1. **External visual motion is strictly necessary** under cross-morphology transfer (`7 != 6` joint DoFs breaks source-fitted joint Jacobians; proprioception alone is blind to hidden command-frame yaw).
2. **Standard monolithic in-context sequence encoders (`joint_transformer`) memorize the training factor grid** rather than performing system identification — scaling training data **6×** (`224` $\rightarrow$ `1,344` trials across 24 tasks) sharpens familiar performance (`75.0%`, `2.14°` yaw error) while causing a **`-68.8 pp` OOD cliff** (`0/32` compositional OOD, `4/32` continuous-angle OOD).
3. **Factorizing the reader with SO(2) equivariance (`factored_equiv`) enables zero-shot physical cross-morphology transfer** for a 100% frozen VLA, recovering **`79.4%` of the privileged oracle ceiling (`52.1%` vs. `65.6%`)** with **`43 wins / 0 losses`** vs. nominal — and even **passively (`39.6%`, zero active probe)** statistically beating active monolithic Transformers (`29.2%`).

### Direct Differentiation vs. Concurrent Work (*Yang, Finn & Sadigh — Cross-Controller Transfer via In-Context Behavior Adaptation*)
| Dimension | Yang, Finn & Sadigh (Concurrent) | **Ours (ICML Submission)** |
| --- | --- | --- |
| **Base Policy** | Retrained multi-controller VLA / expert demos | **100% frozen off-the-shelf VLA ($\pi_{0.5}$)**, zero fine-tuning, zero target demos |
| **Morphology Shift** | Same-DoF controller variations | **Zero-shot physical cross-morphology (`7-DoF Panda/Sawyer` $\rightarrow$ `6-DoF UR5e`)** where proprio & joint Jacobians fail (`7 != 6`) |
| **Context Encoder** | Monolithic Perceiver / Transformer | **Direct critique & falsification** of monolithic encoders on withheld compositions (`0/32`) & continuous rotations (`4/32`), solved by **SO(2)-equivariant factored visuomotor reader** |
| **Causal Audit** | Aggregate transfer accuracy | **Direction 3 Diagnostic Audit:** frame-0 leak check (`90°`), cleared context (`= nominal`), swapped-donor collapse (`3.1%`), task-clustered 95% bootstrap CIs |

---

## 2. What Our Current Verified Results Already Prove

All numbers below are verified on Baige `NVIDIA RPBZZZ6` (Pro6K) with `exit 0` and SHA-256 archived evidence (`10.43` GPUh total across Phase 3 baseline + 8-GPU scaled runs; official LIBERO-Pro2.0 test split strictly untouched):

1. **Figure 1 — The Monolithic Memorization Cliff ([`figures/fig1_memorization_cliff_ood.png`](figures/fig1_memorization_cliff_ood.png)):**
   - Across `96` paired held-out 6-DoF `UR5e` queries (`1,056` live closed-loop episodes, 8 dev tasks × 12 embodiments):
   - `joint_transformer_active` scores **`75.0%` (`24/32`)** on familiar regimes, then crashes to **`0.0%` (`0/32`)** on compositional OOD and **`12.5%` (`4/32`)** on continuous-angle OOD (`-68.8 pp` cliff).
   - `factored_no_equiv_active` similarly collapses to **`0.0%` (`0/32`)** comp OOD and **`3.1%` (`1/32`)** cont OOD.
   - Our **`ours_retained_active` (`luminance_flow`)** holds **`59.4%` (`19/32`)** familiar, **`43.8%` (`14/32`)** comp OOD, and **`53.1%` (`17/32`)** cont OOD (`48.4%` combined OOD vs. `6.2%` monolithic).
   - Even **without an active probe**, **`ours_retained_passive`** achieves **`39.1%` (`25/64`)** combined OOD (`34.4%` comp, `43.8%` cont).

2. **Figure 2 — Overall Performance & Airtight Causal Controls ([`figures/fig2_overall_and_causal_controls.png`](figures/fig2_overall_and_causal_controls.png)):**
   - `ours_retained_active` (`52.1%`, `50/96`) achieves **`43W / 0L`** vs. `nominal` (`7.3%`, `+44.8 pp`, 95% CI `[+21.8, +66.7]`), **`43W / 0L`** vs. `ours_cleared` (`7.3%`), **`48W / 1L`** vs. `ours_swapped_donor` (`3.1%`), `+38.5 pp` vs. `analytic_proprio_sysid` (`13.5%`) and `factored_no_equiv_active` (`13.5%`), `+22.9 pp` vs. `joint_transformer_active` (`29.2%`, 95% CI `[+3.1, +41.7]`), and `+11.5 pp` vs. `analytic_visual_sysid_active` (`40.6%`).
   - `ours_retained_passive` (`39.6%`, `38/96`) statistically beats `joint_transformer_active` (`+10.4 pp`, 95% CI `[+2.1, +21.9]`, `22W/12L`).

3. **Figure 3 — Scaling 6× Sharpens Memorization in Monolithic Models, Slashes Passive Error in Ours ([`figures/fig3_scaling_memorization_vs_sysid.png`](figures/fig3_scaling_memorization_vs_sysid.png)):**
   - Going from 8 tasks (`224` trials) $\rightarrow$ 24 tasks (`1,344` trials):
     - `joint_transformer` familiar yaw error drops `6.3°` $\rightarrow$ `2.14°`, while comp OOD stays **`106.0°` $\rightarrow$ `107.47°`** and cont OOD stays **`82.9°` $\rightarrow$ `83.60°`**.
     - `factored_equiv` cuts passive `UR5e` yaw error from `52.9°` $\rightarrow$ **`28.77°`** (`luminance_flow`) and **`20.35°`** (`24-task RGB`), while active yaw error tightens `18.4°` $\rightarrow$ **`15.43°`**.

---

## 3. Experimental Plan to Make This ICML-Bulletproof (LIBERO + ManiSkill)

### Phase A — Close the Passive Visual Gap to `analytic_visual_sysid_passive` (`~4 GPUh`, 1×8 Pro6K)
- **Observation from S1/S2:** Grayscale optical-flow `analytic_visual_sysid_passive` (`61.5%`, `59/96`) still leads `ours_retained_active` (`52.1%`) by `9.4 pp` (95% CI `[-28.1, +4.2]`) and `ours_retained_passive` (`39.6%`) by `21.9 pp`. Meanwhile, in S1 offline audit, our **24-task `factored_equiv_rgb`** achieved **`20.35°`** passive yaw error (beating `luminance_flow`'s `28.77°` by `8.4°`!).
- **Experiment A1:** Live-evaluate our already-trained 24-task `factored_equiv_rgb` checkpoint + train/evaluate a **10-channel hybrid `rgb_plus_lumaflow`** SO(2)-equivariant reader (combining luminance flow + normalized RGB motion difference) on the same 96 `UR5e` dev queries.
- **Success Criterion:** Learned `factored_equiv` matches or exceeds `analytic_visual_sysid_passive` (`>= 60%`) on active and/or passive `UR5e`, leaving zero unaddressed baseline gaps.

### Phase B — Full LIBERO Benchmarks & One-Shot Official Test Split (`~6 GPUh`, 1×8 Pro6K)
- **Experiment B1 (Official Held-Out Test Split):** Freeze the winning reader from Phase A (zero hyperparameter tuning on test) and run the one-shot confirmatory evaluation on the strictly untouched **8 official LIBERO-Pro2.0 test tasks** (`SPLITS.json["test"]`, `96` paired `UR5e` queries × 11 arms = `1,056` live episodes across all 4 LIBERO suites: `LIBERO-Spatial`, `LIBERO-Object`, `LIBERO-Goal`, `LIBERO-Long/10`).
- **Experiment B2 (Per-Suite & Multi-Arm Generality):** Report per-suite breakdown across all 4 LIBERO suites + evaluate a **second held-out target morphology** (e.g., `IIWA` on its `17/24` admitted tasks or `Kinova3`/`Jaco` in robosuite) so cross-morphology transfer is verified on $\ge 2$ unseen physical robots.

### Phase C — Second Simulator Benchmark: ManiSkill3 Cross-Embodiment (`~12–16 GPUh`)
- **Why ManiSkill3?** Proves the phenomenon (monolithic memorization cliff vs. SO(2)-equivariant visuomotor SysID) is **not** specific to MuJoCo/LIBERO or $\pi_{0.5}$, generalizing to an independent physics engine (SAPIEN/PhysX), different camera viewpoints, different robots (`Panda` $\rightarrow$ held-out `xArm6`/`UR5e`/`Fetch`), and a different frozen base policy.
- **Step C0 (Environment & Gate 0, 0 GPUh CPU):**
  - Provision a dedicated `mani-skill` `uv` environment on Baige PFS (currently not installed in the RoboCasa env).
  - Select 3–4 canonical tabletop manipulation tasks (`PickCube-v1`, `PushCube-v1`, `StackCube-v1`, `PullCubeTool-v1`) supporting multi-embodiment arms.
  - Verify Gate 0: bitwise frame-0 equality across regimes, proprioception blindness (`>= 60°`), and joint-DoF mismatch (`7 != 6`).
- **Step C1 (Gate 1 Oracle vs. Nominal, ~2 GPUh):** Verify frozen base policy achieves high oracle success (`>= 70%`) and collapses under shifted command frame (`<= 15%`) on the held-out ManiSkill target robot.
- **Step C2 (Train on Source Arms $\rightarrow$ Zero-Shot Eval on Held-Out Target Arm, ~10–14 GPUh):** Run the identical Direction 3 audit (`ours_retained_active/passive`, `joint_transformer`, `factored_no_equiv`, `analytic_visual`, `analytic_proprio`, `nominal`, `cleared`, `swapped`) across familiar, compositional OOD, and continuous-angle OOD regimes.

### Phase D — ICML Paper Structure (8 Pages + Appendix)
1. **Introduction (1.25 pp):** Frozen VLA deployment gap on unseen morphologies; why monolithic in-context adaptation fails OOD; our SO(2)-equivariant factored visuomotor SysID; summary of LIBERO + ManiSkill results (teaser Fig. 1).
2. **Problem Formulation & Observability Analysis (1.0 pp):** Formalizing hidden embodiment parameters $(\theta \in SO(2), g, s_{\text{grip}})$; why proprioception & source joint Jacobians fail under morphology change (`7 != 6`); why frame-0 leakage must be ruled out.
3. **Method: SO(2)-Equivariant Factored Visuomotor Reader (1.5 pp):** Visual motion representation (`luminance_flow` / hybrid), equivariant rotation estimation + factored gain/jaw heads, active probe vs. zero-probe passive inference, frozen VLA command inversion wrapper.
4. **Experimental Setup & Direction 3 Diagnostic Protocol (1.0 pp):** 7-DoF `Panda`+`Sawyer` $\rightarrow$ 6-DoF `UR5e` (LIBERO-Pro2.0 4 suites, dev + locked test split) + ManiSkill3; familiar / compositional OOD / continuous-angle OOD regimes; causal controls (`cleared`, `swapped`, `frame0_probe`, `analytic_proprio`, `analytic_joint_jacobian`, `analytic_visual`).
5. **Main Results (2.25 pp):**
   - Table 1 & Fig. 1: Held-out `UR5e` live closed-loop results (LIBERO-Pro2.0 test + dev).
   - Fig. 2: Overall performance + task-clustered 95% bootstrap CIs + causal controls.
   - Fig. 3: Scaling law (8 vs. 24 tasks) — memorization sharpening in `joint_transformer` vs. true SysID in `factored_equiv`.
   - Table 2: ManiSkill3 cross-embodiment transfer (confirming cross-simulator & cross-policy generality).
6. **Related Work & Limitations (1.0 pp):** Careful comparison to Yang, Finn & Sadigh, meta-RL/in-context RL, and visual servoing/system ID; honest discussion of analytic visual flow strength, camera calibration assumptions, and real-robot outlook.

---

## 4. Budget & Immediate Next Decision
- **Spent so far:** `10.88 / 50.0` GPUh (Phase 2 + Phase 3 baseline + Phase 3 8-GPU scaled).
- **Estimated remaining for full ICML package:** Phase A (`~4` GPUh) + Phase B (`~6` GPUh) + Phase C ManiSkill3 (`~14` GPUh) = **`~24` GPUh** (total `~35 / 50` GPUh, well within budget).
- **Recommended next step:** Confirm the plan & charts, then run **Phase A** (hybrid `rgb + luminance_flow` + live eval of `24-task RGB`) and **Phase B** (locked LIBERO-Pro2.0 official test split) while spinning up **Phase C0** (ManiSkill3 environment + CPU Gate 0) in parallel.
