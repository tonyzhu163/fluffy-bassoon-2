# Phase 2 Discriminator: Physical MuJoCo TCP Offset (`libero-xemb-tcp-20261120-r1`)

Registered plan, November 20, 2026. Evaluates physical MuJoCo finger collision-geometry tool-center (TCP) offsets $\delta = (\delta_x, \delta_y)$ on frozen JAX OpenPI $\pi_{0.5}$ across the eight held-out LIBERO-Pro2.0 development tasks (`SPLITS.json["development"]`, official test split untouched).

## Hypotheses and Falsification Gates

1. **Gate 0 (CPU Zero-GPUh Observability & Anti-Empty-Air Audit):**
   - **Zero pre-contact leak:** Frame-0 and pre-contact 256×256 RGB (`agentview`, `robot0_eye_in_hand`), all 7 public arm joints (`robot0_joint_pos`, `robot0_joint_vel`), wrist (`robot0_eef_pos`, `robot0_eef_quat_xyzw`), and fingers (`robot0_gripper_qpos`) must match nominal **bitwise** up to first physical finger contact.
   - **Effective failure & directional contact signal:** At selected radius $r^* \in \{0.025, 0.035, 0.045\}$ m, nominal $\pi_{0.5}$ achieves `8/8` at $\delta=(0,0)$ and `<= 8/32` (`<= 25%`) across single-axis shifts $(\pm r^*, 0), (0, \pm r^*)$, while opposite shifts $(+X/-X, +Y/-Y)$ remain post-contact distinguishable rather than collapsing into identical empty-air misses.
   - *Stop / re-plan if:* any frame-0/pre-contact mismatch, reset collision (`ncon` change), failure to drop nominal below `25%`, or indistinguishable post-contact traces.

2. **Gate 1 (Pro6K Motor Realizability across 8 Dev Tasks × 7 Embodiments = 56 Trials):**
   - Evaluates `nominal`, `oracle_grasp_only`, and `oracle_grasp_and_release` on `tcp_nom`, four single-axis offsets, and two withheld compositional diagonal offsets $(\pm r^*/\sqrt{2}, +r^*/\sqrt{2})$.
   - *Go threshold:* winning oracle achieves `>= 36/48` (`>= 75%`) on the 48 shifted trials, `>= +45 pp` over `nominal`, and zero `tcp_nom` regression.

3. **Gate 2 (Contact-Driven Compositional Cross-Episode Adaptation, 3 Seeds):**
   - Trains only on single-axis + nominal offsets across the 8 fitting tasks (`SPLITS.json -> development_scaling.task_subsets["8"]`, 4 resets × 5 offsets = 160 trials); evaluates on the 8 held-out dev tasks (support reset `11` → query reset `13`, 7 offsets = 56 trials).
   - Audits `frame0_probe`, `analytic_joint_jacobian`, `analytic_free_space_visual` (all provably blind before contact), `joint_contact_transformer`, and our 90°-equivariant `factored_contact_reader` (offline threshold `< 1.8 cm` familiar / `< 2.2 cm` compositional OOD), followed by closed-loop live evaluation across 8 paired arms.

## Compute Budget
- Gate 0: 0 GPU-hours (Baige CPU + Mesa llvmpipe).
- Gate 1 + Gate 2: `<= 8.0` GPU-hours total on Baige Distributed Training (1× NVIDIA RPBZZZ6).
