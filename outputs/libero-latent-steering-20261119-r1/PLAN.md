# Multi-perturbation latent velocity steering and decoupled history inference

This workstream generalizes `libero-s20-adapter-20261119-r1` from a 1-bit `BASE`/`S20` adapter to a two-stage decoupled architecture across multiple LIBERO-Pro2.0 perturbations (default `BASE,S20,S15,D18`, configurable via `--cases`):

1. **Stage 1 — Continuous-Latent Velocity Adapter (`train_adapter.py`, `motor.py`):**
   Fit a zero-initialized SiLU MLP + optional zero-FiLM velocity residual $v_\theta(\cdot, z)$ around the frozen JAX OpenPI π0.5 action expert, conditioned on a learnable continuous latent vector $z_c \in \mathbb{R}^{16}$ per training condition $c$. Train on causally aligned `(obs[t], actions[t+1 : min(t+1+10, steps)])` windows from `LIBERO-Pro2.0-Train` at revision `6c4c5524` across the 24 fitting tasks, using balanced multi-condition batches and the BASE task prompt across all conditions.

2. **Stage 2 — Decoupled Causal History Belief Encoder (`train_history.py`):**
   Freeze the Stage-1 adapter (`v_θ` and its prototype table `{z_c}`). Collect default frozen-π0.5 support attempts paired with fresh query resets and train a causal Transformer $g_\psi(H_{<t}, q_t)$ over 5-control public tokens `(issued, delta_eef, gripper, reward, is_wait, boundary, valid)` + frozen query features (`2056`-d `[state[:8], rms_norm(mean_pooled_prefix)]`) to infer the condition belief logits and continuous latent residual. Supports an explicit counterfactual margin loss ablation supervising how beliefs change across paired evidence contexts.

## Falsifiable hypotheses and go/no-go gates

- **Gate 1 (Stage-1 Motor Realizability):** Evaluated on the 8 project development tasks (`SPLITS.json`) using `--stage adapter`. Privileged `privileged_true_z` (oracle latent lookup from reset) must beat both frozen `default` π0.5 and the strongest fitting-selected single fixed latent `best_fixed_z` by **at least 20 percentage points** overall across the selected cases, while incurring **at most 1/16 BASE regression** relative to `default`.
  - *Stop / revise if:* `privileged_true_z - best_fixed_z < 20 pp` (the frozen flow expert cannot realize conflicting recoveries on these cases, so no history encoder could unlock gains over a fixed latent).
- **Gate 2 (Stage-2 History Adaptation):** Evaluated on paired fresh resets using `--stage history` across 3 independent reader seeds `(41, 42, 43)`. `history_retained` (running from reset using attempt-1 public tape) must beat `history_cleared`, `history_current_prefix` (40-control opening prefix only), `history_swapped_donor` (mismatched condition support), and `best_fixed_z` on native query success before opening the 8 official LIBERO-Pro2.0 test tasks or Shift-OOD variants (e.g., `D12`).
  - *Falsification check:* If `history_current_prefix` matches `history_retained`, the shift is solved online within episode 1 and does not justify cross-episode memory; if `history_swapped_donor` matches `history_retained`, the reader is ignoring support evidence.

## Compute and storage

- Run smoke checks on a Baige development GPU (`run.sh` stages `smoke` / `smoke-eval`); run full Stage-1 fitting, Stage-1 evaluation, Stage-2 collection, and Stage-2 evaluation via Baige **分布式训练** (`distributed.sh` / `run.sh`).
- Preserve GLVND + NVIDIA EGL qualification, GPU probe (`--expected-count 1`), `NCCL_P2P_DISABLE=1` on RTX PRO 6000D, and resumable msgpack checkpoints (adapter/encoder weights, optimizer state, RNG state, step, config, and recipe).
- Phase budget ceiling before scaling to 1,024+ Stage-2 trials: **12 GPU-hours** (Stage-1 fit + dev realizability evaluation + small Stage-2 pilot). The 8 official LIBERO-Pro2.0 test tasks remain reserved until Gate 1 and Gate 2 pass on development.
