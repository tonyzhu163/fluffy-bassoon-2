# Independent PPO40 / GRPO40 / SFT test

2026-11-04, prospective before any outcome in this new case bank.

The running six-policy PPO ablation does not contain GRPO. Its TEST512 cannot
answer the user's PPO-versus-GRPO baseline question. Evaluate the original
Stage 1 PPO40 and GRPO40 checkpoints, plus their common SFT initializer, on
512 new common OpenDrawer cases. Training remains unchanged at3,072,000
TRAIN controls per learned checkpoint. No SGD, source restart or favorable
checkpoint selection is included.

- Cases65,311,000–65,311,511, H600, fixed B8; sampler seed75,311,000 + bank.
  These are disjoint from existing training, DEV, TEST512 and meta readout banks.
- All six policy permutations cycle across banks. Repeat SFT on bank0 for
  decoded-action/outcome reproducibility. Preserve complete physical/public
  scene checks, camera diagnostics, actual sampler hashes and strict loads.
- Primary: GRPO40 minus PPO40 first-success difference, exact two-sided paired
  McNemar p at.05, paired case-bootstrap95% interval. SFT and terminal-success
  contrasts are descriptive. This is one training seed per algorithm.
- Use the existing evaluated actors. GRPO lacks a critic, so for this shared
  evaluation-only architecture, copy its667 actor tensors exactly and add
  the eight unused VLM-critic head tensors from the common initial PPO model.
  Qualify every actor tensor byte-for-byte on allocated Linux CPUs before GPU
  launch. This neither updates the actor nor invents a GRPO value function.
- Reuse pinned evaluation workers at`81caf870`, engine`02ce5940`, and vendor
  `8f9bfa76`, which are already used by the running matched PPO comparison.
  Only the wrapper's prospective labels/order/case bank differ. Do not alter
  any existing job, output identity or worktree.
- Nominal controls:926,400 (512 ×3 ×600 plus8 ×600 SFT repeat). Every reset,
  failure and retry remains charged. Use two isolated one-GPU replicas, each
  with16CPUs/64GiB RAM/8GiB SHM, max225minutes. The master waits for both
  results. Automatic retry is off.
- Separate all-attempt incremental ceiling: **7.5GPU-hours**, within the
  user's requested PPO/GRPO evaluation scope and AGENTS.md's below-eight-hour
  launch allowance. This does not borrow either the96GPU-hour meta budget or
  the92GPU-hour PPO ablation budget. No extension beyond7.5without approval.

The current six-episode native qualification remains charged to the existing
meta budget. Recheck current queue capacity and storage before submission;
probe each actual allocation. Diagnose any failure before considering a fresh
attempt identity. Independent Linux CPU reduction audits the completed result.
