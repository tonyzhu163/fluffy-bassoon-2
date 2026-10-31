# Native VLA context pilot: stabilized history tokens

**Superseded by [r4](../native-vla-context-pilot-20261028-r4/RESULTS.md), the completed run.**

User instruction: "run it on baige". This is a bounded continuation of the
direct native-context experiment described in
[the method contract](../../docs/native-vla-context-icrl.md), using the r1
launch template and original pilot scope. No external recurrent learner is
attached: scalar feedback enters native prefix tokens and native prefix
attention adapters are trainable; the base motor stays frozen.

Source: `b5d97b0a7c0f6aed29660a96e2ab10cce9610915`, including reviewed
main via merge `cb3182becab68d0c8221d9faffd6e72bf1c1c88a`. The only
scientific change after the failed r2 admission is a learned nonzero history
type embedding to avoid degenerate zero-token RMS normalization. The
nonfinite-gradient guard remains enabled. The real gate additionally saves
gradient diagnostics before clipping.

Allocated CPU qualification on
`aibox-r855e239307b-6bfb688cdd-dmr8f`: **15 passed in 27.61 seconds**,
including the new zero-feedback / 18-layer RMS regression and the shared
bridge trainer tests. [Qualification log](evidence/cpu-qualification.log).

Baige job **`job-8kkjhb65px67`**, name `mwj_1028_native_context_g1_b5d97b0`:
one RPBZZZ6, 16 CPU, 64 GiB RAM, 8 GiB shared memory, 60-minute scheduler
cap, no automatic retries. Driver budget is 2,700 seconds checked at phase
and update boundaries. Worktree and outputs are separate and immutable:

- `/mowenjie/fluffy-bassoon-2/worktrees/native-vla-context-pilot-20261028-r3`
- `/mowenjie/fluffy-bassoon-2/outputs/native-vla-context-pilot-20261028-r3`

Admission requires one allocated healthy CUDA device, seven policy tests,
real-model empty-history parity, 12 finite fitting updates, a nonzero public
feedback gradient, unchanged frozen weights, and exact stochastic replay
within the documented tolerance. Only after admission: three K4 training
trials and two validation trials, paired oracle competence, 20 supervised
warm-start updates, one on-policy PPO update, and paired normal / wiped /
reward-blind held-out smoke evaluations. Stop if a gate fails or the bounded
budget expires. Preserve checkpoints and raw evidence before any change.

This is implementation and pipeline qualification. The small trial count
cannot establish adaptation, generalization, or an ICML-level contribution.
