# Native VLA context pilot: first GPU admission

Baige job `job-t9agduualjt2`, source
`cb3182becab68d0c8221d9faffd6e72bf1c1c88a`, started 2026-10-28
19:07:55 Shanghai. One RPBZZZ6, 16 CPU, 64 GiB RAM, 8 GiB SHM,
60-minute scheduler limit. The immutable output is
`/mowenjie/fluffy-bassoon-2/outputs/native-vla-context-pilot-20261028-r2`.

The allocated CUDA probe passed and all six context-policy tests passed in
3.99 seconds. The real native-model empty-history parity check passed. The
first gradient clipping check then rejected a nonfinite gradient norm, before
any optimizer step, collection, warm start or PPO. The driver exited 1 after
94.73 seconds; [the failure receipt](evidence/failure.json) is retained.

The suspected mechanism is an exact zero token for initial zero public
feedback: a bias-free native causal transformer keeps that token at zero,
and repeated RMS normalization can produce enormous input gradients. The
candidate fix gives history tokens a learned nonzero type embedding. It
retains the nonfinite-gradient guard and adds per-update gate diagnostics.
Source `b5d97b0a7c0f6aed29660a96e2ab10cce9610915` adds a deep RMS
normalization regression; allocated Linux qualification passed 15 tests.
The real-model confirmation belongs to a fresh r3 run, not this failed run.

No learning or adaptation result was obtained here.
