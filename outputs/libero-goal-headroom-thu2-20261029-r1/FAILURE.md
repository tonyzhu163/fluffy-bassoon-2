# r1 failed: GPU out of memory before the first batch

thu2 job **49204**, node01 RTX 3090 (`GPU-903c12ee-f081-9a51-3c1c-6be0959a8196`),
source `5475e30cacf934e1ab2572c67bbf01d051cab1c2`, 2026-10-29 04:39:30-04:43:51 UTC,
exit 1. No episode completed; `run/` holds only `config.json` and `plan.json`.

Passed before the failure (receipts in `receipts/`): storage probe, allocated-step
GPU probe, 14 focused tests, and the CUDA model check on a real LIBERO observation
(checkpoint SHA256 matches, fingerprint `8a020cc1...fd810` identical to the CPU check
and unchanged after inference, output `[2, 5, 7]`, identical paired rows and starts,
prompt change alters actions). Both renderer workers ran as graphics processes on
the allocated GPU only.

Cause: each LIBERO EGL renderer process holds about 456 MiB on the GPU. 34 workers
(15.5 GiB) plus the 6.9 GiB model left too little for the first batched forward pass
(`torch.OutOfMemoryError` in Gemma RoPE, 144 MiB free of 23.57 GiB).

Recovery under a new identity, `libero-goal-headroom-thu2-20261029-r2`: 16 workers
(about 7.3 GiB of renderer memory), expandable CUDA segments, the same code, plan,
seeds and checkpoint. This is the second and last GPU job of the approved budget.
