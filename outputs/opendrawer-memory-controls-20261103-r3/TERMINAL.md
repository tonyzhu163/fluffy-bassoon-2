# Startup failure; no new episodes

Baige Pro6K / 算法Z组 `job-tjh32aj7mv5x`
(`mwj_1103_memory_controls_1x1_r3`), source
`6ae5e478141cc21d9cc7cba7a08ea62255564b64`.
Created Nov3 11:05:36 Shanghai; terminal FAILED11:07:03. One instance,
one RPBZZZ6 GPU,32 CPUs/128GiB RAM/32GiB SHM,120-minute limit;
automatic fault recovery/retries off. Scheduler/container released.

CPU qualification passed all11 carried window64 prefixes against full replay
on the same CPU backend, with original boundary lengths/rewards/done flags.
Binding verified all1052 carried endpoints and exactly48 missing DPT/ID
episodes (37 reset,11 window64);36,000 fresh policy controls remain.
An initial diagnostic incorrectly compared CPU state byte hashes against GPU
hashes. That diagnostic was corrected; GPU workers retain their original
exact GPU replay signature checks. Qualification source is `306f195f`.

The launch reached the allocation probe, which rejected an unset
`CUDA_VISIBLE_DEVICES`. The container inventory exposed exactly one allocated
RPBZZZ6, but the probe deliberately requires an explicit selector. The earlier
r2 command supplied `CUDA_VISIBLE_DEVICES=0`; this new launcher omitted it.
No worker, simulator episode, policy control or optimizer update began.
The original master-barrier fix was not exercised or invalidated.

The correction preserves an existing scheduler selector and supplies ordinal0
only when the selector is absent, within the single-GPU container allocation.
The count/CUDA/UUID probe remains mandatory. Preserve the r3 worktree/output;
any further launch must use a fresh r4 output and source pin.

Worker clock:37GPU-seconds. Conservatively charge the entire87-second
creation-to-terminal interval, raising cumulative charge62880→62967GPU-s.
Original ceiling86400 leaves23433GPU-s (6.509GPU-h). A one-GPU120-minute
continuation would give a combined upper70167, leaving16233 unused.
The run contract forbids automatic retries; no successor job has been launched.
