# GRPO native pair arm completed

Baige Pro6K job-cemy61val8um completed two iterations, 40 Adam updates on each of four ranks, and independent native trained-actor restoration in 89 seconds. Scheduler status is Succeeded and the allocation is released. Source 48fa7ea4, original broad Sidney SFT, micro 64 × accumulation 8 ×4 = global 2048 and qualified BF16 text/action with FP32 vision and norms were retained.

All 17 final data-file hashes match (11,168,702,667 bytes), including 12 worker sidecars. Eight frozen-VLM after-checks and 32 accumulation-gradient audits passed; all four first logprob replay errors were 0. Every rank had 190 nonzero noncritic first-gradient tensors. All 256 initial scene groups match the other arm, with 8 exact replicas per group. The independent restore recovered weights, owned Adam state/groups, scheduler, scaler and RNG at step 40 without further optimizer updates or simulator controls.

Collection 1 used the initial SFT policy: 176/1024 trajectories had positive returns, with 61/128 constant-return groups. Collection 2 used the policy after 20 updates on a different cohort: 184/1024 positive-return trajectories and51/128 constant groups. These are training-collection counts. The final 40-update policy was not evaluated, and no improvement over the published SFT baseline is claimed.

All four action journals are resolved at 842,240 requested/conservative slots, below 1,000,000. No measured delivered motor count or physics-substep count is claimed; student-corpus admission remains false.

Scheduler charge: 28,924 GPU-seconds (8.03444 GPU-hours). Conservative creation-to-release charge: 28,960; worker charge: 28,832. Output allocation: 122.361 GiB < 135 GiB. Future reservation is closed; trained checkpoints/cache remain protected.

[Full matched result and limits](../RESULT.md), [independent integrity](../FINAL-INTEGRITY.json), [restore](final-restore/VERIFIED.json), [native metrics](native-iteration-metrics.txt), [exit](EXIT.json), [scheduler evidence](EXECUTION.json).
