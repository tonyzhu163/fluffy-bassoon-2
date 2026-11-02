# Four-GPU preparation stopped before native collection

Baige `job-chi92rfq9p7m`, source `0f62915062e02cac45ed1158c860c6f4b31a1da6`, failed at 2026-11-02 16:04:58 Shanghai. All four scheduler allocations and producer clocks are terminal; automatic restarts were zero.

All four actual GPU probes and eight saved-video Flora warmups per GPU passed. Each official engine loaded once at utilization 0.75 and batch capacity 8, leaving 23,608 MiB free against the 21,973 MiB four-lane requirement. The eight existing 600-control videos have 601 frames in each of three views; all 24 video hashes and frame counts were independently checked on allocated Linux CPUs with CUDA masked. The 32 warmup labels are replays of these eight episodes and add no independent cases. This attempt measured no native collection throughput or concurrent native/scorer memory peak.

The first native driver then failed while resolving `loading_qualification`: `/mowenjie/fluffy-bassoon-2/outputs/robocasa-residual-approved-20261101-r1/STRICT-LOADING.json` was absent. The exact bound bytes exist under this run's pinned Git worktree, with SHA256 `5d1723f6e66c65037b192174017225a29f6302bc03ebce2c921912e4513f3a57`. The profile referenced an output-directory copy that deployment had not materialized. Source and numerical qualification did not verify this complete runtime input closure before expensive engine startup.

No native dispatch command or action journal was created. New calibration episodes, calibration grades, common-before episodes, physical actions and optimizer updates are all zero. There are no unresolved physical journals. All five fresh gates, all 540 common-before episodes, COMMON-FINAL and this attempt's initial-policy fingerprint remain absent. Both inactive joint templates retain NOT_ADMITTED status; no partial evidence was promoted.

Producer charges were 531, 528, 528 and 528 GPU-seconds, totaling 2,115 (above the 2,029 GPU-second whole-pod lifetime upper bound). With the cancelled r1 conservative 864 and r2 charged 289, preparation accounts for 3,268 GPU-seconds, or 0.9078 GPU-hours, of the authorized 26,400. The unspent balance is 23,132 GPU-seconds. Prior memory qualification spend remains separate and unchanged.

Before another GPU job, restore or rebind the exact qualification receipt and check the entire profile's immutable inputs, including the common-before initializer, on allocated Linux CPUs before any model engine starts. No successor job or automatic retry was launched. The terminal monitor is removed after this audit and failure report; the chat stays open.

[CPU audit and full receipts](TERMINAL-AUDIT.json), [audit code](TERMINAL-AUDIT.py), [scheduler closure](SCHEDULER-TERMINAL.json).
