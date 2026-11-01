# Single-thread simulator lifecycle retry

2026-10-29 Asia/Shanghai. Preserve the original six-task matched-budget
source-training contract. Prior r3 completed real native-noise parity but
segfaulted before a training episode, consistent with cross-thread native scene
teardown. Use the same frozen motor, source tasks, learner seeds, K4/K8,
10.24M actual source-action cap per learner and original optimizer batches.

For the one physical simulator, install a synchronous Future-compatible
executor before first reset so setup/reset/step/close all use the calling
thread. Keep external pinned sources unchanged. Record the changed runtime
contract; start fresh because r3 has zero collected source experience.
Enable Python faulthandler and source-trial-start receipts for exact failure
location if the native fault persists. Check the focused suite and real parity
inside the training allocation, then immediately collect actual source trials.

One Baige GPU,16CPUs,128GiB RAM,8GiB shared memory, **25-minute cap**.
The combined ceiling is r1reserve2m+r2duration14m49s+r3duration2m49s+r4cap25m
= **44m38s**, below45GPU-minutes. Driver wall budget1,020seconds,180-second
trial-boundary recovery reserve, external timeout1,200seconds, up to300seconds
test allowance. No materially larger continuation is authorized. The launch
receipt records the exact deployed commit and scheduler job identity.
