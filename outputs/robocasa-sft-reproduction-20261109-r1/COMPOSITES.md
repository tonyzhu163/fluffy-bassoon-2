# Complete native composite evaluation: campaign02

Atomic900 passed its prospective practical screen:363/900=40.33% versus39.6%.
The four-longest-task capacity probe completed16,650 controls and3,330 native
batch-one predictions in387.909 seconds of collection:42.922 controls/s.
All four input hashes, sampler calls and clean closure passed independent
audit. These four diagnostic failures are excluded from benchmark results.
Baige jobjob-fjbgnmxm6ivr succeeded16:18:16 Shanghai; its GPU was released.

COMPOSITES-MANIFEST.json assigns all32 composite tasks exactly once to four
exclusive one-GPU containers,50 trials/task,800 cases/category. Longest-first
assignment balances maximum controls at975,000–990,000 per rank. Each native
B1 server has at most four simulator clients; a freed client slot receives
the next task. No RL or optimization runs here.

Retain the original checkpoint, prediction50/execution5,ODE10,pretrain split,
seed7,224-pixel padding,three RGB cameras,native action conversion and first
success termination. Use the qualified narrow1.0.1 source with equivalent
original1.0.0 effective horizons; no double multiplier. Preserve the old
registry cutoff as secondary. Each case retains inputs/action hashes and
the first two videos per task. Preserve final native RNG and sampler counts.

Resources: four containers,each1GPU/21CPU/224GiB/16GiB SHM,8-hour scheduler
limit,retryOFF. Worker limit7h54m leaves6m for cleanup. The maximum3,930,000
controls imply25.43 GPUh at the measured rate;8h per rank gives about25%
margin on the heaviest shard before startup and changing task mix. Recheck
queue capacity and actual device type before scheduling; in-step probes must
pass on four distinct allocated GPUs before model loading. Peer failures
close the remaining workers and preserve evidence; no automatic retries.

Closed earlier cost is18,416 conservative GPU-seconds=5.11556GPUh,including
GPU05's42-minute queue delay. This sweep reserves32GPUh plus480GPU-seconds
startup/release,bringing the conservative reservation to37.24889GPUh against
the cumulative50GPUh task ceiling. Unallocated queue delay is recorded
separately for future jobs. Never exceed the cap by silently retrying.

Fresh source must be committed,pushed,deployed through Git and verified clean
before launch. Do not alter GPU05 or campaign01 source/output identities.
Require all1600 cases,hash/count reconciliation and final sampler receipts.
Report Composite-Seen and Composite-Unseen separately against7.1% and1.2%,
with descriptive intervals; combine with the already audited atomic result
only after complete audit. Monitor advancing episodes at least every20min.

The user's later RL instruction now prioritizes establishing an upstream
recipe before any C5/C50 ablation. No PPO/GRPO training has launched here.
