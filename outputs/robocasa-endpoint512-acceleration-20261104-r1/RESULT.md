# Matched512 endpoint results — policy0 versus policy130

Completed and independently audited on 2026-11-04. Each of the two predeclared
tasks uses 512 fresh paired cases, native 600-control horizons, fixed B8
inference, the original global-bank ABBA policy order and common sampler RNG.
The checkpoints come from the simulator-reward Flow-SDE PPO source at
`0988b5f9`; no training or checkpoint selection ran in this evaluation.

| Primary: first success in 600 controls | Policy0 | Policy130 | Gain (percentage points) | Descriptive paired 95% CI | Wins / losses | Exact paired McNemar p |
| --- | --- | --- | --- | --- | --- | --- |
| CloseToasterOvenDoor | 168/512 (32.81%)| 212/512 (41.41%)| +8.59| [+3.71,+13.67]| 109 / 65| 0.0010559135|
| OpenDrawer | 331/512 (64.65%)| 369/512 (72.07%)| +7.42| [+2.73,+11.91]| 94 / 56| 0.0024055162|

Both primary comparisons pass the predeclared Holm correction across these two
tasks at family alpha .05: Toaster p<.025, then OpenDrawer p<.05, with wins>losses.
The intervals use 10,000 paired case-bootstrap draws and are descriptive, not
simultaneous intervals. The larger test confirms positive endpoint gains for
these two trained checkpoints; it does not establish an average effect across
the five source tasks or across training seeds.

| Secondary: success at control 600 | Policy0 | Policy130 | Gain (points) | Descriptive paired 95% CI | Unadjusted paired p |
| --- | --- | --- | --- | --- | --- |
| CloseToasterOvenDoor | 166/512 (32.42%)| 212/512 (41.41%)| +8.98| [+4.10,+14.06]| 0.0006513359|
| OpenDrawer | 317/512 (61.91%)| 358/512 (69.92%)| +8.01| [+3.32,+12.70]| 0.0011504552|

The independent Linux CPU audit checks all 2,048 policy episodes, 256 policy
banks, 3,072 H50 sampler receipts and 1,228,800 valid controls. Every valid bank
has 12 resolved 400-control vector calls and exactly 600 controls per case.
Case-policy coverage, same-attempt pairs, original global-bank ABBA, source and
checkpoint-marker/READY joins, exact initial likelihoods, physical/public
states, episode outcomes and common sampler hashes all pass. The audit
independently reproduces the first/end-success statistics and Holm decisions.

Renderer pixels were diagnostic under the original declaration. 68/256 policy
panels have some differing pixels; all physical/public states match exactly.
Across all three views the largest component delta is 2 on the 0–255 scale;
the worst view/panel mean absolute delta is 0.000124 and worst differing-component
fraction is 0.01221%. Inputs were never replaced to force equality.
[Image-difference evidence](IMAGE-DIFFERENCES.json).

All-attempt conservative physical controls are 1,269,296, including 40,496 replay
or abandoned-attempt overhead, within the 96,000 recovery reserve. Nine
abandoned intents remain charged at their full requests; no unresolved intent
belongs to a valid final bank or the completed recovery. Zero SGD ran.

The original two one-GPU jobs retained 896 valid policy episodes. The first
shared-container attempt added 80 valid episodes before it was stopped for Ray
discovery/placement problems. The corrected eight-container phase added 928
valid episodes but its PyTorch master ended before the Drawer workers finished.
The final three-container phase completed only the remaining 72 paired Drawer
cases (144 episodes), using a barrier that waits for all shard results. All
three final clocks exited 0 and the last GPUs released 15:31:24 Shanghai.

| Cost component | Conservative GPU-seconds |
| --- | ---: |
| Original two jobs and earlier duplicate | 7,676|
| Four diagnosed acceleration setup attempts | 6,664|
| Eight-container phase | 9,200|
| Three-container completion | 1,665|
| **All attempts** |**25,205 / 27,600 (7.0014 / 7.6667 GPU-hours)**|

The 2,395 unused GPU-seconds are sealed. All allocations are released. The
original 64-case teacher-gate failures remain recorded; this is the separate,
predeclared 512-case endpoint comparison with a two-task multiplicity family.
The separately confirmed critic rollout/replay mismatch still needs repair
before further PPO; this evaluation measures the resulting actor checkpoints
and does not test the fixed critic or admit student fitting.

[Combined result](gpu-r6/RESULT.json), [independent raw audit](FINAL-AUDIT.json),
[cost/release evidence](COST-CLOSEOUT.json), [final migration](continuation-r5/MIGRATION.json),
[execution record](EXECUTION.json), [audit source](AUDIT.py).
