# Continue to matched data quotas

2026-10-30T11:20:09.529026+08:00 Asia/Shanghai. Human: “wait dont limit time like this，how much are we aiming to collect for either”. This supersedes the previous two-hour overall continuation ceiling. Continue the existing experiment to its recorded action targets without another approval pause at each allocation boundary. Existing current allocation remains immutable; a controlled wallstop is a checkpoint boundary, not experiment completion.

| Learner | Executed source-action goal | Intermediate comparisons |
| --- | ---: | --- |
| DSRL-style feature-conditioned SAC |128000|32000/64000/128000|
| Persistent GTrXL/PPO |128000|32000/64000/128000|
| Separately trained episode-reset GTrXL/PPO |128000|32000/64000/128000|

Source total384000. Planned maxima including stage sourcevalidation86400, targets115200, native4800 and prior13026total603426. The1Mcombined RoboTwin executed-action ceiling remains binding; unallocated396574is not a separate allowance or automatic new scientific scope. This is an initial scale comparison, not final-scale meta-pretraining. At H32,128kactions corresponds to at least4000motor decisions perlearner, with more decisions when chunks terminate early.

Initial actual corpus-collection throughput25.87actions/s extrapolates the590400new planned actions to~6.34GPUhours for collection alone, excluding optimizer work, startup, all-task variation and retries. SAC UTD20 throughput has not yet been measured. Meter actual scheduler seconds and action-ledger charges. Report observed learning updates and revised remaining-cost estimates as they become available; do not call collection or an elapsed slice successful learning.

At a controlled wallstop, first verify terminal job identity, full coordinator/replay/optimizer/sampler/RNG/pendingPPO/corpus/stage/evalcursor recovery and exact ledger. Resume from coordinator checkpoint with the same immutablec3b6281source and unchanged contract/qualified LR, using a fresh output/worktree identity and scheduler allocation. Reuse optimizer-calibration admission; no recalibration or diagnosticweight transfer. Review latestmain before a fresh submission and document any relevant compatibility issue; never silently migrate active protocol. Each scheduler slice retains a finite wall limit to recover from platform interruptions, but these slice boundaries no longer require a new approval or stop the overall campaign.

Example first continuation (only after currentjob terminal and recovery audited):

```sh
bash /mowenjie/fluffy-bassoon-2/worktrees/robotwin-staged-20261030-r3/scripts/diagnostics/robotwin_staged_train.sh c3b6281c032e2c712a62191b03219b1e70e12947 robotwin-staged-20261030-r3 /mowenjie/fluffy-bassoon-2/outputs/robotwin-staged-20261030-r2/experiment/coordinator-recovery.pt 120
```

Create the named detached worktree through Git at the exact pin before that submission; do not precreate runtimeoutput. Automatic scheduler retry remainsoff. On failure, diagnose and preserve existing data before a fresh pinned retry. Policy KL rejection remains a failure requiring diagnosis, not routine wall-clock resumption. No parallel duplicate collector; no tied32 substitution. Completion requires all recorded source milestones/evaluation units and integrity audit, or reaching the shared action ceiling with the limitation explicitly reported.
