# Trained N6 pair completed a production round

**PASS:** the original trained persistent PPO and trained-reset PPO completed one more full source round each on one96GB-class GPU. Baige job-fcxa9mm1mh84 ended Success after **450schedulerGPU-seconds (7m30s)**. Runtime57016ba/c353 is unchanged; this advances the original100-round source goal.

| Learner | Rounds | CompleteK4 | Source actions | PPO updates / SGD steps |
| --- | --- | --- | --- | --- |
| Persistent PPO | 52→53 | 312→318 | 499,022→508,540 (+9,518) | 52→53 / 260→265 |
| Trained reset PPO | 54→55 | 324→330 | 516,264→525,625 (+9,361) | 54→55 / 270→275 |

Both collected24 complete episodes, respecting each task's400/500/600 limit and early success. **18,879 new physical source actions**,611 recorded decisions,12 completeK4 trials and two real PPO updates were added. No target/evaluation actions or fresh learners were introduced by this slice. Original canonical caps and100-round schedule remain binding. Source training is not complete.

## Throughput and memory

The measured full parent runtime, including initial journal/checkpoint admission, GPU probe, model/simulator startup, collection, corpus I/O, updates, cleanup and final audit, was439.776seconds:42.93actions/s. The actual450-second scheduler charge gives **41.95actions per GPU-second**. The two concurrent source collection groups took221.415/233.474seconds including their reset/corpus work; these exclude cold startup and final audit and must not replace the full-job number.

Peak GPU memory, including the restored trained Adam states and actual updates, was70,387,761,152bytes = **65.55GiB /70.39GB**, below85GiB/90% abort. Both drivers exited0; owned cleanup left no live simulator sessions and zero GPU context memory. No automatic retries occurred. The unused2,250seconds of this operational reservation are sealed; future original-goal slices require their own measured reservation. Unused action capacity does not cause additional trials.

Historical scalar measurements used different cases and timing boundaries, so this is not a paired causal speedup ratio. It supplies a measured continuation cost and validates the trained batched execution path. A continuous supervisor is being prepared to amortize repeated admission/startup while preserving complete source rounds and evaluation cursors. SAC continues on its original scalar worker.

## Correctness and recovery

The actual pre-CUDA admission passed in54.98seconds, binding terminal predecessor release, both complete checkpoints/Adam/sampler/private+globalRNG, original parent/canonical locks/journal/residuals, assets and qualified LR. One fresh strict-idle GPU probe and both real native-noise parity checks passed (max error0.0).

All12 trial likelihood and value replay errors were **exactly0.0**, within the unchanged2e-4/1e-5 tolerances. The original six-trial/five-epoch/LR1e-6 update guards passed. Durable group acknowledgments and final full coordinator checkpoints preserve empty pendingPPO, task schedules, all completed stage50 units and source/target separation. The frozen motor remains unchanged.

The separate CPU final audit and an independent artifact audit passed. Both final coordinators match manifest SHA and all263ZIPmembers each pass CRC. Independent native-mask inspection of all611 new decisions reproduces9,518+9,361actions, and final canonical heads have zero pending/uncertain charges. All raw RGB/decision/native references were retained. [Independent receipt](receipts/independent-final-audit.json), [allocated audit](receipts/audit.json), [scheduler proof](receipts/scheduler-terminal.png).

A small saved-image inspection across six task-lane frames/18 camera views found coherent, distinct, nonblank RGB geometry. Images have visible grain and bright backgrounds; this sample does not establish that existing OIDN warnings are harmless. Rendering/runtime was not changed. [Image receipt](receipts/rgb-quality-receipt.json).

The prior checkpoint/ownership migration, original authority and actual predecessor costs are documented in [MIGRATION.md](MIGRATION.md). The earlier first-fit AD/DPT panel and source-stage50 comparisons remain separate scientific evaluations; this systems result does not establish memory-based adaptation or an algorithmic success gain.
