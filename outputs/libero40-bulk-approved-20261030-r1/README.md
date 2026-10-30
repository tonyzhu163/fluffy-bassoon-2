# LIBERO collection and controller training

Verified at 2026-10-30 15:43 Shanghai. All 48 canonical source streams are
terminal, exit0 and collection-qualified: 40 training streams, 8 validation
streams. Eight parallel SAC allocations produced the45 fresh streams; three
previously completed canonical streams were preserved.

| Partition | Complete K4 cases | Episodes | Simulator actions |
| --- | ---: | ---: | ---: |
| Training, all40 tasks |527|2,108|326,277|
| Validation,8 fixed tasks/disjoint states |146|584|85,301|
| Selected corpus total |673|2,692|411,578|

A K4 case repeats one task/state/seed for four actual episodes. Different
seeds do not alone establish different nominal physical initial states;
per-stream state/seed counts are in [terminal receipts](terminal-source-receipts.json).
Separately spent prior source qualification remains12,944 actions. This is
an evolving SAC history corpus, not a demonstrated expert/recovery corpus.
Target90/Plus/Pro data did not enter source pretraining.

The shared full and independently trained episode-reset PPO runs each
completed40 iterations,80 optimizer steps and160 episodes, visiting all40
source goals. Both then exited1 on trial41's same-scene initial-observation
guard. Finite losses and complete policy/optimizer/sampler/motor/globalRNG
recoveries are preserved. Both requested the same nominal state, but actual
post-settling physical state was not recorded, so pixels-only drift is not
established. The repair retains the exact guard, restarts the owned simulator
for each episode and saves failed-episode evidence before the guard. Both r2 allocated probes passed319 Linux tests and then reproduced the
observation mismatch before any optimizer update. Saved full-arm reset arrays
establish that all92 float64 simulator-state entries,8 float32 proprio
entries and every main-camera byte are identical. Only two wrist-camera
channels at one pixel differ by1/255. An explicit source40-only camera
rounding tolerance is being implemented; physical/proprio identity stays
exact, all raw frames remain intact. A new allocated qualification must pass
before continuation. Original runs remain
immutable; continuation must preserve all partial physical costs and the
original172k action reservation. Cumulative time charges after the probes:1,925s full/1,920s reset; cumulative
physical charges26,959/27,105 including the extra40 settling actions each.

All40 offline AD/DPT training now proceeds independently of this PPO repair,
using the exact completed ledger and continuous25/100/500 update milestones.
The first allocated preflight caught a nonexistent test filename before any
training; source33827ab fixes that name, with a fresh r2 identity. Corrected job `job-7pdp4ywjmrrg` passed its actual GPU probe and89 Linux
tests (2 probe-environment JAX skips). AD-full has saved25/100/500 update
policies/full recoveries; AD-static has started, with DPT arms following.
No new simulator actions are needed for offline training.

Human approved32 additional LIBERO GPU-hours. See [approval](approval.json)
and the [live compute ledger](compute-ledger.json). After verifying terminal
SAC worker6 at44m31s scheduler duration and a conservative2,719s script charge,
3,200s of its7,200s hold is reassigned to offline training including the failed
preflight. The remaining4,000s hold exceeds its actual charge; total holds
remain115,200s. Frozen source validation's4h/56k action hold remains intact.
No fresh full reservation is granted on a retry.

Later human steering accepted approximately1M source actions per learner on
both benchmarks, with complete trials and benchmark-appropriate episode
horizons. That future recipe requires explicit provenance/costs and does not
silently rewrite this existing1M-combined campaign. Final target budget is
separate. Earlier results still establish no reliable memory or high topline
performance benefit. Monitor: `continue-approved-libero-source-collection`,
20-minute cadence, quiet while unchanged and healthy.
