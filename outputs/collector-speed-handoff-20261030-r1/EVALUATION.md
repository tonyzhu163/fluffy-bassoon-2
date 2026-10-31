# Collection execution qualification

Checked 2026-10-30, Asia/Shanghai. These are bounded execution tests, not source
training campaigns or evidence of in-context adaptation. Existing scientific
runs and their immutable pins remain unchanged.

## Source quantity contract

The shared plan `162d6942e4af267e231d2059d1cf2a77a01a1144` specifies approximately
1M executed source actions for each of SAC, persistent PPO and separately
trained episode-reset PPO, on each benchmark: approximately 3M per benchmark,
6M across LIBERO and RoboTwin. AD and DPT reuse these histories. Validation,
evaluation, qualification, settling and failed/uncertain execution costs remain
visible separately. This target does not silently expand older campaign caps
or approved GPU hours.

RoboTwin's native whole-trial recipe has 100 rounds, six K4 trials per round:
600 trials / 2,400 episodes per learner. Its limits 600/500/400 imply at most
1.08M source actions per learner before early successes. LIBERO's future
balanced all40 whole-trial schedule must still be frozen by its execution owner;
its existing shared-1M campaign retains its original contract.

## RoboTwin

Final replay pin `1382e8dd9cfc40c66f93a49677e8e16cba16415e`, Baige
`job-ynkqj1uxbvg5`, exit 0, 171 scheduler GPU-seconds. Persistent and trained-reset
controllers each collected two K2 trials in two isolated simulator processes.
All eight episodes completed without an external budget cut. The four saved
rollouts replayed with exactly zero maximum joint likelihood and value error
at the original tolerances (2e-4 / 1e-5). Motor weights were unchanged and no
optimizer update ran. The small controller used `controller_microbatch=1`.

| Mode | Executed actions | Collection seconds, including resets | Replay log-prob / value max errors |
| --- | ---: | ---: | --- |
| Persistent PPO | 1,600 | 39.9034 | 0 / 0, both lanes |
| Episode-reset PPO | 1,420 | 43.4383 | 0 / 0, both lanes |

The earlier R3 probe (`8d36771ccb1f47f6674360c435f08e3717aaad1b`,
`job-rfxrtt2e90sl`) paired native full capture with counted final capture on
handover_mic/17001 and place_phone_stand/17002. Initial observations, complete
chunk action/observation hashes, rewards, termination and exact action counts
matched, 400 actions per scene/path. Observed reset-inclusive rates were
11.17 versus 27.76 actions/s. The existing production driver already uses a
qualified final capture path, so that ratio is not a gain over production.

R3's shared-motor process sweep observed 18.95 / 44.92 / 76.84 actions/s at
1 / 4 / 8 lanes, including resets. These are descriptive same-job systems
measurements, not end-to-end source training or GPU-hour savings. The probe
explicitly marked timing comparisons ineligible. Its later full-batch
controller replay failed; the failed run and evidence are retained. R4's B1
controller path resolves that measured replay mismatch without relaxing the
gate, but R4 did not repeat the sweep, K4, SGD, or corpus-inclusive throughput.

R4's log still contains OIDN invalid-handle and unsupported-CUDA-device
messages. Allocated renderer PCI identity passed, but image quality remains a
production qualification concern. Exact policy replay and a successful exit
do not establish image correctness. Compilation was not qualified.

Across R2/R3/R4: 11,369 confirmed executed actions; 11,769 conservatively
accounted including R2's uncertain additional 400; 630 scheduler GPU-seconds.
The original 12,000-action / 1,800-second envelope remains reserved in the old
canonical campaign and is not available as a fresh retry allocation.

Root `79858ce` additionally integrates atomic group admission/quarantine,
ordinary-resume rejection and the explicit B1 dispatcher option. Its backend,
pool and collector files match the R4 executable. The legacy N>1 dispatcher is
explicitly qualification-only because it lacks the production native cutoff,
corpus and durable pre-dispatch accounting contract.

## LIBERO

Final executable `c6f46e8cd80c7cca0877983b659c76ef26b1edbd`, Baige
`job-1l5gxk7pglgo`, exit 0, 131 scheduler GPU-seconds. All 16 live shadow
decisions compared native and cached B1 paths on the same observation and
explicit noise before executing native actions: feature and action errors were
exactly zero. Native serial repeat, pure cached scalar and both parallel B2
lanes matched initial actual simulator state, public observations and every
chunk's complete trace exactly on this one test case.

| Pure execution path | Executed task actions | Seconds including resets | Observed actions/s |
| --- | ---: | ---: | ---: |
| Native scalar | 77 | 9.8125 | 7.847 |
| Cached scalar | 77 | 9.0730 | 8.487 |
| Cached B1 motor kernels, parallel B2 simulators | 154 | 10.7281 | 14.355 |

The interleaved shadow path is excluded from this table. These single-case,
same-job rates are descriptive; the allocation probe marked timing comparisons
ineligible. Larger native JAX batches had failed the original action/feature
tolerances; the candidate retains B1 kernels and shared frozen weights.
There were no optimizer updates or whole-suite learning/evaluation tests.

Four same-owned-process, fresh-environment resets of Spatial00/state19/seed
1916071045 had exact float64 full simulator state (92 values), proprioception
and main RGB. The later three wrist images differed from the first at only
two channels by +1 LSB, RMS 0.00318944. No acceptance profile or wider model
tolerance was applied. Full raw state/RGB NPZ evidence is retained.

R4a charged 485 physical actions (385 task + 100 settling), zero unresolved.
All LIBERO qualification attempts total 759 actions / 411 GPU-seconds within
the original 2,880 / 1,800 envelope. Its canonical sidecar is settled; unused
2,121 actions / 1,389 seconds are sealed and unreleased. Compute was funded
inside the existing approved 32-hour LIBERO pool, not an additional approval.

## Production handoff

Both execution owners and the coordinator received exact tested pins and these
limits. Production batching preparation is based on the RoboTwin owner's
immutable `db0184089fa88a5f96c153f9b80029bb91b77def`, with its six complete K4
trials, per-task cutoffs, raw corpus, pre-dispatch physical ledger, schedule,
RNG, pending-data and atomic recovery contract preserved. New code requires
reviewed Linux CPU tests and a separately costed GPU qualification before
migration at a full checkpoint boundary. No active job/worktree was changed.

Reports alongside this file are thin copies of preserved remote reports.
Large rollouts and full simulator artifacts remain on project storage.
