# Target first-wave result

Audited 2026-10-30 21:35 Shanghai from all six terminal exit0 indexes and all558
committed group-result hashes. All626 planned episodes finished. Actual cost:
189,051 simulator actions including settling;7,957 conservatively charged GPU
seconds (2.21 GPU-hours), inside the six-hour target hold. All six before/after
motor hashes match; all actual initial observations match within paired cases.
No deployment optimizer updates. Fixed500 models were selected before outcomes.

| Zero-support query | LIBERO90 | Plus development panel | Pro development panel |
| --- | ---: | ---: | ---: |
| Native pi0.5 |31/90|4/4|4/4|
| AD full |26/90|4/4|4/4|
| AD trained static |26/90|4/4|4/4|
| DPT full |26/90|4/4|4/4|
| DPT trained static |27/90|4/4|4/4|

| Same nine LIBERO90 cases | K0 query | K1 retained history | K1 same-weight episode reset |
| --- | ---: | ---: | ---: |
| AD |5/9|6/9|5/9|
| DPT |5/9|6/9|5/9|
| Native reference |6/9|not collected|not applicable|
| AD trained-static reference |6/9|not collected|not applicable|

The only full-versus-reset success difference is LIBERO90 task70 for BOTH
AD and DPT. Its support episode succeeded; all six successful retained-history
queries on this panel also had successful support. This does not demonstrate
failed-attempt recovery. Each retained-history method merely reaches native's
6/9 on this selected panel. Across all90 goals, every student underperforms
native by4–5 successes. This screen does not establish high performance, a
reliable memory benefit, or improved sample efficiency over ordinary DSRL.
Ordinary DSRL and the still-running direct PPO arms are not included here.

Plus/Pro results are ceiling outcomes on tiny280-step development panels:
Plus variants667/290 × states0/1; Pro swap/appearance × seeds123/124 at state0.
Pro seed repeats are not independent new tasks/nominal states. LIBERO90 uses
one initialization/seed per goal, one training seed, with previously inspected
development cases disclosed. These are not full Plus/Pro leaderboard scores
or a confirmatory final architecture ranking.

The next scientific step is source-teacher/control diagnosis and the bounded
PPO comparison, before approving a much larger source corpus. Keep native,
zero/tied-Gaussian steering, trained-static and history-erasure comparisons
explicit. More source data or optimizer steps may help, but this result alone
does not justify claiming that scaling will fix the deficit.

Detailed scores, per-case outcomes, exact receipt hashes, physical costs and
worker integrity flags are in [RESULTS.json](RESULTS.json). Original source
histories, model/recovery files, raw observations, contexts and RPC journals
remain at their immutable Baige paths. No new evaluation job is needed to
finish this wave.

Scheduler completion was independently verified for all six jobs. Observed
running durations were1,411/1,211/1,290/1,171/1,311/1,250seconds. Parent
reservations retain each more-conservative charged time plus50seconds;
13,343unused target GPU-seconds were explicitly released, without spending
them. Source/speed/validation holds remain separate.
