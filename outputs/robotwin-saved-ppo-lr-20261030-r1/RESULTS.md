# Real saved-data PPO updates: execution verified, calibration unfinished

Baige **job-ypsxplhesd9x**, source`3e1de8e80d3f333adeba252884e173ce54f8b467`,
exited0 and succeeded2026-10-30 02:28:06 Shanghai. Scheduler50seconds;
process17.03seconds. Actual total GPU allocation accounting is**44m32s/45m**,
leaving28seconds. All jobs are terminal; no further GPU job fits this envelope.
No new robot actions were executed by any optimizer diagnostic.

The same immutable12source episodes/4,273actions/140noise decisions generated
three **independent** five-epoch PPO updates, five optimizer steps each. These
are not three sequential meta-training updates. All original weights, old
likelihoods/values, source sampler and RNG were restored for each lower-rate
fork. The3e-4reference is the already retained bad-update checkpoint from
job-yem1ctw27j26, evaluated here without repeating its SGD.

| LR | Update time | Exact post KL(original || post) | Post joint ratios outside PPO clip | Gate |
| --- | ---: | ---: | ---: | --- |
|3e-4|5.58s inpriorjob|12.84420|100%|fail|
|3e-5|4.94s|0.43548|97.14%|fail|
|3e-6|5.20s|0.01065|75.00%|fail on clip fraction|

The gate was declared before this run: mean analytic joint KL<=.02 and post
clip fraction<=.2, plus exact behavior replay, reproducible reconstruction,
finite nonzero gradients, changed weights and full recovery. Neither candidate
passed it. The3e-6update substantially reduces drift and meets the analytic KL
part; this is not proof of instability at that rate, but it is not a pass of
the full conservative gate. Do not relabel it after seeing the result.

## What caused the apparent KL explosion

The earlier8.86e19value was the update's ratio-based sample KL estimator,
not the exact Gaussian KL. Post-update float64 ratio-based KL is8.37e24for the
3e-4reference, while exact conditional Gaussian KL is12.8442. The joint density
ratios in1024dimensions amplify modest coordinated mean changes enormously.
The exact KL decomposition is mean-shift12.84336 versus variance0.000841.
Standard deviations remain0.60563–0.60739around the original0.60653; this is
not a measured standard-deviation collapse.

At3e-6, the KL decomposition is mean-shift0.01065328versus variance1.34e-7.
Unclipped gradient norms for its five steps are91.17,79.27,52.29,33.22,27.42;
at3e-5they are91.17,22.48,4.44,9.38,17.63. Their identical first gradient is
consistent with the same initialized policy/data/RNG. Both lower-rate forks
repeat independent recurrent reconstruction from empty history with maximum
mean and log-probability difference0.0. The original behavior replay also
matches exactly. These checks find no reconstruction discrepancy in this
batch; they do not exclude every possible recurrent-training issue.

Frozen feature RMS is1.280(max58.02), normalized proprioception RMS0.601,
previous-noise RMS0.737, initial actor-input RMS0.391. All observed arrays are
finite. Full per-decision means/std, actor inputs, raw policy inputs/actions,
float64 joint log-probabilities/ratios, KL components and trial indices are
saved in the three NPZ files underdiagnostic/. This is source data only:
place_phone_stand,place_object_stand,pick_diverse_bottles; no held-out robot
outcomes selected any parameter.

## Recovery and next work

Both candidates preserve a complete independent learner/optimizer/sampler/RNG
recovery, consumed3trials/12episodes and0pending trials. Each has63AdamW states
atstep5, config LR equal to the optimizer LR, and an independently verified
SHA/ZIP-CRC receipt. Original input SHA is unchanged. Raw NPZ/checkpoint bytes
remain untracked; seeRECOVERY-AUDIT.json anddiagnostic/result.json.

SAC remains at139source replay rows and0updates, below the declared500decision
warmup/batch256; its recipe was not reduced to force an update. The main batch8
comparison has not advanced. There is no robot-performance or adaptation result.

The [revised two-hour proposal](../robotwin-staged-20261030-r2/PROTOCOL.md) first
checks new rollback behavior, then tests predeclared1e-6and3e-7rates on this
same source batch. Only a passing setting permits fresh six-task matched source
collection; otherwise it stops before robot actions. PPO updates with excessive
reported approximate KL are rolled back and the pending data preserved. This
is a prepared unlaunched revision; new guard tests have AST checks only and
must run inside the approved allocation. Larger resource approval is pending.

The analytic KL reduction is explicitly
`mean_over_140_decisions(sum_over_1024_noise_coordinates(KL(old_normal || new_normal)))`.
It is conditional on stored public histories; it is not per-dimension KL or
whole-trajectory KL. The current RoboTwin interface is fullH32×32noise, while
LIBERO's current controller repeats one32D vector over nativeH10. This matters
for cross-benchmark interpretation. A tied32D RoboTwin variant requires fresh
rollouts for all compared methods, not relabeling these recorded actions.
