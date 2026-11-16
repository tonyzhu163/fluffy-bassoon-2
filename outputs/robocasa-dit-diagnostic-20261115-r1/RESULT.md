# Frozen DIT diagnostic: teacher quality, advantage weights and history

**Completed November 16 on Baige CPU, with zero additional GPU-hours. The data
show uneven PPO quality, almost uniform DIT weighting, and a small contribution
from prior episodes to the saved action-flow predictions. Stronger PPO alone
is not an established remedy. Qualify the learning signal before more collection.**

## Successful behavior exists, but the learning trajectories improve modestly

Each task contributes 3,360 training episodes from collection updates 0–29.
These are descriptive collection rates, not paired SFT/PPO30 evaluations.

| Task | Successful training episodes | First 10 updates | Last 10 updates |
| --- | ---: | ---: | ---: |
| OpenDrawer | 2,179 / 3,360 (64.9%) | 62.9% | 67.2% |
| CloseToasterOvenDoor | 1,100 / 3,360 (32.7%) | 29.8% | 33.5% |
| CoffeeSetupMug | 1,072 / 3,360 (31.9%) | 30.5% | 33.1% |
| PickPlaceCounterToStove | 1,998 / 3,360 (59.5%) | 58.3% | 61.0% |
| TurnOffStove | 537 / 3,360 (16.0%) | 16.2% | 14.9% |
| TurnOnSinkFaucet | 1,385 / 3,360 (41.2%) | 38.4% | 44.9% |

Thus there are many successful examples to learn from, while teacher quality
and improvement vary by task. The separate paired baseline checks established
PPO30 improvement over SFT on Drawer and Coffee; the other four intervals
included zero. Collection curves above do not replace those controls.
[Source rewards](evidence/inference01/SOURCE-REWARDS.json) ·
[Paired source baselines](../robocasa-six-task-scale-20261113-r1/SIX-TASK-BASELINES.json).

## DIT barely changes the emphasis on successful episodes

**Interpretation update, November 16:** the all-query success label below
includes queries after the single success reward has already been collected;
their remaining return is zero. Those rankings are not action-quality labels.
The first-query results are unaffected. The subsequent
[project audit](../project-diagnosis-20261116-r1/RESULT.md#success-suffixes-matter-for-interpreting-the-value-diagnostic)
reports the pre-success analysis: some modest positive rankings appear, while
weights remain nearly uniform. Original measurements are preserved here.

The frozen DIT models use weights `exp(Q−V)` at temperature 1. Across the
diagnostic task panels, their effective sample size is **99.50–99.86%** of the
number of uniformly weighted queries. This describes the sampled panel, not
every training minibatch. The weights are close to uniform.

The table reports first-query discrimination of eventual episode success.
AUC 0.5 means chance ranking; comparisons match task, collection update and
query position. Brackets are marginal 95% intervals from 4,000 whole-stream
bootstrap draws. The last columns show how weighting changes the successful
example share over **all 12 query positions**, in percentage points; these are
not changes in robot success rate.

| Task | DIT3 first-query AUC [95%] | DIT6 first-query AUC [95%] | DIT3 weighted-success shift | DIT6 weighted-success shift |
| --- | ---: | ---: | ---: | ---: |
| OpenDrawer | 0.430 [0.253, 0.586] | 0.420 [0.290, 0.531] | −0.257pp | +0.188pp |
| CloseToasterOvenDoor | 0.481 [0.323, 0.634] | 0.502 [0.346, 0.672] | +0.046pp | +0.142pp |
| CoffeeSetupMug | 0.505 [0.324, 0.673] | 0.565 [0.383, 0.737] | +0.122pp | −0.039pp |
| PickPlaceCounterToStove | — | 0.502 [0.331, 0.673] | — | −0.453pp |
| TurnOffStove | — | 0.434 [0.235, 0.667] | — | −0.172pp |
| TurnOnSinkFaucet | — | 0.335 [0.201, 0.484] | — | +0.165pp |

No DIT first-query interval is wholly above chance. Faucet's negative interval
is an exploratory result among many comparisons. Equal-task average shifts
in successful-example share are −0.030pp for DIT3 and −0.028pp for DIT6;
these descriptive means do not establish a harmful effect.

There is some outcome information in Q itself: six-task first-query Q AUC is
0.681 [0.556, 0.800] on Toaster, 0.761 [0.594, 0.898] on CounterToStove, and
0.777 [0.623, 0.903] on Faucet. Corresponding Q−V rankings are weaker.
However, advantage is supposed to subtract expected state value: outcome
association is not a counterfactual action-quality test. These measurements
do not prove that subtraction, the critic objective, or a particular action
estimate is wrong.

IC-IQL also has nearly uniform weights and no first-query AUC interval wholly
above chance. Its all-query successful-example shifts range from −0.174pp to
+0.220pp. Several later-query intervals are positive, but later observations
can already reveal success. This is not a strong success-selection signal
unique to either learner. Full/reset value statistics, Q/return errors, all
weight distributions, and every interval are retained in the
[complete statistics](evidence/analysis01/RESULT.json).

## Prior episodes barely affect the archived action-flow outputs

These are the existing 128 held-out native flow-matching queries per actor,
with identical observations, noisy actions and timesteps under full versus
episode-cleared history. They were measured at training completion, not by a
new simulator evaluation.

| DIT fit | Total steering RMS / native RMS | Full/reset difference RMS / native RMS | Relative imitation-loss improvement from history |
| --- | ---: | ---: | ---: |
| Three tasks | 4.22% | 0.234% | +0.0056% |
| Six tasks | 3.95% | 0.214% | −0.0147% |

The adapter changes the native prediction, but most of that change survives
clearing prior episodes. History has little effect on this offline imitation
loss. The quantities above are action-flow velocities, not integrated robot
actions, and small flow changes can still alter a trajectory. The separate
completed native comparison remains the relevant success test: neither DIT
fit established a positive ID/OOD history benefit.
[Archived intervention evidence](evidence/inference01/ARCHIVED-FLOW-EFFECTS.json) ·
[Native results](../robocasa-six-task-scale-20261113-r1/RESULT.md).

## What this resolves

Weak PPO is a plausible limitation, especially on Stove, but successful
behavior is present and the current learner applies little outcome-selective
weighting or cross-episode influence. These are concrete weaknesses in the
current recipe. A useful next experiment would compare uniform, estimated and
known-outcome weighting under a controlled history intervention before scaling
teachers. No new fit, PPO collection or simulator run was launched here.

This diagnostic does not identify why one frozen fit beats another, quantify
the benefit of stronger PPO, or demonstrate causal action advantages. It uses
one fitted seed per model and only 16 held-out streams per task, with unadjusted
intervals. Outcomes and later-query state information must not be treated as
labels for the causal value of individual actions.

## Execution and verification

- Code pin `5b95c5ba59a64d11a0ec5df984a9244a2be75d70`; core value/history code
  unchanged from fit pin `4987585e`. [Launch](LAUNCH.json), [exit](evidence/EXIT.json).
- All 16 validation streams/task; updates 0, 9, 19, 29; all 12 queries/episode;
  up to 11 preceding complete episodes. Four models produced 13,824 records
  from 4,608 unique source queries in 384 episode records. Reuse across models
  does not increase the number of independent source examples.
- All dataset/checkpoint hashes and strict value loads passed. Four long-window
  full/reset Q/V checks matched production incremental inference with maximum
  absolute error `3.58e-7`. Full/reset outputs agree exactly when no prior
  episode exists. Independent explicit-pair calculations verified 144 reported
  AUC/weight-shift statistics. [Audit](evidence/AUDIT.json).
- Both CPU stages exited 0. GPU use is 0; the old campaign remains closed at
  219.5/230 GPU-hours. The complete raw diagnostic evidence is 984,600 bytes in
  cold storage, with every readback hash verified. [Restore inventory](evidence/ARCHIVE.json).
- Statistics SHA256:
  `324882e33c8724e07d360400632fd19ddafa2649394b18ec8cfa07c11be35b1a`.
