# Successful source behavior lengths, 2026-10-31

**Small descriptive shortening on some cases, not a clear large teacher
advantage.** All16 predeclared source cases succeeded under native and the
corresponding frozen per-task SAC mean/sample. Mean execution length fell by
only1.375 primitives for deterministic SAC mean and0.625 for stochastic SAC
sample (approximately0.93% and0.42% of native mean length). Gains vary by goal.
This measures successful trajectory length, **not source-training cost,
adaptation sample efficiency or a general deployment sample-efficiency gain**.

Only the two terminal `index.json` files were read and SHA-checked against the
already verified completion receipts. No raw trace audit, model/framework,
inference, tests, GPU work, new rollouts or active-output changes occurred.
Exact per-case lengths and metadata hashes are in `receipt.json`.

## Frozen source SAC:16 paired cases, all successful

|Condition |Primitive mean / median |Chunk mean / median |Total primitives |
| --- | ---: | ---: | ---: |
|Native |147.750 /121 |15.3125 /12.5 |2,364 |
|Corresponding SAC mean |146.375 /127 |15.1250 /13.0 |2,342 |
|Corresponding SAC sample |147.125 /121 |15.1875 /12.5 |2,354 |

A chunk proposes up to10 simulator actions; success can stop inside a chunk.
These primitive counts exclude the identical10-action settling per episode.
They are the available physical time-to-success measure; no simulator clock
frequency is assumed to convert them into seconds.

|Paired treatment minus native |Mean / median primitive delta |Shorter / tied / longer |
| --- | ---: | ---: |
|SAC mean |−1.375 /−1.5 |11 /1 /4 |
|SAC sample |−0.625 /−5.0 |10 /0 /6 |

Marginal medians and medians of paired deltas are different statistics. The
actor-mean median127 versus native121 does not contradict11 individually
shorter pairs: cases have different baseline lengths and a few larger offsets.

Per goal:each row averages its two fixed source-local cases (states40/41).
The entries in parentheses are paired mean primitive deltas versus native.

|Source goal |Native mean |SAC mean |SAC sample |
| --- | ---: | ---: | ---: |
|libero_10:00 |283.0 |277.0 (−6.0) |269.0 (−14.0) |
|libero_10:01 |245.5 |238.0 (−7.5) |238.0 (−7.5) |
|libero_goal:00 |116.5 |115.0 (−1.5) |116.0 (−0.5) |
|libero_goal:01 |82.5 |83.5 (+1.0) |85.0 (+2.5) |
|libero_object:00 |140.0 |138.0 (−2.0) |140.5 (+0.5) |
|libero_object:01 |125.5 |127.0 (+1.5) |121.0 (−4.5) |
|libero_spatial:00 |74.0 |71.5 (−2.5) |72.0 (−2.0) |
|libero_spatial:01 |115.0 |121.0 (+6.0) |135.5 (+20.5) |

Recorded elapsed wall seconds have means9.32 native,5.73 SAC mean and5.80
SAC sample, with medians8.57/4.95/4.83. **Do not interpret that as a policy
speed gain.** Native executes first and receives first-per-goal motor/feature
qualification and JIT overhead; the method order is fixed. No clean wall-time
comparison is available from these receipts.

Eight corresponding per-task actors and two cases/goal are not16 independent
goals, a joint all40 SAC, or an all40 success evaluation. The small length
contrast can coexist with the ceiling16/16 success score; it does not establish
a strong history teacher or explain OOD transfer.

## Frozen source PPO: per-attempt executed lengths

Each cell contains16 physical cases. Failures time out at400 primitives, so
all-episode lengths include censoring and must not be called success latency.

|Policy / attempt |Successes |All-episode primitive mean / median |Mean chunks |Successful-only primitive mean |
| --- | ---: | ---: | ---: | ---: |
|Full1 |16/16 |158.3125 /138.0 |16.2500 |158.3125 |
|Full2 |16/16 |157.1250 /144.5 |16.0625 |157.1250 |
|Full3 |15/16 |179.4375 /144.5 |18.3125 |164.7333 |
|Full4 |16/16 |162.6875 /136.5 |16.7500 |162.6875 |
|Trained reset1 |16/16 |159.1250 /139.5 |16.3125 |159.1250 |
|Trained reset2 |16/16 |165.2500 /138.0 |16.8750 |165.2500 |
|Trained reset3 |14/16 |184.5625 /146.5 |18.8750 |153.7857 |
|Trained reset4 |15/16 |165.5625 /133.5 |16.8750 |149.9333 |

Attempt4 minus attempt1, restricted to cases successful at both attempts:
full16pairs have mean+4.375 primitives, median−1,9shorter/0ties/7longer;
trained reset15pairs have mean+0.5333, median0,7shorter/2ties/6longer
(one censored case excluded). Neither sequence shows monotonic shortening.
PPO receipts do not record per-episode elapsed seconds. Separately trained
full/reset policies have different weights; these attempt-length summaries
do not isolate memory, and successful-only subsets can hide failures.

Both underlying panels use fixed frozen policies and correlated repeated
attempts. The next interpretation should retain success, censoring, actual
interaction cost and separately recorded source-training cost together.
No extra experiment or collection is justified by a small length reduction
alone, and none was launched by this analysis.
