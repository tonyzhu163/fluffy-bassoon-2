# Frozen policies on the official LIBERO-Pro2.0 test tasks

Passive V2 achieved 76/96 native successes with frozen π₀.₅ on simulated UR5e. Its advantage over passive Horn–Schunck was 20.8 percentage points (24 paired wins, 4 losses, task-clustered 95% interval +13.5 to +30.2). Its advantage over passive DIS was 37.5 points (39 wins, 3 losses, interval +27.1 to +47.9). Passive V2 scored 52/64 on the two OOD regimes, compared with 54/64 for the exact-transform oracle. Swapping the support donor gave 0/96. These results support the value of matched passive support on this policy and task set.

| Method | π₀.₅ overall (96) | π₀.₅ compositional OOD (32) | π₀.₅ continuous OOD (32) | π₀ overall (96) |
| --- | ---: | ---: | ---: | ---: |
| Exact transform | 82 | 26 | 28 | 0 |
| V2, passive support | 76 | 26 | 26 | 0 |
| EquivFlow-Procrustes, active probe | 54 | 21 | 14 | 0 |
| EquivFlow-Procrustes, passive support | 64 | 20 | 24 | 0 |
| V2, online streaming | 20 | 0 | 7 | 0 |
| Horn–Schunck, passive support | 56 | 16 | 20 | 0 |
| DIS + Procrustes, active probe | 47 | 15 | 17 | 0 |
| DIS + Procrustes, passive support | 40 | 12 | 16 | 0 |
| Broyden UVS, active probe | 31 | 8 | 13 | 0 |
| ReGround-style babbling | 6 | 0 | 3 | 0 |
| Perceiver-ICBA-style, passive support | 33 | 0 | 5 | 0 |
| V2, swapped support | 0 | 0 | 0 | 0 |

Active EquivFlow scored 54/96 against 47/96 for active DIS. It gained 7.3 points overall (21 wins, 14 losses, task-clustered 95% interval −8.3 to +22.9). On continuous-angle OOD it scored 14/32 against 17/32 for DIS. The active reader’s lower development angle error did not establish a reliable gain in official native success. The streaming V2 arm, which started each query at identity and learned only from its own history, achieved 20/96 overall and 7/64 OOD. The 76/96 passive result used a prior rollout on a separate support layout. [π₀.₅ summary and paired intervals](evidence/pi05-summary.json) · [Per-query outcomes](evidence/pi05-rows.jsonl)

Frozen π₀ scored 0/96 with every method, including the exact transform. Its official UR5e score provides no separation among readers. On one matched BASE development start, the same loader and controller succeeded on Panda and failed on UR5e. π₀.₅ succeeded on both. This source control leaves the cause of the π₀ transfer failure open. [π₀ summary](evidence/pi0-summary.json) · [Per-query outcomes](evidence/pi0-rows.jsonl) · [Source control](../xemb-pi0-source-smoke-20261122-r1/RESULT.md)

## Evaluation and recovery

Each policy used eight official test tasks on BASE, one support/query layout pair (11/13), 12 command transforms per task, and 12 methods per query. This gives 96 unique queries and 1,152 scored terminal episodes per policy. Native simulator task success is the endpoint. The learned readers used training seed 41. Intervals use 2,000 paired bootstrap draws over the eight tasks. The 96 queries share task layouts and should be interpreted with that clustering.

The two original eight-card jobs at `437da7f6` each completed seven task shards. Both rejected a Goal 6 reset. A single-card job at `ff2ddf0a` replaced that whole shard for each policy using its original support tapes. Recovery included an unscored simulator warm-up and bounded pre-action retries on the same locked start. The original mismatch did not reproduce in 32 CPU creates. The two completed recoveries logged no rejected creates. The strict merge verified task, transform, regime and all 12 methods on every expected index. The six overlapping original π₀ query rows agreed on every method’s outcome and were counted once. The π₀.₅ original Goal 6 shard had no complete rows. [Recovery plan](PLAN.md) · [Linux qualification](evidence/recovery-qualification.txt) · [π₀.₅ shard hashes](evidence/pi05-provenance.json) · [π₀ shard hashes](evidence/pi0-provenance.json)

## Other benchmarks and resource use

On ManiSkill3 xArm6, EquivFlow without vertical correction achieved 5/48 OOD successes with the scripted motor and 0/24 on continuous angles. Corrected V2 previously achieved 11/48 on the same seed schedule. The registered 50% ManiSkill scaling threshold is still unmet. On LIBERO development, the requested S15 comparison produced four paired losses for each active reader and the exact-transform oracle. The CPU SO(3) reference also found a rotation error with exact motion and unequal gains. These are independent limits on the proposed extensions. [ManiSkill EquivFlow](evidence/ms3-equiv-flow-summary.json) · [ManiSkill V2](evidence/ms3-v2-summary.json) · [Camera comparison](../xemb-camera-dev-20261122-r1/RESULT.md) · [Simulator coverage](../xemb-stage2c-cpu-20261121-r1/RESULT.md) · [SO(3) reference](../xemb-stage2d-cpu-20261121-r1/RESULT.md)

The two main jobs, their two single-card recoveries, the ManiSkill run, the source control and the camera pilot used 11.01 launcher GPU-hours on Baige. All seven are terminal. The campaign attempts covered by the ledger total 12.21 launcher GPU-hours including earlier attempts. Scheduler startup and teardown are outside these measurements. [Job IDs, pins, exits and GPU counts](evidence/job-receipts.json)
