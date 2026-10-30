# LIBERO baseline completion and active-continuation receipt

Campaign date2026-10-30; verification timestamp is in `receipt.json`.
Terminal metadata and streamed byte hashes were read directly from saved Baige
artifacts. No model construction/forward, tests, simulator interactions,
allocations, budget edits or active-output changes occurred. The two long runs
below are an explicitly timed snapshot, **not completion results**.

|Component |Baige job(s) |Source pin |State / result |Actions |Charged GPU-seconds |
| --- | --- | --- | --- | ---: | ---: |
|Frozen source PPO validation |`job-ztzwinm2b802` |`9002a2d` |exit0;32 K4 trials/128episodes;full63/64,trained reset61/64 |22,753 |1,203 |
|All90 noise screen |`job-g8vas173mmy5`, `job-834arjyjj6mp` |`53b6c0d` |exit0;360queries;native29,zero25,tied28,clipped29 /90 |122,993 |4,411 |
|Frozen PPO target wave |`job-3ckrfuvifjwk`, `job-mrfdf4ipqicm` |`7780fb3` |exit0;297groups/324episodes;native30,full24,trained reset26 /90;fixed9 K1 all5/9 |108,894 |4,526 |
|Source SAC competence R1 |`job-qioinn358m7l` |`7780fb3` |exit1;probe interpreter/overlay mismatch before rollout |0 |78 |
|Source SAC competence R2 |`job-zxgkfebhs4k8` |`492c289` |exit0;native/mean/sample each16/16 on the source panel |7,540 |532 |
|Six-case target-online onset |`job-df23nkw7ne5j` |`ba4674f` |exit0;96train+36query episodes;4/6 at0/1/4/8/13/16 budgets |35,558 |1,851 |

Unique totals for **these terminal components only**:297,738 actions and12,601
charged GPU-seconds. They exclude original source pretraining. Failed R1 setup
is counted once; R2 is its successful retry, not an additional20k collection.
Scheduler UI durations differ from conservative clocks: teacher R2 UI490s
versus532 charged; PPO target UI2191/2232s versus2247/2279 charged. No refund or
hold change is performed here.

Evidence checks in `receipt.json`: every terminal index/worker-clock SHA;
source-validation32 and source-SAC48 trace hashes; validation and diagnostic
final recovery SHAs; all657 noise/PPO target group index SHAs and their cursor
recoveries; both original source PPO recoveries/exports, exact worker copies
and eight corresponding source-task SAC checkpoint SHAs; six onset case final
recovery SHAs. Runtime/probe hostnames independently confirm the listed jobs,
including corrected competence R2 `job-zxgkfebhs4k8`.

Source-local panels are near ceiling: source PPO's first two attempts are16/16
for both policies; source SAC native/mean/sample all16/16 provides no source
teacher advantage. The latter uses **eight task-specific actors**, not a joint
all40 SAC or an all40 evaluation. PixelSAC uses image64/proprio8; the history
policies use4104D prefix features. These controls do not establish high target
performance, sample-efficiency gains or reliable memory benefit.

Keep native results separate: student firstwave31/90, noise screen29/90,
PPO wave30/90. Use each contemporaneous paired control. The saved31-versus29
check found matching all90 initial/seed/parameter metadata; four outcome flips
have identical first input/noise arrays but decoded first actions differ.
This cross-process sensitivity does not justify pooling the three native runs.

## Active long-run snapshot, 23:54:45 Shanghai

This portion records the parent's relay of `finish_ppo` read-only verification;
changing active learner checkpoints were deliberately not read or hashed here.
Both tasks were selected after the six-case outcomes: **development diagnosis,
not an independently selected confirmatory benchmark**.

|Task / run suffix |Job |Source |Train episodes / chunks / SGD |Training successes |Latest query |Cumulative / NEW actions |
| --- | --- | --- | --- | ---: | --- | ---: |
|0 / `long-20261030-r1-task00` |`job-pytyo9urlxrj` |`cee1ea0` |41 /1,639 /23,180 |1,first at episode41 |budget40 failed |19,685 /10,655 |
|40 / `long-20261030-r1-task40` |`job-5rqftpqn4wxi` |`cee1ea0` |45 /1,800 /26,400 |0 |budget40 failed |21,340 /12,310 |

Both retain their original6,000-second active clocks,100k NEW-action ceilings
and109,030 cumulative-case ceilings. Migration verification reported equal
full numerical-state hashes,2,740 copied parent files/case,640 replay rows and
3,200 updates restored with **zero SGD during restore**. Their original parent
case recovery hashes are independently verified in the terminal onset receipt.
There is no claimed final checkpoint SHA or final GPU cost for either live run.

Count the parent35,558 actions/1,851 GPU-seconds once; copied prefixes are not
new data. At this snapshot, new actions total22,965, so unique onset-plus-new
cost is58,523 actions. Active outcomes and costs can advance after this receipt.

Canonical budget/coordination/summary remain parent-owned. Parent commit
`8541fa9` reports109,060 reserved seconds and6,140 unassigned inside the original
115,200-second ceiling. This is a reference, not a new reservation. Prepared
mean/dispersion diagnostic commits `ef64aaf` and `8595735` remain preserved,
unfunded and unlaunched; this receipt authorizes no further execution.
