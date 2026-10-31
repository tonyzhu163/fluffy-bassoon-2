# Evaluation record

Record each result here with its source commit, cluster, job and receipt path.

## Common-PPO ML10 core controls, 2026-10-29

This separate comparison uses GRU, Transformer-XL, CTM and CMM with a common
250k parameter cap and the same PPO contract: seed 1000, 50 lanes, K5/H500,
80 updates / 10M training interactions, ten PPO epochs, cached100 replay and
4,000 optimizer steps. Actual parameter counts are approximately matched.

| Core | Parameters | Train success | Held-out success | Training GPU h |
| --- | ---: | ---: | ---: | ---: |
| GRU | 245,757 | 20.8% | 0.8% | 0.3792 |
| Transformer-XL | 232,361 | 20.0% | 8.8% | 0.8073 |
| CTM | 237,347 | 16.0% | 0.0% | 1.2897 |
| CMM | 249,095 | 10.0% | 22.4% | 1.1125 |

GRU/XL were evaluated only initially and finally; CTM/CMM have intermediate
evaluations. Endpoint plots therefore compare the completed 10M runs, and
training GPU-hours exclude the unequal monitoring costs. Final evaluations
use K25 with the original K5-trained clock. CMM held-out reset success is
26.4%, above normal 22.4%, so the lead does not establish a memory benefit.
One seed does not establish an architecture ranking.

The [common-PPO result bundle](../../../outputs/metta-common-ppo-chart-20261029-r1/README.md)
contains exact counts, sources, checkpoint hashes, endpoint CSV, figures and
the saved audit. Its CTM run is the same checkpoint used below; the native
RL²-PPO / ECET / AMAGO1 baselines are different from the common-head controls.

## Native-method ML10 comparison, 2026-10-29

All four original Baige jobs completed 80 rounds / 10M training transitions
and 5M evaluation transitions per method at source
`cd0128f03a060aff2d538a91f35058aa822df845` (seed 1000).
The final saved-artifact audit completed at 08:38:51 Asia/Shanghai.

| Method | Baige job | Train success | Held-out success | Active GPU h |
| --- | --- | ---: | ---: | ---: |
| CTM T1 / cached100 | `job-rczwk9dd1v1e` | 16.0% | 0.0% | 2.0236 |
| RL²-PPO / GRU64 | `job-j400lp4d3m6m` | 24.8% | 0.0% | 0.8367 |
| ECET | `job-wrlr0utzsayo` | 24.8% | 21.6% | 9.0709 |
| AMAGO1 / 10 replay passes | `job-dm64v27p0a30` | 12.8% | 0.0% | 2.6583 |

Active GPU-hours include collection, optimization and evaluation on one
RPBZZZ6 per method. Evaluation uses ten balanced lanes and 25 episodes.
ECET held-out normal/reset/blind success is 21.6/21.2/6.4%; this does not
demonstrate a persistent-memory advantage. Different model sizes, learners,
clocks and replay schedules preclude an architecture-only ranking. CTM does
not establish an advantage in performance, generalization or efficiency.
First-K5 companion curves disclose the K5-training/K25-evaluation clock limit.

The [result bundle](../../../outputs/metaworld-ml10-matched-baige-20261028-r1/README.md)
includes the full report, original plots, raw evaluation outcomes and curve CSV.
The [completion audit](../../../outputs/metaworld-ml10-matched-baige-20261028-r1/checks/20261029-0835/completion-audit.json)
records all 160 recomputed evaluation conditions, counters and checkpoint hashes.
## CTM T1 PPO throughput qualification, 2026-10-28

Baige `job-6zp4ujvgjbqw`, source
`91c16961a02aa739c6cda6cb3c8627145d51aeef`, exit 0, passed 142 allocated
CPU/CUDA/simulator tests. On one paired ML10 batch (CTM T1 width84/depth8,
25,000 transitions, one PPO epoch/five optimizer steps), batching heads and
skipping unused reset construction reduced full-trial PPO from 124.22 to
96.38 seconds with matching losses and gradients. Including collection gives
146.54 versus 118.71 seconds, a 1.23× improvement.

Cached 100-step TBPTT took 3.98 seconds of PPO, or 25.96 seconds
including collection (5.64× faster). It retains forward history and full-trial
GAE but truncates recurrent gradients and changes minibatch membership; no
learning-equivalence claim follows. Frozen replay, transition/optimizer counts
and exact next-update recovery passed. No new job was launched for integration.
One batch on an RPBZZZ6 is not a success curve or a hardware-normalized comparison
with METTA. [Independent audit](../../../outputs/ctm-t1-throughput-audit-20261028-r1/RESULTS.md).
The user-selected training default is now 100-step TBPTT; explicit
`--replay-chunk-length 0` retains full-trial replay. This default change does
not alter the immutable qualification source or its measured results.

## METTA paper-network qualification, 2026-10-29

Independent implementation of METTA v1's residual TTT-MLP memory and
cross-episode attention is qualified at source
`b8666986c99a38a46fab78c49908d50ff9f45991`, Baige
`job-hruprb35xn12`, exit0. All36tests passed with no skips in38.27seconds:
float64 independent inner/outer gradient oracle, causal chunk and episode
memory/reset semantics, no-grad/inference-mode fast updates with frozen slow
weights, attention/lesions, fresh-prefix PPO, exact CPU/CUDA next-update
recovery, and real all-ten-family ML10 behavior replay at default width128,
two layers and chunk128.

The network has447,947parameters. A B10/L128 default-network forward and
backward pass took2.229seconds, with1,200,845,312bytes (1.118GiB) peak CUDA
allocation on one RPBZZZ6. This is a single-chunk capacity measurement;
full K5/H500 PPO cost and learning are not qualified. Missing author
architecture details and normalization/reward-scaling departures are
explicit in `docs/metaworld-metta.md`. The code is paper-based and does not
reproduce the authors' unreleased implementation or reported convergence.

R1 `job-tmuzewc13ikp`/`34913a9` passed35checks and failed only because its
simulator fixture requested two lanes from a ten-family balanced pool.
R2 corrects the fixture to ten; network/PPO/runtime source files are
byte-identical. The failure is preserved. Total59+42launcher-active seconds
stayed below the original20minute qualification budget. No long training or
VLA work was launched. Evidence:
`outputs/metta-network-qual-baige-20261029-r2/{AUDIT.json,RESULTS.md,qualification/}`.


## Original four-method ML10 comparison complete, 2026-10-29

All four original Baige runs at source
`cd0128f03a060aff2d538a91f35058aa822df845` finished80rounds/10Mtraining
transitions, exit0. The final audit independently recomputed all160saved
raw evaluation conditions, verified full checkpoint hashes and recovery
metadata, common task pools, work counters and measured costs. Earlier
entries below record status at their respective audit times.

| Method | Final train | Final held-out | Active GPU h | Training-only GPU h |
| --- | ---: | ---: | ---: | ---: |
| CTM T1/cached100 |16.0%|0%|2.0236|1.2897|
| RL²-PPO/GRU64 |24.8%|0%|0.8367|0.4005|
| ECET |24.8%|21.6%|9.0709|8.5437|
| AMAGO1/ten replay passes |12.8%|0%|2.6583|1.2001|

ECET `job-wrlr0utzsayo` completed6,251,200actor+critic optimizer steps,
200Mforward samples and5Mevaluation transitions. Launcher-active cost was
32660seconds/9.0722GPUh; peak CUDA allocation158,009,856bytes. Final
normal/reset/reward-blind train24.8/30.0/23.2%, held-out21.6/21.2/6.4%.
First-five normal train/held-out24/22%. Final checkpoint SHA256
`824de988a0ac8c3c78aeb3d28d2d9e5ea77e919ce2a99697487a09bce331c80a`
matched the complete policy, independent actor/critic optimizer, sampler
and RNG recovery metadata and terminal counters. All80training rows and
40ECETraw evaluation conditions passed saved-artifact checks.

Active time includes collection and all evaluation on one RPBZZZ6per
method; launcher overhead is reported separately. ECET ties RL²'s final
train score at substantially higher cost and is the only method with
nonzero final held-out success here. CTM does not beat RL² on final train
performance or efficiency and has zero final held-out success. One seed,
unequal parameter counts, method hyperparameters and gradient horizons,
ECET's absent public clocks versus other methods' K5-to-K25clock
extrapolation, and AMAGO's accepted replay adaptation prevent an
architecture ranking or exact METTA reproduction claim. ECET's train
memory reset outperforms normal memory; its held-out reset is similar.
The screen does not establish a reliable persistent-memory advantage.

Original K25curves retain their saved clock encoding; first-five companion
scores, frozen-clock interventions and warm continuations remain separate.
Final train/held-out versus active-time, training-only-time and environment
steps, PDFs, curve CSV, hashes and full audit are under
`outputs/metaworld-ml10-matched-baige-20261028-r1/figures/` and
`checks/20261029-0835/`. Detailed results and limitations are in that run's
`RESULTS.md`. All original runs are terminal; the narrow comparison
monitor is closed after the final artifact audit and mirror.

## ML10 controlled debugging: value weight and public clock, 2026-10-29

CTM paired continuation `job-knyy09k3p6rr`, source
`08a9700616bb4cbac70080f7d3462b03bd1bdd53`, completed exit0 in2430
launcher-active seconds. Both branches restored the same complete10M parent,
then each collected1.25M new transitions and used ten PPO epochs,500Adam
steps and12.5M loss samples. First baseline rollout, policy and full Adam
state exactly reproduced the independent diagnostic reference. All16raw
evaluation conditions and both complete recovery checkpoint hashes were
independently audited; each ends at iteration90/11.25Msteps/4500Adamsteps.

Across two paired evaluation blocks, value_coef.5versus.005 gave train
normal16.2versus16.0%, reset17.4versus17.0%; held-out normal.8versus2.0%,
reset.4versus1.2%. The small held-out counts do not establish a generalization
benefit. The lower coefficient reduced the measured initial shared-core
critic/actor ratio142to1.42, but did not improve the prespecified training
success endpoint. First-update cached logprob RMS drift increased.03564to
.04899. Do not promote.005 on this evidence. This is a short continuation
with inherited Adam moments, not a from-scratch coefficient comparison.
Evidence: `outputs/metaworld-ppo-controls-baige-20261029-r1/audit.json`,
`checkpoint-audit.json` and `value-control-success.png`.

CMM's own optimizer diagnostic `job-eouydzi8z936`, source
`0b05ddf5b460b353fa82ae18e3ed2100d1234b3e`, passed78tests and completed
exit0 in291launcher-active seconds. Its shared-core critic/actor gradient
ratio was131.3; cached logprob RMS drift after ten epochs was.04468.
Smaller minibatch2 increased drift to.10835 and PPO time30.44to149.00s,
at the same1.25M loss samples. Full normal/alternative optimizer/sampler/
normalizer/RNG checkpoints were verified. This does not establish a useful
CMM value-weight or minibatch change; CTM's control cannot prove transfer.

Frozen CMM clock diagnostic `job-pgjw4p0j22ut`, source
`a95271a453d70f3b965da7b0979cc1a623000c6a`, passed8allocated tests,
completed exit0 in506launcher-active seconds and used1M evaluation
transitions/zero training or optimizer steps. With the original10M checkpoint,
capping only the normalized episode index at1 after training's five-episode
range changed normal train success10.0to20.8%, reset22.8to24.8%; held-out
normal22.4to20.8%, reset26.4to39.6%. The held-out reset gain is mostly
drawer-open,32to98% within that family. Persistent memory still does not
beat reset; this is not a learned adaptation or architecture win.

All8raw clock conditions were independently recomputed. The original raw
outcome/return/family arrays exactly reproduce the parent; all four clock
pairs retain identical complete first-five-episode trajectory hashes.
Policy and normalizer fingerprints stayed frozen, and the unchanged parent
checkpoint hash was independently rechecked. Original comparison curves
remain unchanged: capped clock is a separate input intervention. The first
five train episodes score24% under either clock; later20change6.5to20.0%.
Therefore out-of-range public-clock input explains much of this CMM decline.
Evidence: `outputs/metaworld-clock-debug-baige-20261029-r1/audit.json`
and `frozen-clock-success.png`.

Matching CTM clock diagnostic `job-vb7p9f0cz56n`, source
`b790031791acf26309a5e6fcafa465e4a331e51f`, completed exit0,8tests,
584launcher-active seconds and1M evaluation steps/zero training or optimizer
steps. Original/capped train normal16.0/18.4%, reset18.4/18.8%; held-out
normal0/1.6%, reset0/0%. All8raw conditions,4complete first-five-prefix
hashes,12original-array checks and independent parent/frozen-model hashes
passed. The clock correction does not explain most of CTM's low score.
Failed qualification r2 `job-l6xdatciygju` is retained:6tests passed, two
CMM test cases called a CLI preset missing from the old CTM runner; no
evaluation ran. Corrected r3 explicitly provides the same constructor
settings and retains all8cases without changing model/runtime code.

The consolidated decision is to retainT1/value_coef.5/minibatch10, report
first-five competence separately from long-trial clock extrapolation, and
keep memory-reset controls. No T/depth/trace optimum or architecture win
has been established. CTM/RL²/CMM first-five train scores are16/28/24%;
ECET's incomplete30/80snapshot is38%. Original sampled training success
also remains low, so the issue is not solely long-evaluation input drift.
All interventions stay separate from original curves. Full decisions,
32independently recomputed clock/control conditions, costs, retained failures
and completion audit: `outputs/metaworld-ml10-debug-20261029-r1/RESULTS.md`
and `completion-audit.json`. The original ECET monitor continues; no further
tuning, seeds, sweeps or VLA work were launched.

## ML10 AMAGO completion, 2026-10-29

Baige `job-dm64v27p0a30`, source
`cd0128f03a060aff2d538a91f35058aa822df845`, completed80collections/10M
training transitions, exit0. The accepted ten replay-pass adaptation used
2083optimizer steps and99,984,000sampled replay transitions. Final
normal/reset/reward-blind training-family success was12.8/12.0/11.2%;
all held-out results were0%. This is one training seed, with no reliable
memory benefit or architecture advantage. Only ECET remains incomplete.

Launcher-active cost was9585s/2.6625GPUh; learner-active cost9569.97s,
including4320.27s training and5195.27s evaluation. All40saved evaluation
conditions were independently recomputed with equal family weighting.
Checkpoint SHA256
`e0ea9c8d542adfded6fdc656bd3fea5f3bfd21e1d75d5399ee8627026fd31532`
and complete policy/optimizer/replay/sampler/scheduler/scaler/exploration
and RNG metadata were checked without executing a model on the dev host.
Evidence: `outputs/metaworld-ml10-matched-baige-20261028-r1/checks/20261029-0222/`.

## ML10 CTM completion and optimizer diagnosis, 2026-10-29

CTM T1/cached100, Baige `job-rczwk9dd1v1e`, source
`cd0128f03a060aff2d538a91f35058aa822df845`, completed80updates/10Msteps,
4000Adam steps,100M PPO sample forwards and5M evaluation steps, exit0.
Final normal/reset/reward-blind fresh training-family success was
**16.0/18.4/14.0%**, with all held-out results0%. One seed; this supports
neither a reliable memory benefit nor an architecture advantage. All40raw
evaluation conditions, repeated final normal arrays, costs and checkpoint
SHA256 `a34dff8decfef013bde1f74310c67709b5ae0689a2b1f89ed49848db51a58b18`
were independently checked. Launcher-active cost7290s/2.025GPUh, peak1.687GiB.
The unchanged CTM setting is weaker and more costly than the completed
RL²-PPO baseline in this single-seed screen. ECET remains incomplete;
AMAGO subsequently finished, as recorded above.

Isolated diagnostic `job-ver4kajpyynd`, source
`98dddd57dbad5abaf302fdc839da7d2cfe33724e`, passed58allocated Linux tests
and completed exit0 in337allocated seconds. On one25k-transition minibatch,
the shared-core weighted critic gradient norm10.619 versus actor0.07479
gives a142x ratio; their cosine was-.0353. This is a measured gradient
imbalance, not a measured factor in Adam parameter step sizes. Global
GAE-target value explained variance was.933, with substantial variation
by family. Gaussian standard deviations.824–.866 and7.0% action saturation
show that the Gaussian sampling scale has not collapsed.

Before updating, cached/full-prefix replay matched within mixed FP32
atol2e-5/rtol2e-6. After ten PPO epochs, cached-versus-fresh saved-action
log-probability RMS drift was0.03564 and value RMS drift0.2952;31.1% of
value-loss positions selected an outside-clip, locally flat loss branch.
These are diagnostic differences, not distribution KLs or causal learning
effects. Output synchronization median positive half-life was152steps;
33.9% of rates implied half-lives above500steps. No complete-memory-horizon
claim follows from synchronization decay alone.

A paired smaller-minibatch update on the identical125k rollout kept ten
epochs/1.25M samples but used250versus50Adam steps. PPO took182.07versus
38.23s; cached log-probability RMS drift increased to.07449. The METTA
published episode-segment loop implies250steps at its defaults, unlike our
full-trial-equivalent minibatch. Matching optimizer-step counts alone is
not yet a validated remedy for the cached approximation.

Both full diagnostic optimizer/sampler/normalizer/RNG checkpoints and sample
counts/hashes were audited; their protocol explicitly excludes them from
the original comparison. Failed r1 `job-pkx5txzwylyn` is preserved:57tests
passed, no optimizer update, max value difference2.098e-5 just exceeded the
absolute2e-5 check. Its raw rollout is byte-identical to r3. The new mixed
tolerance includes a test accepting small FP32 roundoff and rejecting
material drift; the learner is unchanged. Failed scheduler-pod time131s.

Evidence: `outputs/metaworld-ml10-matched-baige-20261028-r1/ctm/`,
`checks/20261029-0147/`, `outputs/metaworld-ppo-debug-baige-20261029-r3/audit.json`
and `checkpoint-audit.json`. A bounded paired continuation control of value
coefficient.5versus.005 completed as `job-knyy09k3p6rr`, with no measured
train-success improvement. The unchanged first update exactly reproduced
the diagnostic's rollout, policy and Adam reference state.

## ML10 RL²-PPO single-seed completion, 2026-10-29

Baige `job-j400lp4d3m6m`, source
`cd0128f03a060aff2d538a91f35058aa822df845`, finished all 80 updates and
10,000,000 training transitions, with launcher exit zero. One RPBZZZ6,
16 CPUs, 64 GiB RAM and eight simulator workers; garage-adapted GRU64,
21,960 trainable parameters, seed1000, ten PPO epochs and 50 K5/H500 lanes.

Final fresh training-family success was **24.8%**, versus **22.8%** with
episode-boundary memory reset and **16.0%** reward blind. Held-out family
success was **0%** in all three conditions. Evaluation used ten balanced
lanes and 25 episodes, with equal family weighting and paired conditions.
These single-seed results establish neither a reliable memory benefit nor
generalization. The other three methods remain in progress; do not infer
an architecture ranking from their incomplete curves.

Measured launcher-active cost was **3,018s / .8383 GPU-hours**; learner
cost was 3,012.27s / .8367 GPU-hours, including 1,441.96s training and
1,536.14s evaluation. The run used 1,600 optimizer steps, 100M PPO sample
forwards and 5M evaluation simulator steps. Median collection and PPO
times were 17.63s and .728s per update; peak CUDA allocation was .668 GiB.

The terminal audit independently recomputed all 40 saved evaluation
conditions from raw binary outcomes, reconciled step/sample/cost counters,
and verified checkpoint SHA256
`a9634a8124ffb129211c00ea43d80028b3abdb2c6a97ddd8d0901d737bb03de3`.
Checkpoint policy/optimizer/sampler/Python/NumPy/Torch/CUDA recovery key
metadata is present; exact next-update recovery was qualified on allocated
Linux before launch. No model was loaded or executed on the development
host for this saved-artifact audit.

Evidence: `outputs/metaworld-ml10-matched-baige-20261028-r1/rl2/RESULTS.md`,
`checks/20261029-0023/receipt.json` and the date-stamped single-seed PNG
curves under `figures/`. Checkpoints remain on shared storage. No follow-on
run, extra seed or resource extension was launched.

## ML10 methods and parallel collection admission, 2026-10-28

All four methods passed separate allocated Baige RPBZZZ6 qualifications.
CTM job `job-ww6cxxyv1rm1` passed 168 CPU/CUDA/simulator/recovery checks;
ECET `job-4uiaykh1zrzh` passed official-encoder parity, unit and real-simulator
checks; AMAGO `job-pdkat27baohg` passed 15 tests plus real simulator and actual
B24/L2000 default capacity. These use source `6448495ddd6837a9f7b6e9db101175da0129ae30`.
RL² `job-0hn0u49y84v7`, source `3a029b6199458f9eb99b2b872c603a5eeb8c2b1a`,
passed 18 checks including exact CUDA recovery and both FP32 backward paths
against an independent CPU float64 oracle. This only revised a too-tight
elementwise gradient check; learner code is unchanged. Its failed r1 receipt
`job-3g0m5zpcn4tb` remains preserved. All successful launchers exited zero.

For one real 50-lane K5/H500 batch (125k transitions), collection / ten-epoch
optimization seconds were CTM T1 cached100 **21.77 / 36.80**, RL²-PPO GRU64
**15.60 / .87**, ECET **17.76 / 365.63**. Optimizer counts differ: 50, 20 and
78,140 respectively; ECET separately optimizes actor and critic. Trainable
parameters are 237,347 / 21,960 / 69,194. These are method-specific learners,
not parameter- or optimizer-step-matched models. The common comparison fixes
environment interactions and epoch count, not computation.

Official AMAGO1 with native SDPA, width256/layers3, B24/L2000 used **.524s**
for a genuine 48k-transition off-policy optimizer step, with **29.94GB** peak
GPU allocation. Its 4,179,817 trainable parameters and original learner are
retained. This capacity gate samples independent windows from one real trial;
it does not measure full 50-lane collection throughput. The user-selected
comparison uses ten sampled replay passes instead of the original update/data
schedule; AMAGO does not have PPO epochs.

Paired frozen CTM collection fell **69.921 to 20.943s** with eight persistent
CPU workers, a **3.34x** measured gain. Complete trajectory hashes agree
exactly, including sampled actions and all recorded tensors. Simulator time
fell 58.754 to 10.585s; inference/action-transfer time was 8.617 versus 8.311s.
Worker startup is additional charged overhead. New synchronization/reset
execution shortcuts passed output, memory and parameter-gradient parity.

Frozen behavior replay errors were below 2e-5. Full checkpoint hashes for
CTM, RL² and ECET were independently reconciled with their summaries.
These one-update results establish execution/capacity only. They do not
establish convergence, useful adaptation or an architecture advantage.
Evidence: `outputs/metaworld-method-qual-baige-20261028-r1/RESULTS.md`,
`ADMISSION-AUDIT-20261028-2334.json`, and the retained r2 RL² receipts.
The subsequent one-seed 10M-step plan is separate; its source adds only
launch/plan changes and preserves identical qualified model code.

## Meta-World execution admission, 2026-10-28

Baige `job-x2x4kdpfn011`, source
`adf741d6e955a624e9f785b9d1572dabf9ff4ab1`, passed the one-GPU training
probe at 19:20:45 Shanghai. CUDA allocation and PID/PCI identity agree with
the scheduler's single NVIDIA RPBZZZ6 allocation. Inside that allocation,
`test_metaworld_pilot.py`, `test_meta_rl_ppo.py` and
`test_meta_rl_checkpoint.py` passed 71 tests with two CUDA-marker skips in
33.22 seconds. New checks include real two-episode ML1 Push and ML10
simulator trials, goal visibility, terminal reward preservation, and exact
next-update checkpoint recovery for CTM, GRU and Transformer-XL.

This qualifies interfaces and recovery, not trained adaptation or benchmark
performance. The bounded six-cell single-seed pilot remains a separate learning
experiment. The initial r1 allocation `job-ikplxyvyh7p9` stopped at the missing
CUDA-selector check before any model work; its worktree and failed receipt are
preserved. Protocol and limits: [method contract](../../../docs/metaworld-ctm.md).
Receipts: `outputs/metaworld-ctm-pilot-baige-20261028-r2/qualification.log`
and `gpu-probe.json`, mirrored from the same run directory under
`/mowenjie/fluffy-bassoon-2/outputs/`.

## Meta-World bounded learning pilot, 2026-10-28

The same Baige job/source finished all six single-seed cells with launcher
exit zero at 21:17:41 Shanghai. Measured allocation was 7,023 seconds,
**1.951 GPU-hours**, including qualification, plotting and setup. Each cell
had a 900-second soft cap plus complete-update/final-evaluation overruns;
all stopped before the 20-update target with full recovery checkpoints.

| Benchmark | Core | Updates | Training steps | GPU-hours incl. evaluation | Fresh train-family success | Held-out success |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ML1 Push | GRU | 13 | 325,000 | .295 | 10% | 20% |
| ML1 Push | CTM T4 | 2 | 50,000 | .379 | 0% | 0% |
| ML1 Push | Transformer-XL | 8 | 200,000 | .327 | 0% | 0% |
| ML10 | GRU | 16 | 400,000 | .281 | 10% | 0% |
| ML10 | CTM T4 | 2 | 50,000 | .342 | 0% | 0% |
| ML10 | Transformer-XL | 9 | 225,000 | .305 | 10% | 0% |

CTM median full-trial PPO optimization took 375–400 seconds per update,
versus GRU 20–21 and XL 65–67 seconds. CUDA peak allocation was respectively
1.591, .109 and .872 GiB. Optimization dominates CTM runtime; its lower-level
cause remains unprofiled. Two CTM updates cannot determine learnability.
These exploratory runtimes use a training-admission probe, whose
`timing_comparison_eligible` flag is false; they support cost planning,
without a hardware-normalized or cross-paper speed claim.

No useful aggregate memory benefit is established. ML1 memory-reset and
reward-blind scores equal normal for all cores. ML10 fresh train-family
normal/reset/blind scores are GRU 10/12/10%, XL 10/10/10% and CTM 0/0/0%;
all held-out scores and episode-success curves are zero. This is one seed
and ten paired five-episode evaluation trials per condition, with unequal
achieved updates. It qualifies execution and accounting, not convergence,
an architecture ranking, a METTA reproduction or VLA readiness. Profile
whole-trial PPO before proposing longer learning runs; no follow-on run was
launched.

The terminal audit independently recomputed all 96 saved evaluation
conditions from raw binary outcomes with equal family/episode weighting,
verified all six checkpoint hashes, and reconciled finite metrics and
training/evaluation/time counters. Raw evidence and both PNG/PDF charts:
`outputs/metaworld-ctm-pilot-baige-20261028-r2/RESULTS.md` and
`AUDIT-20261028-2145.json`, mirrored from the same Baige output directory.

## Native DPT warm-start completion, 2026-10-28

Baige `job-l3coajmm9noh`, source
`79da19500f4e898d078539b4c562c150e610c16a`, completed with exit zero at 18:46:56
Shanghai. The paired training-distribution label gate scored 6/16 successes for
the frozen motor and 15/16 for the inverse-rotation oracle. The CTM then completed
100 flow-matching updates, saving a native warm start and full recovery state.
Validation flow loss fell .386777 to .333308, but the two training trials cover
only +45/0 degrees and the two reserved validation trials only 0 degrees.

Frozen held-out evaluation before PPO has two trials/four attempts per condition:
normal 3/8 (37.5%), memory-wiped 2/8 (25%), reward-blind 3/8 (37.5%). Normal and
reward-blind aggregate attempt curves also match, although individual outcomes
differ. These tiny results do not establish useful reward-driven ICRL or a
benefit over direct PPO; no candidate is promoted. The gate's frozen-motor score
and the candidate's held-out score use different distributions and are not a
matched comparison.

PPO loaded the correct initializer, replayed stored likelihoods/values exactly,
and took one optimizer step before its .05 KL guard stopped further epochs.
The 7/8 success reported on that collection is pre-update training behavior,
not a post-PPO evaluation. Motor fingerprints remained unchanged. The audit
verified seven artifact hashes and five raw evaluation arrays; final summaries
account for 6,400 simulator steps. Full provenance and candidate digest:
[native completion audit](../../../outputs/ctm-dpt-warmstart-20261028-r2/RESULTS.md).

## DPT-inspired compact warm-start pilot, 2026-10-28

Baige `job-qqijcsjw5rhu`, source
`1569aee35495e2e16e651793537b03eb380aacd1`, saved seven separate persistent-CTM
warm-start candidates plus full optimizer/sampler/RNG recovery checkpoints.
All export/resume hashes and 84 initial/final raw behavioral means and attempt
curves were checked. One training seed, width 64/T4, eight attempts, 256 training
histories and 16 validation histories; 32 behavioral trials except eight for grids.

| Benchmark | Updates | Greedy normal | Memory wiped | Reward blind |
| --- | ---: | ---: | ---: | ---: |
| Deterministic bandit | 200 | .92188 | .92188 | .53125 |
| Bernoulli | 200 | .50781 | .50781 | .50781 |
| Nonstationary | 200 | .56641 | .57813 | .51953 |
| Hidden-gain reach | 200 | .00000 | .00000 | .00000 |
| Dark Room | 112 | .03125 | .76563 | .03125 |
| Key-to-Door | 37 | .06250 | .00000 | .06250 |
| Cached rotation | 200 | .50391 | .41797 | .35938 |

Grid scores are dense return per attempt, not success rates. Their 300-second
soft caps ended before the update target. Bandit learns an immediate-feedback
solution that survives memory wiping. Cached rotation shows a tentative scalar
history signal; it is not a live robot result. The harder tasks are weak, and
Dark Room's greedy memory lesion is substantially better than normal. All NLLs
fell, demonstrating why supervised loss alone cannot qualify these candidates.
No matched downstream-PPO benefit or task-generalization claim is supported.
Full interpretation, initial/sampled controls and artifact index:
[pilot audit](../../../outputs/ctm-dpt-warmstart-20261028-r1/RESULTS.md).

The same job's native labeler failed before saving histories because pi0.5
generates eight steps while its decoder returns five. The prefix-only label/loss
fix at `79da195` passed 67 tests with nine Slurm-only skips in an allocated Baige
qualification. The native-only recovery uses a new immutable output identity;
its launch receipt is [r2](../../../outputs/ctm-dpt-warmstart-20261028-r2/PLAN.md).

## Wider-angle cached rotation control, 2026-10-28

The CPU controls inside Baige `job-61p1rncjnr1x` use source
`c3e9cca68b93b6e777cbe5314e130bcaf85c8e14`, distinct protocol
`cached_rotation_a1_plumbing_v2`, theta/phi {-45,0,45} degrees, and the existing
headroom table SHA256 `74910dc7…`. No new simulator outcomes are used. Scores
below are frozen greedy means on the same 256 paired validation trials:

| Core | Finished seeds | Updates/seed | Normal | Memory wiped | Reward blind |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRU T1 | 3 | 500 | 54.26% | 42.38% | 36.36% |
| Transformer-XL T1 | 3 | 500 | 55.05% | 45.08% | 37.50% |
| Persistent CTM T4 | 3 budget stops | 353/350/358 | 56.53% | 48.76% | 37.03% |

Zero correction scores 42.38%, Thompson sampling 45.36% and angle cancellation
oracle 74.76%. Normal greedy seed standard deviation is 2.38 percentage points
for GRU and 2.19 for XL. These results show a history-dependent reward-only
signal in this cached control. They do not establish live visual robot
adaptation. All final raw binary arrays were independently checked against the
saved means and attempt curves. The immutable 900-second cap stopped CTM seed
0 cleanly at 904.41 seconds, with `completed=false`, full recovery checkpoint
and no failure/partial file. All three CTM seeds now have final evaluation artifacts, stopping at
353/350/358 updates under their original caps. Across seeds, CTM normal greedy
standard deviation is 1.20 percentage points.
GRU and XL took about 65.1 and 245.3 seconds per completed 500-update seed;
CTM's 353 updates took 904.4 seconds including evaluation/checkpoint overhead
and concurrent contention. Different achieved updates, ticks and parameter
budgets prevent a matched architecture ranking. Objects/angle support are
shared across the cache split, and the table has one recorded motor outcome
per scene/action. Evidence:
[cached control audit](../../../outputs/robot-cache-poc-v2-baige-20261028-r2/AUDIT.md).

## Live native feature-bridge PPO admission, 2026-10-28

Baige `job-61p1rncjnr1x`, source
`c3e9cca68b93b6e777cbe5314e130bcaf85c8e14`, runs one persistent CTM T4
and one Transformer-XL T1 worker on distinct assigned RPBZZZ6 GPUs. The
allocation-local two-GPU training probe passed; 32 relevant CPU tests passed
in 3.71 seconds inside the allocation. Actual SAPIEN renderer PCI bindings
match both selected GPUs. The workers share the same frozen native pi0.5
checkpoint, normalized inputs and public feedback/residual-action interface.
They have 202,055 and 264,865 trainable parameters, respectively; this is not
a parameter-matched architecture comparison.

At 17:01 Shanghai, both workers completed two full live-trial updates and
eight optimizer steps. Before each of the first two updates, saved joint
Gaussian likelihoods and critic values replayed with maximum absolute error
exactly zero. After the initially zero output heads moved, 111 CTM or 44 XL
parameter tensors received finite nonzero gradients. All four PPO epochs ran
in each update; final measured joint KL was CTM .03114/.02172 and XL
.04002/.02175. Native-motor no-gradient and fingerprint assertions passed,
and complete between-trial optimizer/sampler/global/motor RNG checkpoints
were saved after both updates. Each worker collected 1,280 training simulator
steps across four trials in total. Their private trial-plan hashes match.

Native collection took 153–162 seconds per update, versus 7.81–7.93 seconds
of CTM PPO or 2.12–2.17 seconds of XL PPO. Peak allocated CUDA memory was
7,650,967,040 bytes (CTM) and 7,623,074,304 bytes (XL). The 100-update target
would exceed the fixed 9,000-second driver cap at this cadence; reaching the
cap leaves a resumable partial run. No cap or active contract was changed.

At update zero, each of the three paired held-out conditions scored 5/8
successful attempts for both bridges. This is only two held-out trials per
condition. Initial equality and training updates qualify execution; they do
not establish learned adaptation or a reliable architecture ranking. The
held-out update-10/50/100 diagnostics remain pending. Evidence:
[launch receipt](../../../outputs/pi05-live-bridge-poc-baige-20261028-r1/LAUNCH.md)
and `/mowenjie/fluffy-bassoon-2/outputs/pi05-live-bridge-poc-baige-20261028-r1/`.

## First held-out live bridge evaluation, 2026-10-28

The same native run/source completed update-10 frozen evaluations and reached
21 updates/84 optimizer steps per learner by 18:00 Shanghai, with no failure
receipts and checkpoints advancing. Only two held-out K4 trials are evaluated
per condition. At update 10:

| Core | Normal | Memory wiped | Reward blind |
| --- | ---: | ---: | ---: |
| Persistent CTM T4 | 62.5% | 62.5% | 50.0% |
| Transformer-XL T1 | 75.0% | 75.0% | 62.5% |

Memory wiping does not reduce aggregate score for either learner in these tiny
samples. The live task therefore has no demonstrated memory benefit yet. The
XL normal increase versus the initial 62.5% is one additional successful
attempt; this cannot establish learning or an architecture ranking. All raw
binary arrays exactly reproduce the saved means and attempt curves. Update-50
evaluation remains pending within the original time cap. Evidence:
[18:00 status and raw checks](../../../outputs/pi05-live-bridge-poc-baige-20261028-r1/STATUS-20261028-1800.md).

## Matched execution pilot and cached robot control, 2026-10-28

Baige `job-uomh9zm3mpnd`, source
`a3f18a8b576e1a81f7a926b5b7424b2a7ef5a55a`, completed all 14 pilot cells:
bounded persistent CTM T1/T4/T8, consolidated CMM, Transformer-XL, GRU and looped
Transformer, six-arm Bernoulli trials of 64/128 pulls, seed 1000. Each trained
for only three PPO updates and completed twelve frozen normal/memory-wiped/
reward-blind greedy/sampled diagnostic conditions. The allocation-local four-GPU
probe passed; the test suite passed 256 tests, 13 skips and 27 subtests in 15.92 s.
Policy parameter counts span 95,715–99,035. Maximum observed CUDA allocation was
637,748,736 bytes. This qualifies execution, not architecture performance.

The same allocation completed three 500-update CPU scalar rotation-cache fits
(persistent CTM, Transformer-XL, GRU; one seed each; equal width, not matched
capacity). All greedy normal, memory-wiped and reward-blind returns were exactly
0.6640625, matching zero correction; there is no adaptation result. The source
protocol used theta/phi ±30°, K8 and a development scene split of the existing
headroom table. V2 ±45° is a separate protocol with more task headroom and a
separately recorded PPO configuration; v1 results must not be relabeled.

Saved artifacts and full recovery checkpoints remain at
`/mowenjie/fluffy-bassoon-2/outputs/ctm-screen-pilot-baige-20261028-r2/` and
`/mowenjie/fluffy-bassoon-2/outputs/robot-cache-poc-baige-20261028-r1/`.
Small receipts and raw diagnostics were mirrored locally. The dated record is
[the POC launch plan](../../../outputs/ctm-poc-launch-20261028-r1/PLAN.md).

## Native frozen-pi0.5 KV bridge admission, 2026-10-28

Baige `job-sr0t8ifv2fls`, source
`44c5da3cd30c35daa3473f843d032d0a486ee238`, completed eight synthetic bridge
tests and one real 80-step native motor episode. Native pi0.5, zero persistent
CTM bridge and zero Transformer-XL bridge were compared on each of the same
sixteen live observations with cloned CUDA generators. Model-space actions,
decoded actions, prepared simulator commands and the advanced noise streams
were exactly equal at every chunk. Recurrence advanced sixteen times per
bridge, once per chunk. The common trajectory succeeded and remained successful
at step 80; it is one episode, not three independent episodes.

The allocation-local GPU probe and actual SAPIEN renderer binding passed.
Motor fingerprint `ed862a89…` and both bridge fingerprints were unchanged.
The gate took 158.76 s including setup/fingerprints, with 6.68 GiB peak CUDA
allocation. This establishes native interface and zero-residual parity on this
episode; it does not test trained adaptation. Evidence:
`/mowenjie/fluffy-bassoon-2/outputs/pi05-feature-bridge-gate-baige-20261028-r1/`.

## Privileged bounded steering capacity screen, 2026-10-28

Baige `job-o16q3eimo8bq`, source
`c2c47152303ccfc1ac469fd3e1ad1e1344201e42`, completed the prespecified
tracker and repeated-control schedule in 826.63 driver seconds, exit 0.
The actual allocation passed 151 focused Linux tests. The ManiSkill task
was `PutOnPlateInScene25Main-v3`, objects 0/1, using the frozen RLinf
ManiSkill-finetuned pi0.5 motor. Training scenes selected scalar gains;
the held-out scenes were not used to tune them.

The privileged tracker executed through the same four-Euler, seven-active-
dimension, residual-bound-one sampler as the learner. Success at any point
within each 80-step attempt was:

| Hidden XY rotation | Bounded tracker | Zero bridge | External action oracle |
| --- | ---: | ---: | ---: |
| -45 degrees | 0/2 | 0/2 | 1/2 |
| 0 degrees | 2/2 | 2/2 | 2/2 |
| +45 degrees | 2/2 | 1/2 | 1/2 |
| Total | 4/6 | 3/6 | 4/6 |

All three zero repeats matched their complete 16-chunk trajectories and
outcomes exactly, including native inputs, noise and commands. This is
repeatability within this run; it does not explain the prior cross-method
zero discrepancy. The frozen motor fingerprint remained unchanged.
The prespecified capacity gate **did not pass**, because the tracker failed
both negative-rotation scenes. The positive-rotation gain is descriptive
evidence from two scenes, not learned competence or ICRL. Failure does not
prove the intervention incapable. The crossed-history diagnostic was not
admitted, and no further PPO ran.

Actual cost: 2,880 simulator steps, 576 action chunks, 928 native prefix and
3,712 suffix lane queries, including 96 external-label and 256 tracker-target
motor calls. Complete raw trajectories and recovery state remain on Baige;
small receipts and the independent audit are in the
[dated result](../../../outputs/pi05-tracker-gate-baige-20261028-r2/RESULTS.md).

## CPU test suite after the split, 2026-10-28

Secondary school cluster (thu2), job 48869, node01, 4 CPUs, 8 GiB, no GPU,
COMPLETED, exit 0. Source `9be85caf6e231a5bc5c59f03c77b484732b3e2eb` in a clean
pinned worktree; imports resolved to that worktree (`PYTHONPATH=src`, no
install). The full `tests/` suite ran in two existing environments:

| Environment | Python, torch, NumPy | Result |
| --- | --- | --- |
| `envs/lycoreco-de5277f-r1` (compact learners) | 3.12.14, 2.7.1+cu128, 2.2.6 | 203 passed, 17 skipped, 27 subtests passed, 93.5 s |
| `envs/rlinf-pi05-maniskill-e1a82eb-r1` (ManiSkill motor) | 3.11.16, 2.11.0+cu128, 1.26.4 | 203 passed, 17 skipped, 27 subtests passed, 119.2 s |

The 17 skips are the CUDA tests (`test_meta_rl_cuda.py` and two in
`test_meta_rl_ppo.py`), which need a probed GPU step. Receipts:
`/cluster/nvme5/zhujiayang/fluffy-bassoon-2/outputs/fb2-cpu-tests-9be85ca-20261028-r1/`
(job script, environment versions, pytest logs, JUnit XML).

## CUDA tests after the split, 2026-10-28

Secondary school cluster (thu2), job 48873, step 0, node02, one RTX 3090
(Slurm GPU 4, UUID `GPU-6e52b20d-e67e-9183-8251-f13388c512cd`), 4 CPUs,
16 GiB, COMPLETED in 26 s, exit 0. Same source `9be85ca` in a fresh pinned
worktree. The GPU probe ran inside the step and passed: one visible device, the
expected UUID through the GRES mapping, a CUDA allocation check, and no other
compute process on it before the tests. A foreign `VLLM::Worker_TP1` process
held 14 GiB on another node02 GPU outside Slurm; it did not touch this one.

With `FLUFFY_META_RL_CUDA_QUALIFICATION=1`, `tests/test_meta_rl_cuda.py` (GPU
rollouts match CPU and PPO updates for GRU, CTM control, CMM, looped
Transformer and Transformer-XL on both tasks; GPU checkpoint restore and the
next update) and the two CUDA cases in `tests/test_meta_rl_ppo.py` passed:

| Environment | torch, CUDA | Result |
| --- | --- | --- |
| `envs/lycoreco-de5277f-r1` | 2.7.1+cu128, 12.8 | 17 passed, 7.0 s |
| `envs/rlinf-pi05-maniskill-e1a82eb-r1` | 2.11.0+cu128, 12.8 | 17 passed, 6.5 s |

With the CPU suite above, all tests in this repository pass. Receipts:
`/cluster/nvme5/zhujiayang/fluffy-bassoon-2/outputs/fb2-cuda-tests-9be85ca-20261028-r1/`
(probe JSON, pytest logs, JUnit XML). A ManiSkill motor episode is still
unqualified in this repository.

## Prior evidence in fluffy-bassoon

All earlier compact results were produced by the same code in
fluffy-bassoon and are recorded in its
[meta_rl/EVALUATION.md](https://github.com/tonyzhu163/fluffy-bassoon/blob/0df7d026cb72d1d047bbb261ef95dc1f5d2da9fa/src/fluffy_bassoon/meta_rl/EVALUATION.md) at `0df7d026`. They are prior evidence, not
results of this repository; rerun anything the paper reports. The most relevant
sections:

- Environment and CMM recurrence; full-trial PPO and recovery (2026-10-24).
- Stochastic rewards and controller headroom; raw-clock Bernoulli learning
  screen; shared clock encoding and CUDA qualification.
- Reduced Copy memory qualification; full-length Copy preflight and CMM
  fidelity limit.
- Compact Transformer-XL control qualification.
- Completed five-core compact screen (GRU, CTM control, CMM port, looped
  Transformer, Transformer-XL on the six-arm Bernoulli bandit).
- Standalone core and native admission test cleanup (2026-10-28).

Two caveats carry over. CMM results from before the 2026-10-28 consolidation
used a different core and their checkpoints do not load at later sources. The
"CTM control" in those screens is the CMM with its memory blocks removed, not
the persistent CTM. The persistent-CTM bandit results (T1/T4 legacy and bounded
synchronization) are summarized in fluffy-bassoon `outputs/coordination.md`.

## Frozen pi0.5 noise-arm Stage 0 qualification, 2026-10-29

The focused software tests pass, but the deterministic-cache prerequisite is
unresolved. At source `352edc407738ce14b6a0adb9100732326ca40902`, Baige
`job-gjkse1sb5gyj` passed 31 Linux tests and completed the 68-episode S0a smoke
(two objects, one scene each; 16 fixed initial-noise arms plus native; two
repeats). Only 4/32 fixed-arm pairs have identical complete trajectories,
5/32 have identical first chunks, and 29/32 have identical success outcomes.
Scene identities and input-noise hashes agree; the frozen motor fingerprint
is unchanged. The three outcome-flipping pairs already differ in the first
chunk. Initial observation hashes were not saved, and many repeats change
both reset seed and batch slot, so the saved evidence cannot isolate the
source of the variation.

Between-arm first-chunk action distance is 0.05155 versus native-repeat
0.03860. Native succeeds 4/4 and pooled fixed arms 59/64; these tiny nominal
counts establish no held-out headroom. The smoke used 5,440 task actions,
1,088 motor chunk queries, 228.90 driver seconds and 12.31 GiB peak Torch CUDA
allocation. A deliberate exit 3 stopped S0b/S0c after the failed admission;
there is no DSRL training or ICRL result. Next: audit initial observations
under matched reset seeds, then budget repeated Bernoulli outcomes if
variation persists.

[Results and compact raw evidence](../../../outputs/noise-mode-icrl-20261029-s0-r1/RESULTS.md).


## RoboTwin saved-controller optimizer diagnostics, 2026-10-30

These verify optimizer execution on real saved source trajectories, not robot
performance or successful meta-learning. The frozen Motus pi0.5 H32 motor was
not instantiated; no new robot actions were collected. Twelve complete K4
source episodes (4,273actions,140decisions; three of the six source tasks)
were reused from r4. Restoring the motor's global high float32 matmul mode
reproduced all original behavior likelihoods/values exactly.

Baige`job-yem1ctw27j26`, source`98efb377c525ff2c101b7e9ec0f6eaf115299b66`,
completed one independent batch3/five-epoch PPO update/five optimizer steps in
5.58s and saved complete recovery. Its3e-4setting was unsafe: reported sampled
KL8.86e19; subsequent exact Gaussian KL(old||post)12.8442, clipped joint ratios
100%. Variance contributed only0.000841to KL; mean movement dominated.

Baige`job-ypsxplhesd9x`, source`3e1de8e80d3f333adeba252884e173ce54f8b467`,
completed two independent original-state forks at3e-5and3e-6(five steps each).
Post analytic KL was0.43548/0.01065and post clipped fractions97.14%/75.00%.
Neither passed the predeclared KL<=.02and clip-fraction<=.2gate, although3e-6
reduced drift substantially. Repeat recurrent mean/log-probability reconstruction
was exact for both. Full policy/optimizer/sampler/RNG and Gaussian diagnostics
are retained. The three forks are not sequential source-training updates.
SAC still has139warmup rows/0updates; the batch8main comparison remains
unlaunched and has no robot-performance or adaptation result.

[Diagnostics and raw evidence](../../../outputs/robotwin-saved-ppo-lr-20261030-r1/RESULTS.md).
[Revised proposal, resource approval pending](../../../outputs/robotwin-staged-20261030-r2/PROTOCOL.md).

The RoboTwin KL values above sum all1,024noise coordinates before averaging
stored decisions. Its current fullH32×32interface differs from LIBERO's32D
coordinate repeated over nativeH10. Within-RoboTwin arms share the interface;
cross-benchmark steering dimensionality is not matched. No tied32D RoboTwin
ablation has been launched or inferred from the existing1024D histories.

## Trained RoboTwin continuation software qualification, 2026-10-31

Baige `job-q6dhvcewwwx1` passed all 67 selected CPU tests at clean exact source
`0c16c2b794f8de5b78603206917e72195083c265`, including three real process-spawn
and cleanup regressions. There were zero failures, errors or skips; the phase
took 74.828 seconds. The actual allocation was zero GPUs, four CPUs and 16 GiB.
This validates the original 100-round paired continuation supervisor's tested
software behavior, not completion of the continuous GPU run or robot learning.
The owner still binds current released manifests and operational authority.

The same job passed 43 cached scalar LIBERO adapter fake cases at exact source
`26b55eae2a5ce6545cdc97b3a875cc6e3a7b46fa` in 0.468 seconds. These do not
qualify the later recorded-input CLI, GPU compatibility or source adoption;
the optional GPU comparison remains deferred. No robot actions were collected.

[Qualification and independently checked receipts](../../../outputs/robotwin-trained-continuation-preparation-20261031-r1/CPU-QUALIFICATION.md).

## RoboTwin complete PPO source100 endpoints, 2026-10-31

Persistent and independently trained-reset GTrXL/PPO each completed100 rounds,
600 whole K4 trials /2400 source episodes,100 PPO updates /500 Adam steps.
Actual source actions were964875 and963759. Scientific lineage `db0184089fa88a5f96c153f9b80029bb91b77def`
continued through runtime `57016ba08eed79043fd67bf8b085bd6d42e6d766` and supervisor
`0c16c2b794f8de5b78603206917e72195083c265`; the common Motus pi0.5 motor stayed frozen.
Baige `job-fji2xx8oq53h` finished source and all declared evaluation stages with
exit0 /scheduler and pod Success. Its actual running cost was15451GPU-seconds;
176 owned-cleanup receipts were clean and final GPU memory was0.

Stage100 source validation: persistent2/24, trained reset3/24. Target bottle/stapler
successes per K8 trial were persistent normal7/1, same-weight memory-reset6/1,
reward-blind8/1, separately trained reset8/0. Actual case seeds, initial public
features and raw observations match across controls. Source/final/stage100
checkpoint SHA and263-member ZIP CRCs, unchanged motor/source RNG and zero
pending/uncertain canonical heads passed independent audits.

The small one-seed/two-development-task panels do not establish a memory or
adaptation advantage. The contemporaneous native probes use another seed bank
and are unpaired. SAC was still ongoing at this PPO-only audit; its completed
endpoint is recorded below.
The [complete paired report](../../../outputs/robotwin-trained-continuation-20261031-r1/FINAL-PPO-RESULTS.md)
contains exact checkpoint identities, episode1/2/4/8 traces, costs and audits.

The [native16 saved-result reconciliation](../../../outputs/robotwin-trained-continuation-20261031-r1/NATIVE16-REUSE.md)
subsequently verified that the earlier qualified native panel already contains
the exact16 final target cases: native6/8 bottle and0/8 stapler. All48 PPO and16
trained-reset initial cases, both native result/unit hashes and the common motor/
capture contract match. Distinct2fd/570 executors and stochastic draws remain
explicit. This cross-run control requires no new actions and does not establish
a statistically reliable or memory-specific benefit.

The same-weight saved `memory_reset` condition is a recurrent-cache reset lesion;
it retains explicit previous-noise/reward across episodes. It must not be
called fully memoryless. The separately trained episode-reset arm clears both
cache and explicit feedback. Exact570 source confirms these independent flags.

## RoboTwin original three-arm campaign complete, 2026-10-31

Feature-conditioned DSRL-style SAC completed100 balanced source rounds /600
complete K4 trials /2400 episodes, using1003790 source actions and638500 updates
per actor, critic and temperature optimizer (1915500 optimizer steps total).
This is not an authors' PixelSAC reproduction. All10/25/50/100 evaluation stages
are complete. Baige `job-gv77bzhyeroo` and its pod finished Success/exit0;
actual scheduler running time was32230GPU-seconds, distinct from the supervisor
and pod clocks. No additional source actions or source updates occurred during
the final evaluation.

Stage100 source validation was2/24. Frozen SAC achieved6/8 bottle and2/8 stapler;
online SAC achieved5/8 and0/8, with3320 additional joint target updates using
fresh target replay. Those target updates are not source SGD. Both conditions
are stochastic; their first sampled noise, motor action and next observation
agree, while online updates after the first chunk change later trajectories.
All56 final validation/target reset cases match the PPO/reset actual seeds,
initial public features and raw observations. Independent checkpoint SHA/210
ZIP-member CRCs,640 native action masks totaling19700 final-evaluation actions,
61 zero-exit children, unchanged motor, restored source RNG and closed canonical
pending0/uncertain0 checks passed. Full replay/Adam/RNG/corpus recoveries remain
preserved; this final audit did not rehash every RGB payload.

The three source banks total2932424 actions /1800 whole K4 trials /7200 episodes.
Across the final one-seed/two-development-task panels, neither persistent memory
nor online target adaptation has an established advantage. The
[final campaign report](../../../outputs/robotwin-trained-continuation-20261031-r1/FINAL-CAMPAIGN-RESULTS.md)
and [SAC endpoint](../../../outputs/robotwin-native-continuation-sac-20261030-r1/FINAL-RESULTS.md)
retain episode1/2/4/8 outcomes, recovery identities, actual costs and limits.
The separate fixed-SAC65 CPU converter/export qualification passed150 tests and
390 K4 /651638 actions; it does not qualify mean-teacher competence, final100
student fits or adaptation. Early student fits still use only the early SAC5
subset. Further fits and teacher measurements remain separate resource proposals.

## RoboTwin episode-reset PPO: completed100-round endpoint

The separately trained episode-reset GTrXL controller completed100 source
rounds/600K4 trials/2400 training episodes,963759 source actions and500 Adam
steps. The owned terminal audit passes, all four10/25/50/100 stages complete,
and the frozen Motus pi0.5 motor hash is unchanged. Final heldout development
results are8/8 on adjust_bottle and0/8 on move_stapler_pad; six-source-goal
validation is3/24. This section covers RESET; FULL is reported below and
the sourceSAC final comparison remains unfinished.

Training success is421/2400 (17.54%), with successive25-round blocks
112,106,102,101 /600: no useful upward trend. All100 update records and
963759 actions reconcile. There are31172 actual valid chunk decisions;
losses are finite, gradients nonzero, and logged approximate joint KL is
small (mean0.000232). Static review confirms correct native horizons and
cross-episode K4 GAE. This establishes weak/flat source learning, without
isolating LR, clipping, sample budget, task interference or noise
parameterization as its cause.

The controller preserves within-episode recurrence but clears cache and
previous noise/reward each episode. Its8/8 score is not cross-episode
adaptation evidence. Eight scenes on each of two goals are not a full
benchmark. Its source-stage before/after native probes use different seeds;
the separate matched historical reference is reconciled below. No reliable
native improvement or memory benefit is claimed. Canonical reported resource
charge1021785 comprises source963759, source validation39434 and target
18592, with no pending or uncertain actions. These are action costs, not
additional optimizer exposure.

Evidence: [training/optimizer diagnosis](../../../outputs/robotwin-reset100-diagnosis-20261031-r1/ANALYSIS.md).
Original endpoint:
`/mowenjie/fluffy-bassoon-2/outputs/robotwin-trained-continuation-20261031-r1/ppo_reset/slice-041/experiment`.
Summary SHA256:
`8f5d5e2e9377df90fa955049e4464abf7ba58daa2977046902917b0132d1ff60`.
Final coordinator recovery SHA256:
`48d910980052ce2cec730669fbb6432b86d956aadbff22799b7ecefdeaadcb49`.

## RoboTwin FULL PPO100 and frozen interventions

The shared PPO pair exits0. FULL completes100updates/600K4/2400source
episodes/964875source actions/31211valid chunk decisions/500Adam steps.
Its source success416/2400 is weak and flat (quarter totals116,104,93,103
per600), similar to trainedRESET421/2400. FULL source validation2/24 and
RESET3/24 are coarse seen-goal/state panels.

| Fixed100 condition | adjust_bottle | move_stapler_pad | Total |
| --- | ---: | ---: | ---: |
| Earlier native Gaussian, matched cases |6/8|0/8|6/16|
| FULL normal |7/8|1/8|8/16|
| Same FULL weights, cache reset |6/8|1/8|7/16|
| Same FULL weights, reward input zeroed |8/8|1/8|9/16|
| Separately trained episode reset |8/8|0/8|8/16|

Controller and motor parameters remain frozen during these queries. Cache
reset retains explicit previous-noise/reward feedback; trainedRESET clears
both feedback and cache while preserving within-episode recurrence. The
different controls should not be called memoryless policies. No reliable
memory/reward-use or broad benchmark advantage follows from this panel.
Normal/cache-reset stapler wins occur on different scenes. The source-stage
native probes are a separate seed bank, distinct from the matched historical
native row above.

The read-only metadata diagnosis verifies all100 updates/2400episodes and
finite optimizer metrics; FULL approximate joint KL averages0.000226. The
cause of weak source learning is unresolved; logs do not isolate sample
budget, LR/clipping, task interference or the noise parameterization.
[FULL training diagnosis](../../../outputs/robotwin-ppo100-diagnosis-20261031-r1/ANALYSIS.md),
[RESET independent endpoint](../../../outputs/robotwin-reset100-independent-20261031-r1/RESULT.md).

Independent FULL byte checks match all11 references among32 saved metadata/
checkpoint files. All64 evaluation groups/304episodes are complete; learned
groups are not budget-truncated. At100, requested/resolved seeds and raw
input descriptors match all16 target cases across the four learned
conditions; the three FULL lesions use identical policy/archive bytes and
their first episodes have identical executed histories. Later histories
diverge. This is a saved-descriptor check, not a new simulator replay or full
raw-payload rehash. The summary initial_observation_sha256 field is a feature
hash; extra raw-input checks support endpoint matching beyond that field.

Canonical saved-owner charge is1068449 actions: source964875, sourceVAL40874,
target56300,native6400; pending/uncertain0. Evaluation totals independently
match episode metadata. The independent file audit did not measure scheduler
time; a subsequent UI lookup below supplies it. The driver ceiling is not
elapsed GPU cost and per-arm times are
not added as separate allocations. Summary SHA256
`4fe0a6aff5b8454a71e1e97cdb21c0fa1b94d14e9f61ae9e20ccf9731390d83a`;
final coordinator SHA256
`c439486c327116c8da4fc077217416757bdc2f21ffa819f8ae802cc3f7810fa5`.
[FULL independent verification](../../../outputs/robotwin-ppo100-independent-20261031-r1/RESULT.md).

The owning chat's46856dc reconciliation verifies a distinct earlier native16
panel against all48 FULL-condition and16 trainedRESET target records:
requested/resolved seeds, raw observations and features match, including
bottleE5's shared fallback. Motor/capture qualification, normalization,
RLinf/RoboTwin, H32/noise shape and action-transform contracts agree.
Executors2fd/570 and stochastic draws are distinct. This supplies a historical
case-paired reference6/16, not a contemporaneous native replication. FULL
records3wins/1loss against it; trainedRESET2wins/0losses. The small panel
does not establish a robust gain or memory benefit. No new action or GPU
cost is added; the native panel's5051actions remain counted once there.
[Native16 evidence](../../../outputs/robotwin-trained-continuation-20261031-r1/NATIVE16-REUSE.md).

A separate authenticated scheduler lookup confirms the shared continuation
job-fji2xx8oq53h succeeded, ending06:43:32Shanghai with4h17m31s elapsed:
15451GPU-seconds on one GPU. Count this allocation once for both arms;
it is only the continuation from prior checkpoints, not all100-round
training cost. Earlier allocations remain separate.
[Scheduler receipt](../../../outputs/robotwin-ppo100-pair-scheduler-20261031-r1/receipt.json).

## RoboTwin SAC100: completed source and evaluation

Audited closed slice0060 records100 source rounds/600 completeK4 trials/
2400 episodes/1003790 source actions/32425 decisions. Success is277/2400
(11.54%); quarter totals78/57/76/66 per600 show no sustained improvement.
Per-goal successes per400 are54 handover,14 bottles,54 a2b,100 object stand,
43 phone stand and12 scan. Some goals improve while others decline; this is
not a claim that every source goal has the same curve. SourceVAL10/25/50/100
is1/24,1/24,3/24,2/24. The later final evaluation and whole campaign exit0;
the earlier source100 manifest snapshot retains its at-read evaluating status.

There are638500 joint SAC cycles, exactly20×(32425−500warmup): one actor,
one combined twin-critic and one temperature step each,1915500 combined
optimizer calls. These replay updates reuse32425 distinct robot decisions.
The1277 retained diagnostic snapshots report only the final step of selected
20-step groups. Recorded scalars are finite; early target/value-loss
transients are large. Smaller later critic error does not establish better
robot success. Temperature alpha is not action std, and negative differential
entropy is valid.

This feature-conditioned full1024D noise SAC usesLR3e-4/batch256/UTD20,
no norm cap, gamma.99 per decision and timeout bootstrap. It is a declared
DSRL-style baseline, not the authors' PixelSAC/DSRL-NA reproduction. All
three learners now have100 balanced rounds/2400 episodes: FULL416successes,
RESET421 andSAC277, with differing actual actions and exploration/objectives.
Weak source learning is not confined to recurrent PPO or novel-target
transfer. Shared interfaces and separate optimizer issues remain unisolated;
one seed does not establish a broad algorithm ranking or causal remedy.

The fixed stage100 SAC recovery's declared SHA is
`59588f07be18d4a1339af0c7536c62ef735a713f3c933d7e372fb3f4f4fa7e90`.
The source metadata analysis did not rehash/load checkpoint tensors. It
reconciles closed JSON/JSONL counts and preserves manifest status at read
time. A subsequent independent endpoint check below verifies final completion.
[Final source100 comparison and exact identities](../../../outputs/robotwin-sac-source-diagnosis-20261031-r3/ANALYSIS.md).

| Source iteration | Frozen bottle/stapler | Source-initialized online bottle/stapler |
| ---: | --- | --- |
| 10 |3/8,0/8|1/8,1/8|
| 25 |4/8,0/8|3/8,0/8|
| 50 |3/8,0/8|5/8,1/8|
| 100 |6/8,2/8|5/8,0/8|

All fixed checkpoints are reported without target-based selection. Final
frozenSAC8/16 matches FULL/RESET8/16 on identical actual cases. This small
one-seed/two-goal panel does not establish algorithm equivalence or robust
gains. Online5/16 is descriptive. Each online task independently restores
the source learner, excludes source replay and starts updates from the first
new target row, without a second warmup. Updates occur inside episode1;
indices1/2/4/8 are sequential episode indices, not separate frozen queries
after a declared support budget. Final online cost is1920/3200 actions and
1240/2080 SAC cycles for bottle/stapler, respectively. Frozen costs1693/2707
actions and zero target SGD.

All40 requested evaluation groups/224episodes are complete, with no budget
truncations. All10 supplied hashes match among26 rehashed files; source
snapshots, final target recoveries and acting archive agree. The final
coordinator SHA is`f551e3f28d14702556392e67dafdf62f094c824f9069511219f8dc5af4db9fd8`;
ZIP CRC passes210members without tensor/pickle loading. Frozen/sourceVAL
acting digest and optimizer clock remain constant; source RNG-restored
metadata is true at all4 stages. All16 actual target scenes match FULL/RESET
raw-input/feature/seed descriptors. The16 source-stage native probe rows use
a separate bank; they are not the historical matched native16 reference.

Canonical SAC resource cost is1088029actions: source1003790, sourceVAL41032,
frozen targets21360 and online targets21847; pending/uncertain0. Evaluation
phase costs independently sum from saved episode metadata. Canonical totals
bind the owner's closed snapshot, not a new event-forest replay. Scheduler
job-gv77bzhyeroo succeeded at08:04:24Shanghai, displayed8h57m10s=32230GPU-s
on one GPU. Count this continuation allocation once; prior source-prefix
allocations remain separate. Supervisor elapsed32209.95s is not substituted
for scheduler cost.
[Independent SAC endpoint and limits](../../../outputs/robotwin-sac100-independent-20261031-r1/RESULT.md),
[scheduler proof](../../../outputs/robotwin-sac100-scheduler-20261031-r1/receipt.json).

A follow-up verifies the separate historical native16→FULL/RESET→SAC
pairing chain for every target case, including bottleE5's actual seed fallback.
Nine native result/proof/receipt input hashes match; motor/capture/norm/version/
action-mask/feature contracts agree. Native6/16 versus frozenSAC8/16 has
3wins/1loss; online5/16 has0wins/1loss. Execution dates, executors2fd/570/e825
and stochastic draws differ. This is a historical matched-initial-case
reference, separate from the unpaired source-stage probes. Original native
5051actions stay counted once; no new native rollout is inferred.
[Native/SAC case-chain proof](../../../outputs/robotwin-sac-native16-reconciliation-20261031-r1/RESULT.md).

| Final condition on these16 cases | Bottle /8 | Stapler /8 | Total /16 |
| --- | ---: | ---: | ---: |
| Historical native Gaussian |6|0|6|
| FULL PPO |7|1|8|
| Same FULL weights, cache reset |6|1|7|
| Same FULL weights, reward blind |8|1|9|
| Trained episode-reset PPO |8|0|8|
| Frozen source SAC |6|2|8|
| Source-initialized online SAC |5|0|5|

The observed frozenSAC target progression3/4/3/8 per16 differs from the flat
aggregate source curve. It supports recording a late descriptive steering
gain, while the small panel and one training seed prevent broad ranking,
equivalence, causal memory or sample-efficiency claims. Retain the reactive
SAC and trained-reset controls when evaluating any new history policy.

## RoboTwin source K4 attempt-position readout

The three source100 episode banks each reconcile600 complete(goal,round)
K4 groups and2400 episodes. Saved ordinals and actual collector code establish
attempt1–4; success/action/chunk totals match the source100 analyses.

| Learner | Attempt1 | Attempt2 | Attempt3 | Attempt4 |
| --- | ---: | ---: | ---: | ---: |
| FULL PPO |103/600|106/600|97/600|110/600|
| Trained episode reset |98/600|110/600|106/600|107/600|
| SAC |72/600|70/600|64/600|71/600|

FULL's attempt4minus1 is+1.17percentage points; RESET+1.50 andSAC−0.17.
All100 and first/last25-round strata reveal no strong monotonic FULL-specific
gain. Different scheduled scene seeds, stochastic actions and the explicit
episode clock can affect ordinal rates. PPO weights remain fixed through the
six-trial rollout batch; SAC also updates parameters online. These descriptive
source marginals do not isolate causal memory, same-case recovery or a
held-out support/query result. They do not support a hidden strong inner
learning benefit behind the flat aggregate curves.
[Counts, grouping semantics and bound inputs](../../../outputs/robotwin-k4-attempt-analysis-20261031-r1/ANALYSIS.md).
## RoboTwin final SAC100 AD/BC CPU export, 2026-10-31

The final saved SAC100 corpus has a qualified AD training export of 32,425
decisions / 2,400 episodes / 600 whole K4 trials / 1,003,790 historical source
actions. The iteration-100 source-validation BC export contains 328 decisions /
24 episodes / 6 K4 trials / 10,180 historical validation actions. Actual case IDs,
task/seeds and initial-observation hashes are disjoint across splits. The exact
62-root ancestry, actor/counter provenance and executed-noise labels remain bound.
Independent inspection verified both actual index and NPZ hashes, all 21 ZIP
member CRCs per dataset, and the case metadata. All allocated phases and final
owned cleanup were clean. Baige job-vth2qv6r4lxu finished Success / exit 0 in
1,290 CPU-worker-seconds, with no new GPU use, robot actions, VLA queries or SGD.
The CPU grant closes at 2,588 actual worker-seconds including software/recoveries;
4,612 unused seconds are sealed.

This is saved-data readiness, with no new student or robot performance result.
No 500-update AD fit/readout ran; that GPU phase remains deferred for the human's
ordinary-baseline priority: DSRL/SAC first, native PPO afterward. Earlier student results retain their
early-SAC5 data identity. The [CPU export report](../../../outputs/robotwin-final100-ad-export-20261031-r4/RESULTS.md)
contains exact dataset identities, independent audits, storage and cost closure.
## Direct native VLA context interface (2026-10-28)

The separate native-context arm inserts the eleven public scalar feedback
fields as native prefix tokens and trains rank-four prefix-attention LoRA,
feedback projection/type embeddings and a training-only value head. Its
base pi0.5 and action expert stay frozen. The learner sees the current native
observation and scalar history, without historical images or privileged theta.
See [the contract](../../../docs/native-vla-context-icrl.md).

Real-model admission passed on Baige `job-8kkjhb65px67`, source
`b5d97b0a7c0f6aed29660a96e2ab10cce9610915`, one allocated RPBZZZ6.
The CUDA probe passed, seven policy tests passed in 2.10 seconds, and the
renderer was bound to the allocated PCI `0000:62:00.0` device.

| Check | Observed result |
| --- | --- |
| Empty-history native four-Euler action parity | Exact; maximum error 0 |
| Twelve fixed-example flow fitting updates | Loss 0.306293 to 0.230301, 9.18 seconds |
| Final public-feedback projection gradient norm | 0.0134457, no nonfinite parameter gradients |
| Joint stochastic velocity likelihood replay | Exact; maximum error 0 |
| Frozen original motor tensors | Unchanged SHA256 `4191deb5458127d2d1939807f2f604f5c224bc27488e9a23ae3b08d3462bb0f8` |
| Peak allocated CUDA memory at gate | 10,703,170,560 bytes |
| Trainable adapter/critic parameters | 950,273 |

This proves that the native conditioning path is trainable and replayable on
one real observation. It does not prove live adaptation. The fitting weights
were restored to their initialization after the gate.

The same job stopped before completing collection: native action preparation
returns CPU commands and the collector initially failed to move the previous
command back to the GPU feedback buffer. The earlier r2 attempt had stopped
at a nonfinite backward norm from zero-initialized history tokens; a nonzero
learned history type embedding enabled the successful r3 gate. Both failures
and their output identities are retained. The collector device fix and its
CPU/CUDA integration regression are separate from the admitted r3 source.

Raw admission and failure receipts:
[r3 artifacts](../../../outputs/native-vla-context-pilot-20261028-r3/evidence/),
[r2 failure](../../../outputs/native-vla-context-pilot-20261028-r2/RESULTS.md).
No held-out adaptation, reward-use or architecture-comparison result is yet
established by this arm.

The fixed collector passed both CPU and actual allocated-CUDA integration
tests on Baige `job-lwpvhzys2vtc`, source `0d6dee3`. This bounded pilot
**completed with exit 0 at 20:00:45 Shanghai**, 1,899.94 driver seconds.
Five complete K4 histories and 20 supervised updates were saved; validation
flow loss fell from 0.68696 to 0.51411. A paired +45-degree oracle gate
scored 4/4 versus base 1/4. Original frozen weights passed the final check.

| Held-out smoke condition | Success / attempts | Per-attempt successes across two K4 trials |
| --- | --- | --- |
| Normal history | 2/8 | 2, 0, 0, 0 |
| History wiped between attempts | 5/8 | 2, 1, 1, 1 |
| Reward blind | 2/8 | 2, 0, 0, 0 |

There is no observed memory/reward benefit in this small panel. The normal
and blind arrays match; wiping scores higher. There is no matched frozen-base
or warm-start-only live control. These two held-out trials per condition
cannot establish a statistical adaptation or architecture result.

The one PPO step had **zero loss and zero gradient**: its K4 rollout contained
only failures, and both rewards and old values were zero. Exact replay passed
for all 64 chunks. Code-path completion does not establish RL learning; the
shared Adam optimizer retains warm-start momentum, which can still move
weights despite zero new gradient. The final checkpoint retains optimizer,
sampler and all RNG state, 20 supervised updates and the PPO completion flag;
all five stored history hashes and delayed public feedback fields were
independently verified on allocated CPU resources. Total task rollout cost:
4,160 steps. Monitor deleted; no follow-on launch. See
[complete results and audit](../../../outputs/native-vla-context-pilot-20261028-r4/RESULTS.md).
