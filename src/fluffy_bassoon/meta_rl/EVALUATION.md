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

## RoboTwin SAC100: source complete, final evaluation pending

Audited closed slice0060 records100 source rounds/600 completeK4 trials/
2400 episodes/1003790 source actions/32425 decisions. Success is277/2400
(11.54%); quarter totals78/57/76/66 per600 show no sustained improvement.
Per-goal successes per400 are54 handover,14 bottles,54 a2b,100 object stand,
43 phone stand and12 scan. Some goals improve while others decline; this is
not a claim that every source goal has the same curve. SourceVAL10/25/50
is1/24,1/24,3/24. Stage100 metadata says evaluating; sourceVAL100, final
targets and whole-campaign exit are still pending.

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
The metadata analysis did not rehash/load checkpoint tensors. It reconciles
closed JSON/JSONL counts and preserves the later manifest's status at read
time; it does not certify completed final evaluation.
[Final source100 comparison and exact identities](../../../outputs/robotwin-sac-source-diagnosis-20261031-r3/ANALYSIS.md).
