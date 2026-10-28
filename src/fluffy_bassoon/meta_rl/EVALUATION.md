# Evaluation record

Record each result here with its source commit, cluster, job and receipt path.

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
