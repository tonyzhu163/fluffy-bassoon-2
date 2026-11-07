# RoboTwin PPO: published evidence and our restoration gap

Primary-source review, updated November 8, 2026. Question: why has transferring the
recorded RLinf PPO recipe not produced a verified gain here? This is a focused
literature/configuration audit, not a systematic review or a reproduction of
external results. External percentages below are author-reported.

## What the evidence currently says

RLinf reports successful pi0.5 PPO on RoboTwin. Our short transferred runs have
not established a learning gain. These statements are compatible: our runs use
different joint initializers, a different task, and much shorter training.
The latest Motus result is **one fresh integrated collection/update cycle**,
with15 Adam updates per rank and DEV3/16→4/16. An earlier recovered first
batch gave DEV4/16→2/16. Neither panel establishes improvement or demonstrates
that the released recipe fails.

Three separate questions need separate evidence:

| Question | Current answer |
| --- | --- |
| Can our learner update real rollout data and restore exactly? | Yes, for the qualified Motus first-batch learner boundary. |
| Can the complete optimized collector/learner cycle run repeatedly at scale? | One8-rank B32 cycle passed the combined footprint; repeated cycles remain unverified. |
| Does sustained PPO improve the selected joint motor on our source tasks? | Unresolved; the available endpoint panels show no gain. |

## Direct and adjacent literature

| Primary source | Relevant evidence | Limit on applying it here |
| --- | --- | --- |
| [RLinf RoboTwin training guide](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/robotwin.html) | On clean `adjust_bottle`, pi0.5 SFT 85.94%, PPO 96.09%; pi0 SFT 76.56%, PPO 98.44%. | Task-specific SFT and clean task configuration; not the joint Motus phone task. The linked SFT/PPO repositories have no explanatory model card in the inspected views. The inspected guide does not identify the exact training epoch or across-training-seed uncertainty behind the percentages. |
| [VLRLA, Appendix A.2.1](https://arxiv.org/html/2610.19517v1) | RoboTwin collection uses 25 PPO iterations, each with 256 environments, four rounds, H400/H50 and five optimization epochs. Both successes and failures are retained. Table 1 reports GRPO 50.8% versus base 36.7% on `place_phone_stand`, and a decrease on `handover_mic`. | Its main-table GRPO results and its PPO data-collection recipe are different evidence. The appendix does not identify Motus as its initializer. It does not establish that every task improves or that one iteration is sufficient. |
| [CFG-Policy, Table 3 and Appendix B.4.2](https://arxiv.org/html/2610.14936v1) | Direct pi0/pi0.5 RoboTwin comparison against Flow-SDE+PPO. Pi0.5 PPO endpoints range from 7.81% on Rotate QR Code to 100% on Grab Roller. Runs use task-specific SFT, 50 rounds, predict50/execute25 and 64 evaluation scenes. | Endpoint scores alone do not show each PPO baseline's gain over SFT. This uses a different cadence and critic setup; guidance is an additional method, not a correction we can silently add. |
| [SimpleVLA-RL, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cbfbcb4da14235bd69b134070898ae9d-Abstract-Conference.html) | Reports RoboTwin improvements with OpenVLA-OFT and exploration-enhancing strategies. | Direct benchmark evidence, but a different policy family and training system. Its gains are not pi0.5 PPO reproduction evidence. |
| [piRL](https://arxiv.org/html/2510.25889v1) | Flow-SDE/Flow-Noise policy optimization; LIBERO ablations compare PPO/GRPO, critic placement, noise, denoising and chunk length. | The inspected version evaluates LIBERO and ManiSkill, not RoboTwin. Its ablations motivate diagnostics, not guaranteed RoboTwin fixes. |
| [RLinf-VLA](https://arxiv.org/html/2510.06710v1) | Explains chunk credit assignment, reset modes, valid-action masks and GRPO group returns. Its ablations find task-dependent effects of masking/filtering and sensitivity to rollout size. | This version lists RoboTwin and pi0/pi0.5 integration as future work. Its original OpenVLA/OpenVLA-OFT results are not the later RoboTwin pi0.5 result. |
| [RLinf-VLA v3, August 2026](https://arxiv.org/html/2510.06710v3) | Updated direct RoboTwin evidence: six task-specific OpenVLA-OFT models average24.48%→84.63% with GRPO. Discusses success filtering, valid-action masking and trajectory-length normalization. | Replaces the earlier version's future-work limitation. It does not establish the same result for joint Motus. Its SFT and RL evaluation sampling differs; use matched sampling for our causal before/after comparison. |
| [Prism-GRPO](https://arxiv.org/html/2608.17423v1) | Direct RoboTwin study of tied-group waste: same-scene groups8,512 rollouts/update,five training seeds; adds an execution-quality reward and reports22–56% fewer rollouts to the binary-GRPO peak. | Extra reward information changes the objective. It motivates measuring tied groups, not silently importing quality rewards into the binary-reward control. |
| [SynthDemo-RL](https://arxiv.org/html/2609.21650v1) | Studies reward coverage: direct PPO leaves some initially zero-success LIBERO-PRO tasks unsolved. Its separate single-arm Piper RoboTwin study improves aggregate success after SFT plus RL, with one task unchanged. | Demonstrations, embodiment and tasks differ. It supports checking coverage, not concluding that our nonzero-reward data cannot support PPO. |

The literature supports PPO as a credible method, while showing substantial
dependence on the starting policy, task, exploration and action cadence.
It does not support expecting identical gains after changing those conditions.

## The released configuration versus the executed port

The official experiment YAML was read through the GitHub connector. Its blob
`20cf1d1366c47a039f3011107a7ecba2fd656879` matches our November 5 snapshot at
[`77a52129`](https://github.com/RLinf/RLinf/blob/77a521298f1d215971246c2d7a423720d9e76652/examples/embodiment/config/robotwin_adjust_bottle_ppo_openpi_pi05.yaml).
Matching this file does not establish that all imported dependencies match.

| Field | Released example | Executed Motus pilot |
| --- | --- | --- |
| Initializer | RLinf task-specific `adjust_bottle` SFT | Joint Motus pi0.5 |
| Task | `adjust_bottle` | `place_phone_stand` |
| Predicted/executed chunk | 50/50 | Native32/32 |
| Wrapper horizon | 200 | 192 |
| PPO scene group | 1 | 8, to match the planned GRPO comparison |
| Environments/rounds | 256/4 | 256/4 |
| Global optimization minibatch/epochs | 2048/5 | 2048/5 |
| Per-device execution microbatch | 32 | 256; accumulation2 on four devices,1 on eight devices |
| Evaluation panel | 128 environments | 16 precommitted source DEV cases |
| Sustained training | Config permits repeated outer epochs | Earlier retained-batch recovery; later one fresh8-rank cycle, second collection interrupted |

The shared actor/value learning rates remain 5e-6/1e-4; gamma .99,
lambda .95, clip .2, five Flow-SDE steps and noise .3 are retained. The official
YAML's `max_epochs: 1000` is a configured ceiling, **not evidence that the
reported checkpoint required or completed 1,000 epochs**.

Motus's [model card](https://huggingface.co/motus-robotics/pi0.5_robotwin2)
declares H32, delta arm-joint actions, `adapt_to_pi`, three cameras and joint
clean/randomized training over 50 tasks and 27,500 episodes. Its native
transforms are necessary for this checkpoint. Changing them to resemble the
H50 release would create an incorrect motor rather than improve fidelity.

The [inherited model defaults](https://github.com/RLinf/RLinf/blob/77a521298f1d215971246c2d7a423720d9e76652/examples/embodiment/config/model/pi0_5.yaml)
set `value_after_vlm: false`; the task YAML detaches critic inputs. piRL instead
uses a VLM-side critic for pi0.5, and CFG-Policy also documents VLM features
for that model. There are multiple published/released recipes here. A VLM
critic is a reasonable separately tested candidate, not proof that the
released RoboTwin expert-side critic is broken.

## What actually happened in this repository

The authoritative local records distinguish execution failures from learning
outcomes:

* The full Motus job hit its fixed worker time limit. Its closeout records
  interruption and forced cleanup, with no established CUDA OOM and no complete
  updated checkpoint. This is an execution failure, not an observed converged
  policy failure. [Closeout](../../outputs/robotwin-restoration-20261106-r1/MOTUSPPO04-CLOSEOUT.json).
* Its preserved first rollout buffer was recovered separately. Native PPO
  completed 15 Adam updates per rank, saved a full learner boundary, restored
  owned state independently and reproduced the next update exactly.
  `6144 / 2048 * 5 = 15`: those are minibatch updates over **one** collection,
  not 15 fresh-data iterations.
  [Learner evidence](../../outputs/robotwin-restoration-20261106-r1/MOTUSREPLAY01-COMPLETE.json).
* Paired Motus evaluation then returned base4/16 and updated2/16, with one
  gained case and three lost cases. This is a negative observation. Four
  discordant pairs give an exact two-sided paired sign/McNemar probability
  of .625 under the no-change null, assuming independent pairs. This
  exploratory panel does not establish a general degradation claim or improvement.
  [Endpoint evidence](../../outputs/robotwin-restoration-20261106-r1/MOTUSDEV01-COMPLETE.json).
* A later fresh optimized eight-rank Motus job completed one integrated
  collect/update cycle with15 Adam steps on every rank and a full cold-verified
  checkpoint1. Its DEV panel was3/16 initially and4/16 after the update. The
  fixed worker timeout interrupted the second collection. This qualifies one
  combined cycle, without establishing sustained learning or improvement.
  Native B32 training/B2 evaluation is an explicit numerical condition distinct
  from the earlier four-rank B64/B4 run.
  [Integrated-cycle closeout](../../outputs/robotwin-restoration-20261106-r1/MOTUSOPTIMIZED03-CLOSEOUT.json).
* Earlier Sidney arms did complete two collections and 40 updates/rank.
  Their later fixed panel was base5/16, PPO5/16, GRPO2/16. Thus the full history
  is not exclusively a failed launcher; we also have short runs with no
  observed endpoint PPO gain.
  [Matched training](../../outputs/robotwin-sidney-matched-baselines-20261106-r1/RESULT.md),
  [endpoint readout](../../outputs/robotwin-restoration-20261106-r1/READOUT01-COMPLETE.json).

The physical Motus audit found 130 successes among 1,024 episodes, 128 unique
scene groups and 5,958 positive-duration decisions. It separates 186 terminal
no-op calls from physical experience. Reward coverage is therefore nonzero;
“PPO received only failures” is not an explanation supported by this run.
Student physical masks remain distinct from the unchanged native teacher
loss masks. [Physical audit](../../outputs/robotwin-restoration-20261106-r1/CPURECONSTRUCTION-PREFLIGHT01.json).

## Ranked explanations and discriminating checks

| Explanation | Evidence status | Next check |
| --- | --- | --- |
| Sustained online learning has not been tested adequately | Verified limitation, not a proof that more training will help | Run successive complete collect/update cycles; show new rollout data and checkpoint progression. |
| New motor/task/cadence changes the learning problem | Verified protocol differences; causal effect unknown | Hold these fixed within a port experiment; report it separately from exact release reproduction. |
| Combined scene/model memory and setup overhead obstruct training | Four-rank B64 reserve failed; eight-rank B32 passed one integrated cycle | Qualify repeated real cycles and exact full-state recovery. Actor-only capacity is insufficient. |
| Group8 reduces scene diversity relative to group1 | Verified layout change | Keep the matched group8 arm; add a separately named group1 PPO reference when affordable. |
| Exploration harms the initializer or fails to reach useful alternatives | Plausible, unisolated | Compare policy0 Flow-SDE and ODE on the same cases before changing noise. Different TRAIN/DEV scenes cannot isolate this gap. |
| Critic quality or return/boundary semantics limit learning | Plausible, unisolated | Inspect explained variance, targets, values, advantage spread and episode boundaries; compare critic designs only after a stable control. |
| SAPIEN camera clipping changes policy observations | Concrete difference from current upstream; no camera-only benefit on the small completed panel | Independent315-file readback passed: near0.1/H192:2/8; near0.01/H192:2/8; near0.1/H384:3/8; near0.01/H384:2/8. Preserve the active continuation's protocol; this exploratory panel does not establish equivalence. |
| Numerical changes from larger batches contaminate likelihood ratios | Verified batch effect; contribution to missing learning gain unisolated | B32 exactly matches original behavior likelihoods; B256 puts11.279% of unchanged-policy ratios outside the clip interval. Use B32/accumulation8 and qualify a corrected learning control. |

Our native-batch adapter matches its native reference exactly, but native
batched outputs differ from singleton outputs. The diagnosis did not isolate
BF16 as the cause. This matters when profiling maximum memory usage:
execution partition and actual batch/row must remain explicit numerical
conditions. [Batch evidence](../../outputs/robotwin-restoration-20261106-r1/BATCHMATH01-COMPLETE.json),
[actor selection](../../outputs/robotwin-restoration-20261106-r1/ACTOR-SELECTION.json).

The local replay hook replaces stored rollout log probabilities with an actor
recomputation after shuffling. Its logged gap is mean absolute coordinate error.
Pinned RLinf's chunk-level loss instead sums H32 times14 action coordinates.
The completed read-only checkpoint0 measurement, independently reduced on CPU,
finds exact original behavior probabilities at B32 on all5958 valid decisions.
At B256, unchanged-policy joint ratios range0.520–1.883;11.279% lie outside
[0.8,1.2] and5.572% activate the clipped surrogate before SGD. The replacement
actor reference hid this discrepancy from the zero first-replay check.
This establishes a numerical execution-partition problem; its contribution to
missing learning gains still requires a corrected learning control. Use B32
with gradient accumulation while preserving global2048 and five epochs.
[Likelihood measurement](../../outputs/robotwin-restoration-20261106-r1/MOTUSJOINTAUDIT01-COMPLETE.json).

Training-loop DEV snapshots and restores the current rollout RNG. Fixed scenes
therefore preserve the training stream but do not by themselves impose common
initial action noise across checkpoints. Treat the latest3/16 to4/16 panels as
descriptive. The earlier separately qualified MotusDEV01 comparison explicitly
restored baseline sampling RNG; its pairing evidence remains valid. Future
source qualification will explicitly share case/chunk noise across policies.

piRL motivates the exploration and critic checks above: it separates
stochastic training from ODE evaluation, and its LIBERO ablations show
noise/gradient and chunk-length tradeoffs. Those results do not justify
simultaneously changing our noise, learning rate, critic and chunk length.
Such a change would leave the cause of any improvement unresolved.
[Primary ablations](https://arxiv.org/html/2510.25889v1#S5.SS3).

The public [RLinf issue1268](https://github.com/RLinf/RLinf/issues/1268)
also reports unstable pi0.5 Flow-SDE GRPO on a different RoboTwin task with
absolute-joint H16 actions. No resolution was visible in the inspected issue.
This is an implementation report, not controlled proof of a general algorithm
failure, and its suggestion that updates are too conservative is the reporter's
interpretation. It does not justify raising our learning rate from log scalars.

## Formula and data-path audit

Our pinned native chunk-level PPO uses summed chunk reward, gamma.99 and
lambda.95. With native done flag d at the following chunk boundary,
delta_t=R_t+.99(1-d_{t+1})V_{t+1}-V_t and
A_t=delta_t+.99*.95(1-d_{t+1})A_{t+1}. Returns are A_t+V_t;
advantages are normalized over valid native decisions with sample standard
deviation and epsilon1e-5. Preserve the native likelihood reduction, value
estimands, loss masks and optimizer weighting when changing execution batches.
The independent float64 check operates on the actual post-routing actor buffer.

For pinned GRPO, a scene's8 rollout scores are standardized with sample standard
deviation and epsilon1e-6; the trajectory advantage is broadcast over its valid
chunks. Equal scores give exactly zero advantages. These formulas and the
actual model likelihood reference are separate checks: passing one does not
qualify the other. Physical delivered-target masks are retained for analysis;
they do not replace RLinf's native training loss masks.

The complete actual PPO audit now matches all8 actor buffers to192 simulator
trace batches by commands and observations. All GAE references pass,136/1024
episodes have positive raw score, and every valid decision in those episodes
has positive normalized advantage. Added timeout bootstrap is exactly zero.
Initial critic explained variance versus raw Monte Carlo returns is near zero;
the aggregate training log's negative explained variance does not isolate why
the endpoint failed to improve. [Actual-batch evidence](../../outputs/robotwin-restoration-20261106-r1/MOTUSLEARNINGAUDIT02-COMPLETE.json).
Counterfactual raw-reward GRPO arithmetic passes1e-5 on the same collection,
with73/128 tied groups and55 informative groups. This is substantial dilution,
not absence of all learning signal or a completed Motus GRPO experiment.
[Raw-group evidence](../../outputs/robotwin-restoration-20261106-r1/GRPORAWGROUPAUDIT02-COMPLETE.json).

## Consequence for the next experiment

Prioritize a stable multi-iteration **source-task PPO control**. The user's
latest instruction requires effective PPO/GRPO before more student work. First qualify the
optimized full collector/learner footprint and repeated offload/reset lifecycle.
Then demonstrate iteration0 collection, update1, collection from policy1 and
update2 before extending a predeclared learning curve. More SGD on the same
retained buffer cannot substitute for new on-policy collections. Record
success, critic fit, advantage spread, KL/clip fraction and actual update/data
counts at each boundary. Use a larger fixed DEV panel and multiple training
seeds for a learning claim; choose the affordable scope from measured costs.

An exact released `adjust_bottle` training control conflicts with this
project's held-out split: `adjust_bottle` is currently a transfer task.
Training on it would require an explicit change to the scientific protocol
and subsequent claims. The autonomous default is to preserve the split and
complete the acknowledged Motus source-task port. No held-out training is
authorized by this review.

Consult the [current ledger](../../outputs/robotwin-restoration-20261106-r1/LATEST-BUDGET.json)
for approved compute and remaining allowance; the historical review budget is
superseded. A concrete continuation contract and measured resource request must
precede expenditure above the active cap. Completed offline DIT/IQL fitting and
frozen-runtime qualification establish neither adaptation nor PPO convergence;
further student experiments wait. The single-policy dataset is still not a
chronological AD corpus.

Search queries, source identities, exclusions and the dated implementation
handoff are in the [review record](../../outputs/robotwin-restoration-20261106-r1/RECIPE-REVIEW.json).
