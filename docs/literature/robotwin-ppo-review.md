# RoboTwin PPO: published evidence and our restoration gap

Primary-source review, November 7, 2026. Question: why has transferring the
recorded RLinf PPO recipe not produced a verified gain here? This is a focused
literature/configuration audit, not a systematic review or a reproduction of
external results. External percentages below are author-reported.

## What the evidence currently says

RLinf reports successful pi0.5 PPO on RoboTwin. Our short transferred runs have
not established a learning gain. These statements are compatible: our runs use
different joint initializers, a different task, and much shorter training.
The latest Motus result is **one recovered collection/update cycle**, not
15 online collect/update cycles. Its negative small-panel result deserves
follow-up; it does not demonstrate that the released recipe fails.

Three separate questions need separate evidence:

| Question | Current answer |
| --- | --- |
| Can our learner update real rollout data and restore exactly? | Yes, for the qualified Motus first-batch learner boundary. |
| Can the complete optimized collector/learner cycle run repeatedly at scale? | Not yet verified with the full combined memory footprint. |
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
| Per-device execution microbatch | 32 | 256 with accumulation2 across four devices |
| Evaluation panel | 128 environments | 16 precommitted source DEV cases |
| Sustained training | Config permits repeated outer epochs | One retained Motus collection batch; later learner recovery |

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
| Combined scene/model memory and setup overhead obstruct training | Full-size reserve failed; optimized small-scene lifecycle passed | Qualify co-resident scenes, rollout and actor through repeated real cycles. Actor-only capacity is insufficient. |
| Group8 reduces scene diversity relative to group1 | Verified layout change | Keep the matched group8 arm; add a separately named group1 PPO reference when affordable. |
| Exploration harms the initializer or fails to reach useful alternatives | Plausible, unisolated | Compare policy0 Flow-SDE and ODE on the same cases before changing noise. Different TRAIN/DEV scenes cannot isolate this gap. |
| Critic quality or return/boundary semantics limit learning | Plausible, unisolated | Inspect explained variance, targets, values, advantage spread and episode boundaries; compare critic designs only after a stable control. |
| Numerical changes from larger batches contaminate likelihood ratios | Known hazard, partly controlled | Retain matched execution-partition probability replay and pre-SGD ratio checks; do not assume batching is transparent. |

Our native-batch adapter matches its native reference exactly, but native
batched outputs differ from singleton outputs. The diagnosis did not isolate
BF16 as the cause. This matters when profiling maximum memory usage:
execution partition and actual batch/row must remain explicit numerical
conditions. [Batch evidence](../../outputs/robotwin-restoration-20261106-r1/BATCHMATH01-COMPLETE.json),
[actor selection](../../outputs/robotwin-restoration-20261106-r1/ACTOR-SELECTION.json).

piRL motivates the exploration and critic checks above: it separates
stochastic training from ODE evaluation, and its LIBERO ablations show
noise/gradient and chunk-length tradeoffs. Those results do not justify
simultaneously changing our noise, learning rate, critic and chunk length.
Such a change would leave the cause of any improvement unresolved.
[Primary ablations](https://arxiv.org/html/2510.25889v1#S5.SS3).

## Consequence for the next experiment

Prioritize a stable multi-iteration **source-task PPO control** while retaining
the decoded-student work as the separately requested arm. First qualify the
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

The current 50-hour ledger has only 694 GPU-seconds remaining. That cannot
fund a meaningful four-device online continuation. A concrete continuation
contract and measured resource/budget request must precede expenditure above
the cap. This review launches no accelerator work. Offline DIT/IQL exploration
can continue on the mixed-quality pilot data; it does not establish PPO
convergence, and the single-policy dataset is not a chronological AD corpus.

Search queries, source identities, exclusions and the dated implementation
handoff are in the [review record](../../outputs/robotwin-restoration-20261106-r1/RECIPE-REVIEW.json).
