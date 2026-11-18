# RoboTwin PPO: published evidence and our restoration gap

Primary-source review, updated November 8, 2026. Question: why has transferring the
recorded RLinf PPO recipe not produced a verified gain here? This is a focused
literature/configuration audit, not a systematic review or a reproduction of
external results. External percentages below are author-reported.

The configuration and replay measurements below concern early Motus and Sidney experiments. The [verified RoboTwin findings](../FINDINGS.md#robotwin) include the later native Sidney comparison. [Motor and controller conventions](../robotwin/rl-baselines.md)

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

The expert backend is another dependency distinction. The pinned
[RLinf installer](https://github.com/RLinf/RLinf/blob/c605b9d55ed93a180119b099e6f836abff293b08/requirements/install.sh#L3724)
installs cuRobo for NVIDIA without an exact revision. Our qualified runtime
uses MPLib; an isolated cuRobo backend is qualified separately. In the pinned
[RoboTwin planner](https://github.com/RoboTwin-Platform/RoboTwin/blob/dca9ec682688821c42944eacd7f5dd9bfb15396f/envs/robot/planner.py),
cuRobo builds a table-only obstacle world; the MPLib path takes an early SAPIEN
scene snapshot. Rebuilding MPLib after task actor loading adds grasped objects
to its obstacle set and changes the expert's contact constraints. It is not an
equivalent repair merely because geometry conversion succeeds. The native
joint-action VLA path does not call this expert planner; earlier expert failures blocked
fresh scene selection. The isolated native backend now restores that path,
without explaining the completed weak learning result. [Actual paired controls](../../outputs/robotwin-restoration-20261106-r1/MOTUS-PLANNER-CONTROL01-CLOSEOUT.json).

Current [cuRobo documentation](https://nvlabs.github.io/curobo/latest/)
describes a redesigned API. A restoration must pin and qualify the compatible
legacy MotionGen API, rather than following an unversioned install into a
shared environment. A CPU build/import does not qualify accelerator kernels
or successful expert episodes. Preserve the existing training/evaluation
runtime while preparing an isolated backend control.

| Field | Released example | Earlier Motus pilot | Corrected fresh controls |
| --- | --- | --- | --- |
| Initializer | RLinf task-specific `adjust_bottle` SFT | Joint Motus pi0.5 | Fresh joint Motus pi0.5 |
| Task | `adjust_bottle` | `place_phone_stand` | `place_phone_stand` |
| Predicted/executed chunk | 50/50 | Native32/32 | Native32/32 |
| Wrapper horizon | 200 | 192 | 192 |
| PPO scene group | 1 | 8, to match the planned GRPO comparison | PPO1; GRPO8 |
| Environments/rounds | 256/4 | 256/4 | 256/4 |
| Global optimization minibatch/epochs | 2048/5 | 2048/5 | 2048/5 |
| Per-device execution microbatch | 32 | 256; accumulation2 on four devices,1 on eight devices | 32; accumulation8 on eight devices; exact original behavior check each collection |
| Evaluation panel | 128 environments | 16 precommitted source DEV cases | 16 loop diagnostics;64 paired DEV and selected disjoint64 confirmation complete |
| Sustained training | Config permits repeated outer epochs | Earlier retained-batch recovery; later one fresh8-rank cycle, second collection interrupted | Two fresh collections complete; final cold boundaries independently verified |

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

The isolated legacy backend now passes an actual4/4 old-scene expert control,
versus preserved full-state MPLib0/4, with initial image/state hashes equal and
native table-only collisions retained. Only stale asset-file paths change in
local YAML copies. Independent device/asset/cleanup readback passes; no VLA or
SGD runs. A subsequent policy-free scan independently qualifies64 fresh
expert/reset-valid scenes from84 ordered candidates, excluding all1000TRAIN
and150oldEVAL seeds. The full prefix and20rejections are retained. This
restores independent confirmation data without establishing a learning gain.
[Fresh bank evidence](../../outputs/robotwin-restoration-20261106-r1/MOTUS-FRESH-SCENES05-CLOSEOUT.json). [Expert evidence](../../outputs/robotwin-restoration-20261106-r1/MOTUS-CUROBO-CONTROL02-CLOSEOUT.json).

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
  Its rollout batch32/evaluation batch2 differed from the earlier four-rank
  rollout64/evaluation4 run; its actor microbatch256 was subsequently rejected
  by the unchanged-behavior likelihood audit.
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

The first64-case SFT readouts for the corrected controls stopped before loading
checkpoint2. The failure was our optimizer evidence guard: native RLinf
[`offload_optimizer` and `onload_optimizer`](https://github.com/RLinf/RLinf/blob/c605b9d55ed93a180119b099e6f836abff293b08/rlinf/hybrid_engines/fsdp/strategy/fsdp.py)
index the optimizer's state mapping for every parameter. This can create an empty
mapping for a registered zero-length FSDP shard without an Adam update. The
actual rejected PPO layouts preserve every captured owned moment and parameter
group; native offload/onload reproduces the mapping on a Linux CPU fixture.
Twelve core checkpoint tests qualify a narrow equivalence between absent and
empty state for unowned zero-length shards. Owned or unregistered parameters
remain strict. This diagnoses the interrupted readout, not a learning failure;
The corrected readouts subsequently passed all GPU guards for both checkpoints
and both methods, plus independent cold/physical/noise readback. The selected
GRPO confirmation also passed every execution check but failed the learning gate.
[Actual-layout and reference audit](../../outputs/robotwin-restoration-20261106-r1/MOTUS-EMPTY-STATE-AUDIT-CPU01.json).

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
