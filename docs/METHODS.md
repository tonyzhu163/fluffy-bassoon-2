# Methods and runtime

For decision pretraining, the unit of data is a support history paired with a decision target at a fresh query under the same persistent condition. The source behavior that generates support can differ from the teacher that labels the query. Only the public interaction prefix is an input. The motor stays frozen during offline adapter fitting, and the reader and motor both stay frozen during evaluation. [Research design](PAPER.md)

## LIBERO evaluation

For the standard four-suite comparison, use the [OpenPI LIBERO evaluation](https://github.com/Physical-Intelligence/openpi/tree/main/examples/libero) and its model-specific action conventions. For LIBERO-Pro2.0 use the pinned release and the project's 24/8/8 base-task split. The run directory contains the [environment bootstrap](../outputs/libero-meta-learning-20261117-r1/bootstrap.py), [native policy and simulator integration](../outputs/libero-meta-learning-20261117-r1/native.py), [qualification driver](../outputs/libero-meta-learning-20261117-r1/qualify.py), [split](../outputs/libero-meta-learning-20261117-r1/SPLITS.json) and [execution record](../outputs/libero-meta-learning-20261117-r1/EXECUTION.md). These are campaign entry points. Follow the [published evaluator](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/docs/remote-evaluation-guide.md) for independent-episode scores.

The first RLinf and converted OpenPI panels failed all six distinct development cases. The original OpenPI JAX sampler solved both matched BASE starts (Spatial and Goal) with ten flow steps and five executed controls per replan. After the PyTorch runtime adopted the released quantile transforms, it also solved both nominal starts and reproduced both zero-adapter traces. The four perturbed cases still failed. Report the sampler, checkpoint, statistics and cadence together. [Motor qualification](../outputs/libero-meta-learning-20261117-r1/EXECUTION.md)

For the proposed retry evaluation, carry public support across resets within one trial. Record the command actually issued at the controller boundary and the observed response with their timestamps. Under delayed control the applied command can differ from the issued command. Keep simulator-internal queues and perturbation IDs outside the policy. A timing diagnostic needs control-rate action and motion measurements. The first proposed controller is a categorical retry selector next to the frozen motor. Its benchmark and the later continuous adapter and WAM extension are in the research design. [Model and protocol](PAPER.md#first-model)

The public release includes successful scripted demonstrations under both BASE and S20. The demonstration index identifies the HDF5 groups for fitting and development examples with distinct support and query reset IDs. In the audited bowl archives, the first stored observation matches the simulator after the first action. Use observation `t` to supervise action `t + 1` and subsequent commands. The archive starts after the initial command. To label alternative decisions at an identical failed-policy state, query a teacher there or fork it in the simulator. [Demonstration index](../outputs/libero-pro2-demonstration-pairs-20261118-r1/RESULT.md) · [Replay and action alignment](../outputs/libero-s20-motor-20261119-r1/RESULT.md)

## RoboCasa365 and RoboTwin interfaces

| | RoboCasa365 | RoboTwin |
| --- | --- | --- |
| Task coverage | Atomic tasks for source learning and mechanism controls | Six source and six held-out tasks in the [split](../benchmarks/robotwin/configs/robotwin_vlrla.json) |
| Motor boundary | Human300 π0.5: three camera views, state normalization and action decoding bound to the checkpoint | Three camera views and 14-dimensional bimanual actions. Sidney uses absolute joints; Motus is a distinct delta-joint motor. |
| Timing | For the native Human300 SFT reference, predict 50 actions, execute 5, use 10 ODE steps. Historical RLinf teachers have their own executed chunks and horizons. | Bind controller, action horizon, executed targets and actual physics time. See the [RoboTwin protocol](robotwin/rl-baselines.md). |
| Native entry | [Flow PPO](../benchmarks/robocasa/train_flow_ppo.py) and [configs](../benchmarks/robocasa/configs/) | [RLinf entry](../benchmarks/robotwin/train_ppo.py) and [configs](../benchmarks/robotwin/configs/) |

The [RLinf integration](../src/fluffy_bassoon/rlinf/) extends the upstream workers for rollout data, replay, action accounting and recovery. Each benchmark uses the RLinf checkout and model transforms qualified for its own configuration. A patch in [patches/](../patches/) applies to its named revision. Native reference evaluators and converted RLinf policies need numerical parity checks before using them interchangeably.

## Offline learners

| Objective | Supervision at the query | Implementation |
| --- | --- | --- |
| Algorithm Distillation (AD) | Imitate what a chronological source policy actually did. An improving-process claim requires evidence of source improvement. | [History imitation](../src/fluffy_bassoon/meta_rl/vla_history_train.py) |
| Decision-Pretrained Transformer (DPT) | Fit a preselected teacher’s action at the same saved query observation. Earlier actions and feedback remain those actually experienced. | [Corpus construction](../src/fluffy_bassoon/meta_rl/frozen_steering_history.py) and the same imitation trainer |
| DIT-style | Estimate values using physical-episode returns and TD regularization, then weight imitation of executed actions by estimated advantage. | [Value objectives](../src/fluffy_bassoon/meta_rl/history_value.py) |
| In-context IQL | Fit causal Q and expectile V from offline transitions, then extract a policy by advantage-weighted imitation. This version has no world-model prompt. | [Value and actor fitting](../src/fluffy_bassoon/meta_rl/history_value_train.py) |

These are adaptations to the frozen motor. In particular, the published DIT
offline value estimators use return-labeled context; ours use causal shifted
action/reward history and reserve returns for supervision. The referenced
offline IC-IQL shares a value-trained backbone with a detached actor head;
ours has separate actor/Q/V encoders. Method names do not establish exact
reproduction. [Primary specifications and implementation differences](../outputs/project-diagnosis-20261116-r1/RESULT.md#method-and-evaluation-distinctions).

Run `python -m fluffy_bassoon.meta_rl METHOD --help` for the trajectory-archive trainer, with `ad`, `dpt`, `dit` or `ic_iql`. Supply train/validation files and SHA-256s, output directory, exact source commit and allocated host. The trainer checks the objective in the archive and rejects overlapping stream IDs. The interface is part of the experiment: a direct action head, a velocity residual and a learned KV adapter answer different questions.

The [flow bridge](../src/fluffy_bassoon/meta_rl/flow_bridge_train.py) fits action-derived flow targets through the frozen expert. The [decoded chunk trainer](../src/fluffy_bassoon/meta_rl/chunk_training.py) supports AD, DIT, IC-IQL and cloning controls on masked physical chunks; it has no DPT query-label path. Do not substitute a behavior action for a missing teacher query. Keep episode termination, padding and delivered-action masks separate. Resume requires the optimizer, sampler, RNG and learned state, not just exported actor weights.

For constructed RoboCasa queries, the exporter matches the trial and decision observation across branches. Supplying `--action-stats` adds the selected branch’s H50 × 32 normalized action target and a 50-control validity mask. The input is its sequence of delivered twelve-coordinate commands after binary gripper and mode mapping. The categorical-only format is still supported. [Constructed-query exporter](../src/fluffy_bassoon/meta_rl/constructed_queries.py).

## Environments and tests

The [agent simulator collector](../benchmarks/agent_sim/README.md) exposes one
persistent RoboCasa365 episode through the same MCP tools to Codex or Claude
Code. It records lossless control-rate observations, native commands and
outcomes, plus explicit pre-action subgoals and brief decision summaries.
Its exporter separates causal inputs from current action/context targets and
future outcome labels. A second policy view conditions actions on subtask and
retrospective speed/quality descriptors, with sourced optional segment labels;
deployment must supply desired descriptors. Future frames remain explicit goal
prediction targets. These are robot-action demonstrations, not noise targets
for the existing `vla_history` trainer; a matching feature/action head is required.
Native simulation and agent success require separate runtime qualification.

Use `uv sync --locked` on allocated Linux for the shared Python environment. A quick learner check is:

```sh
uv run pytest -q tests/test_chunk_training.py tests/test_history_value.py tests/test_vla_history.py
```

Benchmark tests additionally need their qualified RLinf/OpenPI or RoboTwin packages, upstream configs and simulator assets. Run [native RLinf tests](../tests/test_rlinf_native_ppo.py) with `EMBODIED_PATH` pointing to the pinned RLinf checkout’s `examples/embodiment`; run the [RoboCasa preprocessing tests](../tests/test_robocasa_pi05_preprocessing.py) in its patched OpenPI environment. See [RoboTwin setup](robotwin/rl-baselines.md#runtime-setup) for its separate overlay and reference controller tests. A CPU pass qualifies software behavior; a GPU rollout requires an allocation-local probe and a real episode. Operational rules are in [AGENTS.md](../AGENTS.md).
