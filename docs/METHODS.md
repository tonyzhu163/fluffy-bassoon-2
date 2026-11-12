# Methods and runtime

The data path is **benchmark → RLinf rollout → offline history learner → frozen-policy evaluation**. The source policy can change while it generates data. A student fit uses recorded interactions; it does not update the native π0.5 motor or interact with held-out evaluation tasks.

## Benchmark interfaces

| | RoboCasa365 | RoboTwin |
| --- | --- | --- |
| Current scope | Atomic tasks for source learning; composites deferred to transfer | Six source and six held-out tasks in the [split](../benchmarks/robotwin/configs/robotwin_vlrla.json) |
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

Run `python -m fluffy_bassoon.meta_rl METHOD --help` for the trajectory-archive trainer, with `ad`, `dpt`, `dit` or `ic_iql`. Supply train/validation files and SHA-256s, output directory, exact source commit and allocated host. The trainer checks the objective in the archive and rejects overlapping stream IDs. The interface is part of the experiment: a direct action head, a velocity residual and a learned KV adapter answer different questions.

The [flow bridge](../src/fluffy_bassoon/meta_rl/flow_bridge_train.py) fits action-derived flow targets through the frozen expert. The [decoded chunk trainer](../src/fluffy_bassoon/meta_rl/chunk_training.py) supports AD, DIT, IC-IQL and cloning controls on masked physical chunks; it has no DPT query-label path. Do not substitute a behavior action for a missing teacher query. Keep episode termination, padding and delivered-action masks separate. Resume requires the optimizer, sampler, RNG and learned state, not just exported actor weights.

For constructed RoboCasa queries, the exporter matches the trial and decision observation across branches. Supplying `--action-stats` adds the selected branch’s H50 × 32 normalized action target and a 50-control validity mask. The input is its sequence of delivered twelve-coordinate commands after binary gripper and mode mapping. The categorical-only format is still supported. [Constructed-query exporter](../src/fluffy_bassoon/meta_rl/constructed_queries.py).

## Environments and tests

The [agent simulator collector](../benchmarks/agent_sim/README.md) exposes one
persistent RoboCasa365 episode through the same MCP tools to Codex or Claude
Code. It records lossless control-rate observations, native commands and
outcomes, plus explicit pre-action subgoals and brief decision summaries.
Its exporter separates causal inputs from current action/context targets and
future outcome labels. These are robot-action demonstrations, not noise targets
for the existing `vla_history` trainer; a matching feature/action head is required.
Native simulation and agent success require separate runtime qualification.

Use `uv sync --locked` on allocated Linux for the shared Python environment. A quick learner check is:

```sh
uv run pytest -q tests/test_chunk_training.py tests/test_history_value.py tests/test_vla_history.py
```

Benchmark tests additionally need their qualified RLinf/OpenPI or RoboTwin packages, upstream configs and simulator assets. Run [native RLinf tests](../tests/test_rlinf_native_ppo.py) with `EMBODIED_PATH` pointing to the pinned RLinf checkout’s `examples/embodiment`; run the [RoboCasa preprocessing tests](../tests/test_robocasa_pi05_preprocessing.py) in its patched OpenPI environment. See [RoboTwin setup](robotwin/rl-baselines.md#runtime-setup) for its separate overlay and reference controller tests. A CPU pass qualifies software behavior; a GPU rollout requires an allocation-local probe and a real episode. Operational rules are in [AGENTS.md](../AGENTS.md).
