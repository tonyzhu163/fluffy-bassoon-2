# Agent demonstration collection

One stdio MCP server for **Claude Code and Codex**, with a persistent RoboCasa365
atomic-task simulator. Models do not run inside the server. Each client uses its
own model/authentication and sees the same robot tools. The current backend uses
RoboCasa's native Human300 12-coordinate delta-action interface. RoboTwin and a
frozen-policy proposal tool are not implemented by this collector.

## Setup

Use an allocated Linux GPU and the repository's qualified RoboCasa365 runtime.
Create a project-specific environment overlay; do not upgrade an environment
used by active jobs. Install this directory's `requirements.txt` with
`uv pip install --python <overlay-python> -r <requirements-path>` and expose this
checkout's `src` on `PYTHONPATH`. This avoids reinstalling Torch/OpenPI just to
serve MCP. CPU contract tests additionally need `pytest`.

Run the allocation-local GPU probe as required by `AGENTS.md`. Preserve scheduler
visibility and bind EGL rendering to the probed GPU. Set `MUJOCO_GL=egl`,
`PYOPENGL_PLATFORM=egl` and the correct `MUJOCO_EGL_DEVICE_ID`. Model inference
is performed by the agent client; the simulator itself needs no VLA checkpoint.
Do not run the native backend on the Mac. The source and simulator revisions,
controller and assets must be qualified together before collecting a corpus.

The following commands run **on the allocated Linux host** where the selected
Python already imports the native simulator. Replace the paths and commit.
Both commands register the same server implementation, with separate outputs.

```bash
SIM_PYTHON=/absolute/path/to/overlay/bin/python
SIM_SOURCE=/absolute/path/to/verified/checkout
SIM_OUTPUT=/absolute/path/to/project/outputs/new-collection-id
SIM_COMMIT=<verified-source-commit>

codex mcp add robot-sim --env PYTHONPATH="$SIM_SOURCE/src" -- \
  "$SIM_PYTHON" -m fluffy_bassoon.agent_sim \
  --output "$SIM_OUTPUT/codex" --client codex --model gpt-6-astra \
  --source-commit "$SIM_COMMIT" --max-controls 600 --max-chunk 50 --feedback hidden

claude mcp add --transport stdio robot-sim --env PYTHONPATH="$SIM_SOURCE/src" -- \
  "$SIM_PYTHON" -m fluffy_bassoon.agent_sim \
  --output "$SIM_OUTPUT/claude" --client claude --model <selected-claude-model> \
  --source-commit "$SIM_COMMIT" --max-controls 600 --max-chunk 50 --feedback hidden
```

600 is an example collection budget, not a new benchmark horizon. Set it to the
declared task contract. `sim_interface` reports the native registry horizon and
the configured limit; do not silently multiply either. When agents run on a
different machine, use an existing authenticated SSH route to launch this same
stdio command inside the allocation (`ssh -T`, no pseudo-terminal); no public
HTTP listener is required. Preserve any existing simulator `PYTHONPATH` entries.

The output root must be fresh. Restarting a client with the same root deliberately
fails, preserving partial episodes. Use a new output root after interruption.
Select the actual model in the client; `--model` only records your declaration.
No global client configuration is changed by repository code.

## Collection prompt

> Use only the robot-sim tools to operate the robot. Read sim_interface, then
> sim_reset task OpenDrawer, seed 7, episode_id attempt-0001. Observe the three
> camera views and robot state. Before each short action sequence, give a
> subgoal, a concise evidence-based decision summary (one or two sentences),
> and the expected visible outcome in sim_execute. Include the latest
> observation_control. Inspect the returned views, revise your approach when
> needed, and finish with sim_finish and your own success claim. Do not read
> the collection files or simulator internals. Preserve unsuccessful attempts.

The fields are explicit public explanations, not a promise to expose hidden
chain of thought. They can contain observations, uncertainty and recovery
lessons. They are collected **before** dispatch, so they cannot be rewritten
after seeing that action's result. Final reflection is saved separately.

`--feedback hidden` records native rewards/success privately and returns only
robot observations and execution status. This matches demonstration collection
without oracle feedback. `--feedback native` additionally returns observed
native feedback, for reward-bearing interaction histories. Neither mode uses
learned rewards. Hidden mode does not stop when success first becomes true.

The server owns one episode at a time. Each action is a 12-vector in the documented
native range; gripper and mode are canonicalized at the native 0.5 threshold.
Unknown shapes, nonfinite values, out-of-range commands, stale observations and
chunks exceeding the remaining allowance fail before dispatch. The backend
records every delivered control, including an early-terminal prefix. The robot
stays still while the agent reasons. This is not real-time control.

## Training data

Each episode contains:

- `events.jsonl`: chronological reset, pre-action context, requested/canonical
  commands, dispatches, control-rate outcomes, visible tool responses and final
  agent reflection. A dispatch without a transition is uncertain, not training
  evidence of an executed action.
- `observations/<control>/`: lossless camera PNGs and public proprioception,
  instruction and simulation timestamp, including the initial and final states.
- `private/`: simulator states/object poses; the initial record also contains
  model XML and episode metadata for offline analysis and replay qualification.
- `result.json`: native return/success separately from the agent's claim.
  This file is absent for interrupted or failed episodes.

Export a completed episode on Linux:

```bash
PYTHONPATH="$SIM_SOURCE/src" "$SIM_PYTHON" -m fluffy_bassoon.agent_sim.export \
  "$SIM_OUTPUT/codex/attempt-0001" "$SIM_OUTPUT/export-attempt-0001"
```

`examples.jsonl` separates **input**, **target**, **outcome**, and **provenance**.
Inputs contain the current observation and earlier visible decision/result
pairs; targets contain the current public context and actions; outcomes contain
native reward/success labels. Never feed the whole example to a student as its
input. `--history-decisions 0` produces a no-history control. Final reflections
and future outcomes do not enter earlier inputs. The exporter currently builds
within-episode histories; it does not reconstruct the client's whole chat or
automatically carry context across resets.

`trajectory.npz` contains T actions/rewards/terminal flags/decision IDs and T+1
states/timestamps. `manifest.json` records state ordering and lossless image
references relative to the original session root. Retain that root with the
export. This is suitable for action imitation and explicit-context supervision.
The older `vla_history` learner predicts motor **noise**: these direct robot
actions must not be relabeled as noise or DPT query labels. Frozen vision feature
extraction and a matching action/context head are a separate training integration.

Keep all failed and successful attempts, and split by task/scene/session before
constructing support/query data. An explanatory sentence is an agent assertion,
not a verified causal account. Token usage and the exact client conversation are
not available through ordinary MCP; session metadata leaves usage null. Join
client telemetry separately if those are required. Simulation and wall time are
recorded separately.

For another simulator, implement `core.Backend` and launch with the trusted local
`--backend-factory module:callable` option. Keep its observation/action contract
and native feedback explicit. Backend factories are operator configuration,
never tool arguments chosen by the model.

## Verification

On allocated Linux, with `src` on `PYTHONPATH`:

```bash
python -m pytest -q tests/test_agent_sim.py
```

These tests protect action/observation alignment, causality, native-feedback
visibility, uncertain-dispatch handling and real MCP stdio/image transport.
Their simulator is a test fixture, so passing them does not qualify native
RoboCasa rendering, manipulation success, or a particular Claude/Codex model.

Connection references: [Codex MCP](https://learn.chatgpt.com/docs/extend/mcp),
[Claude Code MCP](https://code.claude.com/docs/en/mcp).
