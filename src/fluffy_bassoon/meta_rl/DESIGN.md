# Cross-attempt RL environment contract

Implementation status: compact environments, a paper-based vector-input CMM
core, full-trial PPO, atomic between-trial checkpoints and focused tests are
provided. Learner qualification and manipulation integration remain separate
milestones. See EVALUATION.md for the exact qualified source revisions. These tasks are sanity environments, not benchmarks
or evidence of a CTM/CMM advantage.

## Trial lifetime and feedback

One Gymnasium episode is a trial of `attempts` fixed-length inner attempts.
`reset()` samples the hidden setting once and initializes the first attempt.
Inner boundaries reset physical state, keep the hidden setting and return
`terminated=False`. The final boundary returns `terminated=True`. Attempt horizons
are part of the finite task, so they do not produce `truncated=True`; an external
collector time limit must handle truncation and bootstrapping separately.

Each observation includes:

| Field | Meaning |
| --- | --- |
| `state` | Physical state for the next decision, including a fresh start after an inner boundary |
| `transition_state` | Physical result of the preceding action, before any inner reset |
| `previous_action` | Executed action; one-hot for the bandit |
| `previous_reward` | Reward from that action, zero before the first action |
| `attempt_end` | Whether that action ended an attempt |
| `clock` | Zero-based current attempt index and elapsed steps in that attempt |

At reset, both state fields contain the initial state and feedback is zero.
At an inner boundary, terminal state and reward remain available in the same
observation as the next physical start. The policy must consume this feedback
once, retaining memory, before generating its next action. Terminal feedback is
never placed only in diagnostic `info`. At the final boundary both state fields
describe the last transition and the collector resets recurrent memory only
after processing/logging that terminal transition.

Hidden settings are private environment state and never enter observations or
`info`. Seeds belong in run metadata, not policy inputs. Inputs are validated
before state advances. Returned arrays do not alias internal physical state.

The PPO collector divides both clock coordinates by their declared public
maxima (using one for a zero maximum) before flattening observations. All cores
receive this same fixed encoding. Raw environment observations and feedback
values are unchanged. Run contracts record `clock_by_declared_max_v1`; older
raw-clock pilots remain distinct source/protocols and cannot be resumed into
this encoding. Scaling is a conditioning choice, not privileged task information.

## Tasks

`HiddenRewardBandit` has a uniformly sampled rewarded arm, one pull per attempt,
and a binary reward for selecting it. Every pull ends an attempt, regardless of
the reward, so termination does not independently reveal the rewarded choice.

`BernoulliBandit` samples independent uniform arm probabilities once per trial,
then draws a fresh Bernoulli outcome for each pull. Its default environment has
six arms and 32 attempts; the learner CLI requires these settings explicitly
(`--task bernoulli --arms 6 --attempts 32`). Reward is a noisy sample, not a
success label. Evaluation reports reward and pseudo-regret against the best
trial-fixed arm; private probabilities score evaluation only and never enter
policy inputs, returns or PPO targets. Training and evaluation share the same
prior but use disjoint reset seeds, so this is not an OOD test.

`NonstationaryBernoulliBandit` (`--task nonstationary`, default 64 attempts, six
arms) starts like `BernoulliBandit`. Before every pull after the first, with
probability `hazard` (default 0.05), it redraws all arm probabilities from the
same uniform prior. It records the probabilities behind each pull so that
evaluation pseudo-regret is scored against the best arm at that pull. A learner
that forgets stale evidence at the right rate should beat one that averages the
whole trial.

`DarkRoom` and `KeyToDoor` (`--task darkroom`, `--task keydoor`) follow the
dark gridworlds of Algorithm Distillation. Both use a 9×9 grid, five actions
(stay, up, down, left, right) with clipping at the walls, and a state holding
only the agent's cell scaled to [-1, 1]. Each attempt starts in the centre. The
hidden cells are sampled once per trial and persist across attempts. These two
tasks pay **dense** reward: every step can pay, not only the attempt boundary,
because a single end-of-attempt bit would test only one cell per attempt. In
Dark Room (default horizon 20) every step on the goal pays one. In Key-to-Door
(default horizon 50) reaching the key pays one, and reaching the door afterwards
pays one; each pays at most once per attempt. `info["is_success"]` at an attempt
end means "reached the goal" and "opened the door". Evaluation reports the
per-attempt return as `reward_by_attempt`.

`HiddenGainReach` has two-dimensional bounded actions and physical state
`[x, y, goal_x, goal_y]`. A hidden signed gain stays fixed across attempts.
Starts and goals are freshly sampled at each inner boundary. Position updates
by `step_size * gain * action`, clipped to the workspace `[-2, 2]`. Reward is
binary goal proximity at the fixed horizon; reaching early does not end an
attempt. The default gains are -1 and +1. This is an actuator-direction sanity
task, not yet the planned contact-dynamics manipulation suite.

The eight-step default permits probing and reaching within one attempt. A
four-step candidate reduces that slack, but needs a controller headroom check
before use. `python -m fluffy_bassoon.meta_rl.diagnostics` compares known-gain,
identified-gain carry and fresh-identification controllers on common reset
seeds. The latter deliberately forgets all prior evidence at every attempt;
it differs from a neural-state reset that still observes one-step feedback.
Bandit diagnostics compare random choice, reactive win-stay/lose-cycle,
Thompson sampling and a latent-informed oracle. None is a Bayes-optimality claim.
Raw per-trial arrays accompany aggregate curves. These checks establish useful
task headroom, not which architecture can learn an adaptation strategy.

The environment preserves attempted-action feedback when workspace clipping
limits physical motion. The future actor must score its sampled action with a
consistent bounded policy; environment validation does not implement likelihoods.

## Qualification and recovery

Run `PYTHONPATH=src uv run --with pytest python -m pytest
tests/test_meta_rl_environments.py -q` inside an allocated Linux resource with the
project environment. The tests cover Gymnasium behavior, seeded replay, independent
lanes, fixed boundaries, preserved terminal feedback, oracle reachability and
invalid inputs. They do not establish recurrent learning or useful adaptation
headroom against a strong policy without history.

The initial learner must checkpoint/restart at complete trial boundaries. Exact
mid-trial recovery requires preserving environment and RNG state as well as
learner/recurrent state and has not been implemented in these environments.

## Paper-based CMM port requirements

Source: [CMM, section 3.2 and appendix A](https://arxiv.org/html/2610.07907v1).
The port must distinguish raw sliding pre-activation history from its temporary
Transformer-updated version. Concatenate sink, long-term memory and raw history
as memory tokens; apply the paper's positional embeddings and Transformer;
compute neuron activations from transformed short-term history. Carry the
updated long-term matrix and the **raw** sliding history, not the transformed
history. Preserve the activated state and synchronization accumulators too.

Do not interpret the published stores as prescribed episode-reset groups.
Selective short-term resets are a later experiment. Record robot-interface
changes and qualify a small memory task before making reproduction claims.


`cmm.CMMCore` wraps the repository's single CMM implementation
([design](../../lerobot_policy_ctm/architectures/continuous_memory_machine/DESIGN.md)):
the pinned Sakana synapse and bias-free per-neuron maps (eq. 14). Its full state is activated neurons, raw
pre-activation history, normalized long-term tokens, and synchronization numerator
and denominator. The core returns synchronization features; it has no policy head
or frozen visual-language encoder. Initial activated state and history are zero
buffers. Default memory residuals are ungated; the paper's zero-initialized shared
residual scalar is available for the gated setting. Zero LTM tokens implements
the corresponding storage ablation.

Appendix A requirements include position only in Q/K, zero sink values in every
block, bias-free memory projections, width-preserving SiLU FFNs, normalized LTM,
and LayerNorm after the neuron GLU. Synchronization uses fixed unique ordered
neuron pairs, a linear exponential-weight denominator, and RMSNorm. This differs
from the legacy robot CTM's square-root normalization and dual synchronization
banks; comparisons must state those differences. Decay clipping follows the
repository's CTM convention; those details and unspecified numerical defaults are
port choices, not verified equivalence to unavailable upstream CMM source. The
parameter-count check and the few-shot regression calibration are described in
the core's design note; passing recurrence tests is not calibration.

The numerical tests compare an explicit attention equation, raw-history carry,
linear synchronization weights, lane independence, gradients across observations,
and model-state restoration. Run them with the environment tests before RL:

```bash
PYTHONPATH=src uv run --with pytest python -m pytest tests/test_meta_rl_environments.py tests/test_meta_rl_cmm.py -q
```


## Recurrent PPO and recovery

The initial learner replays complete chronological trials for each PPO minibatch;
minibatches select trial lanes, never isolated timesteps. Recurrent gradients cross
inner-attempt boundaries in meta mode. Gamma and GAE lambda are both one, so the
finite-trial objective gives equal weight to attempt rewards. Episode mode resets
state and return propagation at inner boundaries. Both modes receive identical
feedback fields. The two-arm bandit can be solved from immediate feedback, so its
reset-memory result is not evidence for persistent memory.

Categorical actions use their exact logits. Continuous actions use a diagonal
Gaussian followed by tanh; saved raw Gaussian samples permit stable likelihoods
including the Jacobian correction. The continuous exploration bonus is latent
Gaussian entropy, not the entropy of the bounded distribution. Dropout is disabled
during both collection and differentiable replay. The critic shares recurrent
features. CMM decay parameters are projected after optimizer steps; inference
only applies a functional constraint and never modifies parameters.

`python -m fluffy_bassoon.meta_rl.train` requires a clean pinned source and a Linux
Slurm step. It writes effective configuration, per-update metrics, compute counters,
deterministic evaluation on a disjoint seed range, and an atomic complete learner
checkpoint. SIGUSR1/TERM/INT request stopping after the current update and saving
at a trial boundary. A time cap uses the same boundary. Incomplete updates do not
become recovery points. Resume requires the identical source/training contract,
restores optimizer and all RNGs, and writes to a new output identity. Constant-LR
continuation may extend the target update count. Evaluation and failed-run costs
must also be included when comparing complete pipelines.

The compact driver defaults to CPU; CUDA runs require a one-GPU allocation and
the repository's allocation-local GPU probe before launch. No visual encoder,
robot action adapter or simulator episode is qualified by these compact runs.

`--reward-offset c` subtracts a fixed constant only before return/advantage
fitting; observations, sampled rewards, evaluation and interaction counts stay
raw. The critic predicts centered returns. The default is zero. With gamma one
and a fixed T-decision trial, the centered objective is exactly `J - T*c`, so
the maximizing policy is unchanged. A value baseline shifted by the known
remaining-horizon offset yields identical advantages; the analytic test checks
both whole-trial and fixed inner-attempt boundaries. This conditioning choice
must be declared and matched across models. It is not justified for an
action-dependent trial length counted only in physical simulator steps.

`--value-coef` records the shared critic's loss weight, default 0.5. Zero disables
critic fitting while retaining its action-independent baseline prediction; it
does not freeze the shared core or disable actor learning. This is an explicit
optimization ablation, not an architectural change. Match its value and tuning
budget before comparing memory models.


## CTM dynamics control

`--core ctm` uses the same synapses, shallow neuron GLU, zero initialization and
linear-weight/RMSNorm synchronization as the compact CMM, with zero memory-update
blocks and no long-term store. No sink, positions, long-term initialization or
memory-normalization parameters are trained in this condition. The equivalence
test copies shared tensors into a zero-gated CMM and checks complete recurrence
features/states. This is a controlled CTM variant, not the legacy pretrained
LycoReco robot expert or a claim to reproduce its original recipe.

`--core cmm --ltm-tokens 0` retains the memory Transformer and is the CMM-STM
storage ablation. These controls isolate different components and are not
interchangeable. Run manifests record both parameter count and logical retained
state tensor bytes per lane. Width equality does not match either of those
budgets, and logical state bytes are not peak allocated device memory.


## Reduced memory qualification

`python -m fluffy_bassoon.meta_rl.copy_task` implements the sample/query protocol
of CMM appendix D.1: uniformly sampled vectors are presented in order, followed
by zero-data query steps with a query flag; only query predictions incur MSE.
A shared 64-dimensional linear input adapter and a linear output head surround
the compact core. Sequence gradients cover both phases. The default reduced
task uses length eight, four values per item, width 64 and 16 LTM tokens. It
differs from the paper's Copy-100 size, memory, parameter budget and training
schedule and cannot establish reproduction parity or architecture superiority.

The CPU-only Slurm driver reports fixed-seed query MSE, a zero predictor and a
same-checkpoint reset-before-query lesion. Evaluation uses an independent fixed
random stream. Between-sequence checkpoints reuse the complete learner snapshot
format, including optimizer and data RNG, and support exact-source continuation
into a new output identity. Time caps and signals save after a complete update.
This isolates recurrence learning from PPO before a visual robot port.


## Existing looped Transformer control

`--core looped_transformer` adapts the house `PersistentTransformerExpert` to
the same vector observations. It carries eight learned-initialized memory
tokens, uses one output query, and repeats one shared decoder block for the
specified tick count. The original self-attention mask prevents memory tokens
from reading the output query; cross-attention consumes the current observation.
The redundant action readout is removed and the common actor/critic or copy
head uses the query feature. Full recurrent gradients and per-lane resets use
the same policy interface. This is the existing weight-tied looped Transformer,
not Transformer-XL or a long-context baseline qualification. Capacity and ticks
need separate reporting and matching before an architecture comparison.


## Exploration evaluation

New learner runs report both greedy actions and seeded stochastic sampling,
each with normal state lifetime and a reset-between-attempts intervention.
Both use disjoint environment reset seeds and the same fixed policy-sampling
seed for paired conditions. A forked Torch RNG scope restores the caller's RNG
after every evaluation, including errors; evaluation cannot change subsequent
training samples. Raw observed reward remains the metric even with return
centering. Greedy-only historical pilots do not establish whether the sampled
policy learned an exploration strategy.

`python -m fluffy_bassoon.meta_rl.evaluate` evaluates complete compact F32
checkpoints without changing any tensor. It records both training and evaluator
source, checkpoint hash, device, policy seeds, interaction counts and per-trial arrays.
The saved encoding chooses raw legacy clocks or the declared fixed scaling.
When evaluation device and trial counts match a completed run, the greedy result
must match its saved reference before interpreting stochastic results. Checkpoints
may also be evaluated on CPU after CUDA training; that cross-device case does not
claim exact reference matching. The optional `--reward-blind` condition zeros only
`previous_reward` in the policy input. Previous actions, environment transitions,
clocks and actual reward scoring stay intact. Reward-blind policies can therefore
still exploit any reward information independently exposed in state observations;
the six-arm bandit exposes no such state channel.

## Parameter-matched learner screen

`python -m fluffy_bassoon.meta_rl.screen` seals a source-pinned matrix before
execution. The seven configurations are the persistent CTM at T1/T4/T8, the
consolidated CMM at T1, Transformer-XL, GRU, and the house looped Transformer
at T4. The CTM uses bounded synchronization and decay once per observation;
both CTM and CMM have length-32 short-term traces, and CMM has 32 long-term
tokens. Transformer-XL retains the longest declared trial length, with two
independent layers. The looped Transformer retains eight memory tokens.

Each configuration uses the largest width divisible by four that fits the
declared total policy parameter cap, including actor and critic. The default
cap is 100,000 and the minimum accepted fraction is 85%. The comparison matches
parameters, environment data, full-trial gradient lifetime and PPO optimizer
schedule. Retained state bytes, tick count, wall time and FLOPs remain distinct
axes and must be reported, rather than interpreted as matched compute. The
default screen has two stationary six-arm Bernoulli tasks (K64 and K128), five
training seeds, 32 trial lanes, 300 PPO updates, four epochs and minibatches of
16 complete trials. Reward centering is .5 and value-loss weight is .01.

Planning constructs candidate models only inside allocated Linux resources.
A manifest SHA256, exact source and clean checkout are checked by every worker.
Workers own disjoint index shards, with one model at a time per selected GPU.
Every execution attempt receives a fresh directory. Launch and completion
receipts bind cell identity, manifest and checkpoint hashes, runtime allocation,
parameter/state counts, update timing and diagnostic artifacts. Signals and
time caps save the existing complete between-trial checkpoint; `run --resume`
restores into a new attempt directory under the same source and training contract.
The screening worker stops on an incomplete member so a partial result is never
treated as a completed seed. Recovery does not overwrite the old attempt.
CUDA summaries also retain peak allocated and reserved memory. `screen report`
reads completed receipts without constructing models and projects measured
update, evaluation and diagnostic costs to a declared update/seed budget. Missing
configuration groups are explicitly excluded from the projection.

Final diagnostics evaluate greedy actions and three independent policy-sampling
seeds, each under normal memory, reset between attempts, and reward blinding.
These are frozen-checkpoint interventions, not independently trained controls.
The screen is an architecture selection experiment; a trained episode-reset
control and a non-bandit long-memory task remain necessary before stronger
in-context learning or task-generalization claims.

The LIBERO robot learner sections (gain-trial wrapper, native-prefix interface,
supervised warm start, robot PPO transition) stayed in fluffy-bassoon with their
code; see [PROVENANCE.md](../../../PROVENANCE.md).
