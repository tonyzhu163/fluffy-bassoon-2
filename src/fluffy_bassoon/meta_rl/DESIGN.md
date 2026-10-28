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


## Robot initialization boundary

The compact CMM is not a drop-in loader for a LycoReco supervised expert.
The legacy robot CTM uses different synchronization banks/readout dimensions,
neuron processing and initialization. Preserve it as a reference. Any robotic
CMM port needs explicit feature/action adapters and a matched supervised
warm-start against its controls; record which source tensors transfer and which
are fresh. A paper-based CMM and a robot CTM with an added memory module are
separate implementations and must not share an architecture identity.


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


## LIBERO gain-trial wrapper

`libero_trial.LiberoGainTrial` wraps the pinned LeRobot 0.6.1 `LiberoEnv` with
relative seven-dimensional actions, hard reference-state resets, 20 Hz control,
ten settling steps, two raw 256px cameras and its standard robot-state fields.
One positive hidden scalar multiplies XYZ commands within a trial; rotation and
gripper commands are unchanged, and effective XYZ commands are clipped to the
original bounds. This is a controller perturbation, not a friction/contact model.
The policy sees its issued command and observed motion. The gain-adjusted backend
command stays private because exposing both commands would reveal the gain.
During absorbing padding, feedback retains the last real command; ignored
proposals cannot write an external memory channel into future observations.

Each trial samples a gain and a distinct sequence of reference starts using
independent random streams. Reset seeds and reference indexes stay outside
observations/info. The constructor accepts an explicit initial-state pool;
training/evaluation manifests must declare their disjoint pools. The default
all-state pool is for integration qualification and is not an OOD split.
Both terminal physical observations and the next fresh start are returned at
inner boundaries, with reward and an attempt flag. The wrapped backend's info
is sanitized; task description remains visible through the standard task API.

Attempts have a fixed policy horizon. Early success or backend termination
enters an absorbing state without further simulator actions; success is latched
and paid once at the fixed boundary. The observation's absorbing flag identifies
padding. PPO zeroes actor/entropy terms at padded positions under the fixed-trial
mean; advantage normalization uses real actions. Critic fitting and recurrent
updates still span the full trial, and their forward counts include padding.
The collector reports real actions and reset settling actions separately.
`simulator_steps` excludes padding, and `settling_steps`
counts reset actions separately. Backend truncation or reset/step failure ends
qualification and requires a new trial; exact mid-trial recovery is unsupported.

`python -m fluffy_bassoon.meta_rl.libero_smoke` requires a clean source and an
allocated Linux GPU step with allocation-local EGL configuration. The launch
manifest must run the real UUID/isolation/CUDA probe first. The existing renderer
mapping and own-PID graphics guard verify the actual rendered device. The smoke
saves raw frames, reference-state hashes, issued commands, counters and a seeded
initial-state replay check. This does not qualify task success, frozen visual
features or a robot policy. The compact numeric PPO CLI remains separate until
a matched feature/action adapter and real policy rollout are qualified.


## Exploration evaluation

New learner runs report both greedy actions and seeded stochastic sampling,
each with normal state lifetime and a reset-between-attempts intervention.
Both use disjoint environment reset seeds and the same fixed policy-sampling
seed for paired conditions. A forked Torch RNG scope restores the caller's RNG
after every evaluation, including errors; evaluation cannot change subsequent
training samples. Raw observed reward remains the metric even with return
centering. Greedy-only historical pilots do not establish whether the sampled
policy learned an exploration strategy.

`python -m fluffy_bassoon.meta_rl.evaluate` evaluates complete compact CPU/F32
checkpoints without changing any tensor. It records both training and evaluator
source, checkpoint hash, policy seeds, interaction counts and per-trial arrays.
The saved encoding chooses raw legacy clocks or the declared fixed scaling.
When evaluation trial counts match a completed run, the greedy result must match
its saved reference before interpreting stochastic results. A changed trial count
or missing completed reference is recorded as an unchecked compatibility case.


## Common robot native-prefix interface

`native_prefix.py` preserves detached final-layer native K/V in owned BF16 CPU
frames, indexed by chronological trial observations. Every core receives the
same learned-query `NativeKVAttention` readout and normalized state8 token.
Current-frame and inner-boundary terminal-frame reads are distinct; absent
terminal reads are zero. Feedback contains last real command7, reward1,
attempt-end1, absorbing1 and declared-max-normalized public clock2. Hidden gain,
effective command, backend seeds and reference IDs are absent.

The frozen encoder is supplied separately from the trainable policy. The bank
rejects trainable or nonfinite prefix/state tensors and owns its CPU copies.
PPO minibatches select complete trial lanes and reconstruct all trainable reads
from stored native banks; projected trainable features are never cached.
Absorbing padding reuses the last physical frame. Real inner-boundary terminal
frames and fresh reset frames are both retained. Final unused observations are
not encoded. Physical actions, settling actions, policy forwards and prefix
frame/storage costs remain separate counters.

This interface reuses the existing actor likelihood, recurrent reset and PPO
implementation. It is a new shared frontend, not a compatible loader for the
legacy robot CTM expert. Native-bank unit qualification and real frozen-VLM /
LIBERO qualification are separate gates; only EVALUATION.md records completed
evidence.


## New supervised warm-start demonstration features

The meta-RL demonstration cache has its own explicit purpose
`meta_rl_raw_action_demonstrations_v1`, reusing the house checksums, schema-2
completion identity, frozen encoder and scalar PyAV frame reader. It contains
full final-layer native BF16 K/V, saved-normalized state8 and **raw** bounded
relative action7. It is not a replacement for the old normalized-action cache.
Dataset images already use the training orientation and receive no environment
flip. Match visible skills by instruction, not simulator/dataset task numbers.

Extraction selects complete matching episodes. The final selected episode IDs
form a fixed validation set; all other selected episodes are training data.
The manifest preserves both source and compact cache intervals. Every payload
checksum and frame/episode coverage must validate on loading. Missing retired
payloads are rejected regardless of manifest completion state. A failed or
interrupted extraction has no complete manifest. Demonstrations have no reward
field; supervised fitting must not fabricate terminal rewards.


## Supervised warm-start protocol

`warmstart.py` fits raw bounded action means over complete selected training
demonstrations with the same `make_robot_policy` frontend/core used online.
Each episode starts with fresh recurrent state; all within-episode actions
receive full-sequence gradients. Previous-action feedback is teacher-forced
from the preceding raw command. Reward, attempt-end, absorbing and outer-clock
fields remain zero; inner clock is the recorded control-step index divided by
220. Episodes exceeding this horizon are rejected. No future action enters
its own input and no demo-end reward is invented.

The objective divides each raw action error by the saved action.std + 1e-8,
then averages its square over frames and dimensions. The statistics file must
match the frozen encoder's saved processor checksum. Critic and Gaussian
log-standard-deviation stay frozen. Initial latent standard deviation is one
tenth the saved per-dimension action scale, clipped to the actor's exp([-5,2])
bounds. PPO may later make these trainable through the same policy
constructor. All models use the same NumPy episode-draw stream, fixed learning
rate 3e-4 and full-episode AdamW steps. The qualified PyTorch2.7.1 recipe
uses AdamW defaults: weight decay.01, betas(.9,.999), epsilon1e-8. These differ
from the explicitly zero-weight-decay compact PPO recipe. Held-out MSE weights frames equally
and reports each action dimension; it is not simulator success. Checkpoints
contain full optimizer/RNG state at complete episode boundaries and exact
source/cache/split identity. Equal-width warm-starts still have different
parameter/state budgets and cannot establish an architecture advantage.


## Read-only closed-loop development validation

`robot_evaluate.py` loads a complete supervised snapshot only after verifying
its declared checkpoint SHA, phase, raw-action interface and exact tensor
schema. It reconstructs the same frozen-policy frontend/core, verifies encoder
and processor identities, and applies raw bounded actions directly. No action
unnormalization is applied to the new raw-action actor. Both greedy and seeded
stochastic controls use identical physical reset seeds and fixed policy weights.

The initial gate is the visible spatial task at unmodified translation gain1,
one attempt and H220. Its development seed window is validation data, not an
untouched test set. Report each trial, real/settling/policy/encoder counts and
separate action-selection conditions. Raw action/feedback traces, physical
state traces and periodic camera frames support diagnosis. Checkpoint bytes,
policy tensors and the full encoder digest must remain unchanged. Actual
renderer and closed-loop qualification is distinct from CPU loader tests.


## Supervised-to-PPO transition and robot learner

`robot_train.py` verifies the source warm-start SHA, demonstration-cache identity,
visible instruction, frozen encoder/processor identity and demonstrated feedback
contract before robot PPO. It reuses the common collector, likelihoods, PPO loss
and complete checkpoint/RNG implementation. Meta mode carries state/returns
across fixed attempts; episode mode resets both at attempt boundaries.

SFT did not expose terminal views, reward, attempt-end, absorbing or outer-clock
inputs. Their input-projection columns are initialized to zero on entering PPO,
and the unused critic head is zeroed. On the demonstrated input domain this
preserves action means, likelihoods and entropy. New channels remain trainable,
with no added gates or memory architecture. It does not guarantee competent
control across new attempt boundaries; that is an online-learning gate.

The initial implementation uses one actual environment/trial lane, complete
trial gradients and checkpoints after every complete update. It defaults to
K4/H220, one PPO epoch, learning rate3e-5, gamma/GAE-lambda1, value weight.01,
latent entropy weight.01, raw rewards (no centering) and unnormalized advantages. These are qualification
settings, not a validated learning recipe or a matched-budget result. Full-trial
activation capacity and useful robot learning require separate actual checks.

Initialization happens before restoring a PPO resume snapshot, never after it.
Thus resumed learned feedback columns and critic values survive restoration.
The saved contract includes runtime versions, source, all initialization changes,
optimizer, inputs and environment settings. Resume requires the same contract
and a new output identity. Frozen VLM tensors stay outside the optimizer and
learner snapshot; their full digest is verified before and after training.
The original frozen assets and warm-start input remain protected dependencies.
Read-only evaluation accepts both supervised and robot-PPO snapshots, preserving
phase identity and learned tensors.


The single-trial robot learner disables batch advantage normalization. With a
zero initial critic and one terminal outcome, every action can have the same
return. Subtracting that single-trajectory mean would erase its first reward
gradient. The unnormalized option preserves the score-function gradient and
still zeros ignored-action terms with the fixed-trial denominator. Compact
batched runs retain their original normalization by default and record it
explicitly. This is an optimization correction, not an adaptation mechanism.


### Warm-start command-feedback control

`warmstart.py --without-previous-command` keeps the same complete episodes,
encoder, action targets, loss, optimizer, episode order and public inner clock.
It sets all previous-command fields to zero during training and validation, and
initializes their seven recurrent input-projection columns to zero. Those
columns therefore remain zero throughout SFT, so the restored policy can accept
its own issued commands without activating untrained random weights. Robot PPO
can subsequently train these columns normally. This control tests the
teacher-forcing shortcut hypothesis; it is not evidence of necessary memory or
an architecture advantage. Checkpoint feedback contracts distinguish the two
recipes and cannot be mixed on resume.


### Robot evaluation state protocol

Read-only robot evaluation follows the saved PPO mode: episode-trained policies
reset at inner-attempt boundaries; meta-trained policies carry state. Supervised
snapshots retain the existing carry protocol by default. The explicit
`--reset-between-attempts` intervention resets either kind for diagnosis and
cannot disable a trained episode-reset protocol. Configurations and summaries
record the actual reset choice, and traces contain the executed reset mask.
Unknown or missing PPO modes are rejected. Single-attempt evaluations are
unchanged; a state lesion remains distinct from a separately trained control.


### Bounded-action entropy control

Compact reaching can opt into `--bounded-action-entropy`. PPO estimates the
squashed-Gaussian differential entropy as the analytic latent entropy plus the
log-Jacobian at one fresh reparameterized current-policy sample per action.
The sample remains differentiable through its mean and scale. Reusing detached
behavior actions would remove the mean gradient and would not implement this
entropy objective. The saved optimization contract records `bounded_entropy`.

This is an optimization control, not a new memory mechanism. The default latent
entropy recipe, supervised fitting, rollout action sampling and deployment
interfaces remain unchanged. The control adds RNG draws during PPO replay;
full checkpoint RNG restoration must reproduce its next update exactly.
Reported `exploration_entropy` is a Monte Carlo estimate when enabled, with
sampling variance, and does not quantify independent training-seed uncertainty.


### Explicit motor warm-start for compact meta-RL

`train.py --motor-warmstart <snapshot> --motor-warmstart-sha256 <sha>` starts a
new phase from a complete ordinary episode-PPO checkpoint. It requires the same
task, attempts, horizon, observation/action convention, core configuration and
dtype. Actor, core and exploration scale transfer exactly; the value baseline
starts at zero, optimizer state is fresh, and parent RNG is not restored. The
new phase seed and counters remain separate. The saved initialization contract
includes parent SHA/source/iteration and all pretraining counters, so compare
full pretraining plus new-phase costs, not only fine-tuning interactions.

This supports one motor stage; chained initializers are rejected until their
cost accounting is extended. It is distinct from exact resume. For a resumed
warm-start phase, repeat the same parent identity and then restore the learned
PPO snapshot; initialization occurs first so resumed critic/optimizer/RNG state
survives. Across source revisions, review actor/core semantics as well as tensor
schemas before using an initializer; matching keys/shapes alone cannot prove
semantic compatibility. An ordinary-PPO control must use the identical
initializer, critic reset, fresh optimizer and post-initialization budget before
attributing a gap
to meta-training. A warm-start benefit is not an architectural contribution.


### Conventional relative-attention control

Compact `train.py --core transformer_xl` uses two independent Transformer-XL
layers, four heads, a64-token past-activation cache per layer, post-residual
LayerNorm and4x-width ReLU feed-forward blocks, with zero dropout. Relative
attention combines content and sinusoidal distance scores with two shared
trainable query biases. Cache each layer's **input** activations, mask unused
slots independently per lane and reset the complete cache/length at a trial
boundary or the episode-control reset. It is separate from the existing
weight-tied eight-token looped Transformer.

Reference: [Transformer-XL](https://arxiv.org/abs/1901.02860), with the authors'
[relative-attention implementation](https://github.com/kimiyoung/transformer-xl/blob/44781ed21dbaec88b280f74d9ae2877f52b492a5/pytorch/mem_transformer.py).
This is a vector-observation actor/critic adapter, not a language-model checkpoint
reproduction. Segments contain one observation. Full-trial PPO replay retains
cache gradients instead of the language-model implementation's detached segment
memories. Default PyTorch linear initialization and zero relative query biases
are declared optimization choices. A64-attempt bandit trial fits the available
history; longer contexts require separate qualification and cost measurement.

The compact CLI records layers/heads/cache length in its checkpoint contract and
counts actual retained bytes, including the per-lane int64 length. Ticks must be
one; layers are independently parameterized. Match parameter ceilings through a
declared width grid, then report the larger history state and measured compute
separately. Reused validation and a single seed remain exploratory. Numerical
attention/gradient/replay/recovery qualification and learning are separate gates.
The native-PaliGemma robot factory also exposes this core, with a fixed880-token
past-activation cache per layer, covering the default K4/H220 trial. At width64
it has308,303total parameters including the shared readout/actor/critic, and
450,568retained-state bytes per lane including length. The much larger state and
attention cost must be reported separately from parameter counts. The existing
robot-smoke default remains the original four cores; select the new control
explicitly. Shared SFT/no-command/PPO-input initialization and read-only loading
use its input projection. Full native visual/action/recovery and K4/H220 capacity
qualification are separate from the compact checks, and neither proves robot
base competence or successful adaptation. Longer trials need separate context
and capacity review.
