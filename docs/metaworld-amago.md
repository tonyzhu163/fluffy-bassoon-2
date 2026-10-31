# Official AMAGO1 ML10 baseline

This integration uses the actual off-policy [AMAGO1 implementation](https://github.com/UT-Austin-RPL/amago/tree/514c1c2a471418aaf7829ba60ad0582b7764e643)
at `514c1c2a471418aaf7829ba60ad0582b7764e643` (2024-03-18), under its retained MIT
license. It directly invokes `amago.agent.Agent` and
`amago.learning.Experiment.train_step`, including the upstream loss reduction,
backward pass, clipping, AdamW update and target synchronization. It is not PPO.
The original implementation includes four critics (two sampled for TD targets),
seven discount heads, PopArt, a differentiable online actor objective, filtered
behavior cloning, and target actor/critic updates with tau 0.003.

The [ICLR 2024 paper](https://arxiv.org/abs/2310.09971) is the version cited by
[METTA](https://arxiv.org/html/2610.31656v1). METTA does not identify an AMAGO commit
or give an exact configuration for it. This source choice is therefore an
independent AMAGO1 baseline. It cannot establish exact reproduction of METTA's
AMAGO. The AMAGO2 multi-task classification losses and AMAGO3 updates are not
included. The upstream Git tag named `v2` is not synonymous with AMAGO1: it
already includes forthcoming MultiTaskAgent changes.

## Environment and algorithm contract

The adapter reuses `MetaWorldTrial`, benchmark pools and split hashes from
[`metaworld-ctm.md`](metaworld-ctm.md). Meta-World 3.1.1 and MuJoCo 3.3.0 provide
39-dimensional goal-visible state, ten training families, disjoint generated
train-family evaluation instances and the five held-out families. A trial uses
K=5 physical episodes of H=500 steps. The family stays fixed and the task
instance is resampled at physical resets. Episode success is latched and native
returns are retained. Each raw evaluation row saves family labels, per-lane
outcomes, episode-wise and family-wise success, and environment step counts.

AMAGO's observation contains state plus the two normalized public clocks. Its
standard seven-dimensional `rl2s` contains boundary bit, previous scaled reward,
normalized trial clock and previous executed action. No task ID enters policy
input. The boundary signal, terminal reward and executed action survive physical
resets, while memory clears at trial boundaries. Training uses the official
exploration wrapper and its Meta-World example's 2,000,000-step noise anneal per
lane. The initial previous action is zero to match our common trial contract;
upstream uses a -2 sentinel.

The environment and replay store native dense reward multiplied by 0.1. The
unchanged AMAGO1 Agent defaults to `reward_multiplier=10`, so its TD targets see
native reward units. This is explicit in the manifest and does differ from
PPO's training reward scale; PopArt and multi-discount normalization are part
of AMAGO1. The original input normalization is retained and updates during
gradient steps, unlike CTM's frozen collection/update normalization.

The original Meta-World example used an old v2 four-return Gym interface,
hidden goals and a fixed sampled task instance across shots. Its wrapper
cannot run the current simulator without adaptation. Its learner is preserved
unchanged here; the shared V3 environment protocol is a deliberate departure.

Defaults follow the original example and common CLI: Transformer width 256,
three layers, eight attention heads, full timestep FF encoder and actor/critic
heads; sequence length 2000, batch size 24, 1000 gradient updates per epoch,
AdamW lr 1e-4 and weight decay 1e-3, critic coefficient 10, gradient norm 1,
ten-epoch linear warmup, FP32 surrounding FlashAttention's internal BF16.
Parameters are measured and saved; these models do not share CTM's 250k cap.
Report the size difference with results.

Collection is changed from partial 1000-step epochs to complete 2500-step
trials so checkpoints never require serializing simulator state. Each update
collects ten complete balanced-family trials then performs the configured
number of off-policy gradient updates. Thus the update/data ratio differs
from the original twelve-actor example. The manifest records this ratio's
inputs; report it with curves. Uniform replay sampling selects a trial, then
a uniform contiguous window following upstream `RLData.random_slice` rules.
The original 2000-step context is shorter than K5H500. Explicitly selecting
`--max-seq-len 2500` provides full-trial context and changes the baseline
configuration; do not conflate those two curves.

## Isolated runtime and qualification

Prepare a dedicated project NVMe `uv` environment on allocated Linux resources.
Do not replace the CTM environment or an environment used by active runs. For
example, after the scheduler grants a GPU and a write/capacity probe passes:

```bash
uv venv --python 3.11 /project/nvme/envs/fb2-amago1
uv pip install --python /project/nvme/envs/fb2-amago1/bin/python \
  -r third_party/amago/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
uv pip install --python /project/nvme/envs/fb2-amago1/bin/python --no-deps -e .
uv pip install --python /project/nvme/envs/fb2-amago1/bin/python \
  ninja==1.11.1.4 packaging==25.0 setuptools==80.9.0
uv pip install --python /project/nvme/envs/fb2-amago1/bin/python \
  --no-build-isolation flash-attn==2.8.3
```

These pin the requested direct runtime dependencies, not every transitive
dependency. Save `uv pip freeze`, CUDA/compiler and driver versions, GPU model,
CPU/RAM allocation, and the FlashAttention wheel hash in run evidence before
launch. Hardware/package compatibility has not been established by static
inspection. FlashAttention2 supplies the efficient upstream KV cache. The
explicit `--attention vanilla` fallback uses upstream causal attention and
reconstructs context each step; it is much slower and has a different numeric
path. Label and measure it independently. Do not report its timing as Flash
timing. Avoid silently substituting FlashAttention3 or AMAGO3.

Run fake-environment recovery tests in an allocated Linux step first:

```bash
python -m pytest -q tests/test_metaworld_amago.py
FLUFFY_METAWORLD_REAL_QUALIFICATION=1 FLUFFY_AMAGO_ATTENTION=flash \
  python -m pytest -q tests/test_metaworld_amago.py
```

Tests cover real upstream method identity, reward/action/boundary alignment,
no task identity in observations, replay corruption rejection, retained replay
files, and exact next collection, sampled replay batch, optimizer update and
parameter/scheduler recovery after restoring a checkpoint. The opt-in real
test runs one K5H500 Meta-World trial and a genuine off-policy update. Admission
also requires an in-step GPU probe, Flash KV-cache/full-sequence action parity
on the allocated hardware, full default-shape memory/timing measurement, and
a bounded real ten-family training update before any learning run. Tests and
model computation must not run on macOS or a development/login shell.

Example execution inside the approved allocation, after committing and
deploying verified source:

```bash
python -m fluffy_bassoon.meta_rl.metaworld_amago \
  @third_party/amago/ml10-amago1.args \
  --source-commit EXACT_CLEAN_40_HEX_COMMIT --output-dir /nvme/outputs/UNIQUE_RUN_ID \
  --seed 1000 --updates 1000 --max-elapsed-seconds 3600
```

`--grad-updates`, `--batch-size`, context size and architecture overrides are
scientific configuration changes and are saved in `config.json`. Qualify the
configured shape rather than treating small smoke-test throughput as a final
capacity measurement. The parser permits reduced horizons for qualification;
those results must carry their actual K/H and cannot be called K5H500 curves.

## Longer evaluation trials and collection seam

`--eval-attempts` is independent of training K and defaults to five. For a
25-episode diagnostic, keep the training clock encoding: public attempt/time
clocks retain divisors `[4,500]` for K5H500, and the official rl2 trial clock
retains divisor 2500. Values may exceed one after the training horizon. Do not
renormalize the longer evaluation to 25 episodes, which would change all early
policy inputs. `collect_trials(..., clock_attempts=training_K)` implements the
shared encoding for both clocks. The tests establish identical first-training-
trial decision inputs and actions under paired seeds before the shorter
trial's terminal observation diverges from the longer trial's physical reset.

The original learned position embedding has training-horizon+1 rows: 2501 for
K5H500, not 10000. Extending it would change parameters and initialization. A
removable evaluation-only prehook therefore preserves all existing in-range
lookups and clamps out-of-range lookups to row 2500. Raw absolute timestep
indices remain intact in replay and Transformer calls, including indices up
to 12499 during 25-episode action selection. This is a declared positional
extrapolation convention, not a claim that row 2500 was trained to represent
later times. It adds no weights, changes no checkpoint keys and is removed
after evaluation. The original 2000-token memory limit still applies. Report
this convention alongside longer-trial results.

For future parallel simulation, `collect_trials` offers an optional
`batch_step(executed_actions)` callback. It receives ordered already-explored
NumPy actions of shape `[lanes,4]` and returns one ordered Gymnasium five-tuple
per lane. Policy inference, exploration order and feedback/replay assembly
remain in the driver. The default serial stepping path is unchanged, and a
parity test compares the callback path with serial stepping. This seam does
not itself enable a parallel backend or certify parallel-runtime equivalence.

The executable now optionally selects the common persistent subprocess backend
with `--env-workers N --allocated-cpus 16`. Its default remains zero, the serial
reference. Both training and evaluation factories use the selected backend;
collection calls common `batch_reset` and `batch_step` helpers, which retain
transparent serial fallback. Original exploration noise is applied in the
parent before dispatch, and worker results retain lane order. The worker count
must fit both training and evaluation lane counts and the allocation. Execution
configuration and worker manifests are logged outside the scientific checkpoint
contract. A fake-lane test compares stochastic actions, exploration feedback,
raw replay and outcome summaries between serial and worker lanes. Real Linux
worker trajectory parity must pass before enabling this option for runs.

Gaussian exploration retains the official wrapper and annealing law but now
uses a parent-owned NumPy RandomState initialized from the run seed. The adapter
temporarily routes the unchanged wrapper's NumPy calls through that stream and
restores the driver's ambient NumPy state afterward. This protects action noise
from simulator constructor/reset randomness and makes serial/worker execution
comparable. The stream is stored in every checkpoint; Python-random exploration
multipliers and replay-window draws remain in the already checkpointed Python
RNG. A test deliberately perturbs ambient NumPy state and verifies identical
exploration draws and no ambient-stream consumption. This is an explicit RNG
isolation fix, not a change to the exploration distribution or noise schedule.

## Matched ten-pass comparison schedule

The accepted one-seed comparison uses 50 lanes, K5H500, 80 complete collection
rounds and 10,000,000 new training interactions for each method. AMAGO has no
PPO epochs. Its explicitly adapted comparison schedule samples ten passes'
worth of replay transitions per new collection round, preserving its original
off-policy losses, batch24, context2000 and model size. This is selected with
`--replay-epochs 10` or
`@third_party/amago/ml10-amago1-sdpa-50env-10m-comparison.args`. The original
fixed-gradient default remains available and has not been changed silently.

Each round adds 125,000 new transitions. Ten replay passes target 1,250,000
sampled transition presentations; a default optimizer step samples
24 x 2000 = 48,000. The exact rational schedule floors cumulative targets at
each completed round, yielding 26 steps in most rounds and occasional 27-step
rounds. At 80 rounds it performs 2083 optimizer steps and samples 99,984,000
transitions against the 100,000,000 target, a shortfall of 16,000 (0.016%). The
rounding remainder does not accumulate unboundedly. This same calculation
continues after resuming. The driver counts actual valid replay transitions
from the batch's nonpadding state mask and rejects a mismatch with its declared
sample exposure. Logs save actual/target sampled transitions, achieved replay
passes and optimizer steps per round. A sampled transition is counted once,
independently of the seven gamma heads and four critic heads.

Replay samples windows with replacement from all retained history. Ten sampled
passes do not mean each newly collected transition is visited exactly ten
times, and they do not make AMAGO equivalent to ten PPO epochs. Preserve this
method distinction in titles and figures. Parameter counts, retained context,
reward normalization and loss structure still differ between methods.

For this schedule, the default `--warmup-epochs 10` explicitly means the first
ten complete collection rounds. The scheduler duration is 260 optimizer steps,
computed from the same cumulative exposure target. This retains a ten-round
warmup rather than accidentally using a 10,000-step warmup that exceeds the
entire comparison run. `--warmup-grad-updates` overrides that duration when
explicitly requested. This schedule and warmup are comparison adaptations,
not claims of original AMAGO hyperparameter fidelity.

The original upstream CLI uses twelve actors, 1000 environment steps per actor
per epoch, and 1000 optimizer steps, giving 1/12 optimizer updates per new
transition. Retaining 1000 updates with fifty complete K5H500 trials would give
1/125, only 9.6% of that original rate. An optional
`--optimizer-updates-per-transition 1/12` preserves the aggregate original
ratio using 10416,10417,10417-step round cycles. Its separate
`ml10-amago1-sdpa-50env-10m-original-ratio.args` preset uses 833,333 optimizer
steps over 10M interactions and an explicit 10,000-step original warmup. It is
not the primary matched-pass comparison and requires a separately measured
resource budget. Aggregate ratio matching still does not reproduce the
original partial-epoch collection/learning interleaving: this adapter waits
for full trials to allow complete recovery without serialized simulator state.

Use measured single-step time t from the default capacity test for an initial
compute projection: the matched comparison requires approximately
2083*t/3600 GPU-hours of optimization, while the original-ratio schedule
requires approximately 833333*t/3600. Add collection, replay I/O, evaluation,
checkpointing and scheduler overhead. This is a planning estimate, not a
learning curve or a guarantee of sustained throughput. In particular, do not
identify a complete-trial collection round with an original AMAGO epoch.

## Native PyTorch SDPA backend

`--attention torch-sdpa` is an optional engineering adapter for runtimes where
the external FlashAttention2 extension cannot be built or lacks a compatible
kernel. It keeps the vendored source unchanged. After constructing the official
Transformer, it replaces only each parameter-free `FlashAttention` primitive
with `TorchSDPAAttention`. All upstream projections, positional embeddings,
head scaling, normalization, residual blocks, dropout, losses, model dimensions,
context lengths, replay, exploration and optimizer settings remain the same.
The replacement adds no parameters or checkpoint state; state-dict schemas and
values are identical before and after installation.

The primitive follows upstream FlashAttention's BF16 QKV and KV-cache convention,
causal scaled dot-product attention and training-only attention dropout. Full
sequence training uses native PyTorch SDPA. One-step inference writes new keys
and values into the existing upstream cache and attends to its actual populated
length. It uses an explicit per-lane mask: ordinary `is_causal=True` on one query
would incorrectly attend only key zero. Unequal lane lengths zero unused NaN
cache slots before attention. Original cache shifting, trial resets and absolute
positions remain in the upstream trajectory encoder. Partial cache arguments,
out-of-bounds lengths, cached training and multi-token cached queries fail
closed. The primitive reads a maximum length once per layer to slice cache
storage; its overhead is part of measured collection time.

This preserves the intended attention calculation, not bitwise equality across
CUDA kernels. Native PyTorch selects a supported SDPA implementation; available
dispatch flags are recorded, and the exact kernel requires profiler evidence.
No `flash-attn` package, `nvcc` or CUDA extension build is required. Label the
backend as native SDPA in every manifest and curve. Do not call its throughput
FlashAttention2 throughput. No claim of compatibility with a particular GPU
follows from static inspection.

The separate `requirements-sdpa.txt` profile pins Torch 2.11.0+cu128 and NumPy
2.4.6, matching the supplied existing host runtime versions. It does not certify
the AMAGO adapter on that runtime. Install in its own NVMe uv environment or
reuse an explicitly qualified isolated environment while retaining the exact
resolved lock; do not upgrade the CTM environment in place. Core dependencies
use the same pins as the Flash profile, except these two explicit runtime
versions. A complete resolved environment freeze remains required.

```bash
uv pip install --python /project/nvme/envs/fb2-amago1-sdpa/bin/python \
  -r third_party/amago/requirements-sdpa.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
python -m fluffy_bassoon.meta_rl.metaworld_amago \
  @third_party/amago/ml10-amago1-sdpa.args \
  --source-commit EXACT_CLEAN_40_HEX_COMMIT --output-dir /nvme/outputs/UNIQUE_RUN_ID
```

Admission proceeds in separate allocated Linux steps. The default fake tests
now include native frozen projection/output checks against explicit attention
math after BF16 quantization, causality, gradients, unequal-lane cache resets
with unused NaN slots, full-agent cached/full parity, unchanged parameter state
schema, and exact next-update native-backend recovery. The real simulator smoke
test intentionally uses width64/layer1/context32/B1 to check boundaries and one
off-policy update; it never establishes default-shape capacity.

```bash
python -m pytest -q tests/test_metaworld_amago.py
FLUFFY_METAWORLD_REAL_QUALIFICATION=1 FLUFFY_AMAGO_ATTENTION=torch-sdpa \
  python -m pytest -q tests/test_metaworld_amago.py -k real_ml10
FLUFFY_AMAGO_DEFAULT_CAPACITY=1 FLUFFY_AMAGO_ATTENTION=torch-sdpa \
  FLUFFY_AMAGO_CAPACITY_OUTPUT=/nvme/outputs/QUALIFICATION_ID/amago-default-capacity.json \
  python -m pytest -q -s tests/test_metaworld_amago.py -k default_b24_l2000
```

The last test uses the actual default width256, three layers, eight heads,
2000 transition windows (2001 observation states), batch24, official optimizer
and off-policy loss. It first collects one real K5H500 trial using that default
model, then samples 24 independent replay windows with replacement and performs
one real backward/optimizer/target update. It records peak allocated/reserved
CUDA bytes, collection and optimizer time, parameters, GPU and runtime versions,
and losses. This checks default capacity and single-update throughput. Repeated
windows from one real trial do not constitute a ten-family training experiment,
and one update does not certify long-run stability or produce a learning curve.
Failing this gate must be reported; changes to model size, context, batch or
loss must never be silently used to make the default baseline pass.

## Borrowing a read-only qualified runtime

The selected AMAGO import path needs only five additional direct packages over
an existing Torch/NumPy/Gymnasium/Meta-World/MuJoCo runtime:
`gin-config==0.5.0`, `einops==0.8.1`, `wandb==0.19.11`, `tqdm==4.67.1`, and
`gym==0.26.2`. Qualification additionally uses `pytest==8.3.5`.
`requirements-addon.txt` records them. Matplotlib, other environment extras,
FlashAttention and CUDA compiler tools are unnecessary for native SDPA on this
code path. Resolve WandB's ordinary transitive dependencies in the isolated
overlay and save the resulting environment lock.

A new uv environment using `--system-site-packages` can borrow the interpreter's
system packages. It does not necessarily inherit another virtual environment's
site-packages. When the qualified runtime itself is a venv, explicitly expose
its site-packages read-only to the new overlay (for example through a `.pth`
file inside the new environment), then verify actual imported module paths and
versions. Keep overlay packages ahead of the borrowed path. Never point an
installer at the borrowed environment, and constrain/verify NumPy and Torch so
ordinary dependency resolution cannot silently replace the qualified runtime.
Record both the overlay and borrowed package paths in allocation evidence.

The default capacity gate intentionally measures one actual default-shape
backward/optimizer step rather than estimating it from the tiny test. Static
shape arithmetic explains its resource needs: one FP32 critic hidden tensor
of shape 24 x 2001 x 7 x 4 x 256 is about 1.28 GiB, and several coexist in the original
loss graph. A math SDPA attention-score tensor 24 x 8 x 2001 x 2001 in FP32 alone is
about 2.86 GiB per layer; fused native kernels can avoid materializing it.
These are individual tensor sizes, not a peak memory prediction. No wall-time
or 96 GiB fit guarantee follows without the allocated-device measurement. The
qualification includes benchmark creation, one 2500-decision real trial, and
one B24/L2000 update; preserve total launcher wall time in addition to its
separate collection/update timing fields.

## Recovery, cost and results

Every complete update atomically writes `latest.pt` with all online and target
weights, normalization/PopArt buffers, optimizer, learning-rate scheduler,
scaler, Python/NumPy/Torch/CUDA RNG, seeded task sampler, exploration progress,
and the active replay index. Replay stores immutable hashed NPZ trials. FIFO
retention removes entries from the live sampling index but retains old files,
so older checkpoints remain recoverable. Budget disk capacity for the full
run, then garbage collect only after identifying all retained checkpoints.
No pickle trajectories are read from external sources.

Resume into a fresh output ID using `--resume /old/run/latest.pt`. Keep the
referenced replay paths available and unchanged; their hashes are checked
before restoring. Resumption requires identical source and scientific
contract. A partial failed update leaves the preceding complete checkpoint;
`failure.json` records attempted elapsed time separately. The upstream
experiment's original checkpoint path is not used because it omits replay,
scheduler, RNG and environment-sampling recovery state.

`metrics.jsonl` follows the CTM plotting schema: `evaluation.train_eval.normal`
is the train-family success curve; `evaluation.test.normal` is held-out family
generalization. The x-axis `gpu_hours` charges all active process wall time
on one allocated GPU, including simulation, replay I/O, evaluation and
checkpointing; `training_gpu_hours` is a companion excluding evaluation.
Collection and optimization seconds are saved separately. Queue time is
excluded; the scheduler launcher must additionally record startup/idle
allocation overhead. Resumes accumulate saved runtime; failed discarded work
must be added from failure/launcher receipts when reporting total resource
cost. `rollout_success` is exploratory training collection success, distinct
from deterministic train-family evaluation success.

Final paired-seed diagnostics include normal memory, memory reset at physical
episode boundaries, and reward-blind feedback. These assess history use and
do not alone establish reward-driven in-context learning. Never synthesize
learning curves or reuse METTA values as our measurements. Report seed counts
and uncertainty only from actual independent runs. This document specifies an
integration, not a verified learning result; validation outcomes belong in
dated output evidence and verified evaluation records.
