# RL²-PPO on the common ML10 protocol

The runnable method is **RL²-PPO (garage adapted, GRU64)**, implemented in
`fluffy_bassoon.meta_rl.metaworld_rl2`. It is a new PyTorch implementation with
an explicit provenance contract. It is distinct from the existing common-head,
parameter-matched GRU control. Neither should be described as the original
RL² algorithm or an exact reproduction of published ML10 numbers.

## Reference and dependencies

[Duan et al., RL²](https://arxiv.org/abs/1611.02779) trains a recurrent policy
from observation, previous action, reward and terminal feedback, retaining its
state across episodes in a trial. Its outer optimizer is TRPO. The named
baseline here instead follows the maintained
[garage RL²-PPO ML10 example](https://github.com/rlworkgroup/garage/blob/2d594803636e341660cab0e81343abbe9a325353/src/garage/examples/tf/rl2_ppo_metaworld_ml10.py)
at commit `2d594803636e341660cab0e81343abbe9a325353`.
Garage is a reference implementation, not a claim that the original authors
released this exact ML10 experiment with the RL² paper.

The reference uses GRU64, a linear Gaussian-mean readout, constant learned
standard deviations and a fitted linear-feature value baseline. It uses ten
episodes per trial, ten PPO epochs, discount .99, GAE .95, likelihood clip .2,
uncentered advantages and stopped-gradient entropy added to rewards at .02.
Its optimizer defaults to Adam 1e-3. Our default retains these learner choices
except the explicitly listed adaptations below.

Reference implementation details:

- [Gaussian GRU model](https://github.com/rlworkgroup/garage/blob/2d594803636e341660cab0e81343abbe9a325353/src/garage/tf/models/gaussian_gru_model.py)
  and [policy](https://github.com/rlworkgroup/garage/blob/2d594803636e341660cab0e81343abbe9a325353/src/garage/tf/policies/gaussian_gru_policy.py).
- [LinearFeatureBaseline](https://github.com/rlworkgroup/garage/blob/2d594803636e341660cab0e81343abbe9a325353/src/garage/np/baselines/linear_feature_baseline.py),
  [RL2NPO](https://github.com/rlworkgroup/garage/blob/2d594803636e341660cab0e81343abbe9a325353/src/garage/tf/algos/_rl2npo.py),
  [NPO objective](https://github.com/rlworkgroup/garage/blob/2d594803636e341660cab0e81343abbe9a325353/src/garage/tf/algos/npo.py),
  and [optimizer](https://github.com/rlworkgroup/garage/blob/2d594803636e341660cab0e81343abbe9a325353/src/garage/tf/optimizers/first_order_optimizer.py).

Garage is [MIT licensed](https://github.com/rlworkgroup/garage/blob/2d594803636e341660cab0e81343abbe9a325353/LICENSE).
No garage/TensorFlow source is vendored and no additional runtime package is
required. Use the repository's `uv sync --extra metaworld --group dev`, pinned
Meta-World 3.1.1 and MuJoCo 3.3.0. The run records installed dependency versions
and this repository's exact clean commit independently of the reference pin.

## Algorithm contract

Each training trial has five episodes of 500 decisions. The training-family choice
is fixed during a trial and task instances are resampled at physical episode
resets, matching the [common ML10 contract](metaworld-ctm.md). All methods use
the same goal-visible 39-dimensional state, task pools and split hashes.

The policy receives the state, previous executed action, previous scaled reward,
boundary flag and normalized public clocks: 47 inputs in total. One GRU64
transition produces a hidden state; a direct linear readout gives the means
of a four-dimensional Gaussian with initially unit, learned state-independent
standard deviations. There is no observation normalization, current-state actor
bypass, learned initial state or shared neural critic. Glorot input, recurrent
and readout initialization and zero biases are used. GRU hidden memory resets
at trial start and persists over physical episode boundaries. Trial starts use
zero action, reward, boundary feedback and hidden state.

The simulator receives the Gaussian draw clipped to `[-1, 1]`. PPO stores and
scores the **preclip draw**, so the latent policy likelihood remains Gaussian;
it does not score the clipped command as if it were an unclipped sample. The
next input contains the actual clipped command. Evaluation clips the Gaussian
mean and uses the same deterministic action protocol as CTM.

Before PPO, fit a ridge linear value baseline to discounted full-trial returns
augmented by .02 times the current Gaussian entropy. Its features are clipped
observations and their squares, trial time divided by 100 and its second and
third powers, and a constant. Time is reconstructed from the public clock,
including across episode resets. Fit in float64 using regularizer 1e-5, with
five escalating attempts if needed. Coefficients, regularizer used and fit count
are checkpoint buffers. Fit failure aborts without replacing the preceding
complete learner checkpoint.

PPO uses full-trial GAE and complete recurrent gradients across all five
episodes. Current entropy is recomputed for each policy minibatch and stopped
before adding it to rewards. Advantages are uncentered. The objective contains
only clipped policy loss; the fitted value baseline has no gradient optimizer
or value loss. There is no gradient clipping. Adam uses lr 1e-3 and eps 1e-8.
Defaults are ten PPO epochs and minibatches of up to 32 complete trials; with
the common 50-lane collection this is two optimizer steps per epoch (32 and 18
trials). Each round collects 125,000 decisions, so 80 rounds give 10 million
training decisions. Every
collected transition receives the same loss weight. Optional cached-window
TBPTT, advantage centering, neural value loss and gradient clipping are rejected
by this named adapter instead of silently changing its scientific contract.

Default GRU64 has 21,960 trainable parameters and 256 hidden-state bytes per
lane. The fitted value baseline has 98 float64 coefficients; its buffers are
additional checkpoint state, not trainable policy weights. This reference-sized
method does not claim parameter matching with CTM. `--rl2-match-budget` selects
the largest multiple-of-four width within 85–100% of the declared cap and
records the changed width and parameter-matching flag. That variant must have
its own curve label. `--rl2-width` selects an explicit nonmatched width.

## Explicit adaptations

- The original RL² paper uses TRPO; this method uses PPO and targets continuous
  robot control. It cannot support conclusions about RL²-TRPO.
- Compared with the garage example, this comparison uses Meta-World v3,
  goal-visible state, five episodes, fixed horizons, per-episode instance
  resampling and dense native rewards scaled by .1. Family-level inference
  across resampled instances is a different trial distribution from one fixed
  MDP across all episodes.
- The common full run uses 50 training trials per round and 80 rounds. This
  increases the reference example's ten-trial meta batch. Evaluation defaults
  to 25 episodes per trial and ten lanes; training remains five episodes.
- Common public clocks are extra policy inputs. Feedback records executed
  clipped actions and preserves terminal reward/boundary information through
  physical resets. These are explicit common-protocol choices.
- The implementation uses PyTorch and float64 ridge normal equations. It does
  not assert seedwise numerical equivalence with garage's TensorFlow and NumPy
  implementation. The reference ridge routine uses least-squares on its
  regularized normal matrix; our solve is equivalent for nonsingular matrices.
- PPO performs the requested fixed epoch count. Garage's optimizer can also
  stop early when whole-batch loss changes by less than 1e-6.
- Evaluation, checkpointing and accounting follow the common runner, including
  deterministic all-episode success on fresh training-family and held-out tasks.
  This differs from the reference's evaluator and must remain visible in labels.

## Qualification and launch

All model computation and tests belong inside an allocated Linux step. Run the
GPU probe and common real-simulator qualification before training; the launcher
must record its Slurm cluster, job, hardware, CPU allocation and source commit.
An example payload within an existing allocation is:

```bash
uv run python 'scripts/[train]gpu_probe.py' --expected-count 1 --output "$OUTPUT/gpu-probe.json"
uv run pytest -q tests/test_metaworld_rl2.py tests/test_metaworld_pilot.py
uv run python -m fluffy_bassoon.meta_rl.metaworld_rl2 \
  --source-commit "$COMMIT" --output-dir "$OUTPUT/rl2-seed1000" \
  --seed 1000 --benchmark-seed 0 --num-envs 50 \
  --attempts 5 --horizon 500 --updates 80 \
  --ppo-epochs 10 --minibatch-trials 32 \
  --eval-every 5 --eval-attempts 25 --eval-lanes 10 \
  --max-elapsed-seconds "$MAX_ELAPSED_SECONDS" --device cuda --threads 4
```

Set `$COMMIT` to the clean deployed commit and `$OUTPUT` to a fresh output
identity on shared NVMe. Set `$MAX_ELAPSED_SECONDS` from the allocated job's
approved active-runtime budget, leaving time for final evaluation. The
bounded time limit stops between complete updates, so its final update and
evaluation can exceed the nominal time limit. This command is a configuration
example, not a record that training ran or a request for extra allocation.

The tests check Gaussian latent likelihood and action clipping, absence of an
actor state bypass, preservation of memory and terminal feedback, frozen
behavior replay, independent ridge-regression agreement, full-trial time and
credit assignment, stopped entropy gradients, baseline checkpoint buffers and
exact next-update recovery including optimizer, sampler and all RNG states.
Full-trial PPO uses the GRU's fused sequence operation with the same weights
as one-step collection. Tests compare its outputs and parameter gradients
against serial replay; episode-reset diagnostic replay uses the serial path.
The common runner additionally verifies behavior likelihoods before its first
optimizer step and records raw outcomes, family labels and native returns.

Plot only measured run outputs. Use `method.display_name` and actual widths,
optimization settings and task protocol in labels. The primary train-success
axis is fresh-instance training-family success against measured allocated GPU
hours, charging simulation and evaluation; training-only time is a companion.
Show held-out family success and per-episode adaptation separately. Aggregate
seeds only under the same scientific contract and label uncertainty only when
multiple real seeds support it. No success, generalization or efficiency result
is claimed by this implementation document.
