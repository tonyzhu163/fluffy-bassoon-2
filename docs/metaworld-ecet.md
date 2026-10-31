# ECET on goal-visible Meta-World ML10

`python -m fluffy_bassoon.meta_rl.metaworld_ecet` runs an ECET architecture and
its transition-minibatch PPO learner against the same Meta-World 3.1.1 / MuJoCo
3.3.0 ML10 split as the CTM pilot. This is an **adapted ECET port**, not an
unmodified reproduction or a Transformer-XL renamed as ECET. No training results
are implied by the implementation or the paper's reported scores.

## Source and dependencies

The [official implementation](https://github.com/machinelearningnuremberg/ECET/tree/aff709ae1b35406b16abdf62a607697fb0f19a8c)
is pinned to `aff709ae1b35406b16abdf62a607697fb0f19a8c`. The paper and the
[author's university page](https://ml.informatik.uni-freiburg.de/profile/shala/)
identify that repository. File paths and SHA256 hashes are in
`third_party/ecet/upstream.json`. The main references are `ecet_policy.py`,
`transformer_ep_s.py`, `rl2.py`, `rl2ppo.py`, `vpg.py`, `ppo.py` and
`gaussian_mlp_value_function.py` at that commit.

Upstream is a modified Garage tree that depends on old Gym, TensorFlow, Ray,
`mujoco-py`, and Meta-World commit
`04be337a12305e393c0caf0cbf5ec7755c7c8feb`. Installing it into the current V3
environment would replace the benchmark/runtime. Its custom no-LayerNorm layer
also lacks the current PyTorch TransformerEncoder forward signature. The local
port uses PyTorch's attention and encoder layers directly, retaining the
specified equations and parameterization without installing Garage. It needs
only the project's `metaworld` extra and existing PyTorch/NumPy dependencies.

Upstream `setup.py` declares MIT, but the pinned tree supplies no LICENSE,
COPYING or NOTICE text. We preserve that fact, do not invent copyright text,
and do not vendor original source. `third_party/ecet/fetch_reference.py`
downloads the exact hash-verified reference files to a fresh directory outside
this repository. Qualification can execute the actual upstream encoder function
and no-LayerNorm layer from those files. Fetched files are internal reference
artifacts and are not included in this package's distributions.

## Architecture and learning contract

Each token has 45 scalars: 39D state, previous executed action (4), previous
scaled dense reward (1), and physical-episode boundary (1). Public clocks and
task IDs are omitted. No state normalization is applied, following the default
upstream `normalize_wm=False` configuration. Actor and critic both receive this
same augmented current observation.

The IET receives a contiguous `T=5` transition window from each context episode.
It linearly projects tokens to width 32, adds sinusoidal within-window positions,
and applies two causal 4-head transformer blocks with ReLU feedforward width
128. IET blocks have residual connections and no LayerNorm. The final token
represents that episode. CET applies two transformer blocks to the episode
representations, with LayerNorm, no causal mask, and no episode positions. The
last output is the current-episode query. Its conditioning set is the current
episode and genuine completed episodes from the same trial, so unmasked CET
cannot see future episodes. Past-episode permutation invariance is qualified.

The actor concatenates CET output with an independent width-32 linear
projection of the actual current augmented observation, then uses a 64,64 tanh
MLP. Both stacks and projections use the official TFixup initialization rules.
Dropout is zero during both collection and replay. The independent critic is a
64,64 tanh Gaussian MLP on the current augmented observation, matching the
official default `share_network=False`. Width/depth/context sizes are recorded;
there is no imposed 250k parameter cap. The public runner's architecture defaults
are width 32, four heads, two layers per stack, feedforward 128 and T5/E5. Its
class-constructor defaults differ. The paper's Appendix A.3 Table 1 and A.6.5
instead specify E25/T5 for Meta-World. The common K5 comparison defaults to E5
capacity, retaining at most its four actual past episodes. The published E25
setting is recorded separately in every manifest. TFixup and learned actor/critic
standard deviations are explicitly enabled.

ECET uses an unsquashed Gaussian. The collector stores its latent sample and
scores that sample; it sends the clipped sample to the simulator and feeds back
the executed clipped action. This avoids incorrectly scoring the clipped
replacement under a continuous density. Entropy regularization concerns the
latent Gaussian. The CTM runner uses a tanh Gaussian, so action parameterization
is a declared difference.

The learner retains PPO's clipped actor objective, Gaussian-NLL critic,
separate AdamW optimizers (5e-5 each, weight decay .01), gamma .9, lambda .95,
ratio clip .2, entropy .02, gradient norm 5, ten epochs and transition
minibatches of 32. These optimization numbers are the pinned public example's
defaults, not a claim that its undocumented launch flags reproduce every paper
curve. Actor advantages use full-trial GAE and the independent critic fits
full-trial discounted Monte Carlo returns. All actor epochs precede all critic
epochs, as upstream does. There is no clipped value loss. Advantage centering
is off by default, as in the example's absent `--center_adv` flag; the explicit
local `--normalize-advantages` option uses standard-deviation normalization.

The public example overrides its meta-batch flag to 100 tasks for ML10, collects
25 H500 episodes per task, and its `RL2.train` processes two chunks of 50 workers.
`VPG.train_once` runs its update on each chunk and refreshes the old policy after
that update. Thus one chunk contains 625,000 transitions; ten complete actor
epochs with transition minibatches of 32 make 195,320 actor optimizer steps,
followed by the same number of critic steps. No data is dropped from the final
partial minibatch. The local common comparison collects 50 K5/H500 trials per
update: 125,000 transitions, ten actor passes (39,070 actor steps) and ten critic
passes (39,070 critic steps). Both configurations reuse every transition ten
times per network. The smaller collection batch increases update frequency and
changes the behavior-policy distribution; it is a declared adaptation rather
than an equivalent optimization schedule. Each run records the calculated
data reuse, transition count and optimizer-step count in its contract. The local shuffle uses the saved
PyTorch RNG instead of upstream's saved NumPy RNG; exact recovery concerns the
local implementation, not identical seeded trajectories across repositories.

Each decision samples one contiguous window from every retained past episode;
the current window ends at the current observation. Empty histories use zero
tokens with no padding mask, matching the official encoder: their IET sinusoidal
positions produce nonzero representations, and CET attends to these learned
empty-episode tokens. They are not neutral padding. The test suite explicitly
checks this behavior. An early current-episode prefix is left padded so its last token
and current-state bypass always describe the current observation. The selected
source indices are stored once in behavior collection. PPO replays exactly those
indices and recomputes both IET and CET with current weights. It does not resample
contexts for the importance ratio, detach stale episode embeddings, or apply
100-step recurrent TBPTT. Raw observed history plus integer source indices avoids
duplicating a full E-by-T observation tensor at every decision.
Window sampling has its own ephemeral NumPy Generator, derived from the trial's
reset seed vector. This preserves the uniform sampling distribution while making
the policy stream independent of simulator/global NumPy calls and serial versus
worker placement. Trial seeds already belong to the saved sampler state; no
unrecorded generator survives the complete-trial checkpoint boundary.

## Benchmark adaptations and upstream corrections

The comparison uses ten training families, independently generated disjoint
tasks from those same families for train-family evaluation, and the five held-out
families separately. ML10 has visible goal coordinates; a trial contains K5
episodes of H500 decisions. The family is fixed but the instance is resampled at
physical resets. Native dense reward is multiplied by .1 for PPO and feedback,
as for CTM. Terminal feedback survives physical resets; trial reset clears all
context. Episodes run to H500 even after success. See `docs/metaworld-ctm.md` for
the common sampler, seed, success latch and equal-family aggregation contract.

These are changes from upstream's Meta-World V2 environment, hidden-goal task
defaults, 25-episode-per-task training example, 100-task meta-batch, native reward
scale and reset feedback handling. The local K5/E5 default exposes at most four
genuine past episodes, and therefore is not a test of ECET's published long-trial
advantage. An explicit `--context-episodes 25` retains the published capacity
but, with K5, adds 20 empty episode slots even at the fifth episode; it changes
the unmasked context semantics and must be a separately labeled condition.
Longer `--attempts` / `--context-episodes` runs are separate contracts
and require the task's approved scientific/resource scope. Evaluation defaults
to K25 on ten lanes, independently in both evaluation splits; no test-family gradient
adaptation occurs. The training default is 80 rounds of 125,000 transitions,
or ten million training environment steps with one seed. Upstream's exploration-episode
plus 25-test-episode evaluator is not silently presented as the same protocol.

The pinned `_update_prev_observations` shifts its full-episode archive on **every
decision**, repeatedly inserting the partially collected current episode. This
does not implement the paper's distinct-episode context. The port uses actual
completed episodes. The pinned current-window buffer initially places data at
the start and selects the final zero token; the port left pads the prefix and
always supplies the actual current observation to the bypass. It includes the
last valid random-window start (`H-T`), which upstream's exclusive upper bound
omits. These fixes are explicit departures from literal buggy buffer behavior,
while retaining the paper's hierarchical inference algorithm and the official
encoder computation. Critic clipping is applied after backward here; upstream
calls its clip function before backward. An optional centered-advantage run
uses standard deviation; upstream divides by variance. None of these corrected
variants should be labeled a byte-identical upstream reproduction.

## Execution and qualification

Run all model construction, tests and simulator work on allocated Linux
resources, after the repository's GPU probe in the actual `srun` step. Existing
environment setup is `uv sync --extra metaworld --group dev`; do not replace a
qualified active runtime merely to install Garage. A typical new, clean,
commit-pinned run inside an allocation is:

```bash
python -m fluffy_bassoon.meta_rl.metaworld_ecet \
  --source-commit "$PINNED_COMMIT" --output-dir "$FRESH_OUTPUT_DIR" \
  --seed 1000 --num-envs 50 --attempts 5 --horizon 500 \
  --updates 80 --context-episodes 5 --context-transitions 5 \
  --ppo-epochs 10 --minibatch-size 32 --max-elapsed-seconds 3600
```

This command documents the runner; it does not authorize or claim a run. The
launcher must pass an approved time allocation and use a new output identity.
Before its first optimizer step, the runner reconstructs all sampled contexts
and checks behavior likelihood/value replay within 2e-5. Checkpoints contain
both optimizers, model, environment sampler and Python/NumPy/PyTorch/CUDA RNGs.
The saved source and contract must match exactly on resume. Signals stop between
complete updates. Failures preserve the preceding complete checkpoint.

For source parity, fetch outside the repository, then enable the optional test:

```bash
python third_party/ecet/fetch_reference.py "$FRESH_REFERENCE_DIR"
FLUFFY_ECET_UPSTREAM="$FRESH_REFERENCE_DIR" \
  pytest -q tests/test_metaworld_ecet.py
```

The test suite checks distinct episode windows, lane isolation, no future
access, terminal reward/boundary feedback, current observation from the first
decision, episode-set permutation invariance, gradients through both encoder
levels, Monte Carlo returns, frozen-policy likelihood replay and exact next
update after checkpoint recovery. With
`FLUFFY_METAWORLD_REAL_QUALIFICATION=1`, it also runs two real H500 ML10 episodes
in all ten training families and checks replay. Qualification reports belong
in run artifacts; implementation alone is not a passed qualification.
Set `FLUFFY_META_RL_CUDA_QUALIFICATION=1` for that test to place inference and
replay on the allocated GPU; otherwise it uses CPU within the allocation.

To measure the complete default 50-lane update without paying for unrelated
evaluation episodes, add `--qualify-one-update --updates 1` to the runner.
This mode retains all ten PPO passes and native minibatch/optimizer settings,
writes the same timing, data-reuse, replay-error and resumable-state evidence,
and labels its contract `one_update_throughput`. It performs no evaluation and
produces no learning curve. Its checkpoint cannot resume into the training
contract. The preceding GPU probe and scheduler accounting remain required.

The simulator defaults to serial execution (`--env-workers 0`). After qualifying
serial/worker parity, `--env-workers 8 --allocated-cpus 16` can distribute lane
groups across persistent CPU subprocesses for both collection and evaluation.
All reset/step batches dispatch through the common worker coordinator with an
ordered serial fallback. Worker count, allocation limit and process manifest are
execution metadata outside the scientific checkpoint contract; the policy,
sampled contexts, native minibatch size, ten passes and optimizer steps remain
the same. A failed partial simulator batch is closed and recovery starts from
the preceding complete trial checkpoint.

## Curves and accounting

`metrics.jsonl` uses the CTM curve schema: separate train-family and held-out
success, raw family/lane/episode outcomes and native returns, environment steps,
collection/update timings, cumulative active process GPU hours including
simulation and evaluation, and a companion training-only GPU-hour axis. A
checkpoint resume accumulates previous active runtime. The scheduler launcher
must separately report allocated startup/idle overhead; queue time is excluded.

The requested top-line figure is measured ML10 train-family success against
allocated GPU hours, using independent training seeds and uncertainty across
seeds when available. Held-out-family generalization and success by episode are
separate diagnostics. The final paired interventions clear cross-episode context
or remove reward feedback. The runner never imports paper curves as measured
results. Train-family success or one throughput update does not establish
held-out generalization or an in-context reinforcement learning algorithm.
The paper used 50 million Meta-World training steps and five seeds; a bounded
K5 pilot is not that experiment's training budget.
