# Native RoboTwin Transformer AD/DPT full/static policy

The native modules are `robotwin_history`, `robotwin_history_train` and
`robotwin_history_corpus`. They retain2062 public features,1024 physical noise
coordinates, and all four native feedback/clock scalars. They do not use the
LIBERO +3-scalar/bound1 token adapter or model format.

## Data, validation and selection

Training consumes a checksum-pinned native corpus index with `partition=train`.
An independently recorded `phase=source_validation` bank is exported with
explicit `--partition validation --validation-iteration N` and the matching
immutable per-method iteration-N SOURCE snapshot. Exactly six source-task K4
validation trials are required; target/native/preflight and other stages are
excluded. Acting weights must match that frozen source snapshot, and SAC may
not update from validation feedback. A stage may still be evaluating targets
after its complete source-validation panel exists; this exporter does not
claim whole-stage completion.

Validation is fixed-policy data. Export `behavior_cloning` and `dpt`, never
pretend it contains an AD learning process. The AD student uses actual-action
BC validation as an imitation diagnostic; the DPT student uses labels from the
same pinned source teacher used in its training dataset. These losses do not
rank objectives or establish robot performance. Whole reset case identities
(task and actual K4 seed tuples), requested/actual seed provenance and initial
observation hashes are retained; overlapping cases or wrong partition roles
are rejected before fitting. Seed identities alone do not establish novel geometry.

The original source-only9c27e02 exporter remains qualified for conversion.
Fitting needs the explicit partition/case provenance emitted by the enhanced
exporter. Re-export the SAME immutable early source bank for those metadata;
no new robot data or motor inference is needed. Never silently mix the old
400-step bank with native600/500/400 limits.

## Model and matched recipe

The GTrXL is width128, two layers, four heads, gate bias2, memory capacity300.
Its input is3090: features, previous physical noise, previous reward, episode
boundary and clamped native episode/primitive clocks. Source-feature mean/std
are fitted once on training observations only, then saved as frozen buffers.
No validation or target normalization is fitted.

Full and static policies have the same architecture, initialization seed,
training data/labels, sampled windows and optimizer schedule. Static removes
previous noise/reward/boundary and clears recurrent state at each decision;
it retains the same current features and both observable clocks.

The supervised recipe reuses Gaussian physical-target NLL, AdamW with zero
weight decay, LR3e-4, batch8, gradient norm cap1 and source validation at fixed
intervals. Physical proposals are explicitly clipped to[-1.5,1.5] at deployment.
This is a declared supervised policy family, not PPO's squashed-action
likelihood or an automatic supervised-to-PPO initialization.

Default gradient windows are at most128 positions, drawn uniformly over
task histories and causal window endpoints/lengths. Each sampled window starts
with empty recurrent state; its recorded previous physical feedback remains
available. Memory capacity300 and gradient/visible window128 are distinct
settings. Labels and source policy/version IDs never become query inputs.
Full histories are grouped across actual same-task learning versions as
declared by the exporter. Validation windows are fixed across checkpoints.

Milestones25/100/500 are exports along ONE optimizer trajectory, not625 total
updates or new source interactions. `latest.pt` retains weights, optimizer,
sampler and Python/NumPy/Torch/CUDA RNG, fit-position and elapsed counters.
`--resume-sha256` resumes that exact output and contract. Wall limits are per
invocation; the execution owner's allocation/compute ledger bounds their sum.
No automatic retry or budget increase is launched by the module.

## Commands

After the source and validation exports have qualified on allocated Linux:

```sh
python -m fluffy_bassoon.meta_rl.robotwin_history_train \
  --train-index <source-export/index.json> --train-sha256 <index-SHA256> \
  --validation-index <source-validation-export/index.json> \
  --validation-sha256 <validation-index-SHA256> \
  --objective ad --mode full --seed 0 --output <new-model-output> \
  --updates 100 --milestones 25 100 --batch-size 8 --max-window 128 \
  --width 128 --layers 2 --heads 4 --context-length 300 \
  --learning-rate 3e-4 --eval-every 25 --validation-batches 4 \
  --device cuda:0 --threads 4 --max-seconds 240 \
  --source-commit <exact-clean-pin> --expected-hostname <allocated-host>
```

Run the separately initialized AD static, DPT full and DPT static arms with the
same declared seed, snapshots, budgets and optimizer settings. Execution owner
must reserve the total compute and run the allocation's real GPU/CUDA probe
before GPU fitting. The example is a concrete first100-update configuration,
not launch authorization or a claim that240seconds reaches100updates. Preserve
complete recovery and report any partial endpoint if the operational cap binds.
Source and validation collection costs are shared once across the four fits;
fit positions count repeated sampled presentations separately from unique data.

## Frozen deployment lifecycle

`load_policy` requires a portable native policy and SHA256. The model manifest
binds dataset/source/teacher hashes, motor/feature/noise/horizon interface,
normalization, architecture, mode and optimizer endpoint. No ephemeral memory
is exported. `FrozenNativeHistoryController` requires the exact runtime
interface, sets all weights to eval/no-grad, and uses a per-trial seeded
generator for reproducible Gaussian sampling followed by physical clipping.

1. `reset_trial(seed=...)` empties memory and previous feedback for a new trial.
2. `begin_episode(episode_index=0, episode_limit=...)` starts its first episode.
3. `act(features, deterministic=...)` returns physical noise shaped32x32.
4. Call the simulator, then `observe(actual_noise, reward, done,
   executed_steps=actual_valid_duration)` before another query.
5. After done, begin the next sequential episode. Full memory/feedback persist;
   episode clocks clamp at the K4 training range during K8 evaluation.

The wrapper rejects remapped actions, missing outcomes, illegal horizons and
out-of-order episode/trial transitions. `erase_each_episode` is an inference
lesion; `reward_blind` removes numeric reward only. Neither creates a separately
trained supervised episode-reset baseline. That later control remains absent
until explicitly implemented in the fitter and actually trained.

These modules make a native executable integration path after their allocation
checks. Offline NLL, a model file, or a successful wrapper call is not held-out
robot success, teacher expertise, sample efficiency or evidence of ICRL.
