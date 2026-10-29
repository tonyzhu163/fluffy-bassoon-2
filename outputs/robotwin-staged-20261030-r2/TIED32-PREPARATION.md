# Prospective RoboTwin32D tied-noise arm: implementation and fresh-data costs

2026-10-30 Asia/Shanghai. **Preparation only.** The current full1024D arm is
retained, its declared<=20%clip-fraction gate is retained, and its two-hour
request remains pending. This note authorizes no scientific-scope change,
allocation or use of the remaining28seconds. No tied32D arm is implemented.
Smaller-LR qualification of full1024D and a tied32D ablation are separate work.

## Exact current interface

RoboTwin backend noise_shape=(H32,32), flattened to1024policy coordinates.
SAC/PPO/reset share it; the motor's physical action output has14coordinates
per primitive control step. The measured0.0106534Gaussian KL sums all1024
latent coordinates per stored decision, then averages140decisions. It is not
per-coordinate KL. Native action horizon32 remains fixed.

LIBERO's current expand_noise() requires one32D coordinate and repeats it
across the native motor horizon; pi05_libero declares H10. Native reference
noise is independently sampled across the full horizon. Matching steering
dimensionality would clarify one cross-benchmark difference; motor/SFT,
physical control, task distribution, bounds and warmup still differ. Do not
call the present feature-conditioned full1024D SAC an exact DSRL reproduction.

## Minimal necessary changes for a correctly generated new arm

1. **Separate policy coordinates from motor noise.** Add an explicit immutable
   interface flag, e.g.full_horizon versus tied32. Keep motor_noise_shape=(32,32)
   and H32/action14 unchanged. For tied32 accept z.shape=(32,) and lift with
   repeat(z[None,:],32,axis=0). Default full mode keeps its existing behavior.
   Do not change the motor checkpoint, transforms, features or control stride.
   Preserve an explicit full-noise entry point for native Gaussian evaluation;
   otherwise changing env.noise_shape would accidentally turn native into tied.

2. **Actor/critic dimensions.** Use policy_action_dim=32 for all three tied-arm
   controllers. Frozen public features stay2062D. SAC actor outputs32 means/stds;
   Q inputs become2062+32=2094rather than3086. Persistent/reset GTrXL inputs are
   2062+32+4=2098rather than3090, with32-D actor output and32-D log_std. Keep
   width128/layers2/heads4/memory300, clocks, K4/K8/reset rules and objective
   otherwise unchanged. Initialize new models/optimizers/replay; do not load
   the full1024D input projections/heads or failed trained weights as a warm
   start. A common seed is reproducible but does not guarantee identical
   shared-layer initialization after dimensional changes consume different draws.

3. **Likelihood and KL.** Sample one32D pre-squash u, form z=L*tanh(u), and repeat
   z only at the motor boundary. PPO's joint log probability sums32terms; its
   old/new ratio must not multiply by H32 or treat repeated rows as1024
   independently sampled actions. Exact conditional Gaussian KL also sums32.
   The existing PPO convention omits the fixed32*log(L) density constant;
   it cancels in ratios. Either preserve that convention explicitly or record
   a new absolute-density convention; never mix old and new likelihood records.
   SAC uses its32D squashed-policy density and dimension-dependent default
   target entropy-32(instead of-1024). Keep all other SAC settings unless a
   separate change is declared. Requalify the new interface's optimizer on
   fresh correctly generated data; a full1024D LR receipt is not its admission.

4. **Bounds and references.** To isolate tying, retain RoboTwin L=1.5 initially
   for tied/full learned policies and the declared bounded-uniform SAC warmup.
   LIBERO's learned bound1.0 is a separate factor, not a silent bundled change.
   Retain native independent Gaussian(32,32) reference and add an unlearned
   Gaussian32-vector repeated across H32. Tying changes temporal correlations
   even with zero learned mean, so that reference distinguishes the lift itself
   from learning. Decode parity compares the lifted tensor against that SAME
   full tensor supplied directly, not against a different independent sample.

5. **Recording/recovery.** Save z32/u32, joint behavior log-prob/value/history,
   motor noise(32,32), lift name/version, exact motor commands/executed mask,
   rewards/termination/durations and source/target/task/trial/reset identity.
   Previous-noise recurrent feedback is z32, not an arbitrarily flattened lift.
   Save complete independent actor/Q/targets/alpha/optimizers/replay/sampler/RNG,
   pending PPO and coordinator state. Keep lossless RGB/proprio/next observations
   and immutable behavior snapshots under a new run identity. Interface and
   dimension belong in contracts, policy versions and checkpoint admission.

6. **Single campaign accounting.** Use one RoboTwin1M benchmark cap across all
   interfaces and methods, not two1M ledgers. Before any next source launch,
   freeze a combined campaign contract if both modes are to run. Arm IDs such
   as ppo_full1024/sac_tied32 must distinguish per-arm caps, corpus context and
   counters; current ledger cap names cannot contain'/'. The current driver
   contract binds one noise_shape, so supporting two modes requires separating
   the immutable benchmark ledger contract from per-run interface contracts or
   defining all six arms in one coordinator. Never mutate an active header or
   reset charges. Import prior13,026actions exactly once. If full-mode source
   collection begins first, a later combined plan needs an explicit migration
   preserving all existing charges and recovery, not a second fresh allowance.

## Why fresh SAC and PPO data are required

The old1024D tensor is generally different at every horizon row. Slicing its
first row or averaging rows then repeating changes the motor input, executed
actions and resulting transition/reward. SAC therefore cannot relabel the old
replay as tied actions. PPO's original joint1024D behavior likelihood likewise
is not a32D behavior likelihood; the tied support has a different dimension.
Keep old data for full-arm diagnostics only. A simulator rerun under a new lift
is new interaction and must be charged, even if the initial scene seed matches.

Collect fresh all-six-source K4 trials for SAC, persistent PPO and trained-reset
PPO, with current-policy PPO batches and separate source/validation identities.
Preserve outcomes of both successes and failures. No target task pretraining.
New source models start independently, with no full-arm trained-weight transfer.

## Fresh action and resource arithmetic

Bounds are executed primitive controls, not latent coordinates or optimizer
steps. A tied32D controller still executes up to32controls per chunk; reducing
policy dimension does not make the frozen H32 motor32times cheaper.

- SAC first500warmup decisions cost at most16,000own control actions; the first
  eligible501st decision costs at most16,032cumulative actions. Long400-step
  episodes give roughly15.4k actions for500chunks because episode-final chunks
  are partial. Keep warmup500,batch256,UTD20; no silent lowering to force updates.
- PPO's first8K4-trial batch costs at most12,800own actions/32episodes. Reset
  PPO needs its own fresh batch. Three-arm serial interleaving can require
  roughly45–48kaggregate actions before SAC's first update. At the qualified
  full-mode26.96actions/s this is about28–30minutes of collection alone.
  Tied-mode throughput, success/reset frequency and optimizer cost are unmeasured.
- A hypothetical tied-only128k-per-arm schedule with32k/64k/128k scales retains
  the current384,000source,86,400source-validation,115,200target and4,800native
  maxima. Add one tied-Gaussian episode per two targets per scale:2,400actions.
  Including prior13,026, its planning maximum is605,826actions. This is a fresh
  proposed schedule, not an instruction to replace the current arm.
- Running BOTH full128k three-scale schedules would cost1,196,226including
  prior actions once and the tied reference, exceeding1M. It is not authorized.
- One illustrative cap-feasible comparison is BOTH interfaces at32k/64k only:
  six learners×64k=384,000source; four interface/scale validations=115,200;
  target=153,600; conservatively duplicated native references=6,400;
  tied references=1,600; prior=13,026. **Total673,826**, leaving326,174for later
  decisions/retries. Do not choose an interface/checkpoint using target outcomes.

The last illustrative plan has660,800new actions, about6.81collection GPU-hours
at the old two-scene rate, before I/O/optimizers/startup/task variation. A two-hour
slice would be resumable partial work, not completion. Full-mode saved-data PPO
updates took about5seconds for140decisions/fiveepochs; that does not measure
fresh tied-mode SAC or PPO training. All compute remains conditional on explicit
resource approval; no new allocation has been launched for this note.
